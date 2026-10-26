"""One configuration, every case, through the real agent.

The entry point is ``src.agent.stream_agent`` - the function the chat pane
calls - with the lot "clicked" first through ``src.utils.state``, exactly as
``app.py`` does when the user clicks one. The harness adds two things around
it and changes nothing inside it:

* a LangChain callback on the model, attached through ``build_llm``'s
  ``callbacks`` argument, which is how usage, latency and the served model
  are read off every call and how the budget is enforced before each one;
* the turn's events, read back into a trace: the plan, each tool call with
  its arguments and its result, the answer, and the citation ledger - the
  passages the model was shown, in the order it was shown them, which is
  what retrieval recall is scored on.

Cases run one at a time. The agent keeps the selected lot and the citation
ledger in module state, as the single-user Streamlit process always has, so
two cases in flight would read each other's. Each case is a fresh thread
(``thread_id``) so nothing carries over between questions.

Output follows the layout the Claude eval guides read: ``results.jsonl`` (one
row per case, written as it completes), ``traces/<id>_rep<k>.json``, and
``errors.jsonl`` for attempts that produced nothing scorable. Re-running
skips the (case, rep) pairs already in ``results.jsonl``.
"""

from __future__ import annotations

import functools
import json
import logging
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import UTC, datetime
from pathlib import Path

from eval import judge as judge_mod
from eval import metrics
from eval.configs import RagConfig
from eval.golden import Case, fingerprint
from eval.pricing import rate_for
from eval.spend import SpendMeter, UsageRecorder

logger = logging.getLogger(__name__)

MAX_TOKENS = 4096


def deterministic_grade(answer: str, expect: dict, retrieved: list[dict]) -> dict:
    """Every free check on one answer, from what the row stores.

    Kept apart from the run so that `regrade` can recompute it from
    ``results.jsonl`` when a check changes, without paying for the turn
    again. Retrieval columns exist only when the turn retrieved something:
    a turn answered off the grid values has no ranking to score, and a zero
    there would read as a miss.
    """
    hits = [{"chunk_id": r.get("chunk_id"), "doc_id": r.get("doc_id"), "chunk_text": r.get("text", ""),
             "source_table": r.get("source_table"), "scope": r.get("scope")} for r in retrieved]
    grade: dict = {"used_retrieval": 1.0 if hits else 0.0}
    if hits:
        grade.update(metrics.retrieval_scores(hits, expect, total_relevant=expect.get("relevant_chunks")))
    # What was retrieved *for the lot*: a borough-wide search returning some
    # other site's resolution is not the agent finding one here.
    placed = [h for h in hits if (h.get("scope") or "") in ("lot", "near")]
    grade.update(metrics.answer_checks(
        answer, expect,
        cited_docs=[h["doc_id"] for h in placed],
        cited_sources=[str(h.get("source_table") or "") for h in placed],
    ))
    grade["citations_valid"] = 0.0 if ("does not correspond to any passage" in answer or "do not correspond to any passage" in answer) else 1.0
    grade["answered"] = 0.0 if (not answer.strip() or answer.startswith("⚠️") or answer.startswith("I could not produce")) else 1.0
    return grade


def judge_columns(grade: dict, judged: dict, expect: dict) -> None:
    """Fold a judge verdict into the grade dict, in place."""
    verdict = judged.get("verdict") or {}
    if judged.get("faithful") is None:
        return
    grade["faithful"] = judged["faithful"]
    grade["judge_correct"] = 1.0 if (
        verdict.get("answers_question") and verdict.get("matches_reference") and not verdict.get("unsupported_claims")
    ) else 0.0
    if expect.get("no_change"):
        grade["no_change_per_judge"] = 0.0 if verdict.get("asserts_past_change") else 1.0
    if expect.get("nothing"):
        grade["declines_per_judge"] = 1.0 if verdict.get("declines") else 0.0


def _context_from_trace(trace_path: Path) -> str:
    if not trace_path.exists():
        return ""
    turns = json.loads(trace_path.read_text(encoding="utf-8"))
    return "\n\n".join(f"[{t.get('name', 'tool')}]\n{t.get('content', '')}" for t in turns if t.get("role") == "tool_result")


def regrade(results_path: Path, *, rejudge: bool = False, meter: SpendMeter | None = None,
            judge_model: str = judge_mod.DEFAULT_JUDGE_MODEL, progress=print) -> int:
    """Recompute the deterministic columns of every row in place.

    Judge columns are kept unless ``rejudge``, which runs the judge again on
    the stored answer with the context rebuilt from the row's trace - the
    tool results the model saw - and charges the meter for it.
    """
    if not results_path.exists():
        return 0
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    traces = results_path.parent / "traces"
    # The fixture's labels, not the row's copy of them: a label corrected
    # after the run is what a regrade is for.
    from eval.golden import FIXTURE, load  # noqa: PLC0415

    current = {c.id: c.expect for c in load()} if FIXTURE.exists() else {}
    for row in rows:
        meta = row.get("meta") or {}
        old = row.get("grade") or {}
        if row["prompt_id"] in current:
            meta["expect"] = current[row["prompt_id"]]
            row["meta"] = meta
        expect = meta.get("expect") or {}
        new = deterministic_grade(meta.get("answer", ""), expect, meta.get("retrieved") or [])
        for key in ("faithful", "judge_correct", "no_change_per_judge", "declines_per_judge"):
            if key in old:
                new[key] = old[key]
        if rejudge and meter is not None and meta.get("answer", "").strip():
            try:
                judged = judge_mod.grade(
                    question=row["prompt"], answer=meta["answer"],
                    context=_context_from_trace(traces / f"{row['prompt_id']}_rep{row.get('rep', 0)}.json"),
                    expect=expect, meter=meter, model=judge_model, case_id=row["prompt_id"],
                )
                for key in ("faithful", "judge_correct", "no_change_per_judge", "declines_per_judge"):
                    new.pop(key, None)
                judge_columns(new, judged, expect)
                row["explanation"] = {"judge": (judged.get("verdict") or {}).get("explanation", "")}
                row["judge_model"] = judged.get("judge_model")
                row["judge_usage"] = judged.get("judge_usage")
                row["judge_cost_usd"] = judged.get("judge_usd")
                meta["judge"] = judged.get("verdict") or judged
                progress(f"  {row['prompt_id']:<40} judge: correct={new.get('judge_correct')} faithful={new.get('faithful')}")
            except Exception as exc:  # noqa: BLE001 - keep the old verdict
                progress(f"  {row['prompt_id']:<40} judge failed: {type(exc).__name__}: {exc}")
        row["grade"] = new
    results_path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def _done(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            done.add((row["prompt_id"], int(row.get("rep", 0))))
    return done


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


class Harness:
    """The agent, wired for measurement, for one configuration in one process."""

    def __init__(self, config: RagConfig, meter: SpendMeter, *, out_dir: Path, judge: bool = True,
                 judge_model: str = judge_mod.DEFAULT_JUDGE_MODEL, timeout_s: float = 300.0):
        self.config = config
        self.meter = meter
        self.out_dir = out_dir
        self.judge = judge
        self.judge_model = judge_model
        self.timeout_s = timeout_s

        # Imported here, after `configs.apply()` has put the configuration in
        # the environment: the schema and the provider are read at import.
        from src import agent as agent_mod  # noqa: PLC0415
        from src import config as config_mod  # noqa: PLC0415
        from src import planner as planner_mod  # noqa: PLC0415
        from src.utils import embeddings as emb  # noqa: PLC0415
        from src.utils import state  # noqa: PLC0415

        self.agent = agent_mod
        self.state = state
        self.emb = emb
        self.served_model = (
            config.chat_model if config.closed_orchestrator else config_mod.resolve_model(config.chat_model)[0]
        )
        self.recorder = UsageRecorder(
            meter, model_hint=self.served_model, max_tokens=MAX_TOKENS, run_meta={"config": config.name}
        )
        # The one seam: build_llm with the callback attached. Both the agent
        # and the planner build through it, and both are rebuilt from scratch.
        build = functools.partial(config_mod.build_llm, callbacks=[self.recorder])
        agent_mod.build_llm = build
        planner_mod.build_llm = build
        agent_mod.reset_agent()
        self.label = config_mod.model_label()

    # -- one case --------------------------------------------------------

    def _turn(self, case: Case) -> dict:
        """Run the agent on one case and collect everything it did."""
        state = self.state
        state.clear_rag_buffer()
        state.clear_map_command()
        state.clear_selected_lot()
        if case.lot:
            state.set_selected_lot(case.lot["lot_number"], case.lot.get("lon"), case.lot.get("lat"), case.lot.get("neighborhood"))
        state.start_new_turn()
        self.recorder.reset()
        embed_before = dict(self.emb.USAGE)

        events: list[dict] = []
        final = ""
        plan = ""
        for event in self.agent.stream_agent(case.question, history=None, thread_id=f"eval-{uuid.uuid4()}"):
            kind = event.get("type")
            if kind == "final":
                final = event.get("content", "")
            elif kind == "plan":
                plan = event.get("content", "")
            if kind != "token":
                events.append(event)

        citations = [hit for _, hit in sorted(state.citations().items())]
        embed_after = self.emb.USAGE
        return {
            "answer": final,
            "plan": plan,
            "events": events,
            "citations": citations,
            "embed_calls": embed_after["calls"] - embed_before["calls"],
            "embed_tokens": embed_after["tokens"] - embed_before["tokens"],
        }

    def _trace(self, case: Case, turn: dict) -> list[dict]:
        trace: list[dict] = [{"role": "system", "content": self.agent._SYSTEM_PROMPT}]
        if case.lot:
            trace.append({"role": "user", "content": f"[map: lot {case.lot['lot_number']} selected]\n{case.question}"})
        else:
            trace.append({"role": "user", "content": case.question})
        if turn["plan"]:
            trace.append({"role": "assistant", "content": "Plan:\n" + turn["plan"]})
        for event in turn["events"]:
            if event["type"] == "tool_start":
                trace.append({"role": "tool_call", "name": event["name"], "content": json.dumps(event.get("args", {}), ensure_ascii=False, indent=2, default=str)})
            elif event["type"] == "tool_end":
                trace.append({"role": "tool_result", "name": event["name"], "content": str(event.get("output", ""))})
        trace.append({"role": "assistant", "content": turn["answer"]})
        return trace

    def _context_for_judge(self, turn: dict) -> str:
        parts = []
        for event in turn["events"]:
            if event["type"] == "tool_end":
                parts.append(f"[{event['name']}]\n{event.get('output', '')}")
        return "\n\n".join(parts)

    def run_case(self, case: Case, rep: int, *, set_fingerprint: str) -> tuple[dict | None, dict | None]:
        """(result row, error row) - exactly one of them is not None."""
        started = time.perf_counter()
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        base = {
            "prompt_id": case.id, "prompt": case.question, "tags": case.tags, "rep": rep,
            "config": self.config.name, "model": self.served_model, "at": stamp, "set": set_fingerprint,
        }
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                turn = pool.submit(self._turn, case).result(timeout=self.timeout_s)
        except FutureTimeout:
            totals = self.recorder.totals()
            return None, {**base, "failure": "timeout", "latency_s": round(time.perf_counter() - started, 2),
                          "usage": totals["usage"], "cost_usd": totals["usd"], "llm_calls": totals["llm_calls"]}
        except Exception as exc:  # noqa: BLE001 - the harness must keep going
            totals = self.recorder.totals()
            return None, {**base, "failure": "budget" if self.meter.tripped else "harness",
                          "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-2000:],
                          "usage": totals["usage"], "cost_usd": totals["usd"], "llm_calls": totals["llm_calls"]}
        latency = time.perf_counter() - started
        totals = self.recorder.totals()

        if self.meter.tripped:
            return None, {**base, "failure": "budget", "error": turn["answer"][:300], "latency_s": round(latency, 2),
                          "usage": totals["usage"], "cost_usd": totals["usd"], "llm_calls": totals["llm_calls"]}
        # A model call that raised ends the turn with the agent's "⚠️ <error>"
        # as the answer. That is the endpoint failing, not the build answering
        # badly: an error row, so a resume runs the case again.
        if totals["errors"] and (not turn["answer"] or turn["answer"].startswith("⚠️")):
            return None, {**base, "failure": "serving", "error": "; ".join(totals["errors"])[:500],
                          "latency_s": round(latency, 2), "usage": totals["usage"], "cost_usd": totals["usd"],
                          "llm_calls": totals["llm_calls"]}

        answer = turn["answer"]
        hits = turn["citations"]
        retrieved = [
            {"rank": i + 1, "chunk_id": h.get("chunk_id"), "doc_id": metrics.doc_id_of(h),
             "source_table": h.get("source_table"), "scope": h.get("scope"),
             "similarity": h.get("similarity"), "text": (h.get("chunk_text") or "")[:400]}
            for i, h in enumerate(hits)
        ]
        grade = deterministic_grade(answer, case.expect, retrieved)

        judged: dict = {}
        if self.judge and answer.strip():
            try:
                judged = judge_mod.grade(
                    question=case.question, answer=answer, context=self._context_for_judge(turn),
                    expect=case.expect, meter=self.meter, model=self.judge_model, case_id=case.id,
                )
                judge_columns(grade, judged, case.expect)
            except Exception as exc:  # noqa: BLE001 - a judge failure is recorded, not fatal
                judged = {"error": f"{type(exc).__name__}: {exc}"}

        stop_reasons = [s for s in totals["stop_reasons"] if s]
        truncated = any(s in ("max_tokens", "length") for s in stop_reasons)
        tool_calls = sum(1 for e in turn["events"] if e["type"] == "tool_start")
        rate = rate_for(self.served_model)
        row = {
            **base,
            "status": "truncated" if truncated else "ok",
            "stop_reason": stop_reasons[-1] if stop_reasons else None,
            "grade": grade,
            "explanation": {"judge": (judged.get("verdict") or {}).get("explanation", "")} if judged else {},
            "latency_s": round(latency, 2),
            "llm_latency_s": totals["llm_latency_s"],
            "llm_calls": totals["llm_calls"],
            "tool_calls": tool_calls,
            "tools": [e["name"] for e in turn["events"] if e["type"] == "tool_start"],
            "usage": totals["usage"],
            "cost_usd": totals["usd"],
            "cost_is_estimate": rate.estimate,
            "models_served": totals["models"],
            "embed_calls": turn["embed_calls"],
            "embed_tokens": turn["embed_tokens"],
            "judge_model": judged.get("judge_model"),
            "judge_usage": judged.get("judge_usage"),
            "judge_cost_usd": judged.get("judge_usd"),
            "meta": {
                "answer": answer,
                "expect": case.expect,
                "lot": case.lot,
                "retrieved": retrieved,
                "judge": judged.get("verdict") or judged,
            },
        }
        trace_path = self.out_dir / "traces" / f"{case.id}_rep{rep}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(self._trace(case, turn), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return row, None

    # -- the loop ----------------------------------------------------------

    def run(self, cases: list[Case], *, reps: int = 1, retries: int = 2, progress=print) -> dict:
        """Every (case, rep) not already in ``results.jsonl``.

        A serving failure - a 429, a dropped connection - is retried after a
        pause, ``retries`` times, each attempt recorded in ``errors.jsonl``
        with its usage so the retries are visible in the data and the bill.
        A timeout, a harness error or the budget is not retried.
        """
        results = self.out_dir / "results.jsonl"
        errors = self.out_dir / "errors.jsonl"
        done = _done(results)
        set_fp = fingerprint(cases)
        counts = {"ok": 0, "skipped": 0, "error": 0, "retried": 0}
        progress(f"[{self.config.name}] {self.label} — {len(cases)} case(s) × {reps} rep(s); "
                 f"spent so far ${self.meter.anthropic_usd:.3f} of ${self.meter.cap_usd:.2f}")
        for case in cases:
            for rep in range(reps):
                if (case.id, rep) in done:
                    counts["skipped"] += 1
                    continue
                for attempt in range(retries + 1):
                    row, error = self.run_case(case, rep, set_fingerprint=set_fp)
                    if row is not None or error["failure"] != "serving" or attempt == retries:
                        break
                    error["attempt"] = attempt
                    _append(errors, error)
                    counts["retried"] += 1
                    wait = 30 * (attempt + 1)
                    progress(f"  {case.id:<40} serving error, retrying in {wait}s: {str(error.get('error', ''))[:100]}")
                    time.sleep(wait)
                if row is not None:
                    _append(results, row)
                    counts["ok"] += 1
                    g = row["grade"]
                    progress(
                        f"  {case.id:<40} {row['status']:<9} {row['latency_s']:6.1f}s "
                        f"{row['llm_calls']:>2} llm {row['tool_calls']:>2} tools "
                        f"${row['cost_usd']:.4f}  correct={g.get('answer_correct', '-')} "
                        f"hit@5={g.get('hit@5', '-')} faithful={g.get('faithful', '-')}"
                    )
                else:
                    _append(errors, error)
                    counts["error"] += 1
                    progress(f"  {case.id:<40} ERROR {error['failure']}: {str(error.get('error', ''))[:120]}")
                    if error["failure"] == "budget":
                        progress("  budget reached; stopping this run")
                        return {**counts, "stopped": "budget", "spend": self.meter.summary()}
        return {**counts, "spend": self.meter.summary()}
