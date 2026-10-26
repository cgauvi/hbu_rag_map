"""The comparison, read off the result files, as one Markdown page.

Every number is an absolute first - a rate with its interval, seconds, tokens,
dollars - and only then a comparison, because the absolute is what a user
feels and a percentage without it hides whether the difference is two
seconds or twenty. Rates carry a 95% Wilson interval so that a gap smaller
than the noise is seen for what it is.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from eval import metrics
from eval.configs import CONFIGS

QUALITY = (
    ("answer_correct", "Correct (checks)"),
    ("judge_correct", "Correct (judge)"),
    ("faithful", "Faithful"),
    ("used_retrieval", "Used retrieval"),
    ("hit@5", "Doc in top 5"),
    ("mrr", "MRR"),
    ("ndcg@10", "nDCG@10"),
    ("citations_valid", "Citations valid"),
    ("answered", "Answered"),
)


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rate(rows: list[dict], key: str) -> tuple[float | None, int, tuple[float, float]]:
    values = [float(r["grade"][key]) for r in rows if key in (r.get("grade") or {}) and r["grade"][key] is not None]
    if not values:
        return None, 0, (0.0, 0.0)
    return sum(values) / len(values), len(values), metrics.wilson(sum(values), len(values))


def _fmt_rate(rows: list[dict], key: str) -> str:
    value, n, (lo, hi) = _rate(rows, key)
    if value is None:
        return "—"
    return f"{value:.2f} [{lo:.2f}–{hi:.2f}] n={n}"


def _fmt(value: float | None, spec: str = ".2f") -> str:
    return "—" if value is None else format(value, spec)


def config_summary(name: str, rows: list[dict], errors: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    usage_in = [r["usage"].get("input_tokens", 0) + r["usage"].get("cache_read_input_tokens", 0) + r["usage"].get("cache_creation_input_tokens", 0) for r in rows]
    usage_out = [r["usage"].get("output_tokens", 0) for r in rows]
    cached = [r["usage"].get("cache_read_input_tokens", 0) for r in rows]
    cost = [float(r.get("cost_usd") or 0) for r in rows]
    judge_cost = [float(r.get("judge_cost_usd") or 0) for r in rows]
    summary = {
        "config": name,
        "label": CONFIGS[name].label if name in CONFIGS else name,
        "cases": len(rows),
        "ok": len(ok),
        "truncated": sum(1 for r in rows if r.get("status") == "truncated"),
        "errors": len(errors),
        "error_kinds": dict(sorted((e.get("failure"), 1) for e in errors)),
        "quality": {key: _rate(rows, key) for key, _ in QUALITY},
        "latency_median_s": metrics.median([r["latency_s"] for r in rows]),
        "latency_p90_s": metrics.percentile([r["latency_s"] for r in rows], 0.9),
        "llm_calls_mean": metrics.mean([r["llm_calls"] for r in rows]),
        "tool_calls_mean": metrics.mean([r["tool_calls"] for r in rows]),
        "input_tokens_median": metrics.median(usage_in),
        "output_tokens_median": metrics.median(usage_out),
        "cache_read_share": (sum(cached) / sum(usage_in)) if sum(usage_in) else None,
        "cost_mean_usd": metrics.mean(cost),
        "cost_total_usd": sum(cost),
        "cost_is_estimate": any(r.get("cost_is_estimate") for r in rows),
        "judge_cost_total_usd": sum(judge_cost),
        "models_served": sorted({m for r in rows for m in (r.get("models_served") or [])}),
    }
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["tags"][0]].append(r)
    summary["by_category"] = {
        cat: {"n": len(rs), "answer_correct": _rate(rs, "answer_correct")[0], "judge_correct": _rate(rs, "judge_correct")[0],
              "faithful": _rate(rs, "faithful")[0], "hit@5": _rate(rs, "hit@5")[0], "latency_median_s": metrics.median([r["latency_s"] for r in rs]),
              "cost_mean_usd": metrics.mean([float(r.get("cost_usd") or 0) for r in rs])}
        for cat, rs in sorted(by_cat.items())
    }
    return summary


def build(out_dir: Path, *, configs: list[str] | None = None) -> str:
    names = configs or [n for n in CONFIGS if (out_dir / n / "results.jsonl").exists()]
    summaries = []
    per_config_rows: dict[str, list[dict]] = {}
    for name in names:
        rows = load_results(out_dir / name / "results.jsonl")
        errors = load_results(out_dir / name / "errors.jsonl")
        if not rows and not errors:
            continue
        per_config_rows[name] = rows
        summaries.append(config_summary(name, rows, errors))

    lines = ["# RAG configuration comparison", ""]
    from eval.spend import SpendMeter  # noqa: PLC0415

    meter = SpendMeter.open(out_dir)
    if meter.entries:
        lines.append("Spend to date (every run, this directory): " + ", ".join(f"{k} ${v:.3f}" for k, v in sorted(meter.totals.items())) + ".")
        lines.append("")
    if not summaries:
        lines.append("_No results yet._")
        return "\n".join(lines)

    # -- headline table -------------------------------------------------------
    lines += ["## Headline", "", "| | " + " | ".join(s["label"] for s in summaries) + " |", "|---|" + "---|" * len(summaries)]
    lines.append("| cases (ok / truncated / errors) | " + " | ".join(f"{s['ok']} / {s['truncated']} / {s['errors']}" for s in summaries) + " |")
    for key, label in QUALITY:
        lines.append(f"| {label} | " + " | ".join(_fmt_rate(per_config_rows[s['config']], key) for s in summaries) + " |")
    lines.append("| latency median / p90 (s) | " + " | ".join(f"{_fmt(s['latency_median_s'], '.1f')} / {_fmt(s['latency_p90_s'], '.1f')}" for s in summaries) + " |")
    lines.append("| model calls / tool calls per case | " + " | ".join(f"{_fmt(s['llm_calls_mean'], '.1f')} / {_fmt(s['tool_calls_mean'], '.1f')}" for s in summaries) + " |")
    lines.append("| input / output tokens per case (median) | " + " | ".join(f"{_fmt(s['input_tokens_median'], ',.0f')} / {_fmt(s['output_tokens_median'], ',.0f')}" for s in summaries) + " |")
    lines.append("| input read from cache | " + " | ".join(_fmt(s["cache_read_share"], ".0%") for s in summaries) + " |")
    lines.append("| cost per case / total (USD) | " + " | ".join(f"{'~' if s['cost_is_estimate'] else ''}${_fmt(s['cost_mean_usd'], '.4f')} / {'~' if s['cost_is_estimate'] else ''}${_fmt(s['cost_total_usd'], '.3f')}" for s in summaries) + " |")
    lines.append("| judge cost total (USD) | " + " | ".join(f"${_fmt(s['judge_cost_total_usd'], '.3f')}" for s in summaries) + " |")
    lines.append("| models served | " + " | ".join(", ".join(s["models_served"]) or "—" for s in summaries) + " |")
    lines += ["", "Rates carry a 95% Wilson interval and the number of cases that carry the column. "
              "A `~` marks a cost estimated from a list price the service does not itself report. "
              "`Correct (checks)` is every deterministic check on the case passing (a stated storey count near a unit word, the zone code named, "
              "a resolution number named, a refusal where the corpus is silent); `Correct (judge)` is the judge's reading of the same against the reference; "
              "`Faithful` is 1 when every factual claim is in the tool results, 0.5 when some are, 0 when the answer states something the context does not hold. "
              "`Doc in top 5` / `MRR` / `nDCG@10` score the passages the model was actually shown, in order, against the governing document.", ""]

    # -- per category ---------------------------------------------------------
    lines += ["## By category", ""]
    cats = sorted({c for s in summaries for c in s["by_category"]})
    lines.append("| category | n | " + " | ".join(f"{s['config']} correct / judge / faithful / doc@5 / s / $" for s in summaries) + " |")
    lines.append("|---|---|" + "---|" * len(summaries))
    for cat in cats:
        cells = []
        n = 0
        for s in summaries:
            c = s["by_category"].get(cat)
            if not c:
                cells.append("—")
                continue
            n = max(n, c["n"])
            cells.append(f"{_fmt(c['answer_correct'])} / {_fmt(c['judge_correct'])} / {_fmt(c['faithful'])} / {_fmt(c['hit@5'])} / {_fmt(c['latency_median_s'], '.0f')} / {_fmt(c['cost_mean_usd'], '.3f')}")
        lines.append(f"| {cat} | {n} | " + " | ".join(cells) + " |")
    lines.append("")

    # -- retrieval only -------------------------------------------------------
    retrieval_dir = out_dir / "retrieval"
    if retrieval_dir.exists():
        from eval.retrieval import summarize  # noqa: PLC0415

        lines += ["## Encoder alone (no chat model)", ""]
        embedders = {}
        for path in sorted(retrieval_dir.glob("*.jsonl")):
            rows = load_results(path)
            if rows:
                embedders[path.stem] = summarize(rows)
        if embedders:
            lines.append("| scope · metric | " + " | ".join(embedders) + " |")
            lines.append("|---|" + "---|" * len(embedders))
            for scope in ("at_lot", "near", "corpus"):
                for key in ("hit@1", "hit@5", "mrr", "ndcg@10", "search_s_median"):
                    cells = [(_fmt(e[scope].get(key), ".3f" if key == "search_s_median" else ".2f") if scope in e else "—") for e in embedders.values()]
                    label = f"{scope} · {key}" + (f" (n={next((e[scope]['n'] for e in embedders.values() if scope in e), 0)})" if key == "hit@1" else "")
                    lines.append(f"| {label} | " + " | ".join(cells) + " |")
            lines.append("| query embed s (median) | " + " | ".join(_fmt(e["embed_s_median"], ".3f") for e in embedders.values()) + " |")
            lines.append("")
            lines.append("| hit@5 by category | " + " | ".join(embedders) + " |")
            lines.append("|---|" + "---|" * len(embedders))
            for cat in sorted({c for e in embedders.values() for c in e["hit@5_by_category"]}):
                lines.append(f"| {cat} | " + " | ".join(_fmt(e["hit@5_by_category"].get(cat)) for e in embedders.values()) + " |")
            lines.append("")

    # -- per case -------------------------------------------------------------
    lines += ["## Per case", "", "| case | " + " | ".join(f"{s['config']}: correct · judge · faithful · doc@5 · s · $" for s in summaries) + " |", "|---|" + "---|" * len(summaries)]
    ids: list[str] = []
    for rows in per_config_rows.values():
        for r in rows:
            if r["prompt_id"] not in ids:
                ids.append(r["prompt_id"])
    for cid in ids:
        cells = []
        for s in summaries:
            r = next((r for r in per_config_rows[s["config"]] if r["prompt_id"] == cid), None)
            if r is None:
                cells.append("—")
                continue
            g = r["grade"]
            cells.append(f"{_fmt(g.get('answer_correct'), '.0f')} · {_fmt(g.get('judge_correct'), '.0f')} · {_fmt(g.get('faithful'), '.1f')} · {_fmt(g.get('hit@5'), '.0f')} · {r['latency_s']:.0f} · {float(r.get('cost_usd') or 0):.3f}")
        lines.append(f"| {cid} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Traces: `<config>/traces/<case>_rep0.json` holds the whole turn - prompt, plan, every tool call and result, the answer.")
    return "\n".join(lines)


def write(out_dir: Path, **kwargs) -> Path:
    text = build(out_dir, **kwargs)
    path = out_dir / "report.md"
    path.write_text(text, encoding="utf-8")
    return path
