"""
planner.py — one short plan, written before the turn starts.

A question that spans the cadastre, the grid, the roll and the heritage layers
is answerable in two tool calls, and the failure mode is not that the tools are
missing: it is that the model reaches for the per-lot tools it has used a
hundred times and walks the list one parcel at a time. A plan is cheap enough
to buy its way out of that.

Where it runs, and why not in the graph
--------------------------------------
Outside. ``pre_model_hook`` is the graph's entry point and fires before *every*
model call, so a planner living there would re-plan at each step and need a
sentinel to stop itself; a hand-rolled ``StateGraph`` is more surgery than one
extra call is worth. So ``stream_agent`` calls ``plan_for`` before
``agent.stream``, seeds the result into a custom state key, and the agent's
callable prompt splices it into the system message. One-shot by construction,
its latency visible as its own UI event, and an exception in it cannot take the
turn down with it.

What it costs, and when it is skipped
-------------------------------------
One round trip on a small token budget. ``needs_plan`` keeps that off simple
questions with a Python heuristic rather than a model: an LLM router would cost
a full round trip to decide whether to spend a full round trip, which is a
trade with no upside and a new failure mode. Prefix a question with ``/plan``
to force one, ``/noplan`` to refuse one.

A numbered list, not JSON
-------------------------
Malformed tool-call JSON is already this endpoint's most common failure, and
asking the same model for structured output to *avoid* tool calls would be an
odd bet. A numbered list parses with one regexp, and when it does not parse the
turn simply runs without a plan.
"""

from __future__ import annotations

import logging
import os
import re
import time
import unicodedata

from langchain_core.messages import HumanMessage

from src.config import build_llm

logger = logging.getLogger(__name__)

#: Tokens the planner may spend — and this floor is measured, not guessed.
#:
#: gpt-oss reasons before it writes, and the reasoning comes out of the SAME
#: budget. Against this prompt it spends ~330 tokens thinking and ~120 writing.
#: At 256 the whole budget went to the reasoning and the reply came back
#: EMPTY with finish_reason="length" - an HTTP 200 carrying nothing, which
#: `plan_for` reads as "no plan" and carries on. The planner silently never
#: fires and the only symptom is that nothing improves.
#:
#: Measured 2026-10-02 (openai/gpt-oss-120b):
#:     256  -> completion 256, finish=length, 0 chars      (silent failure)
#:     512  -> completion 450, finish=stop,   128 chars    (a correct plan)
#:     1024 -> completion 469, finish=stop,   139 chars
#: all of them in ~0.4 s, so the headroom is free. Do not lower this to "save"
#: anything: below the floor the cost is the same and the output is nothing.
PLANNER_MAX_TOKENS = int(os.environ.get("HBU_PLANNER_MAX_TOKENS", 768))

#: Steps past this are noise - a plan nobody can follow is worse than none.
MAX_STEPS = 5

#: Off switches the whole thing, for measuring what it is worth.
ENABLED = os.environ.get("HBU_AGENT_PLANNER", "1") != "0"

#: Score at which a question is worth planning. Tuned against the three
#: questions the dossier exists for and a dozen single-lot ones; raise it if
#: the planner starts firing on "what can I build here".
PLAN_THRESHOLD = int(os.environ.get("HBU_PLANNER_THRESHOLD", 4))

#: Below this many characters, a question without a comparison in it is a
#: lookup. "find lot 2 170 935" needs no plan.
_SHORT = 60

_CONJUNCTION = re.compile(r"\b(?:and|et|with|avec|plus|also)\b|,")
_COMPARE = re.compile(r"\b(?:compare|comparer|versus|vs\.?|which of|between|entre)\b")
_SET = re.compile(
    r"\b(?:which lots?|which sites?|quels?|quelles?|all the|list the|"
    r"how many|combien|top \d+|best|meilleurs?)\b"
)
#: A word naming one of the surfaces the dossier joins. Two or more of these in
#: one question is the shape that used to cost six tool calls.
_SURFACE = re.compile(
    r"\b(?:zoned?|zonage|storeys?|stor(?:e?y|ies)|etages?|étages?|height|hauteur|"
    r"assessed|assessment|evaluation|évaluation|worth|value|valeur|"
    r"heritage|patrimo\w*|under-?built|vacant|headroom|"
    r"brownfield|teardown|infill|improvement|thesis|"
    r"cap rate|irr|npv|yield|rebuild|redevelop\w*|"
    r"commerce|commercial|industrial|residential|dwellings?|logements?)\b"
)

_STEP = re.compile(r"^\s*\d+\s*[.)]\s*([a-z_]+)\s*[—\-–:]\s*(.+?)\s*$")


def _fold(text: str) -> str:
    """Lowercase and strip accents, so `évaluation` scores as `evaluation`."""
    lowered = (text or "").lower()
    return "".join(
        ch for ch in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(ch) != "Mn"
    )


def needs_plan(question: str) -> bool:
    """Whether this question is worth a planning round trip.

    Deterministic on purpose: it is debuggable, it is free, and a heuristic
    that misfires costs one wasted call rather than a wrong answer.
    """
    raw = (question or "").strip()
    if raw.lower().startswith("/noplan"):
        return False
    if raw.lower().startswith("/plan"):
        return True

    folded = _fold(raw)
    compare = bool(_COMPARE.search(folded))
    if len(folded) < _SHORT and not compare:
        return False

    score = len(_CONJUNCTION.findall(folded))
    score += 2 * compare
    score += 2 * bool(_SET.search(folded))
    score += len(set(_SURFACE.findall(folded)))
    return score >= PLAN_THRESHOLD


def strip_directive(question: str) -> str:
    """The question without a leading `/plan` or `/noplan`."""
    return re.sub(r"^\s*/(?:no)?plan\b\s*", "", question or "", flags=re.I)


_PROMPT = """\
You are planning a data lookup for an urban-planning assistant. You will NOT \
answer the question and you will NOT call any tool. Write the shortest \
sequence of steps that would answer it.

Tools available:
  describe_data, find_sites, summarize_sites, compare_sites, site_dossier,
  find_lot, find_lot_by_address, same_owner, describe_selected_lot,
  zoning_for_lot,
  lot_efficiency, lot_futures, buildings_on_lot, read_zoning_grid,
  regulations_at_lot, regulations_near, regulations_for_lots,
  search_regulations, council_decisions_near, search_council_minutes,
  data_status

Rules:
- A question combining two or more of {{what the zone permits, assessed value,
  heritage, how under-built it is, a site thesis, what rebuilding is worth}} is
  ONE describe_data call followed by ONE find_sites call. Not one tool per
  condition.
- A question naming a street address starts with find_lot_by_address.
- A question about whether addresses or lots have the SAME OWNER, or who
  owns one, is ONE same_owner call naming every address. Not one address
  lookup per address.
- A question about the by-law's own wording - a footnote, a condition, an
  exception - ends with regulations_at_lot, or regulations_for_lots when it
  covers several lots.
- A question about what was DECIDED near a place - a demolition approved or
  refused, an amendment adopted, what the council recommended, in a date
  range - is ONE council_decisions_near call with the address, kind, outcome
  and dates. Not a zoning lookup.
- At most {max_steps} steps. Fewer is better.

Format exactly, one per line, nothing else:
1. tool_name — what it is for, under 12 words

Question: {question}
Borough in view: {borough}
Selected lot: {lot}
"""

_planner = None
_planner_model: str | None = None


def _get_planner():
    """A second, small-budget client on the same endpoint."""
    global _planner, _planner_model

    current = os.environ.get("HF_MODEL_ID")
    if _planner is not None and _planner_model == current:
        return _planner
    _planner = build_llm(max_new_tokens=PLANNER_MAX_TOKENS)
    _planner_model = current
    return _planner


def reset_planner() -> None:
    global _planner, _planner_model
    _planner = None
    _planner_model = None


def parse(text: str, *, known_tools: set[str]) -> list[tuple[str, str]]:
    """The steps a planner's reply names, dropping anything that is not a tool.

    A hallucinated tool name is the common failure here, and dropping the line
    is better than passing it on: the agent would spend a retry discovering
    there is no such tool.
    """
    steps: list[tuple[str, str]] = []
    for line in (text or "").splitlines():
        match = _STEP.match(line)
        if not match:
            continue
        name, purpose = match.group(1), match.group(2).strip()
        if name in known_tools:
            steps.append((name, purpose))
        else:
            logger.info("Planner named an unknown tool, dropping: %s", name)
        if len(steps) >= MAX_STEPS:
            break
    return steps


def plan_for(
    question: str,
    *,
    known_tools: set[str],
    borough: str = "",
    lot: str = "",
) -> str:
    """A plan for this question, or "" when it does not need or get one.

    Never raises. A planner that fails is a turn that runs unplanned, which is
    exactly how every turn ran before this module existed.
    """
    if not ENABLED or not needs_plan(question):
        return ""

    started = time.monotonic()
    try:
        reply = _get_planner().invoke(
            [
                HumanMessage(
                    content=_PROMPT.format(
                        question=strip_directive(question),
                        borough=borough or "not known",
                        lot=lot or "none",
                        max_steps=MAX_STEPS,
                    )
                )
            ]
        )
        text = reply.content if isinstance(reply.content, str) else ""
        if not text.strip():
            # Almost always the budget: see PLANNER_MAX_TOKENS. Named here
            # because the alternative is a planner that quietly does nothing.
            finish = (reply.response_metadata or {}).get("finish_reason")
            used = (reply.response_metadata or {}).get("token_usage", {})
            logger.warning(
                "Planner returned nothing (finish_reason=%s, completion=%s, "
                "budget=%s) — if finish_reason is 'length', the reasoning used "
                "the whole budget and HBU_PLANNER_MAX_TOKENS is too low.",
                finish, used.get("completion_tokens"), PLANNER_MAX_TOKENS,
            )
            return ""
    except Exception as exc:  # noqa: BLE001 - a plan is an optimisation, not a step
        logger.warning("Planner failed, running unplanned: %s", exc)
        return ""

    steps = parse(text, known_tools=known_tools)
    logger.info(
        "Planned %d step(s) in %.1fs", len(steps), time.monotonic() - started
    )
    if not steps:
        return ""
    return "\n".join(f"{i}. {name} — {why}" for i, (name, why) in enumerate(steps, 1))


def block(plan: str) -> str:
    """The plan as it appears in the system prompt, or nothing."""
    if not plan:
        return ""
    return (
        "\n\nPlan for this turn\n"
        "------------------\n"
        "A planner proposed these steps. Follow them in order. Skip one only "
        "if a result you already have answers it, and if you take a step that "
        "is not here, say in your answer why.\n"
        f"{plan}\n"
    )
