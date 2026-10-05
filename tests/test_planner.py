"""The planner: when it fires, what it parses, and what it costs when it fails.

The planner is an optimisation, and the test that matters most is that it
behaves like one — a turn whose planner raises, times out, or writes nonsense
has to run exactly as it did before this module existed. Everything else here
is about not spending a round trip on a question that does not need one.

The routing heuristic is deliberately Python rather than a model: an LLM router
would cost a full round trip to decide whether to spend a full round trip. That
makes it testable, which is this file.
"""

from __future__ import annotations

import pytest

from src import planner

#: Every tool name the agent actually holds, so a parsed step naming something
#: else is dropped rather than handed on.
KNOWN = {
    "describe_data", "find_sites", "summarize_sites", "compare_sites",
    "site_dossier", "find_lot", "find_lot_by_address", "same_owner", "zoning_for_lot",
    "lot_efficiency", "regulations_at_lot", "regulations_for_lots",
    "search_regulations", "data_status",
}


# ---------------------------------------------------------------------------
# When it fires
# ---------------------------------------------------------------------------

#: The questions the dossier exists for. If the planner does not fire on these
#: it is not earning its round trip.
COMPLEX = [
    "Which lots in Villeray are zoned for 4+ storeys, assessed under $500k, "
    "and not heritage-listed?",
    "Compare 7430 Lajeunesse and 400 Jarry Est on buildable envelope, "
    "assessed value and what the grid permits.",
    "Of the brownfield sites, which ones are in a zone that permits commerce?",
    "How many lots are under-built and worth rebuilding, and what are they "
    "assessed at?",
    "Quels lots sont patrimoniaux et sous-bâtis, avec une évaluation sous "
    "500 000 $ ?",
]

#: Questions a single tool answers. A plan here is a wasted round trip in front
#: of an answer the model already knows how to give.
SIMPLE = [
    "What can I build on this lot?",
    "Find lot 2 170 935",
    "find 7430 rue Lajeunesse",
    "What is the zoning here?",
    "show the zoning layer",
    "hauteur maximale",
    "What boroughs are loaded?",
]


@pytest.mark.parametrize("question", COMPLEX)
def test_a_multi_surface_question_is_planned(question):
    assert planner.needs_plan(question), question


@pytest.mark.parametrize("question", SIMPLE)
def test_a_single_tool_question_is_not(question):
    assert not planner.needs_plan(question), question


def test_an_accented_question_scores_like_its_unaccented_twin():
    """`évaluation` has to score as `evaluation`, or French questions never
    reach the threshold."""
    with_accents = "Quels lots ont une évaluation basse et des étages à bâtir ?"
    without = "Quels lots ont une evaluation basse et des etages a batir ?"

    assert planner.needs_plan(with_accents) == planner.needs_plan(without)


def test_the_user_can_force_a_plan_and_refuse_one():
    assert planner.needs_plan("/plan find lot 2 170 935")
    assert not planner.needs_plan("/noplan " + COMPLEX[0])


def test_the_directive_is_not_part_of_the_question():
    assert planner.strip_directive("/plan what can I build") == "what can I build"
    assert planner.strip_directive("/noplan  hauteur") == "hauteur"
    assert planner.strip_directive("hauteur maximale") == "hauteur maximale"


def test_the_planner_can_be_switched_off_entirely(monkeypatch):
    monkeypatch.setattr(planner, "ENABLED", False)

    assert planner.plan_for(COMPLEX[0], known_tools=KNOWN) == ""


# ---------------------------------------------------------------------------
# What it parses
# ---------------------------------------------------------------------------


def test_a_numbered_list_parses():
    reply = (
        "1. describe_data — the columns for zoning and money\n"
        "2. find_sites — 4+ storeys, under $500k, no heritage\n"
    )

    assert planner.parse(reply, known_tools=KNOWN) == [
        ("describe_data", "the columns for zoning and money"),
        ("find_sites", "4+ storeys, under $500k, no heritage"),
    ]


def test_a_hallucinated_tool_is_dropped_rather_than_passed_on():
    """Handing it on would spend a retry discovering there is no such tool."""
    reply = (
        "1. query_the_database — invented\n"
        "2. find_sites — real\n"
    )

    assert planner.parse(reply, known_tools=KNOWN) == [("find_sites", "real")]


def test_prose_around_the_list_is_ignored():
    reply = (
        "Here is my plan:\n"
        "1. find_lot_by_address — resolve the address\n"
        "Let me know if that works.\n"
    )

    assert planner.parse(reply, known_tools=KNOWN) == [
        ("find_lot_by_address", "resolve the address")
    ]


def test_a_plan_longer_than_the_cap_is_cut():
    reply = "\n".join(f"{i}. find_sites — step {i}" for i in range(1, 12))

    assert len(planner.parse(reply, known_tools=KNOWN)) == planner.MAX_STEPS


def test_an_unparseable_reply_yields_no_plan():
    """And a turn with no plan is a turn as it ran before this existed."""
    assert planner.parse("I would look at the zoning.", known_tools=KNOWN) == []
    assert planner.parse("", known_tools=KNOWN) == []


# ---------------------------------------------------------------------------
# What it costs when it goes wrong
# ---------------------------------------------------------------------------


def test_a_planner_that_raises_does_not_fail_the_turn(monkeypatch):
    def _explode():
        raise RuntimeError("the endpoint is down")

    monkeypatch.setattr(planner, "_get_planner", _explode)

    assert planner.plan_for(COMPLEX[0], known_tools=KNOWN) == ""


def test_a_planner_that_writes_nonsense_does_not_fail_the_turn(monkeypatch):
    class _Rambling:
        def invoke(self, _messages):
            class _R:
                content = "I think you should look at the by-law really."
            return _R()

    monkeypatch.setattr(planner, "_get_planner", lambda: _Rambling())

    assert planner.plan_for(COMPLEX[0], known_tools=KNOWN) == ""


def test_a_usable_plan_comes_back_numbered(monkeypatch):
    class _Planner:
        def invoke(self, _messages):
            class _R:
                content = (
                    "1. describe_data — money and zoning columns\n"
                    "2. find_sites — the four conditions\n"
                )
            return _R()

    monkeypatch.setattr(planner, "_get_planner", lambda: _Planner())

    plan = planner.plan_for(COMPLEX[0], known_tools=KNOWN)

    assert plan.splitlines() == [
        "1. describe_data — money and zoning columns",
        "2. find_sites — the four conditions",
    ]


def test_the_question_asked_of_the_planner_has_no_directive(monkeypatch):
    seen = {}

    class _Planner:
        def invoke(self, messages):
            seen["content"] = messages[0].content

            class _R:
                content = "1. find_sites — go"
            return _R()

    monkeypatch.setattr(planner, "_get_planner", lambda: _Planner())
    planner.plan_for("/plan find lot 2 170 935", known_tools=KNOWN)

    assert "/plan" not in seen["content"]
    assert "find lot 2 170 935" in seen["content"]


# ---------------------------------------------------------------------------
# How it reaches the model
# ---------------------------------------------------------------------------


def test_no_plan_adds_nothing_to_the_prompt():
    assert planner.block("") == ""


def test_a_plan_is_spliced_in_as_instructions():
    text = planner.block("1. find_sites — go")

    assert "Plan for this turn" in text
    assert "1. find_sites — go" in text
    # It has to be followable, not decorative: the model is told what to do
    # when a step is unnecessary and when it departs from the plan.
    assert "Skip one only" in text and "say in your answer why" in text


def test_the_prompt_template_formats_without_raising():
    """The one failure this module hides from itself.

    `_PROMPT` is a `.format` template whose Rules section lists a set in
    braces. An unescaped brace raises KeyError inside `plan_for`, where the
    broad except logs "running unplanned" and carries on — so the planner
    stops working and nothing says so. Formatting it here makes that loud.
    """
    text = planner._PROMPT.format(
        question="q", borough="VSMPE", lot="2 170 935", max_steps=planner.MAX_STEPS
    )

    assert "Question: q" in text
    assert "Borough in view: VSMPE" in text
    # The doubled braces rendered as one literal brace rather than being eaten
    # as a field name — which is what an unescaped version would do.
    assert "{what the zone permits" in text
    assert "what rebuilding is worth}" in text
