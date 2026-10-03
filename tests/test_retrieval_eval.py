"""Score retrieval against a frozen question set, so a ranking change is a number.

Marked ``integration``: it needs a reachable database and a working query
encoder, and it is slow enough that `make test` should not pay for it.

    make test-integration -- -k retrieval_eval

What is being measured is *retrieval*, not the answer. No judge model, no
prose labelling, no human in the loop — either the sheet that governs came back
inside the top k or it did not. That is what makes the number comparable across
a change to chunking, to ranking, or to the chat model.

Two of the tests here assert a floor rather than a target. The floors are
deliberately loose: their job is to fail when something is broken (the encoder
is wrong, the corpus is empty, a borough was never published), not to encode
today's quality as a requirement. Read the printed report for quality; read the
assertion for breakage.

The report prints whether or not the assertions pass, because the number is the
point. Run it once before changing ranking and write the result into the
fixture's header — a retrieval change with no before-number cannot be defended.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from src.utils import queries
from src.utils.embeddings import EmbeddingError, embed_query

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).parent / "fixtures" / "retrieval_eval.yaml"

#: Set HBU_EVAL_DENSE_ONLY=1 to re-measure the dense-only baseline from this
#: same fixture, which is how the two numbers stay comparable: one run, one
#: corpus, one encoder, one set of questions.
DENSE_ONLY = os.environ.get("HBU_EVAL_DENSE_ONLY", "") == "1"

#: A hit this weak is the corpus saying it has nothing, which is what a
#: negative question should produce. Calibrate against the printed report
#: rather than guessing: it is a property of the encoder and the corpus, not a
#: constant of nature.
SILENCE_SIMILARITY = 0.55

#: Where recall is counted. 20 is the widest the tools will return.
KS = (5, 10, 20)


def _labelled(question: dict) -> bool:
    return any(
        question.get(key)
        for key in ("expect_feature_id", "expect_doc_id", "must_contain_any", "expect_nothing")
    )


@pytest.fixture(scope="module")
def question_set() -> list[dict]:
    loaded = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    questions = [q for q in (loaded or {}).get("questions", []) if _labelled(q)]
    if not questions:
        pytest.skip(f"{FIXTURE.name} holds no labelled questions")
    return questions


@pytest.fixture
def corpus_ready(monkeypatch, database_url, real_hf_token) -> None:
    """Skip rather than fail when the corpus or the encoder is not there.

    A missing corpus is a loading state, not a regression - the same reading
    the agent's tools take. `make check` says which half is absent.

    The DSN has to be put back first: `_clean_environment` scrubs it from
    every test so the unit suite cannot open a socket, and `database_url`
    captured it before that ran.
    """
    if not database_url:
        pytest.skip("set DATABASE_URL (or HBU_TEST_DATABASE_URL) to run these")
    if not real_hf_token:
        pytest.skip("set HUGGINGFACE_API_TOKEN: the eval embeds for real")
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", real_hf_token)
    from src.utils.db import close_pool

    close_pool()

    try:
        caps = queries.capabilities()
    except Exception as exc:  # noqa: BLE001 - no database is a skip, not an error
        pytest.skip(f"no database: {exc}")
    if not caps.can_retrieve:
        pytest.skip("rag.chunks is not loaded; run the dataplatform's document_index")
    try:
        embed_query("sonde")
    except EmbeddingError as exc:
        pytest.skip(f"the query encoder is not answering: {exc}")


def _ask(question: dict, *, match_count: int) -> list[dict]:
    """Retrieve for one question, by the path the question says to use.

    French is preferred where the fixture gives both, because that is what the
    tools are told to send; the English twin is scored separately through its
    own row, which is the whole point of the pairs.
    """
    text = question.get("question_fr") or question.get("question_en")
    scope = question.get("scope", "corpus")
    if scope != "corpus":
        # at_lot / near are generated rows; they carry their own coordinates.
        pytest.skip(f"{question['id']}: {scope} rows are not generated yet")
    return queries.search_corpus(
        embed_query(text),
        match_count=match_count,
        neighborhood=question.get("neighborhood"),
        query_text=text,
    )


def _rank_of_match(question: dict, hits: list[dict]) -> int | None:
    """1-based rank of the first hit that satisfies the label, or None."""
    want_feature = question.get("expect_feature_id")
    want_doc = question.get("expect_doc_id")
    want_words = [w.lower() for w in (question.get("must_contain_any") or [])]

    for rank, hit in enumerate(hits, 1):
        if want_doc and hit.get("doc_id") == want_doc:
            return rank
        if want_feature:
            # feature_ids arrives as jsonb; psycopg hands it back as a list.
            cited = hit.get("feature_ids") or []
            if want_feature in cited:
                return rank
        if want_words:
            text = (hit.get("chunk_text") or "").lower()
            if any(word in text for word in want_words):
                return rank
    return None


def _report(rows: list[dict]) -> dict:
    scored = [r for r in rows if not r["expect_nothing"]]
    negatives = [r for r in rows if r["expect_nothing"]]

    report: dict = {}
    for k in KS:
        hit = [r for r in scored if r["rank"] is not None and r["rank"] <= k]
        report[f"recall@{k}"] = len(hit) / len(scored) if scored else 0.0
    reciprocal = [1 / r["rank"] for r in scored if r["rank"]]
    report["mrr"] = sum(reciprocal) / len(scored) if scored else 0.0
    quiet = [r for r in negatives if r["top_similarity"] < SILENCE_SIMILARITY]
    report["silence_rate"] = len(quiet) / len(negatives) if negatives else 1.0
    report["scored"] = len(scored)
    report["negatives"] = len(negatives)
    return report


@pytest.fixture
def results(question_set, corpus_ready) -> list[dict]:
    rows = []
    for question in question_set:
        if question.get("scope", "corpus") != "corpus":
            continue
        asked = question.get("question_fr") or question.get("question_en")
        hits = queries.search_corpus(
            embed_query(asked),
            match_count=max(KS),
            neighborhood=question.get("neighborhood"),
            # Without this the lexical arm never fires and the eval silently
            # measures the dense-only path it is supposed to be comparing to.
            query_text=None if DENSE_ONLY else asked,
        )
        rows.append(
            {
                "id": question["id"],
                "expect_nothing": bool(question.get("expect_nothing")),
                "rank": _rank_of_match(question, hits),
                "top_similarity": float(hits[0].get("similarity") or 0.0) if hits else 0.0,
                "returned": len(hits),
            }
        )
    if not rows:
        pytest.skip("no corpus-scope questions to score")
    return rows


def test_the_question_set_is_well_formed():
    """Every row carries a question and a label, or it scores nothing."""
    loaded = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    questions = (loaded or {}).get("questions", [])
    assert questions, "the fixture is empty"

    ids = [q.get("id") for q in questions]
    assert len(ids) == len(set(ids)), "duplicate question ids"

    for question in questions:
        assert question.get("question_fr") or question.get("question_en"), question
        assert _labelled(question), f"{question['id']} has no label and can never score"
        twin = question.get("pairs_with")
        if twin:
            assert twin in ids, f"{question['id']} pairs with a row that is not here"


def test_retrieval_is_reported_and_has_not_collapsed(results, capsys):
    report = _report(results)

    lines = [
        "",
        f"  scored {report['scored']} question(s), {report['negatives']} negative(s)",
        f"  recall@5   {report['recall@5']:.2f}",
        f"  recall@10  {report['recall@10']:.2f}",
        f"  recall@20  {report['recall@20']:.2f}",
        f"  mrr        {report['mrr']:.3f}",
        f"  silence    {report['silence_rate']:.2f} (negatives below "
        f"similarity {SILENCE_SIMILARITY})",
        "",
        "  misses:",
    ]
    for row in results:
        if not row["expect_nothing"] and row["rank"] is None:
            lines.append(f"    {row['id']} (nothing matched in {row['returned']})")
    with capsys.disabled():
        print("\n".join(lines))

    # A floor, not a target. Below this the encoder, the corpus or the borough
    # filter is broken rather than merely imprecise.
    assert report["recall@20"] >= 0.5, report


def test_a_question_naming_a_zone_reaches_that_zones_sheet(corpus_ready, question_set):
    """The failure the lexical arm exists to fix, scored on its own.

    Kept separate from the headline recall because it is the one class of
    question where dense similarity is known to be weak, and averaging it into
    the rest hides exactly the movement worth watching.
    """
    coded = [q for q in question_set if q.get("expect_feature_id")]
    if not coded:
        pytest.skip("no zone-code questions in the fixture")

    reached = []
    for question in coded:
        hits = queries.search_corpus(
            embed_query(question.get("question_fr") or question["question_en"]),
            match_count=max(KS),
            neighborhood=question.get("neighborhood"),
            zones=[question["expect_feature_id"]],
        )
        reached.append((question["id"], bool(hits)))

    missing = [qid for qid, ok in reached if not ok]
    # Narrowed by feature_ids this is a lookup, so anything missing means that
    # zone is not in the corpus at all - usually a borough that was never
    # published. Reported by name, because which one it is, is the answer.
    assert not missing, f"no document is filed under: {missing}"
