"""The free grade computed from a stored row, and `regrade` rewriting rows in place."""

from __future__ import annotations

import json

from eval.runner import deterministic_grade, regrade


def test_a_turn_that_retrieved_nothing_has_no_retrieval_columns():
    grade = deterministic_grade("Zone H02-161 permits 3 storeys.", {"values": {"storeys_max": 3}, "must_mention_any": ["H02-161"]}, [])
    assert grade["used_retrieval"] == 0.0
    assert "hit@5" not in grade
    assert grade["answer_correct"] == 1.0
    assert grade["answered"] == 1.0


def test_a_turn_that_retrieved_is_scored_on_what_it_was_shown():
    retrieved = [
        {"rank": 1, "chunk_id": "zone:0000", "doc_id": "zone", "source_table": "Reglement_urbanisme__VSP_REG_ZONE", "text": ""},
        {"rank": 2, "chunk_id": "res:0000", "doc_id": "res", "source_table": "Reglement_urbanisme__VSP_REG_PPCMOI", "text": ""},
    ]
    grade = deterministic_grade("Resolution CA17 140107 was adopted.", {"doc_ids": ["res"], "must_mention_any": ["140107"]}, retrieved)
    assert grade["used_retrieval"] == 1.0
    assert grade["hit@1"] == 0.0 and grade["hit@3"] == 1.0
    assert grade["answer_correct"] == 1.0


def test_a_resolution_found_by_a_borough_wide_search_is_not_an_invented_change():
    retrieved = [
        {"rank": 1, "chunk_id": "zone:0000", "doc_id": "zone", "source_table": "Reglement_urbanisme__VSP_REG_ZONE", "scope": "lot", "text": ""},
        {"rank": 2, "chunk_id": "other:0000", "doc_id": "other", "source_table": "Reglement_urbanisme__VSP_REG_PPCMOI", "scope": "corpus", "text": ""},
    ]
    grade = deterministic_grade("No projet particulier applies to this lot.", {"no_change": True}, retrieved)
    assert grade["no_invented_change"] == 1.0
    retrieved[1]["scope"] = "lot"
    grade = deterministic_grade("No projet particulier applies to this lot.", {"no_change": True}, retrieved)
    assert grade["no_invented_change"] == 0.0


def test_the_agents_own_citation_warning_marks_citations_invalid():
    grade = deterministic_grade("See [4].\n\n> ⚠️ [4] does not correspond to any passage retrieved in this turn.", {}, [])
    assert grade["citations_valid"] == 0.0


def test_regrade_recomputes_the_free_columns_and_keeps_the_judges(tmp_path):
    path = tmp_path / "results.jsonl"
    row = {
        "prompt_id": "x", "grade": {"answer_correct": 0.0, "faithful": 0.5, "judge_correct": 1.0},
        "meta": {"answer": "Zone H02‑161 permits 3 storeys.", "expect": {"values": {"storeys_max": 3}, "must_mention_any": ["H02-161"]}, "retrieved": []},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert regrade(path) == 1
    back = json.loads(path.read_text(encoding="utf-8").strip())
    assert back["grade"]["answer_correct"] == 1.0  # the typographic hyphen now folds
    assert back["grade"]["faithful"] == 0.5
    assert back["grade"]["judge_correct"] == 1.0
    assert regrade(tmp_path / "missing.jsonl") == 0
