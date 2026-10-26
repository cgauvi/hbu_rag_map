"""The harness's scoring, on inputs whose right answer is known by construction."""

from __future__ import annotations

import pytest

from eval import metrics


def _hits(*docs: str) -> list[dict]:
    return [{"chunk_id": f"{d}:000{i}", "doc_id": d, "chunk_text": f"passage of {d}"} for i, d in enumerate(docs)]


def test_a_document_in_the_top_five_counts_and_its_rank_is_the_reciprocal():
    scores = metrics.retrieval_scores(_hits("a", "b", "want", "c"), {"doc_ids": ["want"]})
    assert scores["hit@1"] == 0.0
    assert scores["hit@3"] == 1.0
    assert scores["hit@5"] == 1.0
    assert scores["mrr"] == pytest.approx(1 / 3)
    assert scores["first_rank"] == 3


def test_a_document_never_retrieved_scores_zero_everywhere():
    scores = metrics.retrieval_scores(_hits("a", "b"), {"doc_ids": ["want"]})
    assert scores["hit@10"] == 0.0 and scores["mrr"] == 0.0 and scores["ndcg@10"] == 0.0
    assert scores["first_rank"] is None


def test_search_at_lot_hits_carry_no_doc_id_and_the_chunk_id_is_read_instead():
    hits = [{"chunk_id": "want:0003", "chunk_text": ""}]
    assert metrics.retrieval_scores(hits, {"doc_ids": ["want"]})["hit@1"] == 1.0


def test_vocabulary_labels_match_folded_text():
    hits = [{"chunk_id": "x:0000", "chunk_text": "Hauteur en ÉTAGES: 3"}]
    assert metrics.retrieval_scores(hits, {"passages_contain_any": ["etage"]})["hit@1"] == 1.0


def test_ndcg_rewards_relevant_chunks_ranked_first():
    best = metrics.ndcg_at([1, 1, 0, 0], 10, total_relevant=2)
    worse = metrics.ndcg_at([0, 0, 1, 1], 10, total_relevant=2)
    assert best == pytest.approx(1.0)
    assert 0 < worse < best


def test_a_case_with_no_retrieval_label_has_no_retrieval_columns():
    assert metrics.retrieval_scores(_hits("a"), {"values": {"storeys_max": 3}}) == {}


# -- answers -----------------------------------------------------------------


def test_a_storey_count_is_stated_only_near_a_unit_word():
    assert metrics.value_stated("The grid permits a maximum of 3 storeys (étages).", "storeys_max", 3.0)
    assert metrics.value_stated("hauteur: 2 à 3 étages", "storeys_max", 3.0)
    # "3" is in the lot number; nothing about storeys is said.
    assert not metrics.value_stated("Lot 2 170 935 is in zone H02-161.", "storeys_max", 3.0)


def test_a_percentage_is_stated_near_its_sign_or_its_term():
    assert metrics.value_stated("taux d'implantation maximal de 60 %", "coverage_max_pct", 60.0)
    assert metrics.value_stated("lot coverage: 60", "coverage_max_pct", 60.0)
    assert not metrics.value_stated("zone H02-160 permits 3 storeys", "coverage_max_pct", 60.0)


def test_decimal_values_are_recognised_in_both_spellings():
    assert metrics.value_stated("2.5 storeys", "storeys_max", 2.5)
    assert metrics.value_stated("2,5 étages", "storeys_max", 2.5)
    # gpt-oss copies the grid's "2.0" out as written.
    assert metrics.value_stated("Maximum storeys: 2.0 storeys", "storeys_max", 2.0)


def test_the_metre_unit_is_a_whole_word():
    assert metrics.value_stated("maximum building height: 10.0 metres", "height_max_m", 10.0)
    assert metrics.value_stated("height max 10 m", "height_max_m", 10.0)
    # "m" inside "maximum" is not a unit; 10 here is a lot number's digits.
    assert not metrics.value_stated("maximum of 10 dwellings", "height_max_m", 10.0)


def test_answer_checks_conjoin_every_check_the_case_carries():
    expect = {"values": {"storeys_max": 3}, "must_mention_any": ["H02-161"]}
    good = metrics.answer_checks("Zone H02-161 allows up to 3 storeys.", expect)
    assert good["answer_correct"] == 1.0
    partial = metrics.answer_checks("Up to 3 storeys are permitted.", expect)
    assert partial["states_storeys_max"] == 1.0
    assert partial["mentions_any"] == 0.0
    assert partial["answer_correct"] == 0.0


def test_a_no_change_case_fails_when_a_resolution_was_retrieved():
    expect = {"no_change": True}
    clean = metrics.answer_checks("No projet particulier applies to this lot.", expect, cited_sources=["Reglement_urbanisme__VSP_REG_ZONE"])
    assert clean["no_invented_change"] == 1.0
    tainted = metrics.answer_checks("Resolution CA17 140107 was adopted.", expect, cited_sources=["Reglement_urbanisme__VSP_REG_PPCMOI"])
    assert tainted["no_invented_change"] == 0.0


def test_an_off_corpus_case_passes_when_the_answer_declines():
    assert metrics.answer_checks("The zoning data does not cover snow removal.", {"nothing": True})["declines"] == 1.0
    assert metrics.answer_checks("Snow is removed on Tuesdays.", {"nothing": True})["declines"] == 0.0


def test_a_case_with_no_checks_has_no_answer_correct_column():
    assert metrics.answer_checks("anything", {}) == {}


def test_a_zone_code_written_with_a_typographic_hyphen_still_matches():
    # gpt-oss copies "H02-086" out as "H02‑086" (U+2011); lot numbers carry U+202F.
    assert metrics.mentions_any("zone H02‑086 permits", ["H02-086"])
    assert metrics.mentions_any("lot 2 246 896", ["2 246 896"])


# -- adding up ---------------------------------------------------------------


def test_wilson_narrows_with_n_and_stays_inside_the_unit_interval():
    lo5, hi5 = metrics.wilson(4, 5)
    lo50, hi50 = metrics.wilson(40, 50)
    assert 0 <= lo5 < hi5 <= 1
    assert (hi50 - lo50) < (hi5 - lo5)
    assert metrics.wilson(0, 0) == (0.0, 0.0)


def test_percentile_interpolates():
    assert metrics.percentile([1, 2, 3, 4], 0.5) == 2.5
    assert metrics.percentile([], 0.9) is None
