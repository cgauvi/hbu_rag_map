"""The question set is well formed, and the generator's labelling helpers are right."""

from __future__ import annotations

import pytest

from eval import golden


def test_resolution_numbers_are_lifted_out_of_titles_both_ways():
    assert golden._resolution_tokens("Résolution CA17 140107") == ["CA17 140107", "140107"]
    assert golden._resolution_tokens("Règlement #97-040") == ["97-040"]
    assert golden._resolution_tokens("Règlements #8389, 9023 et 9316") == ["8389"]
    assert golden._resolution_tokens("") == []


def test_a_resolutions_other_identifiers_are_lifted_out_of_its_text():
    text = "DOSSIER : PP15-14012  RÉSOLUTION : CA16 140067 ... CO9902770 en vigueur"
    tokens = golden._resolution_tokens(text)
    assert "PP15-14012" in tokens and "CA16 140067" in tokens and "CO9902770" in tokens


@pytest.mark.skipif(not golden.FIXTURE.exists(), reason="the fixture has not been generated yet")
def test_every_case_carries_a_question_and_a_label():
    cases = golden.load()
    assert len(cases) >= 15, "fewer than fifteen cases: one flaky case swings the score"
    for case in cases:
        assert case.question.strip(), case.id
        assert case.category in {"storeys", "uses", "coverage", "ppcmoi", "ppcmoi-negative", "combo", "corpus", "off-corpus"}, case.id
        labelled = any(case.expect.get(k) for k in ("doc_ids", "values", "must_mention_any", "must_mention_all", "passages_contain_any", "nothing", "no_change"))
        assert labelled, f"{case.id} can never score"
        if case.lot:
            assert set(case.lot) >= {"lot_number", "neighborhood", "lon", "lat"}, case.id
    assert len({c.id for c in cases}) == len(cases)


@pytest.mark.skipif(not golden.FIXTURE.exists(), reason="the fixture has not been generated yet")
def test_the_fingerprint_is_stable_for_the_same_set():
    cases = golden.load()
    assert golden.fingerprint(cases) == golden.fingerprint(golden.load())
    assert len(golden.fingerprint(cases)) == 12
