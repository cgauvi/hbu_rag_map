"""Batching the re-embed under a token cap."""

from __future__ import annotations

from eval.corpus import _batches


def _rows(*lengths: int):
    return [("VSMPE", f"c{i}", "x" * n) for i, n in enumerate(lengths)]


def test_without_a_cap_batches_are_by_count():
    batches = list(_batches(_rows(10, 10, 10, 10, 10), batch=2, max_tokens=None, chars_per_token=3.0))
    assert [len(b) for b in batches] == [2, 2, 1]


def test_under_a_cap_batches_stop_before_the_estimated_tokens_pass_it():
    # 300 chars at 3 chars/token = 100 tokens each; cap 250 -> two per batch.
    batches = list(_batches(_rows(300, 300, 300, 300, 300), batch=64, max_tokens=250, chars_per_token=3.0))
    assert [len(b) for b in batches] == [2, 2, 1]


def test_a_single_chunk_over_the_cap_still_goes_alone():
    batches = list(_batches(_rows(3000, 30), batch=64, max_tokens=250, chars_per_token=3.0))
    assert [len(b) for b in batches] == [1, 1]
