"""Pricing and the budget: a dollar figure for a usage, and the stop before the cap."""

from __future__ import annotations

import pytest

from eval import pricing
from eval.spend import BudgetExceeded, SpendMeter, UsageRecorder


def test_sonnet_pricing_prices_cache_reads_at_a_tenth():
    plain = pricing.cost_usd("claude-sonnet-5-5", {"input_tokens": 1_000_000, "output_tokens": 0})
    cached = pricing.cost_usd("claude-sonnet-5-5", {"input_tokens": 0, "cache_read_input_tokens": 1_000_000, "output_tokens": 0})
    assert plain == pytest.approx(2.00)
    assert cached == pytest.approx(0.20)


def test_langchain_usage_folds_cached_tokens_into_the_total_and_is_unfolded_here():
    usage = {"input_tokens": 12_000, "output_tokens": 100,
             "input_token_details": {"cache_read": 10_000, "cache_creation": 0}}
    flat = pricing.flat_usage(usage)
    assert flat["input_tokens"] == 2_000
    assert flat["cache_read_input_tokens"] == 10_000
    assert pricing.cost_usd("claude-sonnet-5-5", usage) == pytest.approx(2_000 * 2 / 1e6 + 10_000 * 0.2 / 1e6 + 100 * 10 / 1e6)


def test_an_unknown_model_is_priced_at_zero_and_flagged():
    rate = pricing.rate_for("nobody/such-model")
    assert rate.estimate and rate.input == 0.0


def test_a_served_id_with_a_snapshot_date_still_finds_its_rate():
    assert pricing.rate_for("claude-haiku-5-5-20260101").input == pricing.rate_for("claude-haiku-5-5").input


def test_a_price_override_wins(monkeypatch):
    monkeypatch.setenv("HBU_EVAL_PRICE_OPENAI_GPT_OSS_120B", "1.0,2.0")
    rate = pricing.rate_for("openai/gpt-oss-120b")
    assert (rate.input, rate.output) == (1.0, 2.0)


def test_the_meter_persists_and_sums_per_service(tmp_path):
    meter = SpendMeter.open(tmp_path, cap_usd=1.0)
    meter.charge(service="anthropic", model="claude-sonnet-5-5", usage={"input_tokens": 100_000, "output_tokens": 0})
    meter.charge(service="huggingface", model="openai/gpt-oss-120b", usage={"input_tokens": 1_000_000, "output_tokens": 0})
    again = SpendMeter.open(tmp_path / "spend.jsonl", cap_usd=1.0)
    assert again.anthropic_usd == pytest.approx(0.20)
    assert again.totals["huggingface"] == pytest.approx(0.15)
    assert again.entries[1]["estimate"] is True


def test_two_meters_on_one_ledger_see_each_others_spend(tmp_path):
    """Two configurations run in two processes; the cap is on the sum."""
    first = SpendMeter.open(tmp_path, cap_usd=0.30)
    second = SpendMeter.open(tmp_path, cap_usd=0.30)
    first.charge(service="anthropic", model="claude-sonnet-5-5", usage={"input_tokens": 100_000, "output_tokens": 0})
    second.charge(service="anthropic", model="claude-sonnet-5-5", usage={"input_tokens": 50_000, "output_tokens": 0})
    with pytest.raises(BudgetExceeded):
        second.check(0.01)  # 0.20 + 0.10 spent in all, 0.30 cap
    assert len(SpendMeter.open(tmp_path).entries) == 2


def test_the_cap_stops_the_call_that_would_pass_it(tmp_path):
    meter = SpendMeter.open(tmp_path / "s.jsonl", cap_usd=0.25)
    meter.charge(service="anthropic", model="claude-sonnet-5-5", usage={"input_tokens": 100_000, "output_tokens": 0})
    meter.check(0.04)
    with pytest.raises(BudgetExceeded):
        meter.check(0.06)
    assert meter.tripped


def test_the_recorder_charges_the_meter_from_the_message_usage(tmp_path):
    meter = SpendMeter.open(tmp_path / "s.jsonl", cap_usd=5.0)
    recorder = UsageRecorder(meter, model_hint="claude-sonnet-5-5", max_tokens=4096)

    class Msg:
        usage_metadata = {"input_tokens": 1_000, "output_tokens": 50}
        response_metadata = {"model_name": "claude-sonnet-5-5", "stop_reason": "end_turn"}

    class Gen:
        message = Msg()

    class Response:
        generations = [[Gen()]]

    recorder.on_chat_model_start({}, [[]], run_id="r1")
    recorder.on_llm_end(Response(), run_id="r1")
    totals = recorder.totals()
    assert totals["llm_calls"] == 1
    assert totals["usage"]["input_tokens"] == 1_000
    assert totals["usd"] == pytest.approx(1_000 * 2 / 1e6 + 50 * 10 / 1e6)
    assert meter.anthropic_usd == totals["usd"]
    assert totals["stop_reasons"] == ["end_turn"]


def test_the_recorder_refuses_to_start_an_anthropic_call_past_the_cap(tmp_path):
    meter = SpendMeter.open(tmp_path / "s.jsonl", cap_usd=0.01)
    recorder = UsageRecorder(meter, model_hint="claude-sonnet-5-5", max_tokens=4096)
    with pytest.raises(BudgetExceeded):
        recorder.on_chat_model_start({}, [[]], run_id="r1")


def test_huggingface_calls_are_never_capped(tmp_path):
    meter = SpendMeter.open(tmp_path / "s.jsonl", cap_usd=0.0)
    recorder = UsageRecorder(meter, model_hint="openai/gpt-oss-120b", max_tokens=4096)
    recorder.on_chat_model_start({}, [[]], run_id="r1")  # no raise
