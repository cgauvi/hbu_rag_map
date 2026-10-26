"""What a call costs, from its reported usage.

Rates are USD per million tokens. The Anthropic rows are the first-party
list prices (cache writes at 1.25x input, cache reads at 0.1x). The
HuggingFace rows are **estimates** and say so in the result: the Inference API
routes a request to one of several providers and bills at that provider's
rate, which the response does not state. The figures are the common
Together/Fireworks list prices for the same weights; override them with
``HBU_EVAL_PRICE_<slug>=in,out`` when you know the rate you are actually
billed.

Nothing here is a budget. ``spend.py`` is, and it prices through this module.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rate:
    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0
    #: A list price for a service that does not report its own: shown with a
    #: tilde in reports.
    estimate: bool = False
    note: str = ""


RATES: dict[str, Rate] = {
    # Anthropic, first party. Cache write 1.25x input; cache reads as listed
    # (Opus 5.5 lists $0.20, below the 0.1x rule the other two follow).
    "claude-opus-5-5": Rate(4.00, 20.00, cache_read=0.20, cache_write=5.00),
    "claude-sonnet-5-5": Rate(2.00, 10.00, cache_read=0.20, cache_write=2.50),
    "claude-haiku-5-5": Rate(0.10, 0.50, cache_read=0.01, cache_write=0.125),
    # HuggingFace Inference API, billed at the routed provider's rate.
    "openai/gpt-oss-120b": Rate(
        0.15, 0.60, estimate=True,
        note="Together / Fireworks list price for gpt-oss-120b; the HF router "
             "bills at the provider it chose.",
    ),
    "openai/gpt-oss-20b": Rate(0.05, 0.20, estimate=True, note="provider list price"),
    "Qwen/Qwen2.5-72B-Instruct": Rate(0.90, 0.90, estimate=True, note="provider list price"),
    "google/gemma-3-27b-it": Rate(0.10, 0.20, estimate=True, note="provider list price"),
    # Encoders. Voyage states a price; the HF feature-extraction endpoint is
    # billed per compute second and a one-sentence query rounds to nothing.
    "voyage-3.5": Rate(0.06, 0.0),
    "BAAI/bge-m3": Rate(0.0, 0.0, estimate=True, note="HF Inference API, per-second billing"),
}


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").upper()


def rate_for(model: str) -> Rate:
    """The rate for a model id, honouring an ``HBU_EVAL_PRICE_<slug>`` override.

    Unknown models get a zero rate flagged as an estimate rather than an
    error: a report with a missing price column still has every other column,
    and the flag is what keeps the zero from reading as "free".
    """
    override = os.environ.get(f"HBU_EVAL_PRICE_{_slug(model)}", "")
    base = RATES.get(model) or RATES.get(_strip_date(model))
    if override:
        parts = [float(p) for p in override.split(",")]
        inp, out = parts[0], (parts[1] if len(parts) > 1 else 0.0)
        return Rate(inp, out, cache_read=inp * 0.1, cache_write=inp * 1.25)
    if base is None:
        return Rate(0.0, 0.0, estimate=True, note=f"no rate on file for {model}")
    return base


def _strip_date(model: str) -> str:
    """``claude-sonnet-5-5-20260401`` -> ``claude-sonnet-5-5``: a served id may
    carry a snapshot date the rate table does not."""
    return re.sub(r"-\d{8}$", "", model)


def cost_usd(model: str, usage: dict | None) -> float:
    """Dollars for one call's usage.

    ``usage`` is LangChain's ``usage_metadata`` shape, or the Anthropic SDK's,
    or the harness's own flat form - the four keys below are read from any of
    them. Cached input tokens are **excluded** from ``input_tokens`` when the
    details say so, because the Anthropic SDK reports them separately while
    LangChain folds them into the total.
    """
    if not usage:
        return 0.0
    rate = rate_for(model)
    inp = int(usage.get("input_tokens") or 0)
    out = int(usage.get("output_tokens") or 0)
    details = usage.get("input_token_details") or {}
    cache_read = int(details.get("cache_read") or usage.get("cache_read_input_tokens") or 0)
    cache_write = int(
        details.get("cache_creation") or usage.get("cache_creation_input_tokens") or 0
    )
    if details:
        # LangChain's input_tokens is the total including the cached part.
        inp = max(0, inp - cache_read - cache_write)
    return (
        inp * rate.input
        + out * rate.output
        + cache_read * rate.cache_read
        + cache_write * rate.cache_write
    ) / 1_000_000


def flat_usage(usage: dict | None) -> dict:
    """The four counts a result row carries, from any of the usage shapes."""
    usage = usage or {}
    details = usage.get("input_token_details") or {}
    cache_read = int(details.get("cache_read") or usage.get("cache_read_input_tokens") or 0)
    cache_write = int(
        details.get("cache_creation") or usage.get("cache_creation_input_tokens") or 0
    )
    inp = int(usage.get("input_tokens") or 0)
    if details:
        inp = max(0, inp - cache_read - cache_write)
    return {
        "input_tokens": inp,
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
    }


def add_usage(total: dict, more: dict) -> dict:
    for key, value in more.items():
        total[key] = int(total.get(key) or 0) + int(value or 0)
    return total
