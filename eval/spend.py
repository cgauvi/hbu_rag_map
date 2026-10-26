"""The budget, and the callback that reads usage off every model call.

Two concerns in one file because they are one mechanism: the callback is where
a call's usage first becomes known, so it is where the ledger is charged and
where the cap is checked before the next call is allowed to start.

The ledger is an append-only JSONL file that persists across runs. The cap is
on **Anthropic spend in total**, not per run - "do not blow a $5 budget" is a
statement about the key, and a second run that forgets the first would. It is
append-only so that two configurations running at once (each in its own
process) add to one ledger rather than overwrite each other's; the cap check
re-reads the file, so what another process spent counts. HuggingFace and
Voyage spend is recorded beside it, estimated where the service does not
state a price (see ``pricing.py``), and reported, not capped.

What stops a run is `BudgetExceeded`, raised *before* a call rather than after
it: when the amount already spent plus a worst-case price for the call about
to be made would pass the cap. The worst case is the model's own ``max_tokens``
of output plus the last call's input, which is pessimistic by design - the
alternative is discovering the overrun in the bill.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from eval.pricing import add_usage, cost_usd, flat_usage, rate_for

DEFAULT_BUDGET_USD = 5.00
BUDGET_ENV = "HBU_EVAL_BUDGET_USD"
LEDGER_NAME = "spend.jsonl"


class BudgetExceeded(RuntimeError):
    """The next call would take Anthropic spend past the cap."""


def _read_entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line torn by a crash mid-write
    return entries


def _totals(entries: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for e in entries:
        totals[e["service"]] = totals.get(e["service"], 0.0) + float(e.get("usd") or 0.0)
    return totals


@dataclass
class SpendMeter:
    """The running total, on disk, shared by every process writing to it."""

    path: Path
    cap_usd: float = DEFAULT_BUDGET_USD
    totals: dict[str, float] = field(default_factory=dict)
    entries: list[dict] = field(default_factory=list)
    #: Set when the cap stops a call, so a caller that only sees the agent's
    #: apologetic final message can still tell what happened.
    tripped: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def open(cls, path: Path, cap_usd: float | None = None) -> SpendMeter:
        """``path`` is the ledger file, or the directory it lives in."""
        if path.is_dir() or path.suffix == "":
            path = path / LEDGER_NAME
        cap = cap_usd if cap_usd is not None else float(os.environ.get(BUDGET_ENV, DEFAULT_BUDGET_USD))
        meter = cls(path=path, cap_usd=cap)
        meter.refresh()
        return meter

    def refresh(self) -> None:
        """Re-read the ledger: another process may have charged it."""
        self.entries = _read_entries(self.path)
        self.totals = _totals(self.entries)

    # -- reading ---------------------------------------------------------

    @property
    def anthropic_usd(self) -> float:
        return float(self.totals.get("anthropic", 0.0))

    @property
    def remaining_usd(self) -> float:
        return self.cap_usd - self.anthropic_usd

    def summary(self) -> dict:
        return {
            "cap_usd": self.cap_usd,
            "anthropic_usd": round(self.anthropic_usd, 4),
            "remaining_usd": round(self.remaining_usd, 4),
            **{f"{k}_usd": round(v, 4) for k, v in sorted(self.totals.items()) if k != "anthropic"},
            "charges": len(self.entries),
        }

    # -- writing ---------------------------------------------------------

    def charge(self, *, service: str, model: str, usage: dict | None, **meta: Any) -> float:
        """Record one call's usage and return what it cost."""
        usd = cost_usd(model, usage)
        entry = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "service": service,
            "model": model,
            "usage": flat_usage(usage),
            "usd": round(usd, 6),
            "estimate": rate_for(model).estimate,
            **meta,
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.entries.append(entry)
            self.totals[service] = self.totals.get(service, 0.0) + usd
        return usd

    def check(self, reserve_usd: float) -> None:
        """Raise before a call that could pass the cap."""
        self.refresh()
        if self.anthropic_usd + reserve_usd > self.cap_usd:
            self.tripped = True
            raise BudgetExceeded(
                f"Anthropic spend is ${self.anthropic_usd:.3f} of a ${self.cap_usd:.2f} cap; "
                f"the next call could cost up to ${reserve_usd:.3f}. Stopping."
            )


# ---------------------------------------------------------------------------
# The callback
# ---------------------------------------------------------------------------


def _service_for(model: str) -> str:
    return "anthropic" if model.startswith("claude-") else "huggingface"


@dataclass
class LLMCall:
    model: str
    started: float
    ended: float = 0.0
    usage: dict = field(default_factory=dict)
    usd: float = 0.0
    stop_reason: str | None = None
    error: str | None = None

    @property
    def latency_s(self) -> float:
        return (self.ended or time.perf_counter()) - self.started


class UsageRecorder(BaseCallbackHandler):
    """Attached to the chat model: one `LLMCall` per request it makes.

    ``raise_error`` is what lets `BudgetExceeded` out of the callback and into
    the agent, which turns it into the turn's final message; the meter's
    ``tripped`` flag is how the runner tells that message from an answer.
    """

    raise_error = True

    def __init__(self, meter: SpendMeter, *, model_hint: str, max_tokens: int, run_meta: dict | None = None):
        self.meter = meter
        self.model_hint = model_hint
        self.max_tokens = max_tokens
        self.run_meta = run_meta or {}
        self.calls: list[LLMCall] = []
        self._open: dict[str, LLMCall] = {}
        self._last_input = 0

    # LangChain calls on_chat_model_start for chat models, on_llm_start for
    # the older text interface; the agent's models are all chat models.
    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._start(run_id)

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._start(run_id)

    def _start(self, run_id) -> None:
        if _service_for(self.model_hint) == "anthropic":
            self.meter.check(self._reserve())
        call = LLMCall(model=self.model_hint, started=time.perf_counter())
        self._open[str(run_id)] = call
        self.calls.append(call)

    def _reserve(self) -> float:
        """The most the next call could cost: last input again, full output."""
        guess_input = max(self._last_input, 20_000)
        return cost_usd(
            self.model_hint,
            {"input_tokens": int(guess_input * 1.25), "output_tokens": self.max_tokens},
        )

    def on_llm_end(self, response, *, run_id, **kwargs):
        call = self._open.pop(str(run_id), None)
        if call is None:
            return
        call.ended = time.perf_counter()
        message = None
        try:
            message = response.generations[0][0].message
        except (IndexError, AttributeError):
            pass
        usage = dict(getattr(message, "usage_metadata", None) or {})
        meta = dict(getattr(message, "response_metadata", None) or {})
        served = meta.get("model_name") or meta.get("model") or self.model_hint
        call.model = str(served)
        call.usage = usage
        call.stop_reason = meta.get("stop_reason") or meta.get("finish_reason")
        self._last_input = int(usage.get("input_tokens") or self._last_input)
        call.usd = self.meter.charge(
            service=_service_for(call.model),
            model=call.model,
            usage=usage,
            latency_s=round(call.latency_s, 3),
            **self.run_meta,
        )

    def on_llm_error(self, error, *, run_id, **kwargs):
        call = self._open.pop(str(run_id), None)
        if call is not None:
            call.ended = time.perf_counter()
            call.error = f"{type(error).__name__}: {error}"

    # -- what a case reads back ----------------------------------------------

    def totals(self) -> dict:
        usage: dict = {}
        for call in self.calls:
            add_usage(usage, flat_usage(call.usage))
        return {
            "llm_calls": len(self.calls),
            "usage": usage,
            "usd": round(sum(c.usd for c in self.calls), 6),
            "llm_latency_s": round(sum(c.latency_s for c in self.calls), 3),
            "models": sorted({c.model for c in self.calls}),
            "stop_reasons": [c.stop_reason for c in self.calls],
            "errors": [c.error for c in self.calls if c.error],
        }

    def reset(self) -> None:
        self.calls = []
        self._open = {}
