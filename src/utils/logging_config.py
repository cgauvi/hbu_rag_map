"""
logging_config.py — Centralised logging setup for the zoning map assistant.

In-memory log buffer
--------------------
All configured loggers write to ``LogBuffer``, a thread-safe circular deque
that the Streamlit log pane reads on every rerun.  Each entry is a dict::

    {
        "ts":      str,   # "HH:MM:SS.mmm"  (UTC)
        "level":   str,   # "DEBUG" | "INFO" | "WARNING" | "ERROR" | "TOOL_IN" | …
        "logger":  str,   # logger name  (e.g. "src.agent")
        "message": str,   # formatted message text
    }

Log level
---------
Set the ``LOG_LEVEL`` environment variable (default ``INFO``).
At ``DEBUG`` level, LangChain / LangGraph internals are also captured.

Usage
-----
    from src.utils.logging_config import setup_logging, LogBuffer, clear_log_buffer

    setup_logging()        # call once at process start (safe to call repeatedly)
    clear_log_buffer()     # call on "New Conversation"
    add_log_entry(...)     # write a structured entry without going through logging
"""

from __future__ import annotations

import logging
import os
from collections import deque
from datetime import UTC, datetime

# ---------------------------------------------------------------------------
# In-memory buffer
# ---------------------------------------------------------------------------

MAX_LOG_ENTRIES = 1_000

# Each entry: {"ts": str, "level": str, "logger": str, "message": str}
LogBuffer: deque[dict] = deque(maxlen=MAX_LOG_ENTRIES)

_LEVEL_ORDER: dict[str, int] = {
    "DEBUG": 0,
    "INFO": 1,
    "WARNING": 2,
    "ERROR": 3,
    "CRITICAL": 4,
    # Custom structural levels used by the agent
    "TOOL_IN": 1,
    "TOOL_OUT": 1,
    "LLM_OUT": 1,
}


def clear_log_buffer() -> None:
    """Remove all entries from the in-memory log buffer."""
    LogBuffer.clear()


def add_log_entry(level: str, source: str, message: str) -> None:
    """Append a structured entry directly to LogBuffer.

    Use this for agent/tool events that bypass the Python ``logging`` hierarchy
    (e.g. LLM responses captured from the LangGraph stream).
    """
    ts = datetime.now(tz=UTC).strftime("%H:%M:%S.%f")[:-3]
    LogBuffer.append({"ts": ts, "level": level, "logger": source, "message": message})


def get_log_entries(min_level: str = "DEBUG") -> list[dict]:
    """Return a snapshot of LogBuffer filtered to ``min_level`` and above."""
    threshold = _LEVEL_ORDER.get(min_level.upper(), 0)
    return [e for e in LogBuffer if _LEVEL_ORDER.get(e["level"], 0) >= threshold]


# ---------------------------------------------------------------------------
# Custom handler
# ---------------------------------------------------------------------------


class _InMemoryHandler(logging.Handler):
    """Write Python log records into ``LogBuffer``."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = datetime.fromtimestamp(record.created, tz=UTC).strftime(
                "%H:%M:%S.%f"
            )[:-3]
            LogBuffer.append(
                {
                    "ts": ts,
                    "level": record.levelname,
                    "logger": record.name,
                    "message": self.format(record),
                }
            )
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Logger configuration
# ---------------------------------------------------------------------------

# Loggers whose records should be routed into LogBuffer.
_MANAGED_LOGGERS = [
    "src",
    "langchain",
    "langchain_core",
    "langgraph",
    "httpx",
    "huggingface_hub",
    "psycopg",
    "psycopg_pool",
]

_handler = _InMemoryHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))

_setup_done = False


def setup_logging() -> None:
    """Configure project and LangChain loggers to write to LogBuffer.

    Safe to call multiple times — subsequent calls are no-ops unless the
    LOG_LEVEL env var has changed, in which case call ``reset_logging()`` first.
    """
    global _setup_done
    if _setup_done:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    for name in _MANAGED_LOGGERS:
        lgr = logging.getLogger(name)
        lgr.setLevel(level)
        if _handler not in lgr.handlers:
            lgr.addHandler(_handler)
        # Prevent duplicate records via the root logger
        lgr.propagate = False

    # When DEBUG is active, enable LangChain's own verbose output which logs
    # prompts and completions through the 'langchain' logger hierarchy.
    if level <= logging.DEBUG:
        try:
            import langchain
            langchain.debug = True  # type: ignore[attr-defined]
        except Exception:
            pass

    _setup_done = True


def reset_logging() -> None:
    """Remove all managed handlers and allow ``setup_logging()`` to run again."""
    global _setup_done
    for name in _MANAGED_LOGGERS:
        lgr = logging.getLogger(name)
        lgr.handlers = [h for h in lgr.handlers if h is not _handler]
    _setup_done = False
