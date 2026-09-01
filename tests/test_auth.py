"""The shared password in front of the map.

Worth its own file because this is the only thing standing between a public URL
and the database — `auth.require_password()` runs above every query and every
Inference API call in `app.py`, and until now nothing tested it.

Driven through Streamlit's own harness rather than by stubbing `st`: the gate's
behaviour *is* its rendering — it draws a form and calls `st.stop()`, and a
mocked `st` would assert that those functions were called rather than that the
script actually stopped.

These are unit tests. The gate returns or stops before `app.py` reaches a
database, so nothing here needs one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils import auth

#: The repo root, so the generated script can import `src` the way the suite
#: does. AppTest runs it in this process but from a synthetic path.
ROOT = Path(__file__).resolve().parent.parent

#: A one-line app: the gate, then a marker that only renders past it.
PAST = "past the gate"
SCRIPT = f"""
import sys
sys.path.insert(0, r"{ROOT}")
import streamlit as st
from src.utils import auth
auth.require_password()
st.write({PAST!r})
"""


def _run(**session_state):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(SCRIPT, default_timeout=30)
    for key, value in session_state.items():
        at.session_state[key] = value
    return at.run()


def _got_past(at) -> bool:
    return any(PAST in str(m.value) for m in at.markdown)


def test_an_unset_password_disables_the_gate(monkeypatch):
    """What lets `make run` and this suite work on a laptop with no secret."""
    monkeypatch.delenv(auth.PASSWORD_ENV, raising=False)

    at = _run()

    assert not at.exception
    assert _got_past(at)
    assert not at.text_input, "a gate was drawn with no password configured"


def test_an_empty_password_is_the_same_as_unset(monkeypatch):
    """ecs.tf injects the variable unconditionally; empty means 'no gate'."""
    monkeypatch.setenv(auth.PASSWORD_ENV, "   ")

    assert _got_past(_run())


def test_a_configured_password_stops_the_script(monkeypatch):
    """Nothing below the gate runs — no query, no Inference API call."""
    monkeypatch.setenv(auth.PASSWORD_ENV, "s3cret")

    at = _run()

    assert not at.exception
    assert not _got_past(at)
    assert [t.label for t in at.text_input] == ["Password"]


def test_the_right_password_opens_the_gate_and_sets_the_flag(monkeypatch):
    """The one test `test_app.py` leans on.

    That suite seeds `auth._STATE_KEY` rather than filling this form on every
    test, so what makes the shortcut honest is this: a real login sets exactly
    that key and nothing else is needed to get past.
    """
    monkeypatch.setenv(auth.PASSWORD_ENV, "s3cret")

    at = _run()
    at.text_input[0].set_value("s3cret")
    at.button[0].click().run()

    assert not at.exception
    assert _got_past(at)
    assert at.session_state[auth._STATE_KEY] is True


def test_a_seeded_flag_is_enough_to_get_past(monkeypatch):
    """The other half of the same contract, from the other direction."""
    monkeypatch.setenv(auth.PASSWORD_ENV, "s3cret")

    at = _run(**{auth._STATE_KEY: True})

    assert _got_past(at)
    assert not at.text_input


def test_the_wrong_password_is_refused(monkeypatch):
    monkeypatch.setenv(auth.PASSWORD_ENV, "s3cret")

    at = _run()
    at.text_input[0].set_value("hunter2")
    at.button[0].click().run()

    assert not at.exception
    assert not _got_past(at)
    assert any("Incorrect password" in str(e.value) for e in at.error)


@pytest.mark.parametrize("entered", ["mot-de-passé", "клавиатура", "🔑"])
def test_a_non_ascii_attempt_fails_rather_than_raising(monkeypatch, entered):
    """`compare_digest` rejects a str holding anything outside ASCII.

    Comparing as str would raise TypeError here and surface as a crash rather
    than a refusal, which is why `auth.py` encodes both sides to bytes first.
    """
    monkeypatch.setenv(auth.PASSWORD_ENV, "s3cret")

    at = _run()
    at.text_input[0].set_value(entered)
    at.button[0].click().run()

    assert not at.exception, "a non-ASCII password crashed the gate"
    assert not _got_past(at)


def test_a_non_ascii_password_can_be_the_right_one(monkeypatch):
    """Encoding both sides means an accented secret still works."""
    monkeypatch.setenv(auth.PASSWORD_ENV, "mot-de-passé")

    at = _run()
    at.text_input[0].set_value("mot-de-passé")
    at.button[0].click().run()

    assert not at.exception
    assert _got_past(at)


def test_terraforms_placeholder_refuses_every_login(monkeypatch):
    """A task that booted before anyone set a value must accept nothing.

    The placeholder is written down in a .tf file, so treating it as a password
    would be worse than having no gate at all — it would look like one.
    """
    monkeypatch.setenv(auth.PASSWORD_ENV, "PLACEHOLDER")

    at = _run()

    assert not _got_past(at)
    assert not at.text_input, "the placeholder drew a form it can never accept"
    assert any("make app-password" in str(e.value) for e in at.error)
