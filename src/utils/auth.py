"""
auth.py — the shared password in front of the map.

One password, the same for everyone, checked in the app rather than at the
load balancer. That is a deliberate limitation and worth stating plainly:

  - It authenticates *access*, not people. Nothing here tells you who asked a
    question, and revoking one person's access means changing the password for
    everyone.
  - Over a plain HTTP listener it crosses the wire in the clear. The listener
    is where that gets fixed (an ACM certificate on the ALB), not here.

What it does buy is that a public URL does not mean a public database: nothing
below the gate runs — no query, no Inference API call — until the password
matches.

Unset, ``HBU_APP_PASSWORD`` disables the gate entirely, which is what makes
``make run`` and the test suite work unchanged on a laptop. The deployed task
always has it set, injected from Secrets Manager by the ECS agent.
"""

from __future__ import annotations

import hmac
import os

import streamlit as st

#: The environment variable ecs.tf injects the secret into.
PASSWORD_ENV = "HBU_APP_PASSWORD"

#: Where the successful check is remembered. Streamlit replays the whole script
#: on every interaction, so this is read many times per session and written once.
_STATE_KEY = "_authenticated"

#: Terraform creates the secret before anyone sets a value. A task that boots
#: with the placeholder still in place should refuse every login rather than
#: accept a password that is written down in a .tf file.
_PLACEHOLDER = "PLACEHOLDER"


def _expected() -> str | None:
    """The configured password, or None when the gate is switched off."""
    value = os.environ.get(PASSWORD_ENV, "").strip()
    return value or None


def require_password() -> None:
    """Render the gate and stop the script unless this session is past it.

    Returns normally when the caller may proceed. Otherwise it draws the form
    and calls ``st.stop()``, so nothing after it in the script runs.
    """
    expected = _expected()
    if expected is None:
        return

    if st.session_state.get(_STATE_KEY):
        return

    if expected == _PLACEHOLDER:
        st.error(
            "This deployment has no access password set yet — Terraform's "
            "placeholder is still in place, so no password can be accepted.\n\n"
            "Run `make app-password ENV=<env>` in `hbu_infra`, then redeploy."
        )
        st.stop()

    st.title("🏙️ HBU Zoning Map")
    st.caption("Enter the shared access password to continue.")

    with st.form("login"):
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")

    if submitted:
        # compare_digest rather than ==, so the time the comparison takes does
        # not leak how many leading characters were right. Bytes rather than
        # str, because compare_digest rejects a str containing anything outside
        # ASCII — and a password with an accent in it should fail the login,
        # not raise a TypeError.
        if hmac.compare_digest(entered.encode("utf-8"), expected.encode("utf-8")):
            st.session_state[_STATE_KEY] = True
            # The rerun is what makes the gate disappear: this run has already
            # drawn the form, and the next one returns early above it.
            st.rerun()
        else:
            st.error("Incorrect password.")

    st.stop()
