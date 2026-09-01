"""The app script itself, driven by Streamlit's own test harness.

Marked ``integration``: these need a reachable database with geometry in it,
because what they check is the part unit tests cannot — that the rerun loop
converges, that a click resolves to a lot, and that the zoning pane reaches the
PDF. Run against the local container:

    make db-up && make test-integration

``st_folium`` is a custom component, so under AppTest it returns whatever the
stub here returns. That is the point: it is the browser's half of the
conversation, and stubbing it is how a pan or a click is simulated at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: AppTest resolves a relative path against its *caller's* directory, which is
#: tests/ — the script is one level up.
APP = Path(__file__).resolve().parent.parent / "app.py"

pytestmark = pytest.mark.integration

# A few blocks of Villeray, and a point inside a real VSMPE lot.
VIEWPORT = {
    "_southWest": {"lat": 45.535, "lng": -73.625},
    "_northEast": {"lat": 45.545, "lng": -73.610},
}
CLICK = {"lat": 45.5405, "lng": -73.6195}


@pytest.fixture(autouse=True)
def _database_url(monkeypatch, database_url):
    if not database_url:
        pytest.skip("set DATABASE_URL (or HBU_TEST_DATABASE_URL) to run these")
    # Put it back: conftest's autouse cleaner scrubs it for the unit suite.
    monkeypatch.setenv("DATABASE_URL", database_url)
    from src.utils.db import close_pool

    close_pool()


@pytest.fixture
def browser(monkeypatch):
    """Stub st_folium, and record how many times the script rendered the map."""
    import streamlit_folium

    calls: list[dict] = []

    def stub(_fmap, **kwargs):
        calls.append(kwargs)
        return stub.reply

    stub.reply = {}
    monkeypatch.setattr(streamlit_folium, "st_folium", stub)
    return stub, calls


def _app():
    """An AppTest for the script, already past the access gate.

    `auth.py` is documented to switch itself off when `HBU_APP_PASSWORD` is
    unset, which is what let this suite run unchanged — but a filled-in `.env`
    sets it and `app.py` calls `load_dotenv()` above the gate, so on any
    developer machine every test in this file stopped at the password form.
    The symptom named nothing: a bare `KeyError: 'Lots'` from a sidebar that
    was never drawn, and `at.exception` empty because stopping is not an error.

    Seeding the flag rather than filling the form, for one reason: the gate
    stops the script, so a login costs a full extra script run, and
    `test_a_reported_viewport_is_adopted_and_then_settles` below counts runs.
    What keeps the shortcut honest is `test_auth.py`, which drives the real
    form and asserts a successful login sets exactly this key — and, from the
    other side, that this key alone is enough to get past.

    Unconditional because it has to be: the gate being off is what this used to
    assume, and assuming it again is how the failure comes back.
    """
    from streamlit.testing.v1 import AppTest

    from src.utils import auth

    at = AppTest.from_file(str(APP), default_timeout=180)
    at.session_state[auth._STATE_KEY] = True
    return at


def test_the_first_render_survives_a_map_that_has_not_reported_yet(browser):
    """st_folium hands back corner dicts full of None before the browser answers."""
    stub, _calls = browser
    stub.reply = {
        "bounds": {"_southWest": {"lat": None, "lng": None},
                   "_northEast": {"lat": None, "lng": None}},
        "zoom": None, "center": None, "last_clicked": None,
    }

    at = _app().run()

    assert not at.exception
    assert at.session_state.viewport is None


def test_a_reported_viewport_is_adopted_and_then_settles(browser):
    """One rerun to learn where the map is, and no more — this loop can spin."""
    stub, calls = browser
    stub.reply = {"bounds": VIEWPORT, "zoom": 17,
                  "center": {"lat": 45.540, "lng": -73.6175}, "last_clicked": None}

    at = _app().run()

    assert not at.exception
    assert at.session_state.viewport == (-73.625, 45.535, -73.610, 45.545)
    assert at.session_state.map_zoom == 17
    # Initial render, then one rerun that adopts the viewport and stops.
    assert len(calls) == 2


def test_a_click_selects_the_lot_underneath_it(browser):
    stub, _calls = browser
    stub.reply = {"bounds": VIEWPORT, "zoom": 17,
                  "center": {"lat": 45.540, "lng": -73.6175}, "last_clicked": CLICK}

    at = _app().run()

    assert not at.exception
    selected = at.session_state.selected_lot
    assert selected is not None, "the click did not resolve to a lot"
    assert selected["lot_number"]
    assert selected["area_m2"] > 0
    # The heading names it, so the user can see which parcel answered.
    assert any(f"Lot {selected['lot_number']}" in str(m.value) for m in at.markdown)


def test_the_selected_lot_reaches_the_agent_tools(browser):
    """A tool asked 'what is selected' must see the click, not a stale process value."""
    stub, _calls = browser
    stub.reply = {"bounds": VIEWPORT, "zoom": 17,
                  "center": {"lat": 45.540, "lng": -73.6175}, "last_clicked": CLICK}

    at = _app().run()

    from src.utils import state

    assert state.get_selected_lot()["lot_number"] == at.session_state.selected_lot["lot_number"]
    assert state.get_viewport() == at.session_state.viewport


def test_the_zoning_pane_reaches_the_grid_pdf(browser):
    """Click to grid: the whole reason the two halves share a database."""
    stub, _calls = browser
    stub.reply = {"bounds": VIEWPORT, "zoom": 17,
                  "center": {"lat": 45.540, "lng": -73.6175}, "last_clicked": CLICK}

    at = _app().run()

    assert not at.exception
    assert any("Grille des spécifications" in str(m.value) for m in at.markdown), \
        "no grid was resolved for the clicked lot"

    kinds = [type(element).__name__ for element in at._tree]
    assert "DownloadButton" in kinds, "the grid PDF is not downloadable"
    assert "Dataframe" in kinds, "the grid's values are not tabulated"


def test_the_map_layers_reflect_what_the_database_has(browser):
    stub, _calls = browser
    stub.reply = {"bounds": VIEWPORT, "zoom": 17,
                  "center": {"lat": 45.540, "lng": -73.6175}, "last_clicked": None}

    at = _app().run()

    from src.utils import queries

    caps = queries.capabilities()
    labels = {box.label: box for box in at.sidebar.checkbox}
    assert labels["Lots"].disabled is not caps.lots
    assert labels["Buildings"].disabled is not caps.buildings


def test_nothing_deprecated_is_rendered(browser):
    """The width/use_container_width migration, guarded against sliding back."""
    stub, _calls = browser
    stub.reply = {"bounds": VIEWPORT, "zoom": 17,
                  "center": {"lat": 45.540, "lng": -73.6175}, "last_clicked": None}

    at = _app().run()

    deprecations = [
        str(w.value) for w in at.warning if "use_container_width" in str(w.value)
    ]
    assert not deprecations, deprecations
