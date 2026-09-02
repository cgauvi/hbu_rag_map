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
    """The map's position is learned, and learning it does not spin.

    Under the tile renderer that costs *no* rerun at all: nothing Python draws
    depends on where the map is looking, so the report is recorded where it
    lands and the script ends. Under the GeoJSON renderer one rerun is owed,
    because its layers are queried from the viewport above the map. Either
    way the loop stops — which is the property this guards.
    """
    stub, calls = browser
    stub.reply = {"bounds": VIEWPORT, "zoom": 17,
                  "center": {"lat": 45.540, "lng": -73.6175}, "last_clicked": None}

    at = _app().run()

    assert not at.exception
    assert at.session_state.viewport == (-73.625, 45.535, -73.610, 45.545)
    assert at.session_state.view_zoom == 17
    assert at.session_state.view_center == [45.540, -73.6175]

    renderer = at.session_state.map_signature[0]
    assert len(calls) == (1 if renderer == "tiles" else 2)


def test_a_pan_does_not_move_what_the_map_is_built_at(browser):
    """The anchor is what `st_folium` hashes into its component key.

    A pan that moved it would rebuild the map object, change the key, remount
    the iframe and refetch every tile — which is the pane visibly redrawing
    itself as the user drags. The browser keeps its own position instead, and
    reports it back for the notes and the tools to read.
    """
    stub, calls = browser
    stub.reply = {"bounds": VIEWPORT, "zoom": 17,
                  "center": {"lat": 45.540, "lng": -73.6175}, "last_clicked": None}

    at = _app().run()
    if at.session_state.map_signature[0] != "tiles":
        pytest.skip("the GeoJSON renderer queries from the viewport by design")

    from src.utils import basemap

    assert at.session_state.map_center == list(basemap.DEFAULT_CENTER)
    assert at.session_state.map_zoom == basemap.DEFAULT_ZOOM
    # ... and the map was handed to the component exactly once for it.
    assert len(calls) == 1


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
    assert any("Zoning grid" in str(m.value) for m in at.markdown), \
        "no grid was resolved for the clicked lot"

    kinds = [type(element).__name__ for element in at._tree]
    assert "DownloadButton" in kinds, "the grid PDF is not downloadable"
    assert "Dataframe" in kinds, "the grid's values are not tabulated"


def _urls(at) -> list[str]:
    """Every ``url`` an element carries — link buttons, mostly.

    AppTest has no wrapper class for `st.link_button` or `st.iframe`, so they
    arrive as ``UnknownElement`` and the proto is the only place to read them
    off. That is why this reaches for the proto rather than for a nice
    accessor: there is not one.
    """
    return [
        element.proto.url
        for element in at._tree
        if getattr(element, "proto", None) is not None
        and getattr(element.proto, "url", "")
    ]


def _framed(at) -> list[str]:
    """Every ``src`` an element carries, which here means the PDF viewer."""
    return [
        element.proto.src
        for element in at._tree
        if getattr(element, "proto", None) is not None
        and getattr(element.proto, "src", "")
    ]


def test_the_grid_is_offered_as_a_link_not_only_as_an_image(browser):
    """The link is the half a reader keeps: a LIEN_GRILLE they can paste into a
    report, or open in a tab beside the map."""
    stub, _calls = browser
    stub.reply = {"bounds": VIEWPORT, "zoom": 17,
                  "center": {"lat": 45.540, "lng": -73.6175}, "last_clicked": CLICK}

    at = _app().run()

    assert not at.exception
    pdfs = [url for url in _urls(at) if ".pdf" in url]
    assert pdfs, "the zoning grid is reachable only as a picture of itself"
    # The city's own URL is among them, because it is the citable one and it
    # outlives this deployment.
    assert any(url.startswith("http") for url in pdfs)


def test_the_grid_is_served_from_this_app_so_it_can_be_framed(browser):
    """An http:// city link cannot be framed by an https:// page, which is the
    whole reason `tiles` publishes the PDF at all."""
    from src.utils import tiles

    stub, _calls = browser
    stub.reply = {"bounds": VIEWPORT, "zoom": 17,
                  "center": {"lat": 45.540, "lng": -73.6175}, "last_clicked": CLICK}

    at = _app().run()

    assert not at.exception
    if tiles.port() is None:
        pytest.skip("the tile server did not take a port, so nothing is published")

    served = [url for url in _urls(at) if tiles.GRID_PREFIX in url]
    assert served, "the grid has no address on this app's own origin"
    # And the viewer is pointed at exactly that address.
    assert [src for src in _framed(at) if tiles.GRID_PREFIX in src], \
        "the grid is linked but not embedded"


def test_the_published_grid_is_the_one_the_route_will_serve(browser):
    """The link on the page and the bytes behind it agree — the pane publishes
    what it fetched, so the id it links to resolves."""
    from src.utils import documents, tiles

    stub, _calls = browser
    stub.reply = {"bounds": VIEWPORT, "zoom": 17,
                  "center": {"lat": 45.540, "lng": -73.6175}, "last_clicked": CLICK}

    at = _app().run()

    assert not at.exception
    if tiles.port() is None:
        pytest.skip("the tile server did not take a port, so nothing is published")

    served = [url for url in _urls(at) if tiles.GRID_PREFIX in url]
    if not served:
        pytest.skip("the clicked lot's zone links no grid in this snapshot")

    doc_id = served[0].split("/")[-1].split(".pdf")[0]
    assert (documents.published(doc_id) or b"").startswith(b"%PDF")


def test_a_click_that_finds_no_lot_resolves_the_zone_instead(browser, monkeypatch):
    """Zoning covers ground the cadastre does not — a park, a right of way —
    and the grid that applies there is a real answer rather than an absence."""
    from src.utils import queries

    stub, _calls = browser
    stub.reply = {"bounds": VIEWPORT, "zoom": 17,
                  "center": {"lat": 45.540, "lng": -73.6175}, "last_clicked": CLICK}
    # The point is inside a real lot, so the miss is arranged rather than
    # hunted for: what is under test is the fallback, not the cadastre.
    monkeypatch.setattr(queries, "lot_at_point", lambda *a, **k: None)

    at = _app().run()

    assert not at.exception
    assert at.session_state.selected_lot is None
    zone = at.session_state.selected_zone
    assert zone is not None, "a click on zoned ground resolved nothing"
    assert any(f"Zone {zone['zone']}" in str(m.value) for m in at.markdown)
    assert any("Zoning grid" in str(m.value) for m in at.markdown)


def test_selecting_a_lot_drops_a_zone_selected_on_its_own(browser):
    """Two selections disagreeing about which zone is under discussion is the
    one thing this pane exists to prevent."""
    stub, _calls = browser
    stub.reply = {"bounds": VIEWPORT, "zoom": 17,
                  "center": {"lat": 45.540, "lng": -73.6175}, "last_clicked": CLICK}

    at = _app()
    at.session_state["selected_zone"] = {"zone": "C01-001", "attributes": {}}
    at.run()

    assert not at.exception
    assert at.session_state.selected_lot is not None
    assert at.session_state.selected_zone is None


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
