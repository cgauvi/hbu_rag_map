"""
app.py — Streamlit front end: an interactive zoning map with a chat panel.

Two inputs, one state. The map is directly interactive — pan and zoom draw lots
and buildings, and clicking a lot selects it — while the chat panel reaches the
same data through tools and can move the map back. Both write the selection to
the same place, so "what can I build here" asked after a click means the lot
that was clicked.

**The geometry arrives as vector tiles**, off the small HTTP server
`src/utils/tiles.py` runs beside Streamlit in this same process. That is the
one thing about this file worth knowing before reading it, because it decides
what the rest does *not* do: no layer is fetched here, nothing about a
viewport is cached, and panning is a thing the browser finishes by itself.
Before tiles, every shape in view was queried, turned into GeoJSON, embedded
in the map document and shipped down the websocket on every rerun — which is
why the viewport reads have a cap, and why a borough at zoom 15 was a tab that
stopped responding. ``HBU_MAP_RENDERER=geojson`` still selects that path, and
the app falls back to it by itself if the tile server could not take its port.

The click is resolved **server-side**, from its coordinates rather than from
whatever shape the browser reports being hit. A click near a boundary, on a lot
the viewport limit left undrawn, or on a simplified edge all still land on the
right parcel that way — and under the tile renderer it is also what keeps the
selection honest, since the browser is holding clipped tile geometry rather
than parcels.

A note on state: the map's real position belongs to the browser and comes back
through ``st_folium`` on every rerun. This file keeps it in ``st.session_state``
and mirrors it into ``src.utils.state`` at the top of each run, so the agent's
tools see this session's viewport and selection rather than a process-wide one.
"""

import os
from datetime import date

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.utils.logging_config import (  # noqa: E402
    LogBuffer,
    clear_log_buffer,
    setup_logging,
)

setup_logging()

IS_DEV = os.getenv("APP_ENV", "dev").lower() == "dev"

st.set_page_config(
    page_title="HBU Zoning Map",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.utils import auth  # noqa: E402

# The gate goes here, immediately after set_page_config and before anything
# touches the database: a wrong password should cost a form render, not a
# connection. It is a no-op when HBU_APP_PASSWORD is unset, which is every
# local run; the deployed task always has it, injected from Secrets Manager.
auth.require_password()

from src.utils import basemap, queries, state, tiles  # noqa: E402

# ---------------------------------------------------------------------------
# The tile server
#
# `cache_resource`, not `cache_data`: this is a socket and a thread, one per
# process, and it must not be copied per session. Streamlit calls this on the
# first script run and hands back the same answer to every session after it.
#
# A None port is not fatal. The usual cause is that something already holds
# 8502 — most often a previous `streamlit run` that has not exited — and the
# right response is to draw the map the old way and say so, rather than to
# show an empty map with no explanation.
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _tile_port() -> int | None:
    return tiles.start()


def _renderer(caps) -> tuple[str, str | None]:
    """``("tiles"|"geojson", why)`` — which renderer draws this session's map.

    ``HBU_MAP_RENDERER`` forces either one. Unset, tiles are used when the
    database is new enough for ``ST_AsMVT`` and the server took its port, and
    the reason for any fallback is returned so the pane can show it: a map
    that quietly halves its own capacity is a map nobody debugs.
    """
    configured = os.getenv("HBU_MAP_RENDERER", "auto").strip().lower()
    if configured == "geojson":
        return "geojson", None
    if not caps.mvt:
        return "geojson", (
            "This PostGIS is older than 3.1, which has no `ST_AsMVT` — "
            "drawing GeoJSON, capped at "
            f"{queries.DEFAULT_FEATURE_LIMIT} shapes per layer."
        )
    if _tile_port() is None:
        return "geojson", (
            f"The tile server could not take port {tiles.DEFAULT_PORT} — "
            "drawing GeoJSON, capped at "
            f"{queries.DEFAULT_FEATURE_LIMIT} shapes per layer. Set "
            "`HBU_TILE_PORT` to a free port, or stop whatever holds this one."
        )
    return "tiles", None

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "messages": [],
    "log_entries": [],
    "selected_lot": None,      # a row from queries.lot_at_point / lot_by_number
    "map_center": list(basemap.DEFAULT_CENTER),
    "map_zoom": basemap.DEFAULT_ZOOM,
    "viewport": None,          # (west, south, east, north), as the browser sees it
    "fit_bounds": None,
    "layers": {
        "lots": True, "buildings": True, "zones": False,
        "capacity": False, "streets": False, "massing": False,
    },
    "filters": {"min_area_m2": None, "max_area_m2": None},
    "neighborhood": None,
    "scrape_date": None,
    "agent_note": None,
    "last_click": None,
}

for _key, _value in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value.copy() if isinstance(_value, (dict, list)) else _value

# A nested default gains keys as the app gains layers and filters, and a session
# that started before one was added still holds the older shape - which is a
# KeyError the first time the sidebar reads the new toggle. Backfilled rather
# than replaced, so what the user had switched on stays on across a reload.
for _key, _value in _DEFAULTS.items():
    if isinstance(_value, dict):
        for _sub_key, _sub_default in _value.items():
            st.session_state[_key].setdefault(_sub_key, _sub_default)


# ---------------------------------------------------------------------------
# Cached data access
#
# Streamlit replays the whole script on every interaction, and st_folium reports
# a viewport on every one of them. Caching on a *rounded* bounding box means a
# nudge of the map redraws from memory while a real move re-queries.
# ---------------------------------------------------------------------------


def _cache_key(bounds, precision: int = 4) -> tuple:
    return tuple(round(float(v), precision) for v in bounds)


def _reported_bounds(result: dict) -> tuple[float, float, float, float] | None:
    """``(west, south, east, north)`` from st_folium, or None.

    The component reports the corner dicts before the browser has filled them
    in — present, but with ``None`` for lat and lng — so their existence is not
    enough to go on.
    """
    corners = result.get("bounds") or {}
    south_west, north_east = corners.get("_southWest"), corners.get("_northEast")
    if not isinstance(south_west, dict) or not isinstance(north_east, dict):
        return None
    values = (
        south_west.get("lng"), south_west.get("lat"),
        north_east.get("lng"), north_east.get("lat"),
    )
    if any(v is None for v in values):
        return None
    return tuple(float(v) for v in values)


def _reported_center(result: dict) -> list[float] | None:
    center = result.get("center") or {}
    if not isinstance(center, dict):
        return None
    lat, lng = center.get("lat"), center.get("lng")
    return None if lat is None or lng is None else [float(lat), float(lng)]


def _moved_enough(old, new, frac: float = 1 / 3) -> bool:
    """Did the viewport move or zoom enough to be worth a redraw?

    st_folium re-reports its bounds on every rerun, most of them a few metres
    off the last — comparing them exactly, or even at ~100 m, turns that jitter
    into a permanent rerun loop (the blinking map). This asks the only question
    that matters: did the centre shift by more than ``frac`` of the span, or
    the span itself change by more than a quarter (i.e. a zoom)?
    """
    if not old:
        return True
    ow, os_, oe, on = old
    nw, ns, ne, nn = new
    span_x, span_y = max(oe - ow, 1e-9), max(on - os_, 1e-9)
    d_cx = abs((nw + ne - ow - oe) / 2)
    d_cy = abs((ns + nn - os_ - on) / 2)
    d_span = max(
        abs((ne - nw) - (oe - ow)) / span_x,
        abs((nn - ns) - (on - os_)) / span_y,
    )
    return d_cx > span_x * frac or d_cy > span_y * frac or d_span > 0.25


@st.cache_data(ttl=300, show_spinner=False)
def _capabilities():
    return queries.capabilities()


@st.cache_data(ttl=300, show_spinner=False)
def _partitions(table: str):
    return queries.neighborhoods(table), queries.scrape_dates(table)


@st.cache_data(ttl=120, show_spinner=False)
def _lots(bounds_key, zoom, scrape_date, neighborhood, min_area, max_area):
    return queries.lots_in_bbox(
        bounds_key, zoom=zoom, scrape_date=scrape_date, neighborhood=neighborhood,
        min_area_m2=min_area, max_area_m2=max_area,
    )


@st.cache_data(ttl=120, show_spinner=False)
def _buildings(bounds_key, zoom, scrape_date, neighborhood):
    return queries.buildings_in_bbox(
        bounds_key, zoom=zoom, scrape_date=scrape_date, neighborhood=neighborhood
    )


@st.cache_data(ttl=120, show_spinner=False)
def _zones(bounds_key, zoom, scrape_date, neighborhood):
    return queries.zones_in_bbox(
        bounds_key, zoom=zoom, scrape_date=scrape_date, neighborhood=neighborhood
    )


@st.cache_data(ttl=120, show_spinner=False)
def _streets(bounds_key, zoom, scrape_date, neighborhood):
    return queries.streets_in_bbox(
        bounds_key, zoom=zoom, scrape_date=scrape_date, neighborhood=neighborhood
    )


@st.cache_data(ttl=120, show_spinner=False)
def _massing(bounds_key, zoom, scrape_date, neighborhood, only_underbuilt):
    return queries.massing_in_bbox(
        bounds_key, zoom=zoom, scrape_date=scrape_date, neighborhood=neighborhood,
        only_underbuilt=only_underbuilt,
    )


@st.cache_data(ttl=120, show_spinner=False)
def _capacity(bounds_key, zoom, scrape_date, neighborhood, only_underbuilt):
    return queries.capacity_in_bbox(
        bounds_key, zoom=zoom, scrape_date=scrape_date, neighborhood=neighborhood,
        only_underbuilt=only_underbuilt,
    )


# Whether a gold layer has any rows for this borough-snapshot at all. Cached
# hard, because it answers a question about a *partition*: it changes when the
# pipeline runs, not when the map moves.
@st.cache_data(ttl=300, show_spinner=False)
def _partition_has_rows(layer, scrape_date, neighborhood):
    return queries.partition_has_rows(
        layer, scrape_date=scrape_date, neighborhood=neighborhood
    )


# Borough-wide rather than by viewport, so it is one aggregate over a partition
# and not a number that changes as the user pans. Cached for longer than the
# map layers for the same reason: nothing about it depends on where the map is.
@st.cache_data(ttl=900, show_spinner=False)
def _capacity_totals(neighborhood, scrape_date):
    return queries.capacity_totals(
        neighborhood=neighborhood, scrape_date=scrape_date
    )


@st.cache_data(ttl=900, show_spinner=False)
def _top_capacity_lots(neighborhood, scrape_date):
    return queries.top_capacity_lots(
        neighborhood=neighborhood, scrape_date=scrape_date
    )


@st.cache_data(ttl=900, show_spinner=False)
def _top_npv_gain_lots(neighborhood, scrape_date):
    return queries.top_npv_gain_lots(
        neighborhood=neighborhood, scrape_date=scrape_date
    )


@st.cache_data(ttl=300, show_spinner=False)
def _lot_capacity(lot_uid, scrape_date, neighborhood):
    return queries.lot_capacity(
        lot_uid, scrape_date=scrape_date, neighborhood=neighborhood
    )


@st.cache_data(ttl=300, show_spinner=False)
def _zoning_for_lot(lot_number):
    return queries.zoning_for_lot(lot_number)


@st.cache_data(ttl=300, show_spinner=False)
def _footprints_on_lot(lot_number):
    return queries.buildings_on_lot(lot_number)


@st.cache_data(ttl=3600, show_spinner="Fetching the grille des spécifications…")
def _zoning_pdf(url):
    """Fetch and rasterise a grid PDF.

    Returns ``(content, filename, pages, render_error)`` rather than raising for
    an unrenderable file: a PDF that cannot be rasterised can still be
    downloaded, and that is a better outcome than an empty pane.
    """
    from src.utils import documents  # noqa: PLC0415

    document = documents.fetch(url)
    try:
        return document.content, document.filename, documents.render_pages(document.content), None
    except documents.DocumentError as exc:
        return document.content, document.filename, [], str(exc)


def _select_lot(lot: dict | None) -> None:
    """Adopt a lot as the selection, on both sides of the app."""
    st.session_state.selected_lot = lot
    if lot:
        state.set_selected_lot(
            lot["lot_number"], lot.get("lon"), lot.get("lat"), lot.get("neighborhood")
        )
    else:
        state.clear_selected_lot()


# ---------------------------------------------------------------------------
# Mirror this session's map state into the module the tools read
# ---------------------------------------------------------------------------

state.set_viewport(
    st.session_state.viewport,
    st.session_state.map_zoom,
    tuple(st.session_state.map_center),
)
if st.session_state.selected_lot:
    _selected = st.session_state.selected_lot
    state.set_selected_lot(
        _selected["lot_number"], _selected.get("lon"), _selected.get("lat"),
        _selected.get("neighborhood"),
    )
else:
    state.clear_selected_lot()

# ---------------------------------------------------------------------------
# Apply anything the last agent turn asked of the map
#
# Ahead of the sidebar so a layer the chat turned on is already checked when the
# sidebar draws its boxes, rather than a rerun behind them.
# ---------------------------------------------------------------------------

_command = state.take_map_command()
if _command:
    if _command.get("center"):
        st.session_state.map_center = list(_command["center"])
    if _command.get("zoom"):
        st.session_state.map_zoom = int(_command["zoom"])
    if _command.get("fit_bounds"):
        st.session_state.fit_bounds = _command["fit_bounds"]
    if _command.get("layers"):
        st.session_state.layers.update(_command["layers"])
    if _command.get("filters"):
        st.session_state.filters.update(_command["filters"])
    if _command.get("select_lot"):
        try:
            _select_lot(queries.lot_by_number(_command["select_lot"]))
        except Exception:  # noqa: BLE001 — a stale selection must not break the page
            pass
    st.session_state.agent_note = _command.get("note")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

_LOG_COLOURS = {
    "DEBUG": "#888888", "INFO": "#0277bd", "TOOL_IN": "#00695c",
    "TOOL_OUT": "#00695c", "LLM_OUT": "#4527a0",
    "WARNING": "#e65100", "ERROR": "#c62828", "CRITICAL": "#6a1b9a",
}
_LOG_ORDER = {
    "DEBUG": 0, "INFO": 1, "TOOL_IN": 1, "TOOL_OUT": 1, "LLM_OUT": 1,
    "WARNING": 2, "ERROR": 3, "CRITICAL": 4,
}


def _render_logs(container, entries, threshold: int) -> None:
    rows = [e for e in entries if _LOG_ORDER.get(e.get("level", "INFO"), 0) >= threshold]
    if not rows:
        container.caption("No entries yet.")
        return
    html = []
    for entry in rows[-200:]:
        colour = _LOG_COLOURS.get(entry.get("level", "INFO"), "#ffffff")
        message = (
            str(entry.get("message", ""))
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        html.append(
            f'<span style="color:#78909c">[{entry.get("ts", "")}]</span> '
            f'<span style="color:{colour};font-weight:bold">'
            f'[{entry.get("level", "INFO")}]</span><br>{message}'
        )
    container.markdown(
        '<div style="border:1px solid #e0e0e0;border-radius:6px;padding:10px;'
        'font-family:monospace;font-size:11px;line-height:1.6;max-height:400px;'
        'overflow-y:auto;white-space:pre-wrap;">' + "<br>".join(html) + "</div>",
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.title("🏙️ HBU Zoning Map")
    st.caption("Montreal zoning · PostGIS + pgvector · HuggingFace")

    try:
        caps = _capabilities()
        connected, connect_error = True, None
    except Exception as exc:  # noqa: BLE001 — this pane exists to report anything
        caps, connected, connect_error = queries.Capabilities(), False, str(exc)

    # Resolved here rather than beside the map, because the sidebar reports it
    # and because starting the tile server is the first thing this script does
    # that can fail on its own.
    renderer, renderer_note = _renderer(caps)

    if connected:
        present = [n for n in ("lots", "buildings", "features", "chunks") if getattr(caps, n)]
        st.success(f"Connected · {', '.join(present) or 'no tables yet'}")
        # The operator's reading, so the advisory ones are included: a missing
        # silver join is worth seeing here even though the app works without it.
        missing = caps.missing()
        if missing:
            st.caption("Not loaded: " + ", ".join(missing))
        # Which renderer is drawing, and — when it is the fallback — why.
        # Worth a line of the sidebar because the two differ in what the map
        # can hold, not in how it looks: the GeoJSON path stops at
        # HBU_MAP_FEATURE_LIMIT shapes and says so only in the notes under the
        # map, long after someone has decided the data is missing.
        if renderer == "tiles":
            _hits, _misses = tiles.cache_stats()
            st.caption(
                f"Vector tiles · port {_tile_port()} · "
                f"cache {_hits}/{_hits + _misses}"
            )
        else:
            st.caption("GeoJSON renderer" + (f" — {renderer_note}" if renderer_note else ""))
    else:
        st.error("Not connected")
        st.caption(connect_error)
        with st.expander("How to connect", expanded=True):
            st.markdown(
                "Set one of these, then press **Reconnect**:\n\n"
                "- `DATABASE_URL` — a local `postgis`+`pgvector` container "
                "(`make db-up`) or an open tunnel\n"
                "- `eval \"$(make -s db-app-env ENV=dev)\"` in `hbu_infra`\n"
                "- nothing at all, with AWS credentials — the endpoint is read "
                "from SSM `/hbu-dev/db/*`"
            )

    if st.button("🔌 Reconnect", width="stretch"):
        from src.utils.db import close_pool  # noqa: PLC0415

        close_pool()
        st.cache_data.clear()
        st.rerun()

    st.divider()

    if caps.lots:
        try:
            hoods, dates = _partitions("lots")
        except Exception:  # noqa: BLE001
            hoods, dates = [], []

        if hoods:
            options = [None, *hoods]
            st.session_state.neighborhood = st.selectbox(
                "Borough",
                options=options,
                index=options.index(st.session_state.neighborhood)
                if st.session_state.neighborhood in options else 0,
                format_func=lambda v: "All loaded" if v is None else v,
            )
        if dates:
            st.session_state.scrape_date = st.selectbox(
                "Snapshot",
                options=dates,
                index=dates.index(st.session_state.scrape_date)
                if st.session_state.scrape_date in dates else 0,
                format_func=lambda d: d.isoformat() if isinstance(d, date) else str(d),
            )
            if st.session_state.scrape_date != dates[0]:
                st.caption(f"⚠️ Not the newest snapshot — {dates[0]} is.")

    st.divider()
    st.subheader("Layers")
    st.session_state.layers["lots"] = st.checkbox(
        "Lots", value=st.session_state.layers["lots"], disabled=not caps.lots
    )
    st.session_state.layers["buildings"] = st.checkbox(
        "Buildings",
        value=st.session_state.layers["buildings"] and caps.buildings,
        disabled=not caps.buildings,
        help=None if caps.buildings
        else f"{queries.SCHEMA}.buildings is not in this database yet.",
    )
    st.session_state.layers["zones"] = st.checkbox(
        "Zoning", value=st.session_state.layers["zones"], disabled=not caps.features
    )
    st.session_state.layers["streets"] = st.checkbox(
        "Streets",
        value=st.session_state.layers["streets"] and caps.streets,
        disabled=not caps.streets,
        help="Sides of the roadway from the city's géobase double — two lines "
        "per street, one per curb, which is the grain a lot's frontage is "
        "measured against. Hover one for its name and its length inside this "
        "borough." if caps.streets
        else f"{queries.SILVER_SCHEMA}.neighborhood_streets is not in this "
        "database yet — run the neighborhood_streets asset.",
    )
    st.session_state.layers["capacity"] = st.checkbox(
        "Utilisation",
        value=st.session_state.layers["capacity"] and caps.redevelopment_gap,
        disabled=not caps.redevelopment_gap,
        help="Shade every lot by how much of its permitted floor area is "
        "actually standing. Blue is emptier; purple exceeds what today's "
        "zoning would allow." if caps.redevelopment_gap
        else f"{queries.GOLD_SCHEMA}.lot_redevelopment_gap is not in this "
        "database yet — run the lot_redevelopment_gap asset.",
    )
    st.session_state.layers["massing"] = st.checkbox(
        "Proposed massing",
        value=st.session_state.layers["massing"] and caps.massing,
        disabled=not caps.massing,
        help="The highest-and-best-use building of each lot, drawn inside its "
        "setback envelope. Amber where the solved footprint had to be shrunk "
        "to fit." if caps.massing
        else f"{queries.GOLD_SCHEMA}.lot_building_massing is not in this "
        "database yet — run the massing asset.",
    )
    if st.session_state.layers["massing"] or st.session_state.layers["capacity"]:
        st.session_state.only_underbuilt = st.checkbox(
            "Under-built lots only",
            value=st.session_state.get("only_underbuilt", False),
            help="Keep the lots that could hold more floor than the assessment "
            "roll says stands on them today. Applies to both the Utilisation "
            "shading and the proposed massing, so the two cannot disagree "
            "about which parcels are in scope.",
        )

    if st.session_state.layers["capacity"]:
        with st.expander("Legend — utilisation"):
            for _color, _label in basemap.capacity_legend_rows():
                st.markdown(
                    f'<span style="display:inline-block;width:0.9rem;'
                    f'height:0.9rem;background:{_color};border:1px solid #666;'
                    f'vertical-align:middle;margin-right:.5rem"></span>'
                    f"{_label}",
                    unsafe_allow_html=True,
                )

    with st.expander("Lot size filter"):
        _min = st.number_input("Min area (m²)", min_value=0.0, value=0.0, step=50.0)
        _max = st.number_input("Max area (m²)", min_value=0.0, value=0.0, step=50.0)
        st.session_state.filters = {"min_area_m2": _min or None, "max_area_m2": _max or None}

    st.divider()

    from src.config import DEFAULT_MODEL_ALIAS, MODELS  # noqa: E402

    _aliases = list(MODELS)
    _current = os.environ.get("HF_MODEL_ID", DEFAULT_MODEL_ALIAS)
    _chosen = st.selectbox(
        "Chat model",
        options=_aliases,
        index=_aliases.index(_current) if _current in _aliases else 0,
        format_func=lambda a: MODELS[a].description,
    )
    if _chosen != _current:
        os.environ["HF_MODEL_ID"] = _chosen
        from src.agent import reset_agent  # noqa: PLC0415

        reset_agent()
        st.session_state.messages = []
        st.rerun()

    if not os.environ.get("HUGGINGFACE_API_TOKEN"):
        st.warning(
            "`HUGGINGFACE_API_TOKEN` is not set. The chat panel and the "
            "regulation search both need it; the map does not."
        )

    if st.button("🔄 New conversation", width="stretch"):
        st.session_state.messages = []
        state.clear_rag_buffer()
        clear_log_buffer()
        st.rerun()

    if IS_DEV:
        st.divider()
        if st.checkbox("🪵 Log pane", value=False):
            _level = st.selectbox("Min level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
            _render_logs(st.container(), st.session_state.log_entries, _LOG_ORDER.get(_level, 0))

# ---------------------------------------------------------------------------
# Nothing below works without a database, so say so once and stop.
# ---------------------------------------------------------------------------

if not connected:
    st.title("🏙️ HBU Zoning Map")
    st.error("Not connected to the database — see **How to connect** in the sidebar.")
    st.code(connect_error or "", language="text")
    st.stop()

if not caps.can_map:
    st.title("🏙️ HBU Zoning Map")
    st.warning(
        "Connected, but no geometry is loaded. The map needs at least one of "
        f"`{queries.SCHEMA}.lots`, `{queries.SCHEMA}.buildings` or "
        f"`{queries.SCHEMA}.features`.\n\n"
        # Required only: a missing silver join is never why the map is empty.
        f"Missing: {', '.join(caps.missing(include_advisory=False))}"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

map_col, side_col = st.columns([0.56, 0.44], gap="medium")

with map_col:
    st.subheader("Map")

    center = tuple(st.session_state.map_center)
    zoom = int(st.session_state.map_zoom)
    bounds = st.session_state.viewport

    lots = buildings = zones = capacity = streets = massing = None
    tile_layers: dict[str, str] = {}
    tile_visibility: dict[str, bool] = {}
    notes: list[str] = []
    key = None
    scrape, hood = st.session_state.scrape_date, st.session_state.neighborhood
    underbuilt = bool(st.session_state.get("only_underbuilt", False))

    if renderer_note:
        notes.append(renderer_note)

    if renderer == "tiles":
        # One URL per layer the database can serve, whether or not its box is
        # ticked. `show` is what the tick controls, so turning a layer on is a
        # thing Leaflet does to tiles it already knows how to ask for — no
        # rerun, no query, no rebuild of the map object.
        #
        # There is no `bounds` in any of this, and that is the change. The
        # filters below are the whole of what a tile URL varies on, so panning
        # never touches Python and the map object below stays identical across
        # a pan — which is also what stops the pane blinking.
        _available = {
            "lots": caps.lots,
            "buildings": caps.buildings,
            "zones": caps.features,
            "capacity": caps.redevelopment_gap,
            "streets": caps.streets,
            "massing": caps.massing,
        }
        for _layer, _present in _available.items():
            if not _present:
                continue
            _filters: dict[str, object] = {"scrape_date": scrape, "neighborhood": hood}
            if _layer == "lots":
                _filters["min_area"] = st.session_state.filters["min_area_m2"]
                _filters["max_area"] = st.session_state.filters["max_area_m2"]
            if _layer in ("capacity", "massing") and underbuilt:
                _filters["underbuilt"] = 1
            tile_layers[_layer] = tiles.layer_url(_layer, _filters)
            tile_visibility[_layer] = bool(st.session_state.layers[_layer])

        # The zoom gates are Leaflet's now — a layer below its minimum is not
        # requested at all — so these say why a ticked layer is not on screen.
        for _layer, _floor in (
            ("lots", basemap.MIN_LOT_ZOOM),
            ("buildings", basemap.MIN_BUILDING_ZOOM),
            ("capacity", basemap.MIN_CAPACITY_ZOOM),
            ("streets", basemap.MIN_STREET_ZOOM),
            ("massing", basemap.MIN_MASSING_ZOOM),
        ):
            if tile_visibility.get(_layer) and zoom < _floor:
                notes.append(
                    f"{basemap.TILE_LAYER_NAMES[_layer]} draws from zoom "
                    f"{_floor} (now {zoom})."
                )

        # "The asset has not run for this borough" is the one thing an empty
        # tile cannot say for itself, and it is worth saying: a blank massing
        # layer reads as "nothing can be built here", a blank utilisation layer
        # as "every lot is fully used", and a blank streets layer as a borough
        # with no roads — each the opposite of what a missing partition means.
        # A partition-level EXISTS, cached, rather than an inference from a
        # viewport that no longer exists.
        for _layer, _asset in (
            ("capacity", "lot_redevelopment_gap"),
            ("streets", "neighborhood_streets"),
            ("massing", "lot_building_massing"),
        ):
            if not tile_visibility.get(_layer):
                continue
            try:
                if not _partition_has_rows(_layer, scrape, hood):
                    notes.append(
                        f"No {basemap.TILE_LAYER_NAMES[_layer].lower()} data for "
                        f"{scrape or 'the latest snapshot'} — has the {_asset} "
                        "asset run for this partition?"
                    )
            except Exception:  # noqa: BLE001 — an advisory note, never a failure
                pass

    elif bounds:
        key = _cache_key(bounds)

        if st.session_state.layers["lots"] and caps.lots:
            if zoom >= basemap.MIN_LOT_ZOOM:
                lots = _lots(
                    key, zoom, scrape, hood,
                    st.session_state.filters["min_area_m2"],
                    st.session_state.filters["max_area_m2"],
                )
                basemap.decorate(lots, "lots")
                if lots.truncated:
                    notes.append(f"Lots capped at {lots.count} — zoom in for the rest.")
            else:
                notes.append(f"Lots draw from zoom {basemap.MIN_LOT_ZOOM} (now {zoom}).")

        if st.session_state.layers["buildings"] and caps.buildings:
            if zoom >= basemap.MIN_BUILDING_ZOOM:
                buildings = _buildings(key, zoom, scrape, hood)
                basemap.decorate(buildings, "buildings")
                if buildings.truncated:
                    notes.append(f"Buildings capped at {buildings.count}.")
            else:
                notes.append(f"Buildings draw from zoom {basemap.MIN_BUILDING_ZOOM}.")

        if st.session_state.layers["zones"] and caps.features:
            zones = _zones(key, zoom, scrape, hood)
            basemap.decorate(zones, "zones")

        if st.session_state.layers["streets"] and caps.streets:
            if zoom >= basemap.MIN_STREET_ZOOM:
                streets = _streets(key, zoom, scrape, hood)
                basemap.decorate(streets, "streets")
                if streets.truncated:
                    notes.append(f"Streets capped at {streets.count}.")
                elif not streets.features:
                    notes.append(
                        "No streets here for "
                        f"{scrape or 'the latest snapshot'} — has the "
                        "neighborhood_streets asset run for this partition?"
                    )
            else:
                notes.append(
                    f"Streets draw from zoom {basemap.MIN_STREET_ZOOM} (now {zoom})."
                )

        if st.session_state.layers["capacity"] and caps.redevelopment_gap:
            if zoom >= basemap.MIN_CAPACITY_ZOOM:
                capacity = _capacity(
                    key, zoom, scrape, hood,
                    bool(st.session_state.get("only_underbuilt", False)),
                )
                basemap.decorate(capacity, "capacity")
                if capacity.truncated:
                    notes.append(f"Utilisation capped at {capacity.count}.")
                elif not capacity.features:
                    # Same reasoning as the massing note below: an empty layer
                    # here would read as "every lot is fully used", which is
                    # the opposite of what a missing partition means.
                    notes.append(
                        "No utilisation data here for "
                        f"{scrape or 'the latest snapshot'} — has the "
                        "lot_redevelopment_gap asset run for this partition?"
                    )
            else:
                notes.append(
                    f"Utilisation draws from zoom {basemap.MIN_CAPACITY_ZOOM} "
                    f"(now {zoom})."
                )

        if st.session_state.layers["massing"] and caps.massing:
            if zoom >= basemap.MIN_MASSING_ZOOM:
                massing = _massing(
                    key, zoom, scrape, hood,
                    bool(st.session_state.get("only_underbuilt", False)),
                )
                basemap.decorate(massing, "massing")
                if massing.truncated:
                    notes.append(f"Massing capped at {massing.count}.")
                elif not massing.features:
                    # Distinguishable from "the layer is off": the asset may
                    # simply not have run for this borough-date, and a silently
                    # empty layer would read as "nothing can be built here".
                    notes.append(
                        "No massing here for "
                        f"{scrape or 'the latest snapshot'} — has the "
                        "lot_building_massing asset run for this partition?"
                    )
            else:
                notes.append(
                    f"Massing draws from zoom {basemap.MIN_MASSING_ZOOM} (now {zoom})."
                )

    # Rebuild the folium map only when something it draws actually changed.
    # st_folium reloads its <iframe> whenever the map object differs — and
    # folium stamps a fresh random id into every map it builds — so handing it
    # the *same* object across reruns is what stops the pane blinking on every
    # viewport report. This signature is everything build_map reads.
    #
    # Note what is *not* in it under the tile renderer: the centre, the zoom
    # and the rounded viewport. None of the three appears in a tile URL, so
    # under tiles they only decide where the map *opens* — and leaving them in
    # would reload the iframe on every scroll-wheel notch and every pan, which
    # is the blinking this signature exists to prevent. A later rebuild reads
    # them from session state, which the browser has kept current, so the map
    # comes back where the user left it. Under the GeoJSON renderer all three
    # decide which shapes are embedded, so all three stay.
    _sel_lot = (st.session_state.selected_lot or {}).get("lot_number")
    _map_sig = (
        renderer,
        None if renderer == "tiles" else (_cache_key(center, 6), zoom, key),
        tuple(sorted(tile_layers.items())),
        tuple(sorted(tile_visibility.items())),
        st.session_state.layers["lots"],
        st.session_state.layers["buildings"],
        st.session_state.layers["zones"],
        st.session_state.layers["capacity"],
        st.session_state.layers["streets"],
        st.session_state.layers["massing"],
        underbuilt,
        str(st.session_state.scrape_date), st.session_state.neighborhood,
        st.session_state.filters["min_area_m2"], st.session_state.filters["max_area_m2"],
        _sel_lot, repr(st.session_state.fit_bounds),
    )
    if st.session_state.get("_map_sig") != _map_sig or "_map_obj" not in st.session_state:
        st.session_state._map_obj = basemap.build_map(
            center=center,
            zoom=zoom,
            lots=lots,
            buildings=buildings,
            zones=zones,
            capacity=capacity,
            streets=streets,
            massing=massing,
            tile_layers=tile_layers,
            tile_visibility=tile_visibility,
            selected=st.session_state.selected_lot,
            fit_bounds=st.session_state.fit_bounds,
        )
        st.session_state._map_sig = _map_sig
    fmap = st.session_state._map_obj
    st.session_state.fit_bounds = None  # a fit is a one-shot, not a mode

    from streamlit_folium import st_folium  # noqa: E402

    result = st_folium(
        fmap,
        height=620,
        use_container_width=True,
        # All four are load-bearing: the click selects a lot, the bounds decide
        # what is queried next, and the zoom gates which layers draw at all.
        returned_objects=["last_clicked", "bounds", "zoom", "center"],
        key="zoning_map",
    ) or {}

    # --- the browser reports where it ended up ---------------------------
    new_bounds = _reported_bounds(result)
    if new_bounds is not None:
        new_zoom = int(result.get("zoom") or zoom)
        new_center = _reported_center(result)
        # Rerun when the map first reports itself, when the zoom changed, or
        # when a pan moved the centre by more than a third of the viewport.
        # _moved_enough is deliberately loose: st_folium re-reports slightly
        # different bounds every time its iframe lays out, and testing that at
        # ~100 m turned the jitter into a permanent rerun loop.
        if _moved_enough(bounds, new_bounds) or new_zoom != zoom:
            st.session_state.viewport = new_bounds
            st.session_state.map_zoom = new_zoom
            if new_center:
                st.session_state.map_center = [round(new_center[0], 6), round(new_center[1], 6)]
            st.rerun()

    # --- a click selects a lot -------------------------------------------
    clicked = result.get("last_clicked")
    if isinstance(clicked, dict) and clicked.get("lat") is None:
        clicked = None
    if clicked and clicked != st.session_state.last_click:
        st.session_state.last_click = clicked
        try:
            hit = queries.lot_at_point(
                float(clicked["lng"]), float(clicked["lat"]),
                scrape_date=st.session_state.scrape_date,
            )
        except Exception as exc:  # noqa: BLE001
            hit, _ = None, st.warning(f"Could not resolve that click: {exc}")
        if hit:
            _select_lot(hit)
            st.rerun()
        else:
            st.caption("No lot at that point in this snapshot.")

    if st.session_state.agent_note:
        st.caption(f"↳ {st.session_state.agent_note}")
    for note in notes:
        st.caption(note)
    if renderer == "tiles":
        # No counts, and their absence is the feature rather than a gap: under
        # this renderer nothing on the Python side ever sees a shape, which is
        # exactly why the map can hold a borough. What is worth saying is which
        # layers are on and whether the zoom lets them draw — the notes above
        # cover the second half.
        on = [
            basemap.TILE_LAYER_NAMES[layer]
            for layer in basemap.TILE_LAYER_ORDER
            if tile_visibility.get(layer)
        ]
        st.caption(
            f"Vector tiles: {', '.join(on) if on else 'no layers on'} · zoom {zoom}"
        )
    else:
        drawn = ", ".join(
            f"{fs.count} {fs.layer}" for fs in (lots, buildings, zones) if fs and fs.count
        )
        st.caption(f"Drawn: {drawn or 'nothing in view'} · zoom {zoom}")

# ---------------------------------------------------------------------------
# Right panel
# ---------------------------------------------------------------------------

with side_col:
    lot_tab, capacity_tab, rules_tab, chat_tab = st.tabs(
        ["📍 Lot & zoning", "📊 Capacity", "📖 Regulations", "💬 Chat"]
    )

    # --- Lot & zoning ----------------------------------------------------
    with lot_tab:
        lot = st.session_state.selected_lot
        if not lot:
            st.info(
                "Click a lot on the map, or ask the chat for one by number.\n\n"
                "Its attributes, the zoning grid that applies to it, and the "
                "grid's PDF appear here."
            )
        else:
            st.markdown(f"### Lot {lot['lot_number']}")
            left, right = st.columns(2)
            left.metric("Area", f"{float(lot.get('area_m2') or 0):,.0f} m²")
            right.metric("Snapshot", str(lot.get("scrape_date")))
            st.caption(
                f"{lot.get('neighborhood')} · "
                f"{float(lot['lat']):.5f}, {float(lot['lon']):.5f}"
            )

            if caps.buildings:
                footprints = _footprints_on_lot(lot["lot_number"])
                if footprints:
                    covered = sum(float(f.get("overlap_m2") or 0) for f in footprints)
                    area = float(lot.get("area_m2") or 0)
                    ratio = f" — {covered / area * 100:.0f}% of the lot" if area else ""
                    st.markdown(
                        f"**Built:** {len(footprints)} footprint(s), "
                        f"{covered:,.0f} m²{ratio}"
                    )
                    st.caption(
                        "Measured from the footprints. The grid's *taux "
                        "d'implantation* below is what is permitted."
                    )
                else:
                    st.markdown("**Built:** no footprint on this lot")

            # --- is it used efficiently, and what else fits ---------------
            if caps.redevelopment_gap and lot.get("lot_uid") is not None:
                potential = _lot_capacity(
                    int(lot["lot_uid"]), lot.get("scrape_date"),
                    lot.get("neighborhood"),
                )
                st.divider()
                if not potential:
                    st.caption(
                        "No highest-and-best-use row for this lot in this "
                        "snapshot."
                    )
                elif potential.get("hbu_status") != "solved":
                    # Never a bare blank: hbu_status is the reason, and each
                    # one is a fact about the lot rather than a gap in the
                    # data.
                    st.markdown("**Potential:** no programme solved")
                    st.caption(
                        {
                            "no_candidate_column":
                                "Every zoning column reaching this lot "
                                "authorises none of the uses the solver "
                                "prices (housing, commerce, industry) — "
                                "usually équipements collectifs.",
                            # The former name of no_candidate_column, from
                            # when the solver priced dwellings alone. Rows
                            # written before the rename carry it until their
                            # partition is re-materialized.
                            "no_residential_column":
                                "Every zoning column reaching this lot "
                                "authorises something other than housing; "
                                "this snapshot predates the solver pricing "
                                "commerce and industry.",
                            "no_governing_column":
                                "Candidate columns exist but none governs — "
                                "usually a lot with no measured frontage under "
                                "a grid that states a minimum width.",
                            "infeasible":
                                "No governing column has a feasible "
                                "programme — a minimum this parcel cannot meet.",
                            "solver_error":
                                "The governing column could not be turned into "
                                "a model.",
                        }.get(
                            potential.get("hbu_status"),
                            str(potential.get("hbu_status")),
                        )
                    )
                else:
                    used = potential.get("used_pct")
                    built = float(potential.get("existing_floor_area_m2") or 0)
                    permitted = float(potential.get("hbu_floor_area_m2") or 0)

                    if used is None:
                        verdict, note = "—", ""
                    elif float(used) > 100:
                        verdict = f"{float(used):,.0f}% of what zoning allows"
                        note = (
                            "More floor stands here than today's grid would "
                            "permit — a legal non-conformity, not headroom."
                        )
                    elif float(used) >= 95:
                        verdict = f"{float(used):,.0f}% used"
                        note = "Effectively built out under the current grid."
                    else:
                        verdict = f"{float(used):,.0f}% used"
                        note = "Under-built against the governing envelope."

                    st.markdown(f"### Efficiency — {verdict}")
                    left, right = st.columns(2)
                    left.metric("Standing today", f"{built:,.0f} m²")
                    right.metric(
                        "Zoning would hold", f"{permitted:,.0f} m²",
                        delta=f"{permitted - built:+,.0f} m²",
                    )
                    if note:
                        st.caption(note)
                    if not potential.get("has_assessment"):
                        # The gap table reads a missing existing floor as zero,
                        # so this lot's whole envelope is counted as headroom.
                        # Say so rather than letting 0 m² read as surveyed.
                        st.caption(
                            "⚠️ The assessment roll has no unit on this lot, so "
                            "*standing today* is read as nothing built."
                        )

                    st.markdown("**What else could go here**")
                    rows = []
                    for label, key_m2 in (
                        ("Residential", "residential_headroom_m2"),
                        ("Commercial", "commercial_headroom_m2"),
                        ("Industrial", "industrial_headroom_m2"),
                    ):
                        extra = float(potential.get(key_m2) or 0)
                        if extra <= 0:
                            continue
                        rows.append({
                            "Use": label,
                            "Additional m²": f"{extra:,.0f}",
                            "Additional sq ft": f"{extra * 10.7639:,.0f}",
                        })
                    if rows:
                        st.dataframe(rows, width="stretch", hide_index=True)
                    else:
                        st.caption("No additional floor area under this grid.")

                    gap = potential.get("dwelling_gap")
                    existing_d = potential.get("existing_num_dwellings")
                    hbu_d = potential.get("hbu_num_dwellings")
                    if hbu_d is not None:
                        line = f"**Dwellings:** {int(existing_d or 0)} today → {int(hbu_d)}"
                        if gap is not None and int(gap) != 0:
                            line += f" ({int(gap):+d})"
                        st.markdown(line)

                    shape = []
                    use = potential.get("hbu_dominant_use")
                    if use and use != "none":
                        shape.append(str(use).replace("_", " "))
                    if potential.get("floors"):
                        shape.append(f"{int(potential['floors'])} storeys")
                    if potential.get("height_m"):
                        shape.append(f"{float(potential['height_m']):.1f} m")
                    if potential.get("grid_zone"):
                        shape.append(f"zone {potential['grid_zone']}")
                    if shape:
                        st.caption("Proposed: " + " · ".join(shape))

                    # --- the developer's arithmetic -----------------------
                    # The programme above is the *most profitable* governing
                    # envelope on discounted net profit, so the money behind
                    # the choice belongs on the pane: what it costs, what the
                    # finished building is worth, and whether building beats
                    # holding. Absent on a snapshot solved before the
                    # discounting existed, and the pane says nothing then.
                    npv = potential.get("npv_cad")
                    if npv is not None:
                        st.markdown("**Developer economics** *(land excluded)*")
                        cols = st.columns(2)
                        cols[0].metric(
                            "Discounted net profit", f"${float(npv):,.0f}"
                        )
                        capital = potential.get("total_capital_cost_cad")
                        if capital is not None:
                            cols[1].metric(
                                "Construction cost", f"${float(capital):,.0f}"
                            )
                        gain = potential.get("redevelopment_npv_gain_cad")
                        if gain is not None:
                            if float(gain) > 0:
                                st.caption(
                                    f"Redeveloping beats holding the standing "
                                    f"building by **${float(gain):,.0f}** at "
                                    f"the solve's discount assumptions."
                                )
                            else:
                                st.caption(
                                    f"Holding the standing building beats "
                                    f"redeveloping by ${-float(gain):,.0f} — "
                                    f"the envelope has room, the economics "
                                    f"say keep it."
                                )

                    # The caveat the README insists on: a footprint capped on
                    # the lesser of two *areas* may have no shape this parcel
                    # can take, and then the floor area above is overstated.
                    fit = potential.get("footprint_fit_pct")
                    if fit is not None and float(fit) < 99.5:
                        st.warning(
                            f"The solved footprint only fits this parcel at "
                            f"{float(fit):.0f}% — the floor areas above are "
                            f"overstated for this lot's shape."
                        )

            st.divider()
            if not caps.features:
                st.warning(f"`{queries.SCHEMA}.features` is not loaded — no zoning to show.")
            else:
                zoning = _zoning_for_lot(lot["lot_number"])
                if not zoning:
                    st.info("No zoning polygon covers this lot in this snapshot.")
                else:
                    if len(zoning) > 1:
                        st.warning(
                            f"This lot straddles {len(zoning)} zones — pick one. "
                            "They are ordered by how much of the lot each covers."
                        )
                    labels = [
                        z["zone"] + (
                            f" · {z['overlap_m2'] / z['lot_area_m2'] * 100:.0f}%"
                            if z.get("overlap_m2") and z.get("lot_area_m2") else ""
                        )
                        for z in zoning
                    ]
                    index = labels.index(st.radio("Zone", labels, horizontal=True)) \
                        if len(zoning) > 1 else 0
                    zone = zoning[index]
                    attributes = zone.get("attributes") or {}

                    rows = [
                        {"Field": label, "Value": str(attributes[key])}
                        for key, label in queries.ZONING_FIELDS
                        if str(attributes.get(key, "")).strip()
                    ]
                    if rows:
                        st.dataframe(
                            rows, width="stretch", hide_index=True,
                            height=min(36 * len(rows) + 38, 420),
                        )
                    else:
                        st.caption("The zoning row carries no grid values in this snapshot.")

                    # --- the grid PDF ------------------------------------
                    url = zone.get("zoning_pdf_url")
                    if not url and caps.chunks:
                        url = queries.zoning_pdf_url_fallback(zone["zone"])

                    st.divider()
                    if not url:
                        st.caption("No grille des spécifications is linked from this zone.")
                    else:
                        st.markdown(f"**Grille des spécifications — zone {zone['zone']}**")
                        try:
                            content, filename, pages, render_error = _zoning_pdf(url)
                        except Exception as exc:  # noqa: BLE001
                            content = filename = None
                            pages, render_error = [], None
                            st.error(f"Could not fetch the grid: {exc}")
                            st.markdown(f"[Open it at the source]({url})")

                        if content:
                            st.download_button(
                                "⬇️ Download the grid (PDF)",
                                data=content,
                                file_name=filename,
                                mime="application/pdf",
                                width="stretch",
                            )
                            for number, png in enumerate(pages, 1):
                                st.image(png, width="stretch", caption=f"Page {number}")
                            if render_error:
                                st.caption(f"Cannot render inline: {render_error}")
                            st.caption(f"[Source]({url})")

    # --- Regulations -----------------------------------------------------
    # --- Borough capacity ------------------------------------------------
    with capacity_tab:
        if not caps.redevelopment_gap:
            st.info(
                f"`{queries.GOLD_SCHEMA}.lot_redevelopment_gap` is not in this "
                "database yet. It is what compares the floor area standing on "
                "each lot against what its zoning envelope would hold — run "
                "the `lot_redevelopment_gap` asset for this partition."
            )
        else:
            totals = _capacity_totals(
                st.session_state.neighborhood, st.session_state.scrape_date
            )
            if not totals or not totals.get("num_lots"):
                st.info("No redevelopment-gap rows for this borough and snapshot.")
            else:
                scope = st.session_state.neighborhood or "every loaded borough"
                st.markdown(f"### How much more {scope} could hold")
                st.caption(
                    f"{int(totals['num_lots']):,} lots · snapshot "
                    f"{st.session_state.scrape_date or 'latest'} · under the "
                    "governing zoning envelope of each lot"
                )

                res = float(totals.get("residential_headroom_m2") or 0)
                com = float(totals.get("commercial_headroom_m2") or 0)
                ind = float(totals.get("industrial_headroom_m2") or 0)
                dwellings = int(totals.get("additional_dwellings") or 0)

                top_left, top_right = st.columns(2)
                top_left.metric("Additional dwellings", f"{dwellings:,}")
                top_right.metric(
                    "Additional floor area",
                    f"{(res + com + ind) * 10.7639:,.0f} sq ft",
                    help=f"{res + com + ind:,.0f} m² across all three classes.",
                )

                # "0 m²" and "never proposed" are different findings, and the
                # table cannot show the same 0 for both. The solver prices all
                # three families and picks the most profitable governing
                # envelope per lot — so a class no solved lot was given any
                # floor of is one that never won a single storey at current
                # rents and costs, which is a finding about the economics
                # rather than about the envelopes being full.
                modelled = {
                    "Residential": int(totals.get("num_with_residential") or 0),
                    "Commercial": int(totals.get("num_with_commercial") or 0),
                    "Industrial": int(totals.get("num_with_industrial") or 0),
                }
                st.dataframe(
                    [
                        {
                            "Use": label,
                            "Additional m²": (
                                f"{value:,.0f}" if modelled[label] else "none proposed"
                            ),
                            "Additional sq ft": (
                                f"{value * 10.7639:,.0f}" if modelled[label] else "—"
                            ),
                            "Lots proposing it": f"{modelled[label]:,}",
                        }
                        for label, value in (
                            ("Residential", res),
                            ("Commercial", com),
                            ("Industrial", ind),
                        )
                    ],
                    width="stretch",
                    hide_index=True,
                )
                unmodelled = [k for k, v in modelled.items() if not v and k != "Residential"]
                if unmodelled:
                    st.info(
                        f"**No lot's most profitable programme includes "
                        f"{' or '.join(k.lower() for k in unmodelled)} floor.** "
                        f"The solver prices commerce and industry at the "
                        f"borough's surveyed rents against their construction "
                        f"costs, and at those numbers no envelope earns more "
                        f"with that class than without it — an economics "
                        f"finding, not a zoning one. (On a snapshot solved "
                        f"before commerce and industry were priced, this same "
                        f"zero means *not modelled*: re-run the pipeline to "
                        f"tell the two apart.)"
                    )

                # The developer's verdict over the borough, where it is
                # positive — the sum nobody should read without the count
                # beside it.
                gain = totals.get("redevelopment_npv_gain_cad")
                if gain is not None and float(gain) > 0:
                    st.metric(
                        "Discounted gain from redeveloping where it pays",
                        f"${float(gain) / 1e6:,.0f}M",
                        help=(
                            f"Sum of redevelopment_npv_gain_cad over the "
                            f"{int(totals.get('num_npv_gain_positive') or 0):,} "
                            "lots where building the highest and best use is "
                            "worth more, discounted, than keeping what "
                            "stands. Land excluded — the owner holds it "
                            "either way."
                        ),
                    )

                # What the headline rests on. Without these three counts the
                # totals above are a number with no error bar - see
                # queries.capacity_totals.
                st.markdown("**What this total rests on**")
                solved = int(totals.get("num_solved") or 0)
                share = solved / int(totals["num_lots"]) * 100 if totals["num_lots"] else 0
                st.markdown(
                    f"- **{solved:,}** lots ({share:.0f}%) have a solved "
                    f"programme. The rest contribute nothing — a lot with no "
                    f"answer is not a lot with no room.\n"
                    f"- **{int(totals.get('num_underbuilt') or 0):,}** lots hold "
                    f"less than their envelope allows.\n"
                    f"- **{int(totals.get('num_over_built') or 0):,}** lots already "
                    f"exceed today's grid. They contribute **zero**, not a "
                    f"negative, so they cannot cancel a neighbour's headroom.\n"
                    f"- **{int(totals.get('num_without_assessment') or 0):,}** lots "
                    f"have no assessment unit, so their whole envelope counts "
                    f"as headroom."
                )

                # The total showing its own working. A handful of very large
                # parcels can carry a third of a borough's headroom, and a
                # reader given only the sum cannot see whether the sites
                # driving it are ones anyone would build on.
                top = _top_capacity_lots(
                    st.session_state.neighborhood, st.session_state.scrape_date
                )
                if top:
                    top_sum = sum(int(r.get("additional_dwellings") or 0) for r in top)
                    share = top_sum / dwellings * 100 if dwellings else 0
                    st.divider()
                    st.markdown(
                        f"**The {len(top)} lots carrying most of it** — "
                        f"{top_sum:,} dwellings, {share:.0f}% of the total"
                    )
                    st.dataframe(
                        [
                            {
                                "Lot": r.get("lot_number") or "—",
                                "Lot area (m²)": f"{float(r.get('lot_area_m2') or 0):,.0f}",
                                "Today": int(r.get("existing_num_dwellings") or 0),
                                "Proposed": int(r.get("hbu_num_dwellings") or 0),
                            }
                            for r in top
                        ],
                        width="stretch",
                        hide_index=True,
                    )
                    st.caption(
                        "Check the areas. A parcel of several hectares is the "
                        "scale of a park, a rail yard or a cemetery rather "
                        "than a development site — the solver fills whatever "
                        "the zoning envelope allows and does not know the "
                        "difference, so a few such lots can move the borough "
                        "total by a third."
                    )

                # The same list on the developer's number instead of the
                # planner's: not where the *room* is but where the *money*
                # is, and the two disagree exactly where a big envelope does
                # not pencil.
                top_gain = _top_npv_gain_lots(
                    st.session_state.neighborhood, st.session_state.scrape_date
                )
                if top_gain:
                    st.divider()
                    st.markdown(
                        f"**Where redeveloping beats holding, by discounted "
                        f"gain** — top {len(top_gain)} lots, land excluded"
                    )
                    st.dataframe(
                        [
                            {
                                "Lot": r.get("lot_number") or "—",
                                "Lot area (m²)": (
                                    f"{float(r.get('lot_area_m2') or 0):,.0f}"
                                ),
                                "Programme": (
                                    str(r.get("hbu_dominant_use") or "—")
                                ).replace("_", " "),
                                "Gain": (
                                    f"${float(r.get('redevelopment_npv_gain_cad') or 0):,.0f}"
                                ),
                                "Dwellings": int(r.get("hbu_num_dwellings") or 0),
                            }
                            for r in top_gain
                        ],
                        width="stretch",
                        hide_index=True,
                    )
                    st.caption(
                        "The same caveat as above applies: the biggest gains "
                        "often sit on park- or rail-yard-scale parcels."
                    )

                net = totals.get("net_floor_area_gap_m2")
                if net is not None:
                    st.divider()
                    st.markdown("**The other question**")
                    st.metric(
                        "Net change if every assessed lot were rebuilt to zoning",
                        f"{float(net):,.0f} m²",
                    )
                    st.caption(
                        "Signed, over the lots the roll reached — over-built "
                        "parcels count against it here. A negative figure means "
                        "the borough as built holds more floor than its current "
                        "by-law would permit, which is a finding about the "
                        "by-law rather than an error."
                    )

                st.divider()
                st.caption(
                    "Zoning capacity only. This is what the grids permit, not "
                    "what is financeable, serviceable or politically available "
                    "— and it is a scrape of the by-law rather than the "
                    "by-law. For anything with consequences, the borough is "
                    "the authority."
                )

    with rules_tab:
        buffer = state.RagBuffer
        if not buffer.get("hits"):
            st.info(
                "Ask the chat about the by-law and the passages it retrieved "
                "appear here in full, with their sources."
            )
            if caps.chunks:
                try:
                    status = queries.corpus_status()
                    if status:
                        st.caption("Corpus loaded:")
                        st.dataframe(status, width="stretch", hide_index=True)
                except Exception:  # noqa: BLE001
                    pass
            else:
                st.caption(
                    f"`{queries.SCHEMA}.chunks` is not in this database — the "
                    "dataplatform's `document_index` asset creates it."
                )
        else:
            scope = {
                "lot": f"lot {buffer.get('lot_number')}",
                "near": "the surrounding area",
                "corpus": "the whole corpus",
            }.get(buffer.get("scope"), "")
            st.caption(f"Searched {scope} for: *{buffer.get('query')}*")
            for number, hit in enumerate(buffer["hits"], 1):
                title = f"[{number}] {hit.get('source_table', '')}"
                if hit.get("similarity") is not None:
                    title += f" · similarity {float(hit['similarity']):.3f}"
                if hit.get("distance_m") is not None:
                    title += f" · {float(hit['distance_m']):.0f} m"
                with st.expander(title, expanded=number == 1):
                    st.write(hit.get("chunk_text", ""))
                    if hit.get("url"):
                        st.caption(f"[Source PDF]({hit['url']})")

    # --- Chat ------------------------------------------------------------
    with chat_tab:
        with st.container(height=420, border=False):
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        status_box = st.empty()

        if not st.session_state.messages:
            st.caption(
                "**Try:** *What can I build on this lot?* · *Find lot 2 170 935* · "
                "*Which lots in view are over 500 m²?* · *Show the zoning layer*"
            )

        selected = st.session_state.selected_lot
        user_input = st.chat_input(
            f"Ask about lot {selected['lot_number']}…" if selected
            else "Ask about zoning, a lot number, or what is in view…"
        )

        if user_input:
            import src.agent as agent_module  # noqa: PLC0415

            state.start_new_turn()
            state.clear_rag_buffer()
            answer = ""

            try:
                with status_box.container(), st.status("Working…", expanded=False) as status:
                    for event in agent_module.stream_agent(
                        user_input, history=st.session_state.messages
                    ):
                        if event["type"] == "tool_start":
                            status.update(label=f"⚙️ {event['label']}", state="running")
                        elif event["type"] == "final":
                            answer = event["content"]
                        new_logs = list(LogBuffer)
                        if new_logs:
                            st.session_state.log_entries.extend(new_logs)
                            clear_log_buffer()
                    status.update(label="✅ Done", state="complete")
            except Exception as exc:  # noqa: BLE001
                answer = f"⚠️ Something went wrong: {exc}"

            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": answer})
            clear_log_buffer()
            st.rerun()
