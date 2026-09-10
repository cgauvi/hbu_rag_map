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
and mirrors it into ``src.utils.state``, so the agent's tools see this
session's viewport and selection rather than a process-wide one.

**The position the browser reports is not the position the map is built at**,
and the difference is what makes panning smooth. `streamlit_folium` keys its
component on a hash of the map's JavaScript, so a centre baked into that
script is a new component key every time it changes — and a new key remounts
the iframe, which throws Leaflet away and refetches the basemap and every
vector tile. Leaflet reports a new centre at the end of every drag. Building
the map where the browser said it was therefore rebuilt the map on every
drag, and *that* was the pane redrawing itself as the user moved.

So there are two positions. ``map_center``/``map_zoom`` are the anchor the
folium object is built at, and a pan does not touch them; ``viewport``,
``view_center`` and ``view_zoom`` are where the browser actually is, read for
the notes under the map and by the agent's tools. The anchor is moved onto the
live view only when the map is being rebuilt for some other reason anyway — a
layer, a borough, a snapshot, a fit — so the one remount that does happen
happens where the user was looking. The selected lot is handed to the
component as a feature group for the same reason: it is the thing that changes
on a click, and a click should not cost a reload.
"""

import json
import logging
import os
import re
from collections.abc import Mapping
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

#: Writes into the same `LogBuffer` the sidebar's log pane reads, so a warning
#: from the page is visible without a terminal.
logger = logging.getLogger(__name__)

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
    # A zone clicked directly, with no lot under the cursor. Zoning covers
    # ground the cadastre does not - a park, a right of way, the far side of a
    # rail cut - and the grid that applies there is a real answer, so a click
    # that finds no parcel resolves the zone instead of reporting nothing.
    # Cleared whenever a lot is selected: a lot carries its own zoning list,
    # and two selections disagreeing about which zone is under discussion is
    # the one thing this pane exists to prevent.
    "selected_zone": None,     # a row from queries.zoning_at_point
    # The *anchor*: where the folium object is built, which is deliberately
    # not where the browser currently is. See "Where the map is" below.
    "map_center": list(basemap.DEFAULT_CENTER),
    "map_zoom": basemap.DEFAULT_ZOOM,
    # Where the browser actually is, as it last reported it.
    "viewport": None,          # (west, south, east, north), as the browser sees it
    "view_center": None,       # [lat, lon]
    "view_zoom": None,
    "map_signature": None,     # what the map was last built *from* — see below
    "fit_bounds": None,
    # From `basemap` rather than written out here, because `DEFAULT_ZOOM` is
    # derived from which of these are on: a layer switched on by default that
    # is below its detail zoom opens the map on an opaque wash of aggregate
    # cells, and — when it is one of the layers drawn above the lots — on a
    # cadastre nobody can see to click. See `basemap.DEFAULT_ZOOM`.
    "layers": dict(basemap.DEFAULT_LAYERS),
    "filters": {"min_area_m2": None, "max_area_m2": None},
    # The Opportunities layer's two screens: one site thesis or every one, and
    # whether to keep only the shortlist of each. Travel in the tile URL, so
    # toggling them costs Leaflet a fetch and Python nothing.
    "opportunity_filters": {"site_thesis": None, "top_only": False, "good_only": False},
    # Which side the Land use layer colours by: the roll, or the solve. In
    # the tile URL like the filters above, so switching costs a fetch and
    # no rerun. Today's side first, because it is the one that is a fact.
    "land_use_side": "existing",
    "neighborhood": None,
    "scrape_date": None,
    "agent_note": None,
    "last_click": None,
    # What row selection each Overview table was last acted on, keyed by the
    # dataframe's widget key. A selected row stays highlighted for as long as
    # it is selected, so a fit reissued from it on every rerun would haul the
    # view back to that lot each time the user panned off it. See
    # `_lot_clicked_in_table`.
    "table_clicks": {},
    # The set of layer names the map's own control last reported as ticked,
    # as a `set[str]` once anything has been reported and None before that.
    # Kept so a report is acted on once rather than on every rerun: a layer
    # Python cannot adopt — one whose sidebar box is forced off — would
    # otherwise be re-adopted and re-refused for ever, which is a rerun loop
    # rather than a wrong tick. None rather than an empty set because "no
    # layers on" is a real answer the map can give.
    "reported_layers": None,
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
def _surface_parking(bounds_key, zoom, scrape_date, neighborhood, only_underbuilt):
    return queries.surface_parking_in_bbox(
        bounds_key, zoom=zoom, scrape_date=scrape_date, neighborhood=neighborhood,
        only_underbuilt=only_underbuilt,
    )


@st.cache_data(ttl=120, show_spinner=False)
def _capacity(bounds_key, zoom, scrape_date, neighborhood, only_underbuilt):
    return queries.capacity_in_bbox(
        bounds_key, zoom=zoom, scrape_date=scrape_date, neighborhood=neighborhood,
        only_underbuilt=only_underbuilt,
    )


@st.cache_data(ttl=120, show_spinner=False)
def _land_use(bounds_key, zoom, scrape_date, neighborhood, use_side):
    return queries.land_use_in_bbox(
        bounds_key, zoom=zoom, scrape_date=scrape_date, neighborhood=neighborhood,
        use_side=use_side,
    )


@st.cache_data(ttl=120, show_spinner=False)
def _opportunities(bounds_key, zoom, scrape_date, neighborhood, site_thesis, top_only, good_only):
    return queries.opportunities_in_bbox(
        bounds_key, zoom=zoom, scrape_date=scrape_date, neighborhood=neighborhood,
        site_thesis=site_thesis, top_only=top_only, good_only=good_only,
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
def _lot_zone_pieces(lot_uid, scrape_date, neighborhood):
    """Every piece of one lot, largest first.

    One row on the great majority of parcels. Where it is longer than one, a
    zoning boundary crosses the parcel and each piece is a site of its own -
    its own area, its own street, its own programme and its own yield - so the
    pane offers the reader the choice of which to open and says plainly that
    the others exist. See `queries.lot_zone_pieces_of`.
    """
    return queries.lot_zone_pieces_of(
        lot_uid, scrape_date=scrape_date, neighborhood=neighborhood
    )


@st.cache_data(ttl=300, show_spinner=False)
def _lot_capacity(lot_uid, scrape_date, neighborhood, feature_id=None):
    return queries.lot_capacity(
        lot_uid,
        scrape_date=scrape_date,
        neighborhood=neighborhood,
        feature_id=feature_id,
    )


@st.cache_data(ttl=300, show_spinner=False)
def _lot_opportunity(lot_uid, scrape_date, neighborhood, feature_id=None):
    """The lot's row of the shortlist table - both theses and what is behind
    them. Its own read for the reason `_lot_program` is: a database with the
    gap and not the shortlist is a real state between two pipeline runs."""
    return queries.lot_opportunity(
        lot_uid,
        scrape_date=scrape_date,
        neighborhood=neighborhood,
        feature_id=feature_id,
    )


@st.cache_data(ttl=900, show_spinner=False)
def _site_thesis_totals(neighborhood, scrape_date):
    return queries.site_thesis_totals(
        neighborhood=neighborhood, scrape_date=scrape_date
    )


@st.cache_data(ttl=900, show_spinner=False)
def _futures_totals(neighborhood, scrape_date):
    return queries.futures_totals(
        neighborhood=neighborhood, scrape_date=scrape_date
    )


@st.cache_data(ttl=900, show_spinner=False)
def _top_site_opportunities(neighborhood, scrape_date):
    return queries.top_site_opportunities(
        neighborhood=neighborhood, scrape_date=scrape_date, limit=12
    )


@st.cache_data(ttl=300, show_spinner=False)
def _lot_program(lot_uid, scrape_date, neighborhood, feature_id=None):
    """The whole proposed programme for one lot.

    Beside `_lot_capacity` rather than folded into it, and cached the same
    way, because they are two reads of two tables answering two questions —
    "is there room" and "what, exactly". The Deal pane calls both: this one for
    the proposal and that one for the single thing it cannot say, which is
    what stands there today.
    """
    return queries.lot_program(
        lot_uid,
        scrape_date=scrape_date,
        neighborhood=neighborhood,
        feature_id=feature_id,
    )


@st.cache_data(ttl=300, show_spinner=False)
def _zoning_for_lot(lot_number, scrape_date):
    """The zones covering a lot, in the lot's own snapshot.

    ``scrape_date`` is the lot's rather than the sidebar's, the same pairing
    `_lot_capacity` makes: the row on screen came from one load of the
    cadastre, and the zones that govern it are that load's. Left out, the
    query answers across every snapshot in the database and the same zone
    comes back once per date - which is what used to make a lot appear to
    straddle two zones that were one zone twice.
    """
    return queries.zoning_for_lot(lot_number, scrape_date=scrape_date)


@st.cache_data(ttl=300, show_spinner=False)
def _lot_coverage(lot_number, scrape_date):
    """How much of the lot is built on, in the lot's own snapshot.

    ``scrape_date`` is passed for the same reason `_zoning_for_lot` passes it,
    and the cost of leaving it out was larger here: the row on screen came from
    one load of the cadastre, and a coverage measured across every load in the
    database counted the same footprint once per snapshot. On a two-snapshot
    borough that read as two buildings covering 106% of a parcel one building
    covers half of.
    """
    return queries.lot_coverage(lot_number, scrape_date=scrape_date)


@st.cache_data(ttl=300, show_spinner=False)
def _zoning_at_point(lon, lat, scrape_date):
    return queries.zoning_at_point(lon, lat, scrape_date=scrape_date)


@st.cache_data(ttl=300, show_spinner=False)
def _lot_documents(lot_uid):
    """The by-law sheets governing one lot, from `rag.lot_documents`.

    No ``scrape_date`` beside the id, unlike `_lot_capacity` and
    `_zoning_for_lot`: a ``lot_uid`` is already one lot in one snapshot, so the
    date it would be paired with is the one it was taken from.
    """
    return queries.lot_documents(lot_uid)


@st.cache_data(ttl=3600, show_spinner="Fetching the zoning grid…")
def _zoning_pdf(url):
    """Fetch and rasterise a grid PDF.

    Returns ``(doc_id, content, filename, pages, render_error)`` rather than
    raising for an unrenderable file: a PDF that cannot be rasterised can still
    be downloaded and still be framed, and either is a better outcome than an
    empty pane.
    """
    from src.utils import documents  # noqa: PLC0415

    document = documents.fetch(url)
    try:
        pages, render_error = documents.render_pages(document.content), None
    except documents.DocumentError as exc:
        pages, render_error = [], str(exc)
    return document.doc_id, document.content, document.filename, pages, render_error


def _select_lot(lot: dict | None) -> None:
    """Adopt a lot as the selection, on both sides of the app."""
    st.session_state.selected_lot = lot
    # A lot resolves its own zoning below, so the zone-only selection a bare
    # click may have left behind is stale the moment a parcel is chosen.
    st.session_state.selected_zone = None
    if lot:
        state.set_selected_lot(
            lot["lot_number"], lot.get("lon"), lot.get("lat"), lot.get("neighborhood")
        )
    else:
        state.clear_selected_lot()


def _select_zone(zone: dict | None) -> None:
    """Adopt a zone that no lot was found under."""
    st.session_state.selected_zone = zone
    st.session_state.selected_lot = None
    state.clear_selected_lot()


def _frame_lot(lot_number: str) -> bool:
    """Select a lot by number and ask the next map build to frame it.

    The two steps `map_tools.show_lot_on_map` takes for the chat, without the
    tool plumbing in between. Both halves are load-bearing: the fit is what
    moves the view, and the selection is what draws the outline over it and
    fills the Lot and Deal panes with the parcel the reader just pointed at.

    The snapshot is passed through because the tables are read off one. Left
    off, `lot_by_number` answers from the newest date the parcel appears in,
    and a reader looking at an older snapshot would be framed on a boundary
    that is not the one the row was measured from.
    """
    try:
        lot = queries.lot_by_number(
            lot_number, scrape_date=st.session_state.scrape_date
        )
    except Exception:  # noqa: BLE001 - a table row must not break the page
        logger.exception("could not resolve lot %s from a table click", lot_number)
        return False
    if not lot:
        return False
    _select_lot(lot)
    bounds = basemap.bounds_of(lot.get("geometry"))
    if bounds:
        st.session_state.fit_bounds = basemap.pad_bounds(bounds)
    return True


#: The Overview tables whose rows are lots, and the read behind each, keyed by
#: the dataframe's widget key. Both sides of a click go through this dict - the
#: pane draws row *n* of `rows_of(...)` and the handler above the map resolves
#: row *n* from the same cached call - so the two cannot disagree about the
#: order, which is the one way a click could frame the wrong parcel.
_LOT_TABLES = {
    "overview_top_capacity": _top_capacity_lots,
    "overview_top_npv_gain": _top_npv_gain_lots,
    "overview_top_sites": _top_site_opportunities,
}


def _site_label(row: Mapping) -> str:
    """What to call one row of a shortlist, now that a row is a piece of a lot.

    A zoning boundary does not have to follow a lot line, so since the piece
    grain a parcel two zones cut in two contributes *two* rows to every table
    here - two programs, two yields, two theses - and both are labelled with
    the same cadastral number. Left at that, a reader sees what looks like the
    same lot twice and no way to tell which half each row is about.

    So a split parcel says so on the row: ``1 740 794 · C04-083 (1 of 2)``.
    The lot number leads because that is what a person searches, reads out and
    matches against a title; the zone follows because it is what makes the row
    a different site from its sibling; the count is what says to expect the
    sibling at all.

    An unsplit parcel - the great majority - is just its number, unchanged.
    Nothing is appended where there is nothing to disambiguate.
    """
    lot = row.get("lot_number") or "—"
    zones = row.get("num_lot_zones")
    zone = row.get("feature_id")
    try:
        count = int(zones) if zones is not None else 1
    except (TypeError, ValueError):
        count = 1
    if count <= 1 or not zone or zone == "-":
        return str(lot)
    rank = row.get("zone_rank")
    try:
        position = int(rank)
    except (TypeError, ValueError):
        # `zone_rank` is not on every table - the gap and the shortlist carry
        # `is_primary_zone` instead, which is the same fact for the only two
        # positions that matter to a label.
        position = 1 if row.get("is_primary_zone") else 2
    return f"{lot} · {zone} ({position} of {count})"


def _zone_piece_key(lot: Mapping) -> str:
    """Where the Lot pane files which piece of a parcel the reader picked.

    Keyed on the lot so two parcels cannot share a choice, and read back by
    `_selected_zone` from every other pane - the Deal block and the Programme
    block ask the same question of the same parcel and must not answer it
    differently.
    """
    return f"zone-piece-{lot.get('lot_uid')}"


def _selected_zone(lot: Mapping) -> str | None:
    """The zone the reader is looking at on this parcel, or None for its primary.

    None is not "unknown": it is what every read here passes when the parcel is
    not split, and it resolves to the largest piece - the answer these panes
    gave before a parcel could have more than one. So a lot one zone covers
    whole never goes near this and is unchanged.
    """
    return st.session_state.get(_zone_piece_key(lot))


def _zone_piece_picker(lot: Mapping) -> str | None:
    """Which piece of a split parcel the Lot pane is about, chosen by the reader.

    Returns the zone number to read the answers for, or ``None`` to take the
    parcel's primary piece - which is what every read here does when it is
    handed no zone, and what the pane showed before a parcel could have more
    than one.

    **Drawn only where there is something to choose.** A lot one zone covers
    whole gets no picker, no caption and no extra click; the pane is unchanged
    for the great majority of parcels. Where a zoning boundary crosses the
    parcel it gets a radio of its pieces, each labelled with its zone, its area
    and the street it faces - because those three are what make one piece a
    different site from its sibling, and a reader choosing between them is
    choosing between two development sites rather than two views of one.

    The caption below it is the part that matters most: without it a reader
    seeing 2 440 m² on a 27 044 m² parcel would take the pane to be wrong.
    """
    lot_uid = lot.get("lot_uid")
    if lot_uid is None:
        return None
    try:
        pieces = _lot_zone_pieces(
            int(lot_uid), lot.get("scrape_date"), lot.get("neighborhood")
        )
    except Exception:  # pragma: no cover - a missing table is not a broken pane
        logger.exception("could not read the zone pieces of lot %s", lot_uid)
        return None
    if len(pieces) < 2:
        return None

    total = float(lot.get("area_m2") or sum(
        float(p.get("piece_area_m2") or 0) for p in pieces
    ))
    st.markdown("**This lot is in more than one zone**")
    st.caption(
        f"A zoning boundary crosses it, so its {total:,.0f} m² are "
        f"{len(pieces)} separate sites — each with its own envelope, its own "
        "street and its own programme. Everything below is about the one "
        "selected."
    )

    def _label(piece: Mapping) -> str:
        area = float(piece.get("piece_area_m2") or 0)
        share = float(piece.get("pct_of_lot") or 0)
        street = piece.get("primary_street_name")
        frontage = piece.get("primary_frontage_m")
        facing = (
            f" · {frontage:,.0f} m on {street}"
            if street and frontage is not None
            else " · no street of its own"
        )
        return f"{piece.get('feature_id')} — {area:,.0f} m² ({share:.0f}%){facing}"

    chosen = st.radio(
        "Zone",
        options=[str(piece.get("feature_id")) for piece in pieces],
        format_func=lambda zone: next(
            _label(piece)
            for piece in pieces
            if str(piece.get("feature_id")) == zone
        ),
        key=_zone_piece_key(lot),
        label_visibility="collapsed",
    )
    return chosen


def _site_area_m2(row: Mapping) -> float:
    """The ground one shortlist row is about, in square metres.

    `piece_area_m2` where the row has one, `lot_area_m2` otherwise. The two
    are equal wherever one zone covers a parcel whole - most of a borough -
    and where they differ it is the piece that matters, because the piece is
    what the program beside it was solved over. Reading the parcel here would
    put 27 044 m² beside a building priced on 2 440.
    """
    for name in ("piece_area_m2", "lot_area_m2"):
        value = row.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _lot_table(rows: list[dict], key: str) -> None:
    """One of the Overview's lot tables, with its rows clickable.

    `key` is the whole of the wiring: Streamlit files the row selection under
    it, `_LOT_TABLES` says which read the rows came from, and
    `_lot_clicked_in_table` puts the two together.
    """
    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=key,
    )


def _lot_clicked_in_table() -> str | None:
    """The lot a row click in an Overview table is asking for, if any.

    Read *above* the map rather than where the table is drawn, and that is why
    this is a function and not four lines in the pane. Streamlit files a
    dataframe's selection in session state under its widget key, so the click
    that caused this rerun is already legible before the pane it came from is
    redrawn - which means the map built below is framed on the parcel on this
    run. Handled where the table is instead, the click would cost a second
    rerun and a second remount of the map, at which point clicking a row is as
    expensive as changing borough.

    Only a *change* fires; `table_clicks` is what remembers. Selecting a row is
    a state and not an event - it stays highlighted until something else is
    picked - so acting on it every rerun would re-fit the map on top of every
    pan the user made afterwards.
    """
    for key, rows_of in _LOT_TABLES.items():
        event = st.session_state.get(key) or {}
        rows = tuple(event.get("selection", {}).get("rows") or ())
        if rows == st.session_state.table_clicks.get(key):
            continue
        # Recorded before the lookup rather than after it: a row whose lot
        # cannot be resolved has still been clicked, and retrying it for as
        # long as it stays highlighted is a query per rerun for ever.
        st.session_state.table_clicks[key] = rows
        if not rows:
            continue
        # A row index outlives the rows it indexed - the borough or the
        # snapshot can change under a highlighted selection - so the bound is
        # checked rather than assumed.
        table = rows_of(
            st.session_state.neighborhood, st.session_state.scrape_date
        ) or []
        if rows[0] < len(table):
            return table[rows[0]].get("lot_number")
    return None


# ---------------------------------------------------------------------------
# The grille des specifications
# ---------------------------------------------------------------------------

#: How tall the embedded viewer is. The side column is 44% of the page and the
#: map beside it is 620, so this is about as much of a portrait sheet as can be
#: shown without the pane becoming the page.
GRID_VIEWER_HEIGHT = int(os.environ.get("HBU_GRID_VIEWER_HEIGHT", 600))


def _embed_pdf(content: bytes, key: str, height: int) -> bool:
    """The viewer, drawn by pdf.js rather than by the browser's PDF plugin.

    Returns whether a viewer was drawn, because the caller opens its fallback
    when one was not.

    This used to be an ``<iframe>`` pointed at the grid's own URL, and that is
    what Microsoft Edge draws *"This page has been blocked by Microsoft Edge"*
    over. Streamlit renders every declared iframe with a ``sandbox`` attribute,
    Chromium refuses to instantiate plugin content inside a sandboxed frame,
    and a browser's built-in PDF viewer is plugin content - so Edge shows its
    interstitial and Chrome shows nothing at all. No header on this side
    changes that: the block is on the frame, not on the response.

    ``st.pdf`` has no plugin in it. It is pdf.js rendering to a canvas in the
    page's own DOM - a CCv2 component, so not inside an iframe at all - and the
    text layer, the search and the page zoom that the rasterised images cannot
    have all survive, on every browser.

    The *bytes* go in rather than the ``/tiles/grid`` URL. Streamlit's media
    file manager hashes them, serves them from the app's own origin and hands
    back the same address for unchanged content on every rerun, so the viewer
    needs no tile port, no CORS and no tile key to appear. The route is still
    what the "Open the grid" button points at, which is the job an iframe was
    never doing.
    """
    if not hasattr(st, "pdf"):  # pragma: no cover - Streamlit older than st.pdf
        return False
    try:
        st.pdf(content, height=height, key=key)
    except st.errors.StreamlitAPIException as exc:
        # `st.pdf` is a thin wrapper over the `streamlit-pdf` component and
        # raises when that extra is not installed. A missing dependency is a
        # deployment fault rather than a document fault, so it is logged rather
        # than drawn: the rasterised pages below open by themselves instead.
        logger.warning("The inline PDF viewer is unavailable: %s", exc)
        return False
    return True


def _grid_url(url: str) -> str | None:
    """The same-origin address of the grid at *url*, or None.

    None when the tile server never took its port - it is the one place this
    app publishes anything over HTTP - in which case the pane falls back to the
    rasterised pages, which need no server at all.
    """
    from src.utils import documents  # noqa: PLC0415

    if _tile_port() is None:
        return None
    return tiles.grid_url(documents.document_id(url))


def _zone_labels(zoning: list[dict]) -> list[str]:
    """Radio labels for the zones covering a lot: the zone, then its share.

    They have to be distinct, because the selection is read back with
    `list.index` and `st.radio` shows one entry per label - two identical ones
    would make the second unreachable. `queries.zoning_for_lot` already returns
    one row per zone, so the only way a label repeats is the one it cannot
    collapse: `source_table` carries no borough namespace, so C01-001 exists in
    every borough that publishes a VSP_REG_ZONE, and a lot answered by two
    boroughs' loads gets the borough appended to tell them apart. Not appended
    unconditionally, since on the ordinary lot it is the same borough twice and
    reads as noise.
    """
    seen: dict[str, int] = {}
    for zone in zoning:
        seen[zone["zone"]] = seen.get(zone["zone"], 0) + 1
    labels = []
    for zone in zoning:
        label = zone["zone"]
        if seen[label] > 1 and zone.get("neighborhood"):
            label += f" ({zone['neighborhood']})"
        if zone.get("overlap_m2") and zone.get("lot_area_m2"):
            label += f" · {zone['overlap_m2'] / zone['lot_area_m2'] * 100:.0f}%"
        labels.append(label)
    return labels


#: What separates one use code from the next inside a ``USAGE`` column. A
#: semicolon is what the scrape stores - "C.4;H" is one zone permitting two
#: things - and the comma is here because a handful of rows use it instead.
#: Neither is a delimiter this app chose; both are the by-law's punctuation as
#: the scrape found it.
_USE_SEPARATORS = re.compile(r"[;,]")


def _permitted_uses(zoning: list[dict]) -> list[str]:
    """The use codes the given zones permit, first-seen order.

    A grid states its uses as a list of by-law codes - "H.1-3", "C.4", "E.2" -
    packed into one string per column, and a lot on a zone boundary is governed
    by two such lists at once. What comes back is their union, because the
    question the Lot pane asks is what may be built *on this lot*, and either
    zone permitting a use is enough for it to be an answer.

    Deduplicated, and in the order met rather than sorted. `zoning_for_lot`
    returns zones by how much of the lot each covers, so the codes listed first
    are the dominant zone's; sorting would put "C.4" ahead of "H" on a lot that
    is 95% residential and read as though commerce were the point of it.

    Exclusions are not in here - see `queries.ZONING_USE_ATTRIBUTES`, which is
    also where the columns come from, so a renamed one cannot go on being read
    here after it has stopped being rendered in the table.
    """
    uses: dict[str, None] = {}
    for zone in zoning:
        attributes = zone.get("attributes") or {}
        for key in queries.ZONING_USE_ATTRIBUTES:
            for code in _USE_SEPARATORS.split(str(attributes.get(key) or "")):
                code = code.strip()
                if code:
                    uses.setdefault(code, None)
    return list(uses)


def _render_zoning_summary(zoning: list[dict], *, count: bool = True) -> None:
    """What zoning does to the selection, without the grid's values.

    The values themselves - seventeen rows of minima, maxima and terms of art -
    are a reading of the by-law, so they are drawn in the pane that is about
    the by-law, beside the sheet they are read off. What stays here is the part
    that is about the *parcel*: how many separate sets of rules land on it, and
    which uses they permit. Those two are what decides whether the grid is
    worth opening at all, and they are the two a reader can hold in their head
    while they look at the floor areas above.

    `count` is off for a click that resolved a zone and no lot. "1 grid zone"
    is a statement about a lot, and on that path there is no lot to make it
    about - the heading above has already named the one zone there is.
    """
    if count:
        st.markdown(
            f"**{len(zoning)} grid zone{'' if len(zoning) == 1 else 's'}** — "
            + " · ".join(_zone_labels(zoning))
        )
    uses = _permitted_uses(zoning)
    if uses:
        st.markdown("**Permitted uses** — " + ", ".join(uses))
    else:
        st.caption("No permitted use is stated on this zoning in this snapshot.")
    st.caption(
        "The grid's values, and the sheet they are read off, are under "
        "**Regulations**."
    )


def _use_sides(potential: dict, coverage: dict | None) -> list[dict]:
    """The rows of the Lot pane's today-against-proposed table.

    Split from the renderer so the agent and a test can read the same rows
    the pane draws. Each row is one measure with a value on each side, and a
    side with nothing to say is a dash rather than a zero: the roll not
    reaching a lot and the roll counting no dwellings on it are different
    facts, and only the second is a 0.

    The footprint on today's side is the *measured* one - the silver clip
    the Footprint line above the table already reports - and not a figure off
    the gap table, which carries none; the proposed side is the solver's.
    """
    def area(value):
        return "—" if value is None else f"{float(value):,.0f} m²"

    def count(value):
        return "—" if value is None else f"{int(value):,}"

    today_use = potential.get("existing_dominant_income_class")
    today_words = potential.get("existing_dominant_use_description")
    today = " · ".join(str(v) for v in (today_use, today_words) if v) or "not on the roll"
    proposed_use = potential.get("hbu_dominant_use")
    if potential.get("hbu_status") != "solved":
        proposed = "no programme"
    else:
        proposed = str(proposed_use or "—")
        if today_use and today_use != "none" and proposed_use and proposed_use != "none":
            proposed += " · same use" if today_use == proposed_use else " · changes use"

    floor_today = (
        "not reported" if queries.floor_area_unreported(potential)
        else area(potential.get("existing_floor_area_m2"))
    )
    footprint_today = (
        area(coverage.get("covered_area_m2"))
        if coverage and coverage.get("num_footprints") else "—"
    )
    return [
        {"Measure": "Use", "Today": today, "Proposed": proposed},
        {
            "Measure": "Floor area",
            "Today": floor_today,
            "Proposed": area(potential.get("hbu_floor_area_m2")),
        },
        {
            "Measure": "Footprint",
            "Today": footprint_today,
            "Proposed": area(potential.get("footprint_m2")),
        },
        {
            "Measure": "Dwellings",
            "Today": count(potential.get("existing_num_dwellings")),
            "Proposed": count(potential.get("hbu_num_dwellings")),
        },
    ]


def _render_use_comparison(potential: dict | None, coverage: dict | None) -> None:
    """What stands against what is proposed, one measure per row.

    The rest of the Lot pane reads the two sides apart - the footprint above,
    the roll's use in words, the efficiency arithmetic below - and each of
    those is the right place for its caveat. This is the one place the two
    sides sit in a row, which is the reading the map's Land use layer offers
    per lot on hover and the one a reader comparing parcels actually makes.
    """
    if not potential:
        return
    st.markdown("**Today against the proposal**")
    st.dataframe(_use_sides(potential, coverage), width="stretch", hide_index=True)
    st.caption(
        "Today is the assessment roll and the measured footprint; proposed is "
        "the solved programme. The **Land use** layer colours the map by "
        "either side."
    )


def _render_zoning_attributes(zone: dict) -> None:
    """The grid's values as a table, for a zone reached either way."""
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


def _render_zoning_values(zoning: list[dict], *, key: str) -> None:
    """The grid's values for a lot's zones, one zone at a time.

    This is the table the Lot pane used to draw. It is here because it is a
    reading of the by-law rather than a fact about the parcel, and because it
    belongs beside the sheet: checking a number against the PDF used to mean
    changing tabs to do it.

    One zone at a time, for the same reason the documents below are drawn one
    at a time. A lot straddling two zones is a lot where *which* row governs is
    the question being asked, and stacking both invites reading a maximum off
    the wrong one.

    The heading goes *above* the picker rather than under it, which is not a
    detail on a lot that straddles. This pane then carries two radios whose
    options read alike - the zones, and the sheets those zones cite, which on
    the ordinary straddling lot are the same list twice - and what tells them
    apart is the thing each sits under. Under a heading, the top one is
    plainly choosing whose values are in the table; unlabelled, it is a second
    document picker that does not change the document.
    """
    if not zoning:
        return
    st.markdown("**Grid values**")
    index = 0
    if len(zoning) > 1:
        labels = _zone_labels(zoning)
        # Indices as the options rather than the labels, the choice
        # `_render_lot_documents` makes below and for the same reason: two
        # zones can carry the same label - the same number in two boroughs,
        # both covering the same share - and `st.radio` would then make the
        # second unreachable.
        index = st.radio(
            "Zone",
            range(len(zoning)),
            format_func=lambda i: labels[i],
            horizontal=True,
            key=key,
        )
    zone = zoning[index]
    st.caption(f"Zone {zone['zone']}, as the by-law states it.")
    _render_zoning_attributes(zone)


def _render_zoning_grid(zone: dict, *, has_chunks: bool) -> None:
    """One zone's grid: its values, then the sheet they are read off.

    Reached only from the Regulations pane, for a click that landed on zoned
    ground carrying no parcel. A lot goes the other way round - through
    `_render_lot_documents`, which asks the corpus what governs it rather than
    asking one zone row what it links.

    The values come first and *before* the link is resolved, so a zone whose
    scrape dropped its ``LIEN_GRILLE`` still reports what it permits rather
    than only that its PDF is missing.
    """
    st.markdown("**Grid values**")
    st.caption(f"Zone {zone['zone']}, as the by-law states it.")
    _render_zoning_attributes(zone)
    st.divider()

    url = zone.get("zoning_pdf_url")
    if not url and has_chunks:
        # Nothing on the row, but the corpus may have embedded the sheet under
        # this zone's number anyway.
        url = queries.zoning_pdf_url_fallback(zone["zone"])

    if not url:
        st.caption("No zoning grid is linked from this zone.")
        return

    st.markdown(f"**Zoning grid — zone {zone['zone']}**")
    # The sheet itself is a French document titled "grille des specifications";
    # the caption names it so a reader can match the page to the by-law's index.
    st.caption("The borough files this sheet as a *grille des spécifications*.")
    _render_pdf_document(url)


def _render_pdf_document(url: str) -> None:
    """The document at *url*: the links to it, then the sheet.

    Split out of `_render_zoning_grid` so the Regulations pane can draw a
    document it reached from `rag.lot_documents`, where there is no zone row to
    carry the link and the sheet is not necessarily a grid.

    The links come first and are unconditional, because a hyperlink is the
    thing a reader most often wants to *keep* - a LIEN_GRILLE pasted into a
    report, or a second tab open beside the map while they work. Both are
    offered and they are not redundant: `_grid_url` is this app's own copy,
    openable from an https page as the city's ``http://`` link is not, and the
    city's URL is the citable one that will outlive this deployment.
    """
    if not url:
        # `_render_zoning_grid` checks this before it calls, because it has a
        # zone to name in the message. The documents pane has no equivalent -
        # a corpus row with a null url is a document that was indexed without
        # one - so the shared entry point refuses rather than handing None to
        # the fetcher and printing it back as a link.
        st.caption("This document carries no link in the corpus.")
        return

    try:
        doc_id, content, filename, pages, render_error = _zoning_pdf(url)
    except Exception as exc:  # noqa: BLE001 - one dead link, not the pane
        st.error(f"Could not fetch the document: {exc}")
        st.markdown(f"[Open it at the source]({url})")
        return

    from src.utils import documents  # noqa: PLC0415

    # Re-published on every rerun rather than only on the fetch: `_zoning_pdf`
    # is cached, so a rerun that redraws this sheet does not go through
    # `documents.fetch` and would not otherwise renew the registry entry the
    # "Open the grid" button below is about to point at.
    documents.publish(doc_id, content)
    served = _grid_url(url)

    if served:
        left, right = st.columns(2)
        left.link_button("📄 Open the grid", served, width="stretch")
        right.link_button("🔗 At the city", url, width="stretch")
    else:
        st.link_button("🔗 Open it at the city", url, width="stretch")
    # The raw link, selectable, because "get the hyperlink" is its own task and
    # a button is the one form of a URL that cannot be copied out of.
    st.caption(f"`{url}`")

    # A real PDF viewer - text selection, search, page zoom - which is what the
    # rasterised pages cannot be. It draws from the bytes rather than from
    # `served`, so it appears whether or not the tile server took its port.
    framed = _embed_pdf(content, f"grid-viewer-{doc_id}", GRID_VIEWER_HEIGHT)

    # Collapsed when the viewer above is showing the same sheet, open when it
    # is the only thing there is. `render_error` is not shown as an error in
    # the first case: a document this app cannot rasterise is one the viewer
    # may still display perfectly well.
    with st.expander("Pages as images", expanded=not framed):
        if pages:
            for number, png in enumerate(pages, 1):
                st.image(png, width="stretch", caption=f"Page {number}")
        elif render_error:
            st.caption(f"Cannot render inline: {render_error}")
        else:
            st.caption("No pages to show.")

    st.download_button(
        "⬇️ Download the grid (PDF)",
        data=content,
        file_name=filename,
        mime="application/pdf",
        width="stretch",
        # Keyed on the document, so the two ways into this pane - a lot's
        # documents and a bare zone selection - do not collide on Streamlit's
        # auto-generated widget id. They are exclusive branches, so one sheet
        # is drawn per rerun and the id is unique by construction.
        key=f"grid-download-{doc_id}",
    )


# ---------------------------------------------------------------------------
# The by-law sheets that govern a lot
# ---------------------------------------------------------------------------


def _documents_for_lot(lot: dict, zoning: list[dict], *, caps) -> tuple[list[dict], bool]:
    """Every sheet that applies to *lot*, and whether the join answered.

    Two sources, and the second is a fallback rather than an equal.

    `queries.lot_documents` is the real answer: the corpus records which
    features cite each document, `silver.lot_features` records which features
    cover the lot, and the view multiplies the two. It finds a document whether
    or not any scraped attribute still points at it, and it finds documents on
    layers other than zoning the day the dataplatform starts indexing one.

    The fallback is the ``LIEN_GRILLE`` on the zoning rows the pane has already
    fetched. It needs no corpus at all, which is the case it is here for - a
    borough loaded this morning has its lots and its zones before
    ``document_index`` has run over it - and it sees exactly what a scrape
    happened to link. The flag comes back so the pane can say which of the two
    the reader is looking at, because "no other document applies" and "no other
    document is reachable this way" are different statements.

    Either way one row is one *document*: a sheet cited by two of the lot's
    zones is one PDF with two zones on it, and offering it twice would read as
    two sets of rules.
    """
    if caps.lot_documents and lot.get("lot_uid") is not None:
        rows = _lot_documents(int(lot["lot_uid"]))
        if rows:
            return rows, True

    merged: dict[str, dict] = {}
    for rank, zone in enumerate(zoning, 1):
        url = zone.get("zoning_pdf_url")
        if not url and caps.chunks:
            url = queries.zoning_pdf_url_fallback(zone["zone"])
        if not url:
            continue
        overlap = float(zone.get("overlap_m2") or 0)
        area = float(zone.get("lot_area_m2") or 0)
        row = merged.get(url)
        if row is None:
            merged[url] = {
                "doc_id": None,
                "url": url,
                "title": None,
                "source_table": zone.get("source_table"),
                "neighborhood": zone.get("neighborhood"),
                "scrape_date": zone.get("scrape_date"),
                "zones": [zone["zone"]],
                # `zoning_for_lot` orders by overlap descending, so the first
                # zone to reach a sheet is the one covering most of the lot -
                # which is the share this column means.
                "pct_of_lot": (overlap / area * 100) if area else None,
                "overlap_m2": overlap,
                "coverage_rank": rank,
            }
        else:
            row["zones"].append(zone["zone"])
            row["overlap_m2"] += overlap
    return list(merged.values()), False


def _document_label(document: dict) -> str:
    """A one-line name for a sheet: the zones it governs, then its share."""
    zones = [str(z) for z in (document.get("zones") or [])]
    label = " + ".join(zones[:2]) + ("…" if len(zones) > 2 else "")
    if not label:
        label = str(document.get("title") or document.get("doc_id") or "document")
    pct = document.get("pct_of_lot")
    if pct is not None:
        label += f" · {float(pct):.0f}%"
    return label


def _render_lot_documents(lot: dict, *, caps) -> None:
    """The Regulations pane's top half: what governs the selected lot.

    A retrieval pane and a *documents* pane are not the same thing, and this is
    the half that was missing. What the chat retrieved is a set of passages
    that matched a question; what a zone cites is the sheet that governs the
    parcel whether or not anybody has asked anything. Clicking a lot is not a
    question, so it produced nothing here until now.

    The grid's *values* lead, then the sheets. Both are readings of the
    by-law, which is why they are in this pane and not beside the lot's own
    numbers, and putting them one above the other is what lets a reader check
    a figure against the page it came off without changing tabs.

    One sheet is drawn at a time rather than all of them stacked. A grid is a
    full page of PDF and the pane is 44% of the window, so three of them is
    three scroll-lengths between the reader and the retrieved passages below -
    and a lot with three is a lot on two zone boundaries, where *which* sheet
    governs is the question being asked in the first place.
    """
    st.markdown(f"### By-laws for lot {lot['lot_number']}")

    zoning = (
        _zoning_for_lot(lot["lot_number"], lot.get("scrape_date"))
        if caps.features else []
    )

    # The values above the sheets, and outside the `not documents` branch
    # below rather than inside it: a zone whose scrape carries no link still
    # states a height and a coverage, and letting the table go missing with the
    # PDF would make an unlinked lot read as an unzoned one.
    #
    # Keyed on the lot, like the document radio below, so the zone chosen for
    # the last lot is not carried into the next one - where the index may name
    # a zone that does not cover it.
    _render_zoning_values(
        zoning, key=f"rules-zone-{lot.get('lot_uid') or lot['lot_number']}"
    )
    if zoning:
        st.divider()

    documents, from_join = _documents_for_lot(lot, zoning, caps=caps)

    if not documents:
        if not caps.features:
            st.info(
                f"`{queries.SCHEMA}.features` is not in this database, so "
                "there is no zoning layer to resolve this lot against."
            )
        elif not zoning:
            st.info(
                "No zone covers this lot in this snapshot, so no sheet "
                "applies to it. Turn **Zoning** on in the sidebar to see "
                "where the boundaries run."
            )
        elif not caps.chunks:
            st.info(
                f"The {len(zoning)} zone(s) covering this lot carry no link to "
                f"a sheet, and `{queries.SCHEMA}.chunks` is not in this "
                "database to resolve one from. The dataplatform's "
                "`document_index` asset is what loads the corpus."
            )
        else:
            st.info(
                f"The {len(zoning)} zone(s) covering this lot cite no document "
                "in this snapshot."
            )
        return

    zones = sorted({str(z) for d in documents for z in (d.get("zones") or [])})
    st.caption(
        f"{len(documents)} document(s) · "
        f"{'zone ' + ', '.join(zones) if zones else 'no zone named'} · "
        f"snapshot {lot.get('scrape_date')}"
    )
    if not from_join:
        # Two ways to arrive here and they are different findings: the join is
        # absent, or it is present and empty for this lot. Saying "not in this
        # database" about a view that is there sends the reader to the wrong
        # repo.
        why = (
            "which has no row for this lot in this snapshot"
            if caps.lot_documents
            else "which is not in this database"
        )
        st.caption(
            f"Resolved from each zone's `{queries.ZONING_URL_ATTRIBUTE}` "
            f"rather than from `{queries.SCHEMA}.lot_documents`, {why} — so "
            "any document no scrape linked is not listed."
        )

    index = 0
    if len(documents) > 1:
        # Indices rather than labels as the options: two sheets can legitimately
        # carry the same label - the same zone number in two boroughs, both
        # covering the same share - and `st.radio` would then make the second
        # unreachable.
        index = st.radio(
            "Document",
            range(len(documents)),
            format_func=lambda i: _document_label(documents[i]),
            horizontal=True,
            # Keyed on the lot as well as the pane. A bare "rules-document"
            # would carry the index chosen for the last lot into the next one,
            # where it may name a sheet that is not in the list.
            key=f"rules-document-{lot.get('lot_uid') or lot['lot_number']}",
        )

    chosen = documents[index]
    on = ", ".join(str(z) for z in (chosen.get("zones") or []))
    if chosen.get("source_table") == queries.ZONING_SOURCE_TABLE:
        # The familiar name leads. This app calls the sheet a zoning grid
        # everywhere else, while the corpus's own title for it is whatever the
        # scrape recorded - a filename, as often as not - so the recorded one
        # follows in the caption where it can be matched against an index
        # rather than mistaken for the sheet's subject.
        st.markdown("**Zoning grid**" + (f" — zone {on}" if on else ""))
        caption = "The borough files this sheet as a *grille des spécifications*."
        if chosen.get("title"):
            caption += f" Indexed as *{chosen['title']}*."
        st.caption(caption)
    else:
        title = str(chosen.get("title") or "Document")
        st.markdown(f"**{title}**" + (f" — {on}" if on else ""))
    _render_pdf_document(chosen["url"])


# ---------------------------------------------------------------------------
# The proposed building, in detail
#
# The Lot pane answers "is there room here" — a subtraction, three headroom
# figures and a dwelling count — and stops there on purpose: it is about the
# parcel, and the parcel is what a click selected. This pane answers the
# question that follows, which is a different one and much longer: *what,
# exactly, is being proposed*. Storeys by use and the order they stack in, the
# unit mix by bedroom class, where the stalls go, what each part costs, and the
# printed caps the answer is pressed against.
#
# It is a pane rather than a section under the Lot one because every number
# here is conditional on the same choice — the governing envelope the solver
# picked — and reading them beside the *existing* building would invite exactly
# the confusion the Lot pane already has to caption its way out of twice. Here
# nothing standing today is on screen except where it is named as such.
# ---------------------------------------------------------------------------

#: Why a lot has no solved programme. One dict rather than two copies: this
#: pane and the Lot pane both say it, and the agent's `lot_efficiency` says the
#: same five things in its own voice. A status the dataplatform adds and this
#: map has never heard of falls through to the raw value rather than to a
#: blank, which is the rule `hbu_status` is published under.
_HBU_STATUS_REASONS = {
    "no_candidate_column":
        "Every zoning column reaching this lot authorises none of the uses "
        "the solver prices (housing, commerce, industry) — usually a "
        "community-facilities zone.",
    # The former name of no_candidate_column, from when the solver priced
    # dwellings alone. Rows written before the rename carry it until their
    # partition is re-materialized.
    "no_residential_column":
        "Every zoning column reaching this lot authorises something other "
        "than housing; this snapshot predates the solver pricing commerce "
        "and industry.",
    "no_governing_column":
        "Candidate columns exist but none governs — usually a lot with no "
        "measured frontage under a grid that states a minimum width.",
    "infeasible":
        "No governing column has a feasible programme — a minimum this "
        "parcel cannot meet. Not the parking: a lot the stalls alone stop is "
        "solved again without them and reported solved, with the parking "
        "waived and the stalls it owes counted.",
    "solver_error":
        "The governing column could not be turned into a model.",
}


def _hbu_status_reason(status) -> str:
    return _HBU_STATUS_REASONS.get(status, str(status))


#: CMHC's bedroom classes, in the order a rent schedule prints them, keyed by
#: the spelling the survey uses because that is the key the solver wrote.
#: `3_bedroom_plus` is "3 chambres +" rather than exactly three.
_BEDROOM_LABELS = {
    "studio": "Studio",
    "1_bedroom": "1 bedroom",
    "2_bedroom": "2 bedrooms",
    "3_bedroom_plus": "3+ bedrooms",
}

#: What each name in `binding` means, in the by-law's own vocabulary where it
#: has one. The list answers "why is it not bigger", and every entry on it is a
#: cap that was *reached* rather than a fault — except the first, which answers
#: the different question "why is there nothing at all".
_BINDING_LABELS = {
    "nothing_pencils":
        "**Nothing pencils.** The envelope is whatever the grid prints; what "
        "is zero is the best programme inside it. No mix the caps allow earns "
        "back what it costs at these rents — an economics finding, not a "
        "zoning one.",
    "max_dwellings":
        "The dwelling ceiling this column's usage classes imply.",
    "density_max":
        "*Densité* — the floor-area ratio. Another storey would exceed the "
        "floor area the zone allows.",
    "above_grade_parking":
        "Structured parking above grade is spending the *Densité* the "
        "dwellings wanted: a stall inside the building is floor area.",
    "site_coverage_max":
        "*Taux d'implantation au sol* — the share of the lot the plate may "
        "cover.",
    "setbacks":
        "The zone's margins, not the coverage: the buildable area the four "
        "setbacks leave is the smaller of the two ceilings here. Argued at "
        "the lot line rather than at the plan.",
    "floors":
        "*En étage* — the permitted number of storeys.",
    "height_max":
        "*Hauteur en mètre* — the metric cap, at least as tight here as the "
        "storey rows.",
    "commercial_floor_area":
        "Commercial floor outbid housing for storeys the level rows would "
        "have allowed the dwellings. The housing figure is small for a reason "
        "in the rents rather than in the grid.",
    "industrial_floor_area":
        "Industrial floor outbid housing for storeys the level rows would "
        "have allowed the dwellings.",
    "max_underground_levels":
        "The assumed limit on dug levels — an assumption of the model rather "
        "than a printed norm. It is in the assumptions below.",
    # The six below appear on an INFEASIBLE row instead, and each names a
    # contradiction rather than a cap. They are here because `binding` is
    # where the solver puts them, and a pane showing a bare "infeasible" would
    # be discarding the whole answer.
    "height_range":
        "*Hauteur min* exceeds *Hauteur max* — two rows of the same column "
        "contradicting each other.",
    "height_max_below_floors_min":
        "*En étage min* demands storeys *Hauteur max* has no room for, at the "
        "storey heights assumed below.",
    "floors_min_exceeds_permitted_levels":
        "*En étage min* demands more storeys than the level rows permit.",
    "no_priced_unit_type":
        "CMHC published no rent for any bedroom class in this borough, and "
        "this column authorises nothing else — so there was no model to build.",
    "site_coverage_range":
        "*Taux d'implantation min* exceeds *Taux d'implantation max* — the "
        "column's own two bounds contradict each other.",
    "buildable_area_below_site_coverage_min":
        "*Taux d'implantation min* demands a footprint the setbacks leave no "
        "room for. The column is coherent; this parcel cannot satisfy it.",
}

#: The four kinds of storey, and how each reads in a stack.
_STOREY_USES = {
    "residential": "Housing",
    "commercial": "Commerce",
    "industrial": "Industry",
    "parking": "Parking",
}


def _levels(entry: dict) -> str:
    """One storey run's levels, the way a floor indicator reads them.

    1 is the rez-de-chaussée and the dug levels run -1 downwards; there is no
    level 0, so a run never spans grade and this never prints a range across
    it.
    """
    low, high = entry.get("from_level"), entry.get("to_level")
    if low is None:
        return "—"
    return f"{int(low)}" if high in (None, low) else f"{int(low)} – {int(high)}"


def _render_program_stack(stack: list) -> None:
    """`floor_stack`, top storey first.

    The column is stored bottom upwards, which is the order the solver stacks
    in and the order a `jsonb_array_elements` reader wants. A person reads a
    building off an elevation, from the roof down, so the table is reversed
    here and nowhere else.

    Runs of identical levels rather than one row per storey: the model builds
    one plate and repeats it, so a fifteen-storey tower over retail and a
    parking deck is three rows and not fifteen.
    """
    rows = []
    for entry in reversed(list(stack)):
        if not isinstance(entry, dict):
            continue
        use = str(entry.get("use") or "")
        row = {
            "Level": _levels(entry),
            "Use": _STOREY_USES.get(use, use.replace("_", " ").title() or "—"),
            "Storeys": int(entry.get("floors") or 0),
            "Plate (m²)": f"{float(entry.get('floor_plate_m2') or 0):,.0f}",
            "Floor area (m²)": f"{float(entry.get('floor_area_m2') or 0):,.0f}",
        }
        # A dug level stands no metres — height is measured from grade up — so
        # the entry carries 0 there, and an em dash says so rather than "0.0".
        height = float(entry.get("height_m") or 0)
        row["Height (m)"] = f"{height:,.1f}" if height else "—"
        row["Dwellings"] = int(entry.get("dwellings") or 0) or "—"
        row["Stalls"] = int(entry.get("stalls") or 0) or "—"
        rows.append(row)
    if not rows:
        return
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(
        "Bottom to top the solver stacks parking, commerce, industry then "
        "housing; the table reads down from the roof. That order is a "
        "reporting convention rather than a design — the *Niveaux de bâtiment "
        "autorisés* block is marked per column and not per usage, so the model "
        "counts storeys by type and never places one. Below grade, level −1 "
        "downwards, the floor area is outside the *superficie de plancher* "
        "(article 38 1° of by-law 01-283) and a dug level stands no metres."
    )


def _render_program_binding(program: dict) -> None:
    """The printed caps the answer is pressed against.

    Two headings over one column, because `binding` carries two kinds of name.
    On a solved row every entry is a cap the programme *reached*, and the list
    answers "why is it not bigger". On an infeasible one it is a pair of rows
    contradicting each other, and the list answers "why is there nothing" —
    calling that a cap the programme reached would describe a programme that
    was never built.
    """
    binding = program.get("binding") or []
    if not binding:
        return
    solved = program.get("hbu_status") == "solved"
    st.markdown("**Why not more**" if solved else "**What stopped it**")
    for name in binding:
        st.markdown(f"- {_BINDING_LABELS.get(str(name), f'`{name}`')}")
    if solved:
        st.caption(
            "Each of these is a cap the programme *reached*. Change one of "
            "them in the grid and the answer moves; change a row that is not "
            "on this list and it does not."
        )


def _render_program_choice(program: dict) -> None:
    """How much of a choice this lot had, and which line was believed.

    Three numbers that look like provenance and are not. `num_candidates` says
    whether there was a decision to make at all; `num_zones` above one says the
    lot sits on a zoning boundary, where two publishers drew two lines and
    `pct_of_lot` is which one is believed. A reader who does not know a lot
    straddles two zones reads the programme as *the* answer when it is the
    answer under one of two candidate rule-sets.
    """
    parts = []
    candidates = program.get("num_candidates")
    governing = program.get("num_governing_candidates")
    if candidates is not None:
        parts.append(
            f"{int(candidates)} usage candidate(s) on this lot"
            + (f", {int(governing)} governing" if governing is not None else "")
        )
    if program.get("usages"):
        parts.append("column heads " + ", ".join(str(u) for u in program["usages"]))
    if program.get("column_index") is not None:
        parts.append(f"column {int(program['column_index'])}")
    if program.get("pct_of_lot") is not None:
        parts.append(f"the zone covers {float(program['pct_of_lot']):,.0f}% of the lot")
    if parts:
        st.caption("Chosen from: " + " · ".join(parts) + ".")
    if int(program.get("num_zones") or 0) > 1:
        # Worded for both branches: this is said under a solved programme and
        # under a lot that has none, and "solved under" would be a claim on
        # the second.
        st.caption(
            f"⚠️ {int(program['num_zones'])} zones reach this lot, and it is "
            "answered under the one covering most of it — a sliver of a "
            "neighbour's zoning is two publishers disagreeing about where a "
            "line runs, not two rule-sets the owner may choose between."
        )


#: The three futures, as the pane names them.
_FUTURE_TITLES = {
    "hold": "Keep",
    "enhance": "Enhance",
    "rebuild": "Tear down and rebuild",
}

#: What each investment thesis is called on the pane, and who buys it.
#:
#: `gold.lot_investment_opportunities.investment_thesis` is the *other* axis
#: from the site thesis: not why the parcel is acquirable but what would go on
#: it, read off whichever of the three proposed floor areas dominates. It is
#: the axis that says which buyer to call, so the pane says the buyer and not
#: only the label.
_INVESTMENT_THESIS_TITLES = {
    "residential": (
        "Residential",
        "housing dominates the proposed floor",
        "apartment developers and multi-residential funds",
    ),
    "mixed_use": (
        "Mixed use",
        "housing over commercial, neither one dominant",
        "mixed-use developers; the retail base underwrites the storeys above",
    ),
    "commercial": (
        "Commercial",
        "commercial floor dominates the proposed building",
        "retail and office investors, and owner-occupiers of the ground floor",
    ),
    "industrial": (
        "Industrial",
        "industrial floor dominates the proposed building",
        "industrial and logistics buyers",
    ),
}


def _months(payload, key: str, fallback: int | None = None) -> int | None:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            payload = None
    if isinstance(payload, dict) and payload.get(key) is not None:
        try:
            return int(payload[key])
        except (TypeError, ValueError):
            return fallback
    return fallback


def _money(value, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    value = float(value)
    if signed:
        return f"{'+' if value >= 0 else '−'}${abs(value):,.0f}"
    return f"${value:,.0f}"


def _as_dict(payload) -> dict:
    """A JSONB column that may arrive parsed, as text, or as nothing."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            payload = None
    return payload if isinstance(payload, dict) else {}


def _render_deal(lot: dict, *, caps) -> None:
    """One lot, one pane: what it costs, what it returns, and what gets built.

    The pane a broker works from. It used to be three — a *Buyer* pane, an
    *Owner* pane and an *HBU* pane — and the split was wrong for anyone whose
    question is whether a transaction clears: the price sat on one tab, the
    building that justifies the price on another, and the third priced a
    future for a person who is not in the deal. All of it is one row of
    `gold.lot_investment_opportunities` beside one of `gold.lot_highest_best_use`,
    priced on one footing there, so the pane reads both and re-prices neither.

    The order is the order a deal is argued in: what it takes to buy, what a
    buyer could pay before the return goes, which of the three futures pays
    best at that price, why the parcel is acquirable at all and who would
    build on it, then the building itself in full. The owner's side of it
    survives in two places, both because the deal cannot be stated without
    them: what the standing income is worth to whoever holds it sets the floor
    under the asking price, since a seller keeps the better of that and the
    roll; and the Overview pane still counts the lots whose owner does best by
    keeping, which is the count of lots that will not be listed however well
    they price for a buyer.
    """
    if not lot or lot.get("lot_uid") is None:
        st.info(
            "Click a lot on the map, or ask the chat for one by number. This "
            "pane prices what the ground would take and what a buyer would "
            "get back for it."
        )
        return

    site = (
        _lot_opportunity(
            int(lot["lot_uid"]), lot.get("scrape_date"), lot.get("neighborhood"),
            _selected_zone(lot),
        )
        if caps.investment_opportunities
        else None
    )

    # A road parcel is the public way itself — the roll files it under a CUBF
    # road code, or a geobase double side runs down the inside of it. Nothing
    # may be bought and built there whatever the grid over the block permits,
    # and every figure below would be arithmetic on an artefact of two layers
    # meeting at the curb. The same refusal the Lot pane makes, made once here
    # ahead of both halves of the pane.
    if (site or {}).get("hbu_status") == "road_parcel":
        st.markdown(f"### Lot {lot.get('lot_number')} — **street**, not a deal")
        st.caption(
            "This parcel is the public way itself; nothing may be built on it "
            "and there is nothing to transact."
        )
        return

    _render_deal_terms(lot, site, caps=caps)

    # Why the parcel is acquirable at all - the second axis, and the one that
    # survives a lot the roll never priced. Drawn from here rather than from
    # inside the terms block, which returns early on a lot with no price, and
    # a vacant parcel with no assessment is exactly the site this block is
    # worth reading on.
    _render_site_thesis(lot, caps=caps)

    # The building, in full, under the money it justifies. Missing programme
    # table and missing shortlist table are different states of a database
    # between two pipeline runs, and each loses its own half of the pane
    # rather than the pane.
    if caps.highest_best_use:
        st.divider()
        _render_hbu_program(lot, caps=caps)
    else:
        st.divider()
        st.info(
            f"`{queries.GOLD_SCHEMA}.lot_highest_best_use` is not in this "
            "database yet, so the building behind the price above is not "
            "shown — run the `lot_highest_best_use` asset for this partition."
        )


def _render_deal_terms(lot: dict, site: dict | None, *, caps) -> None:
    """The price, the ceiling over it, and the three futures priced against it.

    Everything here is a buyer's arithmetic, because a buyer is who a lot is
    sold to. Each future is its value to whoever ends up holding the lot less
    the price paid for the ground, so unlike the programme below — which
    excludes land, the solver holding the envelope constant either way — the
    lot cost is inside every number on this block.
    """
    if not caps.investment_opportunities:
        st.info(
            f"`{queries.GOLD_SCHEMA}.lot_investment_opportunities` is not in "
            "this database yet — it is where the price and the three futures "
            "are worked out. Run the `lot_investment_opportunities` asset for "
            "this partition; the building below is drawn without it."
        )
        return
    if not site:
        st.info(
            "No row in the shortlist table for this lot in this snapshot, so "
            "there is no price and no return on it — run the "
            "`lot_investment_opportunities` asset for this partition."
        )
        return

    program = (
        _lot_program(
            int(lot["lot_uid"]), lot.get("scrape_date"), lot.get("neighborhood"),
            _selected_zone(lot),
        )
        if caps.highest_best_use
        else None
    ) or {}

    st.markdown(
        f"### Lot {site.get('lot_number') or lot.get('lot_number')} — the deal"
    )
    subtitle = []
    if site.get("grid_zone"):
        subtitle.append(f"zone {site['grid_zone']}")
    if site.get("lot_area_m2"):
        subtitle.append(f"{float(site['lot_area_m2']):,.0f} m²")
    subtitle.append(f"snapshot {site.get('scrape_date')}")
    st.caption(" · ".join(str(part) for part in subtitle))

    hold_value = site.get("owner_hold_value_cad")
    acquisition = site.get("acquisition_cost_cad")
    assumptions = _as_dict(site.get("screen_assumptions"))
    factor = float(assumptions.get("market_value_factor") or 1.0)
    assessed = site.get("existing_total_assessed_value")
    best = site.get("buyer_best_future")

    if acquisition is None:
        st.warning(
            "The assessment roll never reached this lot, so there is no price "
            "to put on the ground and no deal to price against it. The "
            "building the solver proposes is below; what the land costs is "
            "the missing half."
        )
        return

    # --- what it takes, and what it could take -----------------------------
    #
    # Three numbers and the third is the one a broker works: the price, the
    # ceiling the best future puts over it, and the distance between them.
    # A deal exists in that distance and nowhere else, which is why it is a
    # metric of its own rather than a subtraction left to the reader.
    residual_by_future = {
        "hold": hold_value,
        "enhance": site.get("residual_price_enhance_cad"),
        "rebuild": site.get("residual_price_rebuild_cad"),
    }
    ceiling = residual_by_future.get(best) if best and best != "none" else None
    if ceiling is None:
        priced = [
            float(value) for value in residual_by_future.values() if value is not None
        ]
        ceiling = max(priced) if priced else None
    room = None if ceiling is None else float(ceiling) - float(acquisition)

    price_cols = st.columns(3)
    price_cols[0].metric(
        "Price to pay",
        _money(acquisition),
        help=(
            "The larger of the roll's assessed value (land and building) times "
            f"the market factor of {factor:g}, and what the standing income is "
            "worth discounted — a seller keeps the better of the two. Not an "
            "appraisal."
        ),
    )
    price_cols[1].metric(
        "Most a buyer could pay",
        _money(ceiling),
        help=(
            "The price at which the best future's NPV falls to zero at the "
            "discount rate. Above it the buyer is paying for someone else's "
            "profit."
        ),
    )
    price_cols[2].metric(
        "Room between them",
        _money(room, signed=True) if room is not None else "—",
        help=(
            "The ceiling less the price. It is the whole of the negotiating "
            "range and the whole of the buyer's margin: what is conceded out "
            "of it on price comes out of the return."
        ),
    )

    basis = (
        "the standing income's present value"
        if hold_value is not None
        and float(acquisition) > float(assessed or 0) * factor + 0.5
        else f"the roll's assessed value × {factor:g}"
    )
    st.caption(
        f"The price is set by {basis}. The roll says {_money(assessed)}; at "
        f"×{factor:g} that is {_money(float(assessed or 0) * factor)}; the standing "
        f"income is worth {_money(hold_value)} to whoever holds it. A seller "
        "keeps whichever is larger, so that is the floor a listing starts from."
    )
    if room is not None and float(room) <= 0:
        st.warning(
            "**No room at this price.** The best a buyer can do with this lot "
            f"is worth {_money(ceiling)} to them and the ground is asking "
            f"{_money(acquisition)} — the price would have to come off "
            f"{_money(abs(float(room)))} before any future clears the discount "
            "rate. The columns below show by how much each one misses."
        )
    elif room is not None:
        st.success(
            f"**A deal clears here.** *{_FUTURE_TITLES.get(best, best)}* is the "
            f"play, and it carries {_money(room)} of room over the asking "
            "price at the solve's discount rate."
        )

    # --- who to call -------------------------------------------------------
    #
    # The investment thesis is the axis the site thesis is not: what would be
    # built rather than why the parcel is available. It is the line that turns
    # a shortlist into a call list, so it sits above the arithmetic rather
    # than inside the programme.
    thesis = str(site.get("investment_thesis") or "none")
    if thesis != "none":
        title, meaning, buyers = _INVESTMENT_THESIS_TITLES.get(
            thesis, (thesis.replace("_", " ").title(), "", "")
        )
        line = f"**Would build {title.lower()}**"
        if meaning:
            line += f" — {meaning}"
        st.markdown(line + ".")
        bits = []
        if buyers:
            bits.append(f"The buyers are {buyers}")
        rank, count = site.get("thesis_rank"), site.get("num_ranked_in_thesis")
        if rank is not None:
            bits.append(
                f"This lot ranks {int(rank)} of {int(count or 0)} {title.lower()} "
                "sites in the borough on yield on cost"
                + (", and is on its shortlist" if site.get("is_top_opportunity") else "")
            )
        if bits:
            st.caption(". ".join(bits) + ".")
    elif site.get("hbu_status") == "solved":
        st.caption(
            "No investment thesis: the solved programme carries too little "
            "floor of any one class to file the lot under one."
        )

    # --- the three futures, priced after the purchase ----------------------
    st.divider()
    st.markdown("**What a buyer could do with it**")
    if best == "none":
        st.caption(
            "At this price none of the three clears the discount rate. Each "
            "column still says by how much it misses and the most a buyer "
            "could pay for it."
        )

    cols = st.columns(3)
    enhance_solved = bool(site.get("enhance_solved"))
    rebuild_months = _months(program.get("program_assumptions"), "construction_months")
    rebuild_lease = _months(program.get("program_assumptions"), "lease_up_months")
    enhance_months = _months(
        site.get("enhance_assumptions"), "enhance_construction_months"
    )
    enhance_lease = _months(site.get("enhance_assumptions"), "enhance_lease_up_months")

    def timeline(months, lease) -> str:
        if months is None:
            return "—"
        text = f"{int(months)} months to build"
        if lease:
            text += f", {int(lease)} to fill"
        return text

    futures = [
        {
            "key": "hold",
            "npv": site.get("buyer_npv_hold_cad"),
            "irr": site.get("buyer_irr_hold_pct"),
            "owner_irr": None,
            "yoc": site.get("buyer_yoc_hold_pct"),
            "yield": site.get("buyer_yield_hold_pct"),
            "residual": hold_value,
            "cost": 0.0,
            "timeline": "today",
            "after": (
                f"{int(site.get('existing_num_dwellings') or 0)} dwellings, "
                f"{_money(site.get('existing_annual_stabilised_noi_cad'))} NOI a year"
            ),
            "available": True,
            "why_not": None,
            "parking_waived": False,
            "waived_stalls": None,
        },
        {
            "key": "enhance",
            "npv": site.get("buyer_npv_enhance_cad"),
            "irr": site.get("buyer_irr_enhance_pct"),
            "owner_irr": site.get("owner_irr_enhance_pct"),
            "yoc": site.get("buyer_yoc_enhance_pct"),
            "yield": site.get("buyer_yield_enhance_pct"),
            "residual": site.get("residual_price_enhance_cad"),
            "cost": site.get("enhance_capital_cost_cad"),
            "timeline": timeline(enhance_months, enhance_lease),
            "after": (
                f"{int(site.get('enhance_num_dwellings') or 0)} dwellings, "
                f"+{float(site.get('enhance_added_floor_area_m2') or 0):,.0f} m² on "
                f"{int(site.get('enhance_added_storeys') or 0)} added storey"
                f"{'s' if int(site.get('enhance_added_storeys') or 0) != 1 else ''}"
                if enhance_solved else "—"
            ),
            "available": enhance_solved and site.get("buyer_npv_enhance_cad") is not None,
            "parking_waived": bool(site.get("enhance_parking_waived")),
            "waived_stalls": site.get("enhance_waived_stalls"),
            "why_not": {
                "no_building": "nothing stands on the lot, or the roll states no storey count",
                "not_underbuilt": "the envelope holds no more than what stands",
                "no_program": "no rebuild was solved, so there is nothing to grow toward",
                "no_envelope": "the governing zone's columns could not be rebuilt",
                "INFEASIBLE": "the standing building does not fit today's grid, so nothing can be added under it",
                "ERROR": "the enhancement could not be modelled",
            }.get(str(site.get("enhance_status")), site.get("enhance_status")),
        },
        {
            "key": "rebuild",
            "npv": site.get("buyer_npv_rebuild_cad"),
            "irr": site.get("buyer_irr_rebuild_pct"),
            "owner_irr": site.get("owner_irr_rebuild_pct"),
            "yoc": site.get("buyer_yoc_rebuild_pct"),
            "yield": site.get("buyer_yield_rebuild_pct"),
            "residual": site.get("residual_price_rebuild_cad"),
            "cost": (
                float(site.get("hbu_total_capital_cost_cad") or 0)
                + float(site.get("site_costs_cad") or 0)
            ),
            "timeline": timeline(rebuild_months, rebuild_lease),
            "after": (
                f"{int(site.get('hbu_num_dwellings') or 0)} dwellings, "
                f"{float(site.get('hbu_floor_area_m2') or 0):,.0f} m², "
                f"{_money(site.get('hbu_annual_stabilised_noi_cad'))} NOI a year"
            ),
            "available": site.get("hbu_status") == "solved"
            and site.get("buyer_npv_rebuild_cad") is not None,
            # Off the programme row first - it is the row the flag is written
            # on - and off the shortlist's copy where that read is missing.
            "parking_waived": bool(
                program.get("parking_waived")
                if program.get("parking_waived") is not None
                else site.get("hbu_parking_waived")
            ),
            "waived_stalls": (
                program.get("waived_stalls")
                if program.get("waived_stalls") is not None
                else site.get("hbu_waived_stalls")
            ),
            "why_not": _hbu_status_reason(site.get("hbu_status")),
        },
    ]
    for column, future in zip(cols, futures, strict=True):
        with column:
            title = _FUTURE_TITLES[future["key"]]
            if future["key"] == best:
                st.markdown(f"#### ✅ {title}")
            else:
                st.markdown(f"#### {title}")
            if not future["available"]:
                st.markdown("—")
                st.caption(f"Not priced: {future['why_not']}.")
                continue
            st.metric(
                "NPV after purchase",
                _money(future["npv"], signed=True),
                help="The future's value to its holder, less the price of the ground above.",
            )
            st.metric(
                "IRR",
                _pct(future["irr"]),
                help=(
                    "Unlevered, annual, on the whole stream: the price at day "
                    "one, the budget with soft costs, contingency and builder's "
                    "risk over the build, the income filling over the "
                    "absorption-driven lease-up, the hold, the sale less "
                    "selling costs."
                ),
            )
            st.metric(
                "Yield on all-in cost",
                _pct(future["yoc"]) if future["yoc"] is not None else (
                    f"{float(future['yield']):,.1f}%" if future["yield"] is not None else "—"
                ),
                help=(
                    "Stabilised NOI once the future is reached, over the price "
                    "of the lot plus the all-in budget to reach it - hard cost "
                    "with soft costs, contingency and builder's risk on it."
                ),
            )
            st.metric(
                "Most you could pay",
                _money(future["residual"]),
                help="The price at which this future's NPV is zero at the discount rate.",
            )
            st.markdown(f"**Build cost** {_money(future['cost'])}")
            st.markdown(
                "**All in** "
                + _money(float(future["cost"] or 0) + float(acquisition))
            )
            st.markdown(f"**Time** {future['timeline']}")
            st.caption(future["after"])
            if future.get("parking_waived"):
                st.warning(
                    "**Parking waived.** Nothing pencils here with the stalls "
                    "it owes, so this was solved without the obligation: it is "
                    f"short {int(future.get('waived_stalls') or 0)} stall(s) of "
                    "what the assumed ratios ask, and stands on a variance for "
                    "those. Whatever fits and pays is still in its budget."
                )
            if future["owner_irr"] is not None:
                st.caption(
                    f"To the owner, {_pct(future['owner_irr'])} on the increment: "
                    "no price paid, the standing income given up during the works."
                )

    # --- what is behind the columns ------------------------------------------
    st.divider()
    notes = []
    if enhance_solved:
        notes.append(
            f"**Enhance** keeps the {int(site.get('existing_num_storeys') or 0)}-storey "
            f"building and adds {int(site.get('enhance_added_storeys') or 0)} storey on its "
            f"plate and an annex to {float(site.get('enhance_footprint_m2') or 0):,.0f} m² of "
            f"ground, {float(site.get('enhance_added_floor_area_m2') or 0):,.0f} m² of new "
            f"floor in all, costed at the addition premium; "
            f"{_money(site.get('enhance_disruption_cad'))} of the standing income is lost "
            "during the works. Nothing is dug under it and the new stalls are on the yard "
            f"({int(site.get('enhance_surface_stalls') or 0)})."
            + (
                f" The yard could not take "
                f"{int(site.get('enhance_waived_stalls') or 0)} of the stalls the "
                "building owes, so the addition was solved with its parking waived."
                if site.get("enhance_parking_waived") else ""
            )
        )
    rebuild_bits = [f"construction {_money(site.get('hbu_total_capital_cost_cad'))}"]
    for label, key in (
        ("demolition", "demolition_cost_cad"),
        ("site assessment", "site_assessment_cost_cad"),
        ("remediation", "remediation_cost_cad"),
    ):
        if float(site.get(key) or 0):
            rebuild_bits.append(f"{label} {_money(site.get(key))}")
    notes.append(
        "**Tear down and rebuild** is the programme below, its income starting "
        "only after the build and the lease-up, and it pays "
        + ", ".join(rebuild_bits)
        + "."
    )
    notes.append(
        "**Keep** is the standing building's stabilised income discounted over "
        "the same hold and sold at the same cap, starting today — the future a "
        "buyer gets for the price alone."
    )
    notes.append(
        f"Every column pays {_money(acquisition)} for the ground before any of "
        "the above. That is what separates these numbers from the programme "
        "below, which excludes land: the solver holds the envelope constant "
        "whoever owns it, and a purchase does not."
    )
    for note in notes:
        st.markdown(f"- {note}")

    if site.get("is_heritage_sector") or site.get("demolition_review_required"):
        st.caption(
            "The rebuild column is arithmetic, not a permit: this lot carries "
            "a heritage or demolition-review flag, set out under **Heritage "
            "and review** below."
        )
    st.caption(
        "Every figure is unlevered, at the discount rate, hold and terminal cap "
        "the solve ran with; the IRR and the yield on all-in cost carry soft "
        "costs, contingency, builder's risk and an absorption-driven lease-up "
        "on top of the hard cost. None is an appraisal or a listing price."
    )


#: What each site thesis is called on the pane, and what it means in a line.
_SITE_THESIS_TITLES = {
    "brownfield": (
        "Brownfield",
        "a contamination-risk use stands on it, so the ground is characterised "
        "and cleaned before the change of use",
    ),
    "teardown": (
        "Teardown",
        "an obsolete building fills little of an envelope that allows storeys "
        "above it, so the play is to demolish and rebuild",
    ),
    "infill": ("Infill", "nothing stands on it"),
    "improvement": (
        "Improvement",
        "the building stays and gains a storey on its footprint or a rear "
        "annex on the ground the proposal would cover",
    ),
}


def _pct(value, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value):,.{digits}f}%"


def _render_returns(site: dict, thesis: str) -> None:
    """Yield on all-in cost and IRR for the thesis's own future, both chairs.

    The solve's present values are one number each; these are the proforma
    around them - soft costs, contingency, builder's risk, the site's costs,
    the acquisition, a lease-up the absorption rate sets - and the two tests
    a screen holds a candidate to: the yield against the area's cap rate plus
    a development spread, the IRR against a hurdle. Both the buyer's IRR
    (the whole building, after the price) and the owner's (the increment
    over the building they have) are shown, because they answer different
    questions and are rarely close.
    """
    irr = site.get("site_irr_pct")
    yoc = site.get("site_all_in_yield_on_cost_pct")
    owner_irr = site.get("owner_site_irr_pct")
    if irr is None and yoc is None and owner_irr is None:
        return
    st.markdown("**Returns**")
    cap = site.get("market_cap_rate_pct")
    spread = site.get("site_yoc_spread_bps")
    cols = st.columns(4)
    cols[0].metric(
        "Yield on all-in cost",
        _pct(yoc),
        delta=(
            f"{'+' if float(spread) >= 0 else '−'}{abs(float(spread)):,.0f} bps vs "
            f"{_pct(cap)} cap"
            if spread is not None and cap is not None else None
        ),
        help=(
            "Stabilised NOI over everything a buyer pays: the price, the "
            "site's costs, the hard cost with soft costs, contingency and "
            "builder's risk on it. Held against the area's cap rate: a "
            "development has to earn a spread over buying the same income "
            "already built."
        ),
    )
    cols[1].metric(
        "IRR, buyer",
        _pct(irr),
        help=(
            "Unlevered, annual: the price and the site's costs at day one, "
            "the budget over the build, the income filling over the lease-up, "
            "the hold, the sale at the terminal cap less selling costs."
        ),
    )
    cols[2].metric(
        "IRR, owner",
        _pct(owner_irr),
        help=(
            "The same stream on the increment: no price paid, the standing "
            "income given up during and after the works, the difference in "
            "value at the sale."
        ),
    )
    verdicts = []
    if site.get("clears_cap_rate") is not None:
        verdicts.append(("cap rate", bool(site.get("clears_cap_rate"))))
    if site.get("clears_hurdle") is not None:
        verdicts.append(("IRR hurdle", bool(site.get("clears_hurdle"))))
    cols[3].metric(
        "Screen",
        "✔ good candidate" if site.get("is_good_candidate") else "not yet",
        help=", ".join(
            f"{'clears' if ok else 'misses'} the {name}" for name, ok in verdicts
        ) or None,
    )
    budget_key = "enhance" if thesis == "improvement" else "rebuild"
    rows = []
    if budget_key == "rebuild":
        for label, key in (
            ("Price paid", "acquisition_cost_cad"),
            ("Site costs", "site_costs_cad"),
            ("Hard cost", "hbu_total_capital_cost_cad"),
            ("Soft costs", "rebuild_soft_cost_cad"),
            ("Contingency", "rebuild_contingency_cad"),
            ("Builder's risk insurance", "rebuild_builders_risk_cad"),
            ("Total development cost", "rebuild_total_development_cost_cad"),
        ):
            if site.get(key) is not None:
                rows.append({"Line": label, "Amount": _money(site.get(key))})
        months = site.get("rebuild_lease_up_months")
    else:
        for label, key in (
            ("Price paid", "acquisition_cost_cad"),
            ("Addition, hard cost", "enhance_capital_cost_cad"),
            ("Addition, all in", "enhance_budget_cad"),
            ("Total development cost", "enhance_total_development_cost_cad"),
        ):
            if site.get(key) is not None:
                rows.append({"Line": label, "Amount": _money(site.get(key))})
        months = site.get("enhance_lease_up_months")
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    note = []
    if months is not None:
        note.append(f"{int(months)} months of lease-up at the assumed absorption")
    if site.get("comparable_cap_rate_pct") is not None:
        note.append(
            f"the roll implies {_pct(site.get('comparable_cap_rate_pct'))} on the "
            "lots around this one, which is an assessed and not a market cap"
        )
    if note:
        st.caption("; ".join(note).capitalize() + ".")


def _render_site_thesis(lot: dict, *, caps) -> None:
    """Why this parcel is acquirable - the second axis of the shortlist table.

    `gold.lot_investment_opportunities` files every lot under a *site thesis*
    beside the investment thesis: brownfield, teardown, infill, improvement or
    none, each a predicate over the roll's year and storeys, the solver's
    storeys and footprint, the use code and the grid's own *Patrimoine* rows,
    and each carrying its own cost - demolition, characterisation and
    remediation, or the addition's premium - into its own yield. This block
    says which held on this lot, what it costs to clear, and what the heritage
    rows say. Read by its own query, so a database without the table loses
    this block and nothing else on the pane.
    """
    if not caps.investment_opportunities or lot.get("lot_uid") is None:
        return
    site = _lot_opportunity(
        int(lot["lot_uid"]), lot.get("scrape_date"), lot.get("neighborhood"),
        _selected_zone(lot),
    )
    st.divider()
    st.markdown("**Why this site**")
    if not site:
        st.caption(
            "No row in the shortlist table for this lot in this snapshot - run "
            "the `lot_investment_opportunities` asset for this partition."
        )
        return

    thesis = str(site.get("site_thesis") or "none")
    title, meaning = _SITE_THESIS_TITLES.get(thesis, (thesis, ""))
    rank = site.get("site_thesis_rank")
    count = site.get("num_ranked_in_site_thesis")

    if thesis == "none":
        st.markdown("No site thesis holds on this lot.")
        st.caption(
            "Neither an obsolete building under an unused envelope, nor a "
            "contamination-risk use, nor an empty lot, nor room for a storey "
            "or an annex - at the thresholds this snapshot was screened with, "
            "under *assumptions* below."
        )
    else:
        st.markdown(f"### {title} - {meaning}")
        if rank is not None:
            st.caption(
                f"Rank {int(rank)} of {int(count or 0)} {thesis} sites in the "
                "borough, on this thesis's own yield on cost"
                + (" · on its shortlist" if site.get("is_top_site_opportunity") else "")
                + "."
            )
        else:
            st.caption(
                "Filed under this thesis and **unranked**: at the solve's "
                "assumptions the play does not pay - rebuilding does not beat "
                "holding, or the addition earns nothing - so the site "
                "condition holds and the arithmetic does not."
            )

    # --- the conditions that held ----------------------------------------
    held = []
    year = site.get("existing_year_built")
    storeys = site.get("existing_num_storeys")
    floors = site.get("hbu_floors")
    share = site.get("built_share")
    if site.get("is_brownfield_use"):
        held.append(
            f"the dominant use - {site.get('existing_dominant_use_description') or '?'} "
            f"(CUBF {site.get('existing_dominant_use_code') or '?'}) - is a "
            "contamination-risk activity"
        )
    if year is not None:
        line = f"built {int(year)}"
        if storeys is not None and floors is not None:
            line += (
                f", {int(storeys)} storey{'s' if int(storeys) != 1 else ''} where "
                f"the governing grid takes {int(floors)}"
            )
        if share is not None:
            line += f", {float(share) * 100:,.0f}% of the proposed floor standing"
        held.append(line)
    elif floors is not None and not float(site.get("existing_floor_area_m2") or 0):
        held.append(f"nothing assessed on it, under a grid taking {int(floors)} storeys")
    if held:
        st.markdown("\n".join(f"- {line}" for line in held))
    also = [
        name
        for name, column in (
            ("brownfield", "is_brownfield_site"),
            ("teardown", "is_teardown_site"),
            ("infill", "is_infill_site"),
            ("improvement", "is_improvement_site"),
        )
        if site.get(column) and name != thesis
    ]
    if also:
        st.caption(
            "Also holds: " + ", ".join(also) + ". The theses resolve in a fixed "
            "order - brownfield, teardown, infill, improvement - and the first "
            "names the lot; the others are kept as flags."
        )

    # --- what it costs, on this thesis's own denominator -------------------
    site_yield = site.get("site_yield_on_cost_pct")
    if thesis == "improvement":
        cols = st.columns(4)
        cols[0].metric(
            "Added floor",
            f"{float(site.get('improvement_floor_m2') or 0):,.0f} m²",
            help=(
                f"{int(site.get('improvement_added_storeys') or 0)} storey on the "
                f"standing footprint of "
                f"{float(site.get('existing_footprint_m2') or 0):,.0f} m², plus "
                "an annex on the ground the proposal covers and the building "
                "does not, capped at the floor gap."
            ),
        )
        cols[1].metric("Cost", f"${float(site.get('improvement_cost_cad') or 0):,.0f}")
        cols[2].metric(
            "NOI a year", f"${float(site.get('improvement_noi_cad') or 0):,.0f}"
        )
        cols[3].metric(
            "Yield on cost",
            f"{float(site_yield):,.1f}%" if site_yield is not None else "—",
        )
        st.caption(
            "The addition earns the proposal's NOI per square metre and costs "
            "its capital cost per square metre times the addition premium under "
            "*assumptions* - the shoring, the tie-ins and the occupied site."
        )
    elif thesis != "none":
        cols = st.columns(3)
        cols[0].metric(
            "Yield on cost, all in",
            f"{float(site_yield):,.1f}%" if site_yield is not None else "—",
            help=(
                "The proposal's stabilised NOI over construction, the land at "
                "its assessed value, and the site's own costs below."
            ),
        )
        first_axis = site.get("yield_on_cost_pct")
        if first_axis is not None:
            cols[1].metric(
                "Before site costs",
                f"{float(first_axis):,.1f}%",
                help="The same yield with only construction and land in the denominator.",
            )
        cols[2].metric(
            "All-in cost",
            f"${float(site.get('site_total_project_cost_cad') or 0):,.0f}",
        )
        costs = [
            {"Item": label, "Cost": f"${float(site.get(key) or 0):,.0f}"}
            for label, key in (
                ("Construction", "hbu_total_capital_cost_cad"),
                ("Land, at its assessed value", "existing_total_assessed_value"),
                ("Demolition of what stands", "demolition_cost_cad"),
                ("Environmental site assessment", "site_assessment_cost_cad"),
                ("Remediation", "remediation_cost_cad"),
            )
            if site.get(key) is not None and float(site.get(key) or 0)
        ]
        if costs:
            st.dataframe(costs, width="stretch", hide_index=True)
        gain = site.get("redevelopment_npv_gain_cad")
        if gain is not None:
            st.caption(
                f"Rebuilding {'beats' if float(gain) > 0 else 'trails'} holding by "
                f"${abs(float(gain)):,.0f}, discounted - the verdict the rank "
                "breaks ties on."
            )

    # --- the returns ------------------------------------------------------
    _render_returns(site, thesis)

    # --- heritage ---------------------------------------------------------
    flags = []
    if site.get("is_heritage_sector"):
        flags.append(
            f"The governing zone ({site.get('grid_zone') or '?'}) prints "
            "**Secteur d'intérêt patrimonial: "
            f"{site.get('heritage_sector')}**. Demolition goes to the "
            "borough's demolition committee with a heritage study, so the "
            "lot is kept out of the teardown and brownfield theses."
        )
    if site.get("has_piia_review"):
        flags.append(
            f"The zone is in **PIIA sector {site.get('piia_sector')}**: new "
            "construction, additions visible from the street and rooftop "
            "constructions are subject to a discretionary architectural "
            "review. A replacement building is exactly what that review "
            "judges, so the lot is kept out of the teardown and brownfield "
            "theses - an addition to what stands is still on the table."
        )
    if site.get("demolition_review_required") and not site.get("is_heritage_sector"):
        flags.append(
            "The building **predates 1940**, the year the *Loi sur le "
            "patrimoine culturel* draws for the heritage inventory a demolition "
            "by-law must cover. Whether this building is on the borough's "
            "inventory is not in this database; treat demolition as reviewed."
        )
    if flags:
        st.markdown("**Heritage and review**")
        for flag in flags:
            st.markdown(f"- {flag}")
    elif thesis != "none":
        st.caption(
            "No heritage sector, no PIIA sector and no pre-1940 building on "
            "this lot's row. Every demolition in this borough still goes "
            "through the demolition by-law."
        )

    assumptions = site.get("screen_assumptions")
    if isinstance(assumptions, str):
        try:
            assumptions = json.loads(assumptions)
        except ValueError:
            assumptions = None
    if isinstance(assumptions, dict) and assumptions:
        with st.expander("Every threshold and rate this was screened with"):
            st.caption(
                "Carried on the row, so a shortlist read a month later can be "
                "read back against the rules that produced it. The rates are "
                "stated per square metre, not surveyed per lot."
            )
            st.dataframe(
                [
                    {"Assumption": str(key).replace("_", " "), "Value": str(value)}
                    for key, value in sorted(assumptions.items())
                ],
                width="stretch",
                hide_index=True,
            )


def _render_hbu_program(lot: dict, *, caps) -> None:
    """The whole proposal for one lot, from `gold.lot_highest_best_use`.

    Everything on this pane is one row, so nothing on it can disagree with
    anything else on it. Where the *existing* building appears — the dwelling
    count today, the verdict against holding — it comes from the gap table
    through the same cached read the Lot pane makes, and is labelled as today's
    rather than as the proposal's. Where the *land* appears — what the ground
    costs, what clearing it costs, what the building is worth once both are
    paid — it comes from the shortlist row through the read the Deal block
    above has already made, and is never worked out here: the futures were
    priced on one footing over there, and a rebuild NPV computed twice in two
    places is a pane that can contradict itself.
    """
    program = _lot_program(
        int(lot["lot_uid"]), lot.get("scrape_date"), lot.get("neighborhood"),
        _selected_zone(lot),
    )
    if not program:
        st.info(
            "No highest-and-best-use row for this lot in this snapshot. The "
            "solver writes one for every lot a zone reaches, so this is either "
            "a lot the zoning layer does not cover or a partition the "
            "`lot_highest_best_use` asset has not run over."
        )
        return

    # The gap row, for the one thing the programme table cannot say: what
    # stands there now. Optional — a database with the programme and not the
    # subtraction is a real state between two pipeline runs — and the pane
    # loses the comparison rather than the proposal when it is missing. The
    # same cached call the Lot pane makes, so this costs no query.
    existing = (
        _lot_capacity(
            int(lot["lot_uid"]), lot.get("scrape_date"), lot.get("neighborhood"),
            _selected_zone(lot),
        )
        if caps.redevelopment_gap and lot.get("lot_uid") is not None
        else None
    )
    # The shortlist row, for the one thing the programme table cannot say
    # either: what the ground under it costs. Same cache as the Deal block
    # above, so on the merged pane this is free; optional for the same reason
    # `existing` is, and the money block below drops the land and keeps the
    # construction when it is absent.
    site = (
        _lot_opportunity(
            int(lot["lot_uid"]), lot.get("scrape_date"), lot.get("neighborhood"),
            _selected_zone(lot),
        )
        if caps.investment_opportunities and lot.get("lot_uid") is not None
        else None
    )

    # The one-word verdict belongs in the heading only where there is a
    # building to name. An unsolved row can still carry a dominant use — a
    # rename, a partial re-materialization — and putting it up there would
    # announce a programme the next line goes on to say does not exist.
    solved = program.get("hbu_status") == "solved"
    use = str(program.get("hbu_dominant_use") or "").replace("_", " ")
    heading = "### The proposed building"
    if solved and use and use != "none":
        heading += f" — {use}"
    st.markdown(heading)
    subtitle = [f"Lot {program.get('lot_number') or lot.get('lot_number') or '—'}"]
    if program.get("grid_zone"):
        subtitle.append(f"zone {program['grid_zone']}")
    subtitle.append(f"snapshot {program.get('scrape_date')}")
    st.caption(" · ".join(str(part) for part in subtitle))
    # What the proposal replaces, in one line, before thirty of what it is.
    # The class and then the roll's words for it, because the class is a
    # filing and the words are the fact.
    today_use = (existing or {}).get("existing_dominant_income_class")
    if solved and today_use:
        today = str(today_use)
        if (existing or {}).get("existing_dominant_use_description"):
            today += f" ({existing['existing_dominant_use_description']})"
        if use and use != "none" and today_use != "none":
            change = "keeps the use" if today_use == use else "changes the use"
            st.caption(f"Today {today} — the proposal {change}.")
        else:
            st.caption(f"Today {today}.")

    # A road parcel has a programme row like any other lot, and the Lot pane
    # refuses to report economics on one because every figure would be
    # arithmetic on an artefact of two layers meeting at the curb. The same
    # refusal here, for the same reason, ahead of every figure on the pane.
    if (existing or {}).get("hbu_status") == "road_parcel":
        st.markdown("**Street** — not a development site")
        st.caption(
            "This parcel is the public way itself. Nothing may be built on "
            "it, so no programme is reported here whatever the grid over the "
            "block permits."
        )
        return

    if not solved:
        st.markdown("**No programme was solved for this lot.**")
        st.caption(_hbu_status_reason(program.get("hbu_status")))
        # On an unsolved lot the candidate counts and the binding are most of
        # what there is to say: whether the solver had one column to choose
        # from or none, and which pair of printed rows contradicted each other.
        _render_program_binding(program)
        _render_program_choice(program)
        if program.get("solve_error"):
            st.caption(f"Solver error: `{program['solve_error']}`")
        return

    # Said before a single figure, because every figure below is conditional
    # on it: a programme that only exists with its stalls waived is a building
    # on a parking variance, and the stall table further down would otherwise
    # read "no parking" as a choice the solver made.
    if program.get("parking_waived"):
        st.warning(
            "**Solved with the parking waived.** With the stalls it owes at "
            "the assumed ratios, nothing pencils on this parcel — either "
            "nothing can take them, or nothing they serve can pay for them. "
            "The solver was asked again without the obligation, and this is "
            f"that answer: short {int(program.get('waived_stalls') or 0)} "
            "stall(s) of what is owed, with whatever fits and pays still "
            "built. Everything on this pane assumes that variance."
        )

    # --- the shape --------------------------------------------------------
    cols = st.columns(4)
    cols[0].metric("Storeys", f"{int(program.get('floors') or 0)}")
    cols[1].metric("Height", f"{float(program.get('height_m') or 0):,.1f} m")
    cols[2].metric("Footprint", f"{float(program.get('footprint_m2') or 0):,.0f} m²")
    cols[3].metric(
        "Gross floor area",
        f"{float(program.get('gross_floor_area_m2') or 0):,.0f} m²",
        help=(
            "Footprint × the storeys above grade — the *superficie de "
            "plancher* Densité is tested against. Dug levels are outside it."
        ),
    )

    footprint = float(program.get("footprint_m2") or 0)
    lot_area = float(program.get("lot_area_m2") or lot.get("area_m2") or 0)
    buildable = program.get("buildable_area_m2")
    plate = []
    if lot_area:
        plate.append(f"{footprint / lot_area * 100:,.0f}% of the {lot_area:,.0f} m² lot")
    if buildable is not None and float(buildable):
        plate.append(
            f"{footprint / float(buildable) * 100:,.0f}% of the "
            f"{float(buildable):,.0f} m² the setbacks leave"
        )
    if program.get("primary_frontage_m"):
        plate.append(f"{float(program['primary_frontage_m']):,.1f} m of frontage")
    if plate:
        st.caption("Plate: " + " · ".join(plate) + ".")
    st.caption(
        "One plate, repeated: the model builds a single footprint and stacks "
        "identical storeys on it. *Footprint* is therefore the ground this "
        "proposal covers and *gross floor area* is that ground times the "
        "storeys above grade — the same two measures the **Lot** pane reports "
        "for what stands today."
    )

    # --- the massing, as it was drawn ------------------------------------
    #
    # The rectangle is the only place the proposal has a width, a depth and a
    # bearing: everything above is an area, and an area is not a shape. The fit
    # belongs on the same block because it is that distinction read backwards —
    # a footprint capped on the lesser of two *areas* may have no shape this
    # parcel can take, and then every floor area above is overstated.
    if caps.massing and program.get("massing_status"):
        st.markdown("**As drawn on the parcel**")
        width, depth = program.get("massing_width_m"), program.get("massing_depth_m")
        drawn = []
        if width and depth:
            drawn.append(f"{float(width):,.1f} m × {float(depth):,.1f} m")
        if program.get("rotation_deg") is not None:
            drawn.append(f"long axis {float(program['rotation_deg']):,.0f}°")
        if program.get("aspect_ratio") is not None:
            drawn.append(f"aspect ratio {float(program['aspect_ratio']):,.2f}")
        if drawn:
            st.markdown(" · ".join(drawn))
        if program["massing_status"] == "shrunk":
            fit = program.get("footprint_fit_pct")
            st.warning(
                f"The solved footprint had to be **shrunk** to fit this "
                f"parcel's shape — "
                f"{float(program.get('placed_footprint_m2') or 0):,.0f} m² "
                f"drawn of {footprint:,.0f} m² costed"
                + (f", {float(fit):,.0f}% of it" if fit is not None else "")
                + f". Every floor area on this pane is overstated by that "
                f"much: the drawn building carries "
                f"{float(program.get('placed_gross_floor_area_m2') or 0):,.0f} "
                f"m² against the "
                f"{float(program.get('gross_floor_area_m2') or 0):,.0f} m² "
                f"costed above."
            )
        else:
            st.caption(
                "The costed footprint fits the setback envelope as a *shape* "
                "and not only as an area, so the floor areas above stand. "
                "This is the rectangle the **Proposed massing** layer draws."
            )

        # The same distinction applied to the yard. It is here rather than in
        # the parking block below because it is a fact about *shape* and the
        # block below is about money, and because a reader who has just been
        # told the building fits will want to know whether its cars do.
        parking_status = program.get("parking_status")
        if parking_status in ("fitted", "shrunk", "no_fit"):
            reserved = float(program.get("surface_parking_area_m2") or 0)
            placed_area = float(program.get("placed_surface_parking_m2") or 0)
            if parking_status == "fitted":
                st.caption(
                    f"Its **surface parking** fits too: "
                    f"{placed_area:,.0f} m² of yard, drawn on the parcel "
                    f"outside the building by the **Surface parking** layer. "
                    f"A stall is not part of the massing above - no floor "
                    f"area, no storey, no height - so it is a shape of its own."
                )
            else:
                fit = program.get("surface_parking_fit_pct")
                stalls = program.get("placed_surface_stalls")
                st.warning(
                    f"The programme parks "
                    f"{int(program.get('surface_stalls') or 0)} car(s) on the "
                    f"yard and the yard cannot take them all: "
                    f"{placed_area:,.0f} m² of asphalt fits of "
                    f"{reserved:,.0f} m² reserved"
                    + (f", {float(fit):,.0f}% of it" if fit is not None else "")
                    + (
                        f" - room for {int(stalls)} stall(s), not "
                        f"{int(program.get('surface_stalls') or 0)}"
                        if stalls is not None
                        else ""
                    )
                    + ". The parking cost above is understated by whatever "
                    "those stalls would cost in structure instead."
                )

    # --- the stack --------------------------------------------------------
    stack = program.get("floor_stack")
    if stack:
        st.divider()
        st.markdown("**What stands on each storey**")
        _render_program_stack(stack)
    elif int(program.get("floors") or 0):
        # A partition solved before `floor_stack` existed. The storey counts by
        # use are still on the row, so the pane says the smaller version of the
        # same thing rather than dropping the section.
        st.divider()
        st.markdown("**Storeys by use**")
        counted = [
            (label, int(program.get(key) or 0))
            for label, key in (
                ("Housing", "residential_floors"),
                ("Commerce", "commercial_floors"),
                ("Industry", "industrial_floors"),
                ("Parking above grade", "above_grade_parking_floors"),
                ("Dug levels", "underground_levels"),
            )
        ]
        st.markdown(
            " · ".join(f"{label} {value}" for label, value in counted if value) or "—"
        )
        st.caption(
            "This snapshot was solved before the storey stack was published, "
            "so which use sits on which level is not recorded — only how many "
            "of each there are."
        )

    # --- housing ----------------------------------------------------------
    st.divider()
    st.markdown("**Housing**")
    proposed_d = program.get("num_dwellings")
    today_d = (existing or {}).get("existing_num_dwellings")
    if not int(proposed_d or 0) and not int(program.get("residential_floors") or 0):
        st.markdown("No dwelling in this programme.")
        if program.get("permits_residential"):
            st.caption(
                "The governing column authorises housing — the solver priced "
                "it, and something else was worth more. Which, and against "
                "what, is under *why not more* below."
            )
        else:
            st.caption("The governing column does not authorise housing.")
    else:
        left, right = st.columns(2)
        left.metric(
            "Dwellings proposed",
            f"{int(proposed_d or 0):,}",
            delta=(
                f"{int(proposed_d or 0) - int(today_d or 0):+,} vs today"
                if today_d is not None else None
            ),
        )
        right.metric(
            "Housing floor area",
            f"{float(program.get('residential_area_m2') or 0):,.0f} m²",
            help=(
                "Footprint × the housing storeys — the plate, not the unit "
                "schedule. The narrower rentable schedule the rents were "
                "taken off is "
                f"{float(program.get('unit_area_m2') or 0):,.0f} m²."
            ),
        )
        units = program.get("units")
        mix = (
            [
                {
                    "Bedrooms": _BEDROOM_LABELS.get(str(key), str(key)),
                    "Dwellings": int(count or 0),
                }
                for key, count in sorted(
                    units.items(),
                    key=lambda item: (
                        list(_BEDROOM_LABELS).index(str(item[0]))
                        if str(item[0]) in _BEDROOM_LABELS else 99
                    ),
                )
                if int(count or 0)
            ]
            if isinstance(units, dict) else []
        )
        if mix:
            st.dataframe(mix, width="stretch", hide_index=True)
            st.caption(
                "CMHC's bedroom classes, chosen for the building as a whole "
                "rather than per storey: the solver picked a mix and not a "
                "plan, and dividing it across the storeys would invent the "
                "part it did not choose."
            )
        unpriced = program.get("unpriced_types") or []
        if unpriced:
            st.caption(
                "CMHC published no rent for "
                + ", ".join(
                    _BEDROOM_LABELS.get(str(k), str(k)).lower() for k in unpriced
                )
                + " in this borough, so the solver would not build "
                + ("them" if len(unpriced) > 1 else "it")
                + " — a fact about the survey rather than about the zone."
            )

    # --- commerce and industry -------------------------------------------
    st.divider()
    st.markdown("**Commerce and industry**")
    classes = [
        {
            "Class": label,
            "Authorised": "yes" if program.get(permits_key) else "no",
            "Storeys": int(program.get(floors_key) or 0),
            # "0 m²" and "not authorised" are different findings, and one
            # column cannot show the same value for both — the distinction the
            # Overview pane draws over a borough, drawn here over one lot.
            "Floor area (m²)": (
                f"{float(program.get(area_key) or 0):,.0f}"
                if float(program.get(area_key) or 0)
                else ("none proposed" if program.get(permits_key) else "—")
            ),
        }
        for label, area_key, floors_key, permits_key in (
            (
                "Commerce", "commercial_area_m2", "commercial_floors",
                "permits_commercial",
            ),
            (
                "Industry", "industrial_area_m2", "industrial_floors",
                "permits_industrial",
            ),
        )
    ]
    st.dataframe(classes, width="stretch", hide_index=True)
    idle = [row["Class"] for row in classes if row["Authorised"] == "yes" and not row["Storeys"]]
    if idle:
        st.caption(
            f"{' and '.join(idle)} {'is' if len(idle) == 1 else 'are'} "
            "authorised on the governing column and this programme proposes "
            "none. At the borough's surveyed rents against the construction "
            "cost per square foot, the storey earns more as something else — "
            "an economics finding rather than a statement about the zoning."
        )

    # --- parking ----------------------------------------------------------
    #
    # Four places a stall can go, and they cost an order of magnitude apart, so
    # the split *is* the finding and the total alone would hide it. Each one
    # also answers to a different norm, which is why the table says what each
    # is rather than only how many: a dug level is outside the *superficie de
    # plancher*, a deck is a storey of it, a garage bay is floor area without
    # being a storey, and a stall on the yard is not in a building at all —
    # which is why the last is absent from the stack above and present here.
    st.divider()
    st.markdown("**Parking**")
    total_stalls = program.get("total_stalls")
    if program.get("parking_waived"):
        short = int(program.get("waived_stalls") or 0)
        st.markdown(
            f"**Parking waived** — short {short} stall(s) of the "
            f"{short + int(total_stalls or 0)} the assumed ratios ask for."
        )
        st.caption(
            "Not a choice the solver made: with the stalls in the model there "
            "was no building here that fit or paid, so it was solved again "
            "with the obligation dropped. Whatever fits and pays is still "
            "built and listed below; the shortfall is what a variance would "
            "have to cover."
        )
    if total_stalls is None:
        st.caption("No stall count on this row.")
    elif not int(total_stalls):
        st.markdown("No parking in this programme.")
    else:
        left, right = st.columns(2)
        left.metric("Stalls", f"{int(total_stalls):,}")
        right.metric(
            "Parking cost", f"${float(program.get('parking_cost_cad') or 0):,.0f}"
        )
        st.dataframe(
            [
                {"Where": where, "Stalls": int(program.get(key) or 0), "What it is": note}
                for where, key, note in (
                    (
                        "Underground",
                        "underground_stalls",
                        f"{int(program.get('underground_levels') or 0)} dug "
                        f"level(s), "
                        f"{float(program.get('underground_area_m2') or 0):,.0f} "
                        "m² — built and paid for, outside the floor area",
                    ),
                    (
                        "Parking deck",
                        "above_grade_stalls",
                        f"{int(program.get('above_grade_parking_floors') or 0)} "
                        "storey(s) of it — a storey and floor area both, so "
                        "it answers to Densité and to En étage",
                    ),
                    (
                        "Garage, ground floor",
                        "garage_stalls",
                        f"{float(program.get('garage_area_m2') or 0):,.0f} m² "
                        "of enclosed bay — floor area without being a storey, "
                        "so Densité counts it and En étage does not",
                    ),
                    (
                        "On the yard",
                        "surface_stalls",
                        "not in a building at all: neither a storey nor floor "
                        "area, and the cheapest stall by a factor of eight",
                    ),
                )
                if int(program.get(key) or 0)
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Stall counts follow the assumed ratios rather than a printed "
            "norm — half a stall per dwelling and a rate per 1 000 sq ft of "
            "non-residential floor, both under *assumptions* below — and a "
            "stall the occupants would rent is built beyond them where it "
            "pays. A parkade stall costs several times a surface one, so "
            "where they go is most of what parking does to the arithmetic — "
            "and the two provisions that are floor area take it from the "
            "dwellings."
        )
        # What the stalls give back. The rent is inside the gross revenue in
        # the money block below; the absorption saving is inside the present
        # value there and nowhere else, being a timing effect and not income.
        rented = program.get("rented_stalls")
        if rented is not None:
            income = float(program.get("annual_parking_gross_revenue_cad") or 0)
            line = (
                f"{int(rented)} of them rented, {_money(income)} a year of parking "
                "rent inside the gross revenue"
            )
            saved = float(program.get("lease_up_months_saved") or 0)
            if saved > 0:
                line += (
                    f"; at {float(program.get('parking_coverage') or 0):.2f} stalls a "
                    f"dwelling the housing leases {saved:.1f} month(s) faster, worth "
                    f"{_money(program.get('absorption_value_cad'))} of present value"
                )
            st.caption(line + ".")

    # --- the money --------------------------------------------------------
    #
    # Two denominators, and the pane shows both because they answer different
    # questions. The solver's own is *land excluded*: it chose this envelope
    # over every other one on a lot whose ground it holds constant, so land
    # would cancel out of the comparison and carrying it would only make the
    # number bigger. A transaction does not hold the ground constant — it buys
    # it — so the second denominator adds what the lot costs, and it is the one
    # the Deal block above works from.
    st.divider()
    st.markdown("**What it costs and what it earns**")
    acquisition = (site or {}).get("acquisition_cost_cad")
    costs = [
        {"Item": label, "Capital cost": f"${float(program[key]):,.0f}"}
        for label, key in (
            ("Construction — housing", "construction_cost_cad"),
            ("Construction — commerce", "commercial_cost_cad"),
            ("Construction — industry", "industrial_cost_cad"),
            ("Parking", "parking_cost_cad"),
        )
        if program.get(key) is not None and float(program[key])
    ]
    total_cost = program.get("total_capital_cost_cad")
    if total_cost is not None:
        costs.append({"Item": "Construction, total", "Capital cost": f"${float(total_cost):,.0f}"})
    # What clearing the ground costs, itemised from the shortlist row rather
    # than the programme: the solver prices a building on an empty site and
    # says nothing about emptying it.
    site_extras = [
        (label, float((site or {}).get(key) or 0))
        for label, key in (
            ("Demolition of what stands", "demolition_cost_cad"),
            ("Environmental site assessment", "site_assessment_cost_cad"),
            ("Remediation", "remediation_cost_cad"),
        )
    ]
    for label, value in site_extras:
        if value:
            costs.append({"Item": label, "Capital cost": f"${value:,.0f}"})
    if acquisition is not None:
        costs.append(
            {"Item": "**The lot itself**", "Capital cost": f"${float(acquisition):,.0f}"}
        )
        all_in = (
            float(total_cost or 0)
            + sum(value for _, value in site_extras)
            + float(acquisition)
        )
        costs.append({"Item": "**All in, with the land**", "Capital cost": f"${all_in:,.0f}"})
    if costs:
        st.dataframe(costs, width="stretch", hide_index=True)
    if acquisition is None:
        st.caption(
            "The assessment roll never reached this lot, so what the ground "
            "costs is not on this row and every total above is construction "
            "alone."
        )

    money = st.columns(3 if acquisition is not None else 2)
    noi = (
        program.get("annual_stabilised_noi_cad")
        if program.get("annual_stabilised_noi_cad") is not None
        else program.get("annual_net_operating_income_cad")
    )
    if noi is not None:
        money[0].metric(
            "Stabilised NOI, a year",
            f"${float(noi):,.0f}",
            help=(
                "Net of the assumed operating-expense ratio and vacancy, on "
                f"${float(program.get('annual_gross_revenue_cad') or 0):,.0f} "
                "of gross revenue."
            ),
        )
    if program.get("npv_cad") is not None:
        money[1].metric(
            "Net profit, land excluded",
            f"${float(program['npv_cad']):,.0f}",
            help=(
                "The finished building discounted over the hold with a "
                "terminal sale, less the capital cost beside it. This is the "
                "number the choice of envelope was made on, and the ground is "
                "out of it on both sides."
            ),
        )
    # Taken off the shortlist row rather than subtracted here: the three
    # futures were priced on one footing over there, and a rebuild NPV worked
    # out twice in two places is a pane that can contradict itself.
    if acquisition is not None and len(money) > 2:
        money[2].metric(
            "Net profit, after buying it",
            _money((site or {}).get("buyer_npv_rebuild_cad"), signed=True),
            help=(
                "The same building, less the price of the ground and the cost "
                "of clearing it. This is the rebuild column of the Deal block "
                "above, repeated here against the programme it prices."
            ),
        )
    if program.get("present_value_cad") is not None:
        line = (
            f"Present value of the building "
            f"${float(program['present_value_cad']):,.0f}, against "
            f"${float(total_cost or 0):,.0f} of capital."
        )
        if acquisition is not None:
            line += (
                f" Buying the ground puts {_money(acquisition)} on the cost "
                "side and nothing on the value side, which is the whole of "
                "the difference between the two profits above."
            )
        st.caption(line)
    gain = (existing or {}).get("redevelopment_npv_gain_cad")
    if gain is not None:
        if float(gain) > 0:
            st.success(
                f"Building this beats holding what stands by "
                f"**${float(gain):,.0f}**, discounted — before the ground is "
                "paid for, and by the same amount after it, since both futures "
                "pay the same price for it."
            )
        else:
            st.info(
                f"Holding what stands beats building this by "
                f"${-float(gain):,.0f}. The envelope has room; the economics "
                f"say keep it."
            )

    # --- why not more, and what it was chosen from ------------------------
    st.divider()
    _render_program_binding(program)
    _render_program_choice(program)

    assumptions = program.get("program_assumptions")
    if assumptions:
        with st.expander("Every assumption this was solved with"):
            st.caption(
                "Carried on the row rather than looked up, so a programme can "
                "always be read back against the building it assumed: a row "
                "written at one set of rates cannot be read against another."
            )
            st.dataframe(
                [
                    {"Assumption": str(key).replace("_", " "), "Value": str(value)}
                    for key, value in sorted(assumptions.items())
                ],
                width="stretch",
                hide_index=True,
            )

    st.caption(
        "A developer's programme rather than a planner's: the most profitable "
        "governing envelope on discounted net profit, at surveyed rents and "
        "stated costs. It is what the grids permit and the assumptions price "
        "— not what is financeable, serviceable or politically available — "
        "and it is a scrape of the by-law rather than the by-law. It is what "
        "a buyer is buying, and not a price: the price is the block above."
    )

# ---------------------------------------------------------------------------
# Mirror this session's map state into the module the tools read
# ---------------------------------------------------------------------------

# The *live* view rather than the anchor: a tool asked "which lots are in
# view" must answer about what the user is looking at. Written again below,
# from the browser's own report, as soon as st_folium hands one back — the
# chat renders after the map does, so a tool run in this same script sees this
# run's viewport rather than the previous one's.
state.set_viewport(
    st.session_state.viewport,
    st.session_state.view_zoom or st.session_state.map_zoom,
    tuple(st.session_state.view_center or st.session_state.map_center),
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

#: Set when this run's command moves the map itself, so the anchor sync below
#: leaves the commanded position alone rather than snapping back to wherever
#: the browser was before the agent spoke.
_commanded_view = False

_command = state.take_map_command()
if _command:
    if _command.get("center"):
        st.session_state.map_center = list(_command["center"])
        _commanded_view = True
    if _command.get("zoom"):
        st.session_state.map_zoom = int(_command["zoom"])
        _commanded_view = True
    if _command.get("fit_bounds"):
        st.session_state.fit_bounds = _command["fit_bounds"]
        _commanded_view = True
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
    # Every label below is `basemap.TILE_LAYER_NAMES`, and not one of them is
    # typed out here.
    #
    # These boxes and Leaflet's own control tick the same layer, and a reader
    # who has both open reads two labels as two things. They had drifted once
    # already — this pane said "Streets" where the map said "Street sides",
    # which is the one layer whose name is doing work: the géobase is doubled,
    # and a reader who is not told that reads the pair of lines as a rendering
    # fault. Naming the layer twice is how the telling gets lost in one of the
    # two places. Derived, it cannot: a rename reaches both, or neither.
    #
    # The tile renderer only, strictly — under the GeoJSON renderer the map's
    # names carry their feature counts, so "Lots (412)" is the same label plus
    # what is in view rather than a different one.
    _names = basemap.TILE_LAYER_NAMES
    st.session_state.layers["lots"] = st.checkbox(
        _names["lots"], value=st.session_state.layers["lots"], disabled=not caps.lots
    )
    # Gated on the cadastre as well as on the footprints, because the layer is
    # the intersection of the two: the footprints clipped to the lots they
    # stand on, so a hovered area is the ground covered on *that* parcel rather
    # than the whole of a terrace BDOI drew as one outline. Without lots there
    # is nothing to clip against and the layer would draw an empty borough.
    _can_draw_buildings = caps.buildings and caps.lots
    st.session_state.layers["buildings"] = st.checkbox(
        _names["buildings"],
        value=st.session_state.layers["buildings"] and _can_draw_buildings,
        disabled=not _can_draw_buildings,
        help=(
            "Footprints clipped to the lots they stand on, so the hovered "
            "area is the ground covered on that lot."
            if _can_draw_buildings
            else f"{queries.SCHEMA}."
            + ("buildings" if not caps.buildings else "lots")
            + " is not in this database yet."
        ),
    )
    st.session_state.layers["zones"] = st.checkbox(
        _names["zones"],
        value=st.session_state.layers["zones"],
        disabled=not caps.features,
    )
    st.session_state.layers["streets"] = st.checkbox(
        _names["streets"],
        value=st.session_state.layers["streets"] and caps.streets,
        disabled=not caps.streets,
        help="Sides of the roadway from the city's double-line street network "
        "(Géobase) — two lines per street, one per curb, which is the grain a "
        "lot's frontage is measured against. Hover one for its name and its "
        "length inside this borough." if caps.streets
        else f"{queries.SILVER_SCHEMA}.neighborhood_streets is not in this "
        "database yet — run the neighborhood_streets asset.",
    )
    st.session_state.layers["capacity"] = st.checkbox(
        _names["capacity"],
        value=st.session_state.layers["capacity"] and caps.redevelopment_gap,
        disabled=not caps.redevelopment_gap,
        help="Shade every lot by how much of its permitted floor area is "
        "actually standing. Blue is emptier; purple exceeds what today's "
        "zoning would allow." if caps.redevelopment_gap
        else f"{queries.GOLD_SCHEMA}.lot_redevelopment_gap is not in this "
        "database yet — run the lot_redevelopment_gap asset.",
    )
    # Two tables rather than one: today's class is on the gap table and the
    # solver's on the HBU one, and the layer draws both sides off one row.
    _land_use_ready = caps.redevelopment_gap and caps.highest_best_use
    st.session_state.layers["land_use"] = st.checkbox(
        _names["land_use"],
        value=bool(st.session_state.layers.get("land_use")) and _land_use_ready,
        disabled=not _land_use_ready,
        help="Colour every lot by what it is used for - residential, "
        "commercial, industrial, mixed or none - on the assessment roll "
        "today, or as the solver proposes. Hover a lot for both sides and "
        "the floor, footprint and dwellings on each."
        if _land_use_ready
        else f"{queries.GOLD_SCHEMA}.lot_redevelopment_gap and "
        f"{queries.GOLD_SCHEMA}.lot_highest_best_use are both needed - run "
        "the lot_highest_best_use and lot_redevelopment_gap assets.",
    )
    if st.session_state.layers["land_use"]:
        # Indented for the reason the opportunity filters are: it changes
        # what the layer above shows rather than adding one.
        _pad, _opt = st.columns([0.08, 0.92])
        with _opt:
            _sides = {"existing": "Today (assessment roll)", "hbu": "Proposed (HBU)"}
            _side = st.radio(
                "Colour by",
                options=list(_sides),
                format_func=_sides.get,
                index=list(_sides).index(
                    st.session_state.get("land_use_side") or "existing"
                ),
                horizontal=True,
                help="The roll's dominant use on each lot, or the use the "
                "solved programme would put there. The hover shows both "
                "whichever is coloured.",
            )
            st.session_state.land_use_side = _side
    st.session_state.layers["massing"] = st.checkbox(
        _names["massing"],
        value=st.session_state.layers["massing"] and caps.massing,
        disabled=not caps.massing,
        help="The highest-and-best-use building of each lot, drawn inside its "
        "setback envelope — one colour for all of them. Whether the solved "
        "footprint had to be shrunk to fit is in the hover, per lot."
        if caps.massing
        else f"{queries.GOLD_SCHEMA}.lot_building_massing is not in this "
        "database yet — run the massing asset.",
    )
    st.session_state.layers["surface_parking"] = st.checkbox(
        _names["surface_parking"],
        value=st.session_state.layers["surface_parking"] and caps.surface_parking,
        disabled=not caps.surface_parking,
        help="Where the proposed building parks on the ground, drawn on the "
        "yard it leaves. A separate shape because a surface stall is not part "
        "of the building - no floor area, no storey, no height - so it is "
        "fitted into the lot rather than into the setback envelope, at least "
        "one stall deep. Whether the yard could take every stall the "
        "programme asked it for is in the hover, per lot."
        if caps.surface_parking
        else f"{queries.GOLD_SCHEMA}.lot_surface_parking is not in this "
        "database yet - run the massing asset, which writes it beside "
        "lot_building_massing.",
    )
    st.session_state.layers["opportunities"] = st.checkbox(
        _names["opportunities"],
        value=(
            st.session_state.layers["opportunities"] and caps.investment_opportunities
        ),
        disabled=not caps.investment_opportunities,
        help=(
            "The lots the dataplatform filed under a site thesis - why the "
            "parcel is acquirable: brownfield, teardown, infill or improvement "
            "- coloured by which. A heavier edge marks each thesis's "
            "shortlist. Draws from zoom "
            f"{basemap.MIN_OPPORTUNITY_ZOOM}."
            if caps.investment_opportunities
            else f"{queries.GOLD_SCHEMA}.lot_investment_opportunities is not in "
            "this database yet - run the lot_investment_opportunities asset."
        ),
    )
    if st.session_state.layers["opportunities"]:
        # Indented for the reason the under-built box below is: these narrow
        # the layer above rather than adding one.
        _pad, _opt = st.columns([0.08, 0.92])
        with _opt:
            _options = ["every thesis", *queries.SITE_THESES]
            _current = st.session_state.opportunity_filters.get("site_thesis")
            _chosen = st.selectbox(
                "Site thesis",
                options=_options,
                index=_options.index(_current) if _current in _options else 0,
                help=(
                    "One thesis, or all four. Brownfield: a contamination-risk "
                    "use to clear. Teardown: an obsolete building under an "
                    "unused envelope. Infill: nothing stands on it. "
                    "Improvement: the building stays and gains a storey or an "
                    "annex."
                ),
            )
            st.session_state.opportunity_filters["site_thesis"] = (
                None if _chosen == "every thesis" else _chosen
            )
            st.session_state.opportunity_filters["top_only"] = st.checkbox(
                "Shortlist only",
                value=bool(st.session_state.opportunity_filters.get("top_only")),
                help=(
                    "Keep the first site_top_n of each thesis - the lots the "
                    "dataplatform marked is_top_site_opportunity."
                ),
            )
            st.session_state.opportunity_filters["good_only"] = st.checkbox(
                "Good candidates only",
                value=bool(st.session_state.opportunity_filters.get("good_only")),
                help=(
                    "Keep the lots whose thesis clears the area's cap rate by "
                    "the development spread and the IRR hurdle from a buyer's "
                    "chair, and pays against holding. Drawn with a green edge."
                ),
            )

    if st.session_state.layers["massing"] or st.session_state.layers["capacity"]:
        # Indented, because it narrows the two layers above rather than adding
        # a third. The empty first column is the indent: Streamlit gives a
        # checkbox no way to inset its own label, and nesting it in an
        # expander would hide a filter that changes what the map draws.
        _pad, _opt = st.columns([0.08, 0.92])
        with _opt:
            st.session_state.only_underbuilt = st.checkbox(
                "Under-built lots only",
                value=st.session_state.get("only_underbuilt", False),
                help="Keep the lots that could hold more floor than the "
                "assessment roll says stands on them today. Applies to both "
                "the Utilisation shading, the proposed massing and its "
                "parking, so the three cannot disagree about which parcels "
                "are in scope.",
            )

    def _swatches(rows):
        for _color, _label in rows:
            st.markdown(
                f'<span style="display:inline-block;width:0.9rem;'
                f'height:0.9rem;background:{_color};border:1px solid #666;'
                f'vertical-align:middle;margin-right:.5rem"></span>'
                f"{_label}",
                unsafe_allow_html=True,
            )

    if st.session_state.layers["capacity"]:
        with st.expander("Legend — utilisation"):
            _swatches(basemap.capacity_legend_rows())
            # Deliberately one legend for two zoom bands: below zoom 15 the
            # shading is a cell rather than a lot, but its number is `used_pct`
            # on the same scale and goes through the same bands, so the
            # swatches above describe both.
            st.caption(
                "Below zoom "
                f"{queries.MVT_DETAIL_ZOOM['capacity']} these shade summary "
                "cells rather than individual lots, on the same scale."
            )

    if st.session_state.layers.get("land_use"):
        _side_word = (
            "the solver's proposal"
            if st.session_state.get("land_use_side") == "hbu"
            else "the assessment roll today"
        )
        with st.expander("Legend — land use"):
            _swatches(basemap.land_use_legend_rows())
            st.caption(
                f"Coloured by {_side_word}; switch sides above. Hover a lot "
                "for both uses and the floor area, footprint and dwellings on "
                f"each. Draws from zoom {basemap.MIN_LAND_USE_ZOOM}."
            )

    if st.session_state.layers["opportunities"]:
        with st.expander("Legend — opportunities"):
            _swatches(basemap.opportunities_legend_rows())
            st.caption(
                "A heavier edge marks each thesis's shortlist. Hover a lot for "
                "its rank, its yield with the site's own costs in, and what "
                "the heritage rows say; the Deal pane explains the lot in full."
            )

    # The remaining ramps, and only for the layers that are both switched on
    # and currently summarised. Shown from the *live* zoom rather than the
    # anchor, which is why it reads `view_zoom` out of session state: the
    # sidebar is built before the map on every run, so this run's number is
    # the one the browser last reported.
    #
    # That last point is also why massing is skipped below rather than left to
    # `aggregate_legend_rows` to refuse. `view_zoom` is one interaction behind,
    # so this test could put the cell legend on screen over a map already
    # showing rectangles — which is how a legend for a ramp came to sit beside
    # a two-colour drawing of the fit. The layer is one flat colour now and
    # has no legend at any zoom.
    _view_zoom = int(
        st.session_state.view_zoom or st.session_state.map_zoom
    )
    for _layer in queries.AGGREGATE_LAYERS:
        if _layer in ("capacity", "massing"):
            continue
        if not st.session_state.layers.get(_layer):
            continue
        if _view_zoom >= queries.MVT_DETAIL_ZOOM[_layer]:
            continue
        _name = basemap.TILE_LAYER_NAMES[_layer]
        with st.expander(f"Legend — {_name.lower()} cells"):
            st.caption(basemap.aggregate_legend_units(_layer))
            _swatches(basemap.aggregate_legend_rows(_layer))

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
# A row clicked in one of the Overview tables
#
# Here rather than in the pane the table lives in, because that pane is drawn
# *after* the map: the fit is set now, so this run's map is built with it.
# See `_lot_clicked_in_table`.
# ---------------------------------------------------------------------------

_clicked_row_lot = _lot_clicked_in_table()
if _clicked_row_lot and _frame_lot(_clicked_row_lot):
    # For the same reason the chat's own `fit_bounds` sets it: the anchor sync
    # below must leave the framing alone rather than snap back to wherever the
    # browser was before the row was clicked.
    _commanded_view = True

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

map_col, side_col = st.columns([0.56, 0.44], gap="medium")

with map_col:
    st.subheader("Map")

    # Where the map is
    # ----------------
    # Two positions, and keeping them apart is what stopped this pane
    # redrawing itself on every pan.
    #
    # `map_center`/`map_zoom` are the **anchor**: what the folium object is
    # built at. `st_folium` keys its component on a hash of the JavaScript the
    # map renders to, so a centre baked into that script is a new key every
    # time it changes — and a new key is a *remount*: the iframe is thrown
    # away and Leaflet starts over, refetching the basemap and every vector
    # tile. Leaflet reports a new centre at the end of every drag, so building
    # the map where the browser was meant rebuilding the map on every drag.
    #
    # `view_center`/`view_zoom`/`viewport` are where the browser actually is.
    # They are read for the notes under the map and by the agent's tools, and
    # they are written *after* st_folium rather than before, which costs
    # nothing because the browser is already drawn there.
    #
    # The anchor follows the live view only when the map is being rebuilt for
    # some other reason anyway — see `signature` below — so the one remount
    # that does happen happens where the user was looking.
    center = tuple(st.session_state.map_center)
    zoom = int(st.session_state.map_zoom)
    bounds = st.session_state.viewport
    view_zoom = int(st.session_state.view_zoom or zoom)

    lots = buildings = zones = capacity = streets = massing = None
    surface_parking = opportunities = land_use = None
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
            # `caps.lots` as well, and for the reason the sidebar gives: the
            # layer is the footprints clipped to the lots they stand on, so
            # without a cadastre it draws an empty borough. Offered here only
            # when the sidebar can offer it too — a layer in the map's control
            # that the sidebar holds permanently off is a box the two of them
            # can never agree about.
            "buildings": caps.buildings and caps.lots,
            "zones": caps.features,
            "capacity": caps.redevelopment_gap,
            "land_use": caps.redevelopment_gap and caps.highest_best_use,
            "opportunities": caps.investment_opportunities,
            "streets": caps.streets,
            "massing": caps.massing,
            "surface_parking": caps.surface_parking,
        }
        for _layer, _present in _available.items():
            if not _present:
                continue
            _filters: dict[str, object] = {"scrape_date": scrape, "neighborhood": hood}
            if _layer == "lots":
                _filters["min_area"] = st.session_state.filters["min_area_m2"]
                _filters["max_area"] = st.session_state.filters["max_area_m2"]
            if _layer in ("capacity", "massing", "surface_parking") and underbuilt:
                _filters["underbuilt"] = 1
            if _layer == "opportunities":
                _filters["site_thesis"] = st.session_state.opportunity_filters.get(
                    "site_thesis"
                )
                if st.session_state.opportunity_filters.get("top_only"):
                    _filters["top_only"] = 1
                if st.session_state.opportunity_filters.get("good_only"):
                    _filters["good_only"] = 1
            if _layer == "land_use":
                _filters["use_side"] = st.session_state.get("land_use_side")
            tile_layers[_layer] = tiles.layer_url(_layer, _filters)
            tile_visibility[_layer] = bool(st.session_state.layers.get(_layer))

        # The zoom gates are Leaflet's now — a layer below its minimum is not
        # requested at all — and the note that says so is appended *below*,
        # once st_folium has handed back the zoom the browser is actually at.
        # It cannot be written here: this pane no longer reruns to adopt a
        # zoom, so a note written from the anchor would be one interaction
        # stale — and a stale zoom gate is worse than none, because it names a
        # number the reader can see is wrong.

        # "The asset has not run for this borough" is the one thing an empty
        # tile cannot say for itself, and it is worth saying: a blank massing
        # layer reads as "nothing can be built here", a blank utilisation layer
        # as "every lot is fully used", and a blank streets layer as a borough
        # with no roads — each the opposite of what a missing partition means.
        # A partition-level EXISTS, cached, rather than an inference from a
        # viewport that no longer exists.
        for _layer, _asset in (
            ("capacity", "lot_redevelopment_gap"),
            ("land_use", "lot_redevelopment_gap"),
            ("opportunities", "lot_investment_opportunities"),
            ("streets", "neighborhood_streets"),
            ("massing", "lot_building_massing"),
            ("surface_parking", "lot_building_massing"),
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

        if st.session_state.layers.get("land_use") and caps.redevelopment_gap \
                and caps.highest_best_use:
            if zoom >= basemap.MIN_LAND_USE_ZOOM:
                land_use = _land_use(
                    key, zoom, scrape, hood,
                    st.session_state.get("land_use_side") or "existing",
                )
                basemap.decorate(land_use, "land_use")
                if land_use.truncated:
                    notes.append(f"Land use capped at {land_use.count}.")
                elif not land_use.features:
                    notes.append(
                        "No land use here for "
                        f"{scrape or 'the latest snapshot'} — has the "
                        "lot_redevelopment_gap asset run for this partition?"
                    )
            else:
                notes.append(
                    f"Land use draws from zoom {basemap.MIN_LAND_USE_ZOOM} "
                    f"(now {zoom})."
                )

        if st.session_state.layers["opportunities"] and caps.investment_opportunities:
            if zoom >= basemap.MIN_OPPORTUNITY_ZOOM:
                opportunities = _opportunities(
                    key, zoom, scrape, hood,
                    st.session_state.opportunity_filters.get("site_thesis"),
                    bool(st.session_state.opportunity_filters.get("top_only")),
                    bool(st.session_state.opportunity_filters.get("good_only")),
                )
                basemap.decorate(opportunities, "opportunities")
                if opportunities.truncated:
                    notes.append(f"Opportunities capped at {opportunities.count}.")
                elif not opportunities.features:
                    # Two readings, one of them a fault: no lot in view carries
                    # a site thesis, or the asset has not run here. Named
                    # both, because an empty layer reads as "nothing to see".
                    notes.append(
                        "No opportunity here for "
                        f"{scrape or 'the latest snapshot'} - either no lot in "
                        "view carries a site thesis under the current filter, "
                        "or the lot_investment_opportunities asset has not run "
                        "for this partition."
                    )
            else:
                notes.append(
                    f"Opportunities draw from zoom {basemap.MIN_OPPORTUNITY_ZOOM} "
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

        if st.session_state.layers["surface_parking"] and caps.surface_parking:
            if zoom >= basemap.MIN_PARKING_ZOOM:
                surface_parking = _surface_parking(
                    key, zoom, scrape, hood,
                    bool(st.session_state.get("only_underbuilt", False)),
                )
                basemap.decorate(surface_parking, "surface_parking")
                if surface_parking.truncated:
                    notes.append(
                        f"Surface parking capped at {surface_parking.count}."
                    )
                elif not surface_parking.features:
                    # Two readings and only one of them is a fault, so the note
                    # names both: a borough whose programmes all park in
                    # structure has no rows here and is not missing anything.
                    notes.append(
                        "No surface parking here for "
                        f"{scrape or 'the latest snapshot'} - either the "
                        "programmes park underground, on a deck or in a "
                        "ground-floor bay, or the lot_building_massing asset "
                        "has not run for this partition."
                    )
            else:
                notes.append(
                    f"Surface parking draws from zoom "
                    f"{basemap.MIN_PARKING_ZOOM} (now {zoom})."
                )

    # --- does this run rebuild the map? ----------------------------------
    #
    # Everything the map object is made of except where it is pointed. When
    # this changes the map's JavaScript changes with it, the component's key
    # changes, and the iframe is remounted whatever we do — so that is the
    # moment to move the anchor to wherever the browser has got to, and the
    # remount lands on the view the user was looking at instead of snapping
    # back to the last anchor. When it does *not* change — every pan, every
    # zoom, every click — the map object below is byte-identical to the last
    # one and the component is left alone.
    #
    # `fit_bounds` is in here because it is baked into the script too, and
    # because clearing it is itself a change: without it in the signature the
    # rerun after a fit would rebuild at the pre-fit anchor and throw the
    # framing away.
    signature = (
        renderer,
        tuple(sorted(tile_layers.items())),
        tuple(sorted(tile_visibility.items())),
        key,
        st.session_state.fit_bounds is not None,
    )
    if signature != st.session_state.map_signature:
        st.session_state.map_signature = signature
        if st.session_state.view_center and not _commanded_view:
            st.session_state.map_center = list(st.session_state.view_center)
            st.session_state.map_zoom = view_zoom
            center = tuple(st.session_state.map_center)
            zoom = view_zoom

    # A fresh map object every rerun, and it has to be fresh.
    #
    # The obvious optimisation is to cache it and hand st_folium the same
    # object again, which is what this did. It cannot: **a folium map object
    # survives being rendered once.** `st_folium` rewrites every element's
    # `_id` to a stable `div_N` as it walks the tree, and the things holding a
    # *name* rather than an element do not follow — folium's own
    # `Layer.render` re-adds its `addTo` snippet under the new name and leaves
    # the old one behind, so the second render emits
    # `vector_grid_protobuf_<32 hex>.addTo(map_div)` for a variable that was
    # never declared. That is an uncaught ReferenceError in the map script,
    # which aborts before `initComponent`, which leaves the pane blank. It
    # cost this map every rerun that did not happen to rebuild it.
    #
    # Rebuilding is also free of the blinking the cache existed to prevent,
    # and for a reason worth writing down: st_folium keys its component on
    # `generate_js_hash`, which *strips* the `_<suffix>` off every variable
    # before hashing. Two independently built maps with the same inputs
    # therefore produce the same key, so the iframe is not remounted and the
    # pane does not blink. The random id folium stamps into each map never
    # reaches the browser at all.
    fmap = basemap.build_map(
        center=center,
        zoom=zoom,
        lots=lots,
        buildings=buildings,
        zones=zones,
        capacity=capacity,
        land_use=land_use,
        opportunities=opportunities,
        streets=streets,
        massing=massing,
        surface_parking=surface_parking,
        tile_layers=tile_layers,
        tile_visibility=tile_visibility,
        fit_bounds=st.session_state.fit_bounds,
    )

    # The selection travels beside the map rather than inside it. It is the
    # one thing on this pane that changes on a *click*, and a shape added to
    # the map object would put a click in the same class as a borough change:
    # a new component key and a reloaded iframe. `st_folium` evaluates a
    # feature group into the map already on screen, so the outline appears
    # over tiles that were never refetched. See `basemap.selection_layer`.
    selection = basemap.selection_layer(st.session_state.selected_lot)

    from streamlit_folium import st_folium  # noqa: E402

    result = st_folium(
        fmap,
        height=620,
        use_container_width=True,
        # All five are load-bearing: the click selects a lot, the bounds scope
        # what the agent's tools call "in view", the zoom says which layers
        # Leaflet is drawing at all, and `selected_layers` is the map's own
        # control answering back — see the reconciliation below.
        returned_objects=[
            "last_clicked", "bounds", "zoom", "center", "selected_layers",
        ],
        feature_group_to_add=selection,
        key="zoning_map",
    ) or {}

    # --- the browser reports where it ended up ---------------------------
    #
    # Recorded, not acted on. Under the tile renderer nothing Python draws
    # depends on where the map is looking, so a pan is finished the moment
    # Leaflet finishes it: this stores the position for the notes below and
    # for the agent's tools, and returns no rerun of its own. The component
    # has already caused one rerun by reporting — a second one, which is what
    # this used to do, is the page redrawing itself twice per drag.
    new_bounds = _reported_bounds(result)
    if new_bounds is not None:
        view_zoom = int(result.get("zoom") or view_zoom)
        new_center = _reported_center(result)
        moved = _moved_enough(bounds, new_bounds) or view_zoom != zoom

        # Under the tile renderer every report is worth keeping, because
        # keeping one costs nothing: no query above reads it. Under the
        # GeoJSON renderer it is a cache key, so the jitter has to be filtered
        # out or an idle re-layout re-queries the whole viewport.
        if renderer == "tiles" or moved:
            st.session_state.viewport = new_bounds
            st.session_state.view_zoom = view_zoom
            if new_center:
                st.session_state.view_center = [
                    round(new_center[0], 6), round(new_center[1], 6)
                ]
            # The chat renders after this column, so a tool run later in this
            # same script sees this run's viewport rather than the last one's.
            state.set_viewport(
                new_bounds,
                view_zoom,
                tuple(st.session_state.view_center or st.session_state.map_center),
            )

        # A fit is a one-shot, and this is where it stops being asked for.
        #
        # Not on the run that issued it: a fit changes the map's script, so
        # that run mounts a fresh component, and a fresh component reports
        # nothing but its defaults. The next report is the browser's own, sent
        # the moment the map is initialised — which is *after* the fitBounds
        # in the script has run. So a report arriving while a fit is pending
        # is the fit having landed. Cleared any earlier and the map would be
        # rebuilt without the framing before Leaflet applied it; left set and
        # every later rebuild would drag the user back to the lot the agent
        # framed. The anchor is moved onto the fitted view in the same breath,
        # so the rebuild below lands where the fit put things.
        if st.session_state.fit_bounds:
            st.session_state.fit_bounds = None
            st.session_state.map_zoom = view_zoom
            if st.session_state.view_center:
                st.session_state.map_center = list(st.session_state.view_center)
            st.rerun()

        # The GeoJSON renderer is the exception, and has to be: its layers are
        # queried from the viewport above the map, so a move it cannot see is
        # a map drawn for somewhere else. `_moved_enough` is deliberately
        # loose — st_folium re-reports slightly different bounds every time
        # its iframe lays out, and testing that at ~100 m turned the jitter
        # into a permanent rerun loop.
        if renderer != "tiles" and moved:
            st.session_state.map_zoom = view_zoom
            if new_center:
                st.session_state.map_center = list(st.session_state.view_center)
            st.rerun()

    # --- the map's own layer control speaks back --------------------------
    #
    # The sidebar's boxes and Leaflet's are two switches over one layer, and
    # until this the traffic ran one way: Python told the map what to draw and
    # never asked. A layer ticked on the map therefore drew, correctly, while
    # its sidebar box stayed unticked and the "Vector tiles:" line below went
    # on naming the layers Python had last asked for. The two disagreed on
    # screen about the same layer, which is the thing a checkbox exists not to
    # do.
    #
    # `selected_layers` is `basemap._layer_memory` answering: the names the
    # control is showing as ticked, reported the moment the map mounts and
    # again 250 ms after each click on it. It is the *current* value of the
    # component rather than an event, so a report that arrives on a run that
    # reruns for some other reason — a click resolving a lot, a fit landing —
    # is still there to be read on the next one and cannot be lost.
    #
    # Only the layers this run actually put on the map are reconciled, so a
    # name from a stale report — the control of a previous borough, a layer
    # since gated off by capabilities — cannot switch on a box for a layer
    # that is not there.
    _reported = result.get("selected_layers")
    if renderer == "tiles" and _reported is not None:
        _ticked = {
            entry.get("name")
            for entry in _reported
            if isinstance(entry, dict)
        }
        # Acted on once per distinct report. Adopting the report is what makes
        # the sidebar agree; adopting it *again* on every rerun is how a layer
        # Python declines to adopt becomes a spin, so the comparison is
        # against the last report seen rather than against the layer state.
        if _ticked != st.session_state.reported_layers:
            st.session_state.reported_layers = _ticked
            _sync = {
                _layer: basemap.TILE_LAYER_NAMES[_layer] in _ticked
                for _layer in tile_layers
                if bool(st.session_state.layers.get(_layer))
                != (basemap.TILE_LAYER_NAMES[_layer] in _ticked)
            }
            if _sync:
                # A rerun rather than a quiet write: the sidebar is drawn
                # above this pane and has already been drawn for this run, so
                # its boxes only redraw on the next one. The map that run
                # rebuilds is the map already on screen — `show` now matches
                # the tick the browser is holding — so what the reader sees is
                # the sidebar catching up, not the layers moving.
                st.session_state.layers.update(_sync)
                st.rerun()

    # --- a click selects a lot, or failing that a zone --------------------
    #
    # The lot is tried first because it is the finer answer and the one the
    # rest of the pane is built around. But a click that lands on no parcel is
    # not a click that landed on nothing: zoning covers ground the cadastre
    # does not, and the grid that applies there is exactly what somebody
    # pointing at it is asking for. So the fallback resolves the zone and the
    # pane shows its grille, rather than reporting an absence.
    clicked = result.get("last_clicked")
    if isinstance(clicked, dict) and clicked.get("lat") is None:
        clicked = None
    if clicked and clicked != st.session_state.last_click:
        st.session_state.last_click = clicked
        lon, lat = float(clicked["lng"]), float(clicked["lat"])
        try:
            hit = queries.lot_at_point(
                lon, lat, scrape_date=st.session_state.scrape_date,
            )
        except Exception as exc:  # noqa: BLE001
            hit, _ = None, st.warning(f"Could not resolve that click: {exc}")
        if hit:
            _select_lot(hit)
            st.rerun()
        else:
            zoned = []
            if caps.features:
                try:
                    zoned = _zoning_at_point(lon, lat, st.session_state.scrape_date)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"Could not resolve that click: {exc}")
            if zoned:
                _select_zone(zoned[0])
                st.rerun()
            else:
                st.caption("No lot or zone at that point in this snapshot.")

    # What the layers are actually showing, read off the zoom this run's report
    # carries rather than off the anchor. Written here rather than above the
    # map because that is where the number is: the pane no longer reruns to
    # adopt a zoom, so a note composed before `st_folium` would name the zoom
    # of the interaction before this one.
    #
    # These used to say "Lots draw from zoom 15", because below 15 they did not
    # draw. They do now — as cells of `gold.map_cell_aggregates`, one shape per
    # tile-grid square — so the note's job has changed from explaining an
    # absence to naming a substitution. That distinction is the whole reason it
    # is still here: a shaded square that a reader takes for a parcel is worse
    # than a blank map, because it answers.
    if renderer == "tiles":
        _summarised = [
            _layer
            for _layer in queries.AGGREGATE_LAYERS
            if tile_visibility.get(_layer)
            and view_zoom < queries.MVT_DETAIL_ZOOM[_layer]
        ]
        if _summarised:
            _named = ", ".join(
                basemap.TILE_LAYER_NAMES[_layer] for _layer in _summarised
            )
            _detail = max(
                queries.MVT_DETAIL_ZOOM[_layer] for _layer in _summarised
            )
            if queries.serves_outline(view_zoom):
                # A third thing to say, and it is the one a reader out here
                # most needs: the cells have stopped carrying numbers, so a
                # hover that does nothing is the design rather than a fault.
                notes.append(
                    f"{_named}: outlines of the "
                    f"{queries.aggregate_cell_zoom(view_zoom)}-level summary "
                    "cells — shape and shading only, with nothing to hover. "
                    f"Zoom to {queries.AGGREGATE_OUTLINE_ZOOM} for the cell "
                    f"summaries and to {_detail} for the features themselves."
                )
            else:
                notes.append(
                    f"{_named}: showing {queries.aggregate_cell_zoom(view_zoom)}-level "
                    f"summary cells, not individual features — zoom to "
                    f"{_detail} for the features themselves."
                )

        # The two filters a cell cannot honour, said out loud.
        #
        # `min_area`/`max_area` and the under-built screen are properties of a
        # *lot*, and a cell is not one: the cells were dissolved without them,
        # so below the detail zoom the shading covers every lot in the borough
        # whatever the sidebar says. Silently ignoring a filter that is visibly
        # switched on is the worst of the three options — the map would look
        # filtered and not be — so the note is unconditional whenever both
        # things are true.
        _filters_on = []
        if st.session_state.filters["min_area_m2"] or st.session_state.filters["max_area_m2"]:
            _filters_on.append("the lot area range")
        if underbuilt:
            _filters_on.append("the under-built screen")
        if _filters_on and _summarised:
            _subject = " and ".join(_filters_on)
            _verb = "does not apply" if len(_filters_on) == 1 else "do not apply"
            notes.append(
                f"{_subject[0].upper()}{_subject[1:]} {_verb} to the summary "
                "cells — they are dissolved from every lot. Zoom in for the "
                "filtered view."
            )

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
            f"Vector tiles: {', '.join(on) if on else 'no layers on'} "
            f"· zoom {view_zoom}"
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
    # Deal sits next to Lot rather than at the end, because it is the same
    # selection read one step further on: the parcel, then what a buyer would
    # pay for it and build on it, then the borough, then the by-law behind
    # both. One pane and not two, because the price and the programme are one
    # question - a brokered lot is worth what the building it carries is worth,
    # less what the ground costs.
    lot_tab, deal_tab, capacity_tab, rules_tab, chat_tab = st.tabs(
        [
            "📍 Lot", "💰 Deal", "📊 Overview",
            "📖 Regulations", "💬 Chat",
        ]
    )

    # --- Lot -------------------------------------------------------------
    with lot_tab:
        lot = st.session_state.selected_lot
        zone_only = None if lot else st.session_state.selected_zone
        if not lot and not zone_only:
            st.info(
                "Click a lot on the map, or ask the chat for one by number.\n\n"
                "Its attributes appear here, with the zones covering it and "
                "the uses they permit; the grid's values and the sheet they "
                "are read off are under **Regulations**. A click that lands "
                "on no lot resolves the zone under it instead — turn "
                "**Zoning** on in the sidebar to see where the boundaries run."
            )
        elif zone_only:
            # A zone and no parcel: what it permits, and nothing that would
            # need one. The values and the sheet they are read off are both in
            # the Regulations pane, which resolves them from this same
            # selection. No zone *count* here - that is a statement about a
            # lot, and this is the branch where there is no lot.
            st.markdown(f"### Zone {zone_only['zone']}")
            st.caption(
                f"{zone_only.get('neighborhood')} · snapshot "
                f"{zone_only.get('scrape_date')} · no lot at that point"
            )
            _render_zoning_summary([zone_only], count=False)
        else:
            st.markdown(f"### Lot {lot['lot_number']}")
            left, right = st.columns(2)
            left.metric("Area", f"{float(lot.get('area_m2') or 0):,.0f} m²")
            right.metric("Snapshot", str(lot.get("scrape_date")))
            st.caption(
                f"{lot.get('neighborhood')} · "
                f"{float(lot['lat']):.5f}, {float(lot['lon']):.5f}"
            )

            # Which piece of the parcel this pane is about. A zoning boundary
            # does not have to follow a lot line, so a large lot can be two
            # sites - two envelopes, two streets, two programmes - and every
            # answer below belongs to one of them rather than to the parcel.
            # The picker is drawn only where there is something to pick: on
            # the great majority of lots this is one zone and the pane reads
            # exactly as it always did.
            selected_zone = _zone_piece_picker(lot)

            # The gap row is read before the footprints rather than after,
            # because one of its statuses decides whether the footprints mean
            # anything at all. A parcel the solver calls `road_parcel` *is* the
            # public way - the roll files it under a CUBF road code, or a
            # geobase double side runs down the inside of it - and a building
            # that overlaps one is the cadastre and the footprint layer
            # disagreeing at the curb, not floor standing on a site. Nothing
            # may be built there whatever the grid over the block permits, so
            # the pane reports no coverage, no utilisation and no economics for
            # it: every one of those numbers would be arithmetic on an artefact.
            potential = (
                _lot_capacity(
                    int(lot["lot_uid"]), lot.get("scrape_date"),
                    lot.get("neighborhood"), selected_zone,
                )
                if caps.redevelopment_gap and lot.get("lot_uid") is not None
                else None
            )
            is_road_parcel = bool(
                potential and potential.get("hbu_status") == "road_parcel"
            )

            # Ground covered, not floor built: the two are different numbers
            # and the Efficiency block below reports the other one. Both are
            # labelled for which they are, because a 312 m2 lot carrying a
            # 164 m2 footprint and 460 m2 of floor is not a contradiction - it
            # is a three-storey building - and unlabelled they read as one.
            coverage = None
            if caps.buildings and not is_road_parcel:
                coverage = _lot_coverage(lot["lot_number"], lot.get("scrape_date"))
                built = int((coverage or {}).get("num_footprints") or 0)
                if built:
                    covered = float(coverage["covered_area_m2"])
                    pct = coverage.get("coverage_pct")
                    ratio = f" — {pct:.0f}% of the lot" if pct is not None else ""
                    st.markdown(
                        f"**Footprint:** {built} building(s), "
                        f"{covered:,.0f} m² of ground{ratio}"
                    )
                    st.caption(
                        "The ground under the parts of the footprints that "
                        "fall inside this lot — measured *taux d'implantation*, "
                        "against the one the grid below permits. A footprint "
                        "spanning several lots counts here only for the part "
                        "on this one."
                    )
                else:
                    st.markdown("**Footprint:** no building on this lot")

            # What stands there, in the roll's own words. The gap row
            # carries the MEFQ's description of the use code on the assessment
            # unit holding most of the parcel's value - so this names what the
            # lot *is* today, against the programme proposed below it. Read off
            # the same row as the arithmetic rather than looked up again, so the
            # words and the floor areas cannot end up describing two different
            # units.
            use_description = (
                potential.get("existing_dominant_use_description")
                if potential else None
            )
            use_code = (
                potential.get("existing_dominant_use_code") if potential else None
            )
            if use_description or use_code:
                if use_description:
                    # French, as published - the manual is not issued in
                    # English, and translating it here would put words on the
                    # pane that no source says.
                    st.markdown(
                        f"**Current use:** {use_description}"
                        + (f" · CUBF {use_code}" if use_code else "")
                    )
                else:
                    # A code the codebook does not carry. The number is still
                    # the roll's answer, and saying so beats an empty line.
                    st.markdown(
                        f"**Current use:** CUBF {use_code} — not in this "
                        f"edition of the manual"
                    )
                st.caption(
                    "The use of the assessment unit carrying most of this "
                    "lot's value, not of every unit on it: a lot with a "
                    "triplex over a depanneur reports one of the two."
                )
            if not is_road_parcel:
                _render_use_comparison(potential, coverage)

            # --- is it used efficiently, and what else fits ---------------
            if is_road_parcel:
                st.divider()
                st.markdown("**Street** — not a development site")
                st.caption(
                    "This parcel is the public way itself: a street, a lane, "
                    "a highway or a right of way. Nothing may be built on it, "
                    "so no coverage, utilisation or development economics are "
                    "reported here. A footprint overlapping it is a spatial "
                    "artefact of two layers meeting at the curb, not floor on "
                    "this lot."
                )
            elif caps.redevelopment_gap and lot.get("lot_uid") is not None:
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
                    # The five reasons live beside the Deal pane, which says the
                    # same five things at length. Two copies of this text is
                    # two places for a status the dataplatform renames to be
                    # half-updated.
                    st.caption(_hbu_status_reason(potential.get("hbu_status")))
                else:
                    # Before the figure it qualifies: a permitted floor that
                    # only exists with its stalls waived is a different fact
                    # from one the parcel can park, and the Deal pane says
                    # the same thing at length.
                    if potential.get("parking_waived"):
                        st.warning(
                            "**Parking waived on the proposal.** No programme "
                            "fits this lot with the stalls it owes, so the "
                            "one below was solved without them — "
                            f"{int(potential.get('waived_stalls') or 0)} "
                            "stall(s) owed and none provided."
                        )
                    used = potential.get("used_pct")
                    permitted = float(potential.get("hbu_floor_area_m2") or 0)
                    # A missing existing floor is two different facts, and
                    # only one of them is a zero. Where the roll assessed a
                    # unit on the lot and stated no area for it, the number is
                    # unknown - so the metric, the delta and the verdict all
                    # go blank rather than reporting a building that stands
                    # as nothing standing.
                    unreported = queries.floor_area_unreported(potential)
                    built = (
                        None if unreported
                        else float(potential.get("existing_floor_area_m2") or 0)
                    )

                    if unreported:
                        verdict = "floor area not reported"
                        note = ""
                    elif used is None:
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

                    # *Floor* area on both sides, every storey added up - not
                    # the ground the building covers, which is the footprint
                    # reported above. Labelled "standing today" these two sat
                    # under a footprint figure they are supposed to exceed, and
                    # the pane looked like it contradicted itself.
                    st.markdown(f"### Efficiency — {verdict}")
                    left, right = st.columns(2)
                    left.metric(
                        "Floor area today",
                        "N/A" if built is None else f"{built:,.0f} m²",
                    )
                    right.metric(
                        "Floor area zoning allows", f"{permitted:,.0f} m²",
                        # No subtraction without both sides. An unreported
                        # existing floor made this delta the whole envelope,
                        # which reads as surveyed headroom on a lot nobody
                        # measured.
                        delta=(
                            None if built is None
                            else f"{permitted - built:+,.0f} m²"
                        ),
                    )
                    st.caption(
                        "Floor area is every storey added up, from the "
                        "assessment roll — so it runs to several times the "
                        "footprint on a building of more than one storey."
                    )
                    if note:
                        st.caption(note)
                    if unreported:
                        # The case this pane used to report as 0%: a building
                        # stands, the roll assessed it, and the roll states no
                        # superficie for it. Common on non-residential units,
                        # where the value is carried by the land.
                        st.caption(
                            "⚠️ The assessment roll states no floor area for "
                            "the unit on this lot, so how much of the envelope "
                            "is in use cannot be computed. This is a gap in "
                            "the roll, not an empty lot — the footprint above "
                            "is what is known to stand here."
                        )
                    elif queries.nothing_assessed(potential):
                        # The gap table reads a missing existing floor as zero,
                        # so this lot's whole envelope is counted as headroom.
                        # Say so rather than letting 0 m² read as surveyed.
                        # Asked of the unit count rather than has_assessment,
                        # which gold writes true on every row - so this caption
                        # never once appeared on the 2,481 lots it describes.
                        st.caption(
                            "⚠️ The assessment roll has no unit on this lot, so "
                            "*floor area today* is read as nothing built."
                        )

                    st.markdown("**What else could go here**")
                    if unreported:
                        # Headroom is the same subtraction as the delta above,
                        # split by class, so it cannot be stated either - the
                        # expression reads the missing existing floor as zero
                        # and hands back the whole envelope. Quoting the
                        # envelope is the honest half of it.
                        st.caption(
                            f"The envelope holds {permitted:,.0f} m² in all. "
                            f"How much of that is already standing is not in "
                            f"the roll, so what could be added on top of it "
                            f"cannot be stated."
                        )
                    else:
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
                            st.caption(
                                "No additional floor area under this grid."
                            )

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
                    # One line of a programme that is thirty. The rest — the
                    # storey stack, the bedroom mix, the stalls, the cost of
                    # each part and the caps that stopped it being bigger — is
                    # its own pane, for the reason that pane's header gives.
                    if caps.highest_best_use:
                        st.caption(
                            "The whole proposal — storeys, unit mix, "
                            "commercial and industrial floor, parking, the "
                            "massing as drawn, and what it costs against "
                            "what it earns, and what the ground under it "
                            "costs — is under **Deal**."
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
                zoning = _zoning_for_lot(lot["lot_number"], lot.get("scrape_date"))
                if not zoning:
                    st.info(
                        "No zoning polygon covers this lot in this snapshot. "
                        f"A zone has to cover more than "
                        f"{queries.MIN_ZONE_OVERLAP_M2:g} m² of the lot to "
                        "count; anything less is the cadastre and the zoning "
                        "layer disagreeing, not a rule."
                    )
                else:
                    # `len(zoning)` is a count of *zones*, because the query
                    # returns one row per distinct zone. It used to be a count
                    # of rows, and a lot in one zone across two snapshots was
                    # reported as straddling two.
                    if len(zoning) > 1:
                        st.warning(
                            f"This lot straddles {len(zoning)} zones — that "
                            "many separate sets of rules land on it. "
                            "**Regulations** takes them one at a time, in the "
                            "order they are listed here."
                        )
                    # How many grids reach this lot and what they let anybody
                    # build - not the grids themselves. Seventeen rows of
                    # minima and maxima are a reading of the by-law, so they
                    # are drawn once, in the pane that is about the by-law,
                    # beside the sheet they are read off and the other
                    # documents covering this lot.
                    _render_zoning_summary(zoning)

    # --- Deal: the price, the play, and the building behind it -----------
    #
    # The same selection as the Lot pane and never its own, so the two cannot
    # end up describing different parcels. A zone-only click resolves nothing
    # here: a programme is solved per lot, and there is no envelope, no
    # frontage and no assessment to price without one.
    #
    # `st.tabs` renders every tab on every rerun, so this draws whether or not
    # anyone is looking at it. What that costs is two primary-key lookups,
    # cached for five minutes on the lot — the same shape and the same cache
    # as the three reads the Lot pane already makes on every rerun beside it.
    with deal_tab:
        if not caps.investment_opportunities and not caps.highest_best_use:
            st.info(
                f"Neither `{queries.GOLD_SCHEMA}.lot_investment_opportunities` "
                f"nor `{queries.GOLD_SCHEMA}.lot_highest_best_use` is in this "
                "database yet. The first prices what a buyer would pay and get "
                "back; the second is the building behind that price — the "
                "storeys, the unit mix, the stalls and the money. Run both "
                "assets for this partition. The **Lot** pane still reports the "
                "subtraction without them."
            )
        elif st.session_state.selected_lot:
            _render_deal(st.session_state.selected_lot, caps=caps)
        elif st.session_state.selected_zone:
            st.info(
                f"Zone {st.session_state.selected_zone['zone']} is selected "
                "and no lot is. A deal is priced per parcel — the envelope, "
                "the frontage and the assessment the price is set against are "
                "all the lot's — so pick one on the map to see what it would "
                "take and what it would return."
            )
        else:
            st.info(
                "Click a lot on the map and the whole deal appears here: what "
                "it costs to buy, the most a buyer could pay before the return "
                "goes, which of keeping, enhancing and rebuilding pays best at "
                "that price, why the parcel is acquirable at all, and the "
                "building the solver proposes for it — its storeys, "
                "dwellings, commercial floor, parking and the printed caps "
                "that stopped it being bigger."
                "\n\nTurn **Proposed massing** on in the sidebar to see that "
                "building drawn on its parcel, and **Opportunities** to see "
                "every other lot in the borough carrying the same thesis."
            )

    # --- Overview: the borough's capacity --------------------------------
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
                # Sites and parcels are two numbers now, and both are said: a
                # zoning boundary crossing a lot makes two development sites of
                # it, so the borough has more sites than parcels and every
                # total below is over the sites. `num_split_lots` is what makes
                # the difference legible rather than a discrepancy.
                sites = int(totals.get("num_sites") or totals["num_lots"])
                lots = int(totals["num_lots"])
                split = int(totals.get("num_split_lots") or 0)
                scale = (
                    f"{sites:,} sites on {lots:,} lots "
                    f"({split:,} of them split between zones)"
                    if split
                    else f"{lots:,} lots"
                )
                st.caption(
                    f"{scale} · snapshot "
                    f"{st.session_state.scrape_date or 'latest'} · under the "
                    "zoning envelope governing each piece of ground"
                )
                # Said once, at the top, for the three lot tables below it. A
                # borough total is an argument about particular parcels, and
                # the tables are where it names them - so the row is the way
                # back to the map, and from there to the Lot and Deal panes
                # that read the same parcel one step further on.
                st.caption(
                    "**Click any row below** to select that lot and frame it "
                    "on the map."
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
                # Against the *site* count, because that is what has a
                # programme or does not: a split parcel can have one
                # solved piece and one that is a park.
                denominator = int(totals.get("num_sites") or totals["num_lots"])
                share = solved / denominator * 100 if denominator else 0
                st.markdown(
                    f"- **{solved:,}** sites ({share:.0f}%) have a solved "
                    f"programme. The rest contribute nothing — a site with no "
                    f"answer is not a site with no room.\n"
                    f"- **{int(totals.get('num_underbuilt') or 0):,}** sites hold "
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
                    _lot_table(
                        [
                            {
                                "Site": _site_label(r),
                                "Site area (m²)": f"{_site_area_m2(r):,.0f}",
                                "Today": int(r.get("existing_num_dwellings") or 0),
                                "Proposed": int(r.get("hbu_num_dwellings") or 0),
                            }
                            for r in top
                        ],
                        "overview_top_capacity",
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
                    _lot_table(
                        [
                            {
                                "Site": _site_label(r),
                                "Site area (m²)": f"{_site_area_m2(r):,.0f}",
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
                        "overview_top_npv_gain",
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

    # --- Regulations -----------------------------------------------------
    #
    # Two halves, and the order between them is the argument. On top are the
    # documents that *govern* the current selection, which arrive from a join
    # and are true of the parcel whether or not anybody has asked anything.
    # Below are the passages the last chat turn *retrieved*, which are an
    # answer to a question that was asked. Clicking a lot is not a question,
    # which is why this pane used to sit empty after one.
        # The second axis, borough-wide: how many lots each site thesis
        # filed, how many pay, what the ranked ones yield. Its own read of
        # its own table, so a database with the gap and not the shortlist
        # still draws the totals above.
        if caps.investment_opportunities:
            _site_rows = {
                row["site_thesis"]: row
                for row in (
                    _site_thesis_totals(
                        st.session_state.neighborhood, st.session_state.scrape_date
                    )
                    or []
                )
            }
            if _site_rows:
                st.divider()
                st.markdown("### Why the sites are acquirable")
                st.caption(
                    "Every lot filed under a site thesis - brownfield, teardown, "
                    "infill or improvement - and, of those, the ones that pay at "
                    "the solve's assumptions. Yields carry each thesis's own "
                    "costs: demolition, characterisation and remediation, or "
                    "the addition's premium. A good candidate clears the area's "
                    "cap rate by the development spread and the IRR hurdle, "
                    "with soft costs, contingency and the lease-up in."
                )
                st.dataframe(
                    [
                        {
                            "Site thesis": thesis,
                            # Sites, and the parcels under them. They
                            # differ where a zoning boundary crosses a lot
                            # and both pieces land in the same thesis.
                            "Sites filed": int(row.get("num_lots") or 0),
                            "Lots": int(
                                row.get("num_parcels") or row.get("num_lots") or 0
                            ),
                            "That pay": int(row.get("num_ranked") or 0),
                            "Shortlisted": int(row.get("num_top") or 0),
                            "Median yield": (
                                f"{float(row['median_site_yield_on_cost_pct']):,.1f}%"
                                if row.get("median_site_yield_on_cost_pct") is not None
                                else "—"
                            ),
                            "Best yield": (
                                f"{float(row['best_site_yield_on_cost_pct']):,.1f}%"
                                if row.get("best_site_yield_on_cost_pct") is not None
                                else "—"
                            ),
                            "Verdict, summed": (
                                f"${float(row.get('ranked_improvement_noi_cad') or 0):,.0f}/yr NOI"
                                if thesis == "improvement"
                                else f"+${float(row.get('ranked_npv_gain_cad') or 0):,.0f}"
                            ),
                            "Good candidates": int(row.get("num_good_candidates") or 0),
                            "Median IRR": (
                                f"{float(row['median_site_irr_pct']):,.1f}%"
                                if row.get("median_site_irr_pct") is not None else "—"
                            ),
                            "Ranked area (ha)": (
                                f"{float(row.get('ranked_lot_area_m2') or 0) / 10_000:,.1f}"
                            ),
                            "In a heritage sector": int(row.get("num_heritage_sector") or 0),
                            "Under a PIIA": int(row.get("num_piia_review") or 0),
                        }
                        for thesis in queries.SITE_THESES
                        for row in [_site_rows.get(thesis, {})]
                    ],
                    width="stretch",
                    hide_index=True,
                )
                _future_rows = _futures_totals(
                    st.session_state.neighborhood, st.session_state.scrape_date
                )
                if _future_rows:
                    st.markdown("**Keep, enhance, or tear down and rebuild**")
                    st.caption(
                        "Which of the three a buyer does best out of on each "
                        "lot, at the run's market factor and after paying for "
                        "the ground. The **Deal** pane shows the arithmetic "
                        "for one lot. The owner's column is kept beside it for "
                        "one reason only: it is the count of owners who do "
                        "best out of that same future themselves, and the "
                        "**Keep** row of it is the lots that will not be "
                        "listed however well they price for a buyer."
                    )
                    st.dataframe(
                        [
                            {
                                "Future": _FUTURE_TITLES.get(r.get("future"), r.get("future")),
                                "Buyer's best on": int(r.get("buyer_wins") or 0),
                                "Buyer's NPV, summed": _money(r.get("buyer_npv_cad"), signed=True),
                                "Owner's best on": int(r.get("owner_wins") or 0),
                            }
                            for r in _future_rows
                        ],
                        width="stretch",
                        hide_index=True,
                    )
                _top_sites = _top_site_opportunities(
                    st.session_state.neighborhood, st.session_state.scrape_date
                )
                if _top_sites:
                    st.markdown("**The best of each, interleaved by rank**")
                    _lot_table(
                        [
                            {
                                "Site": _site_label(r),
                                "Thesis": r.get("site_thesis"),
                                "Rank": int(r.get("site_thesis_rank") or 0),
                                "IRR": (
                                    f"{float(r['site_irr_pct']):,.1f}%"
                                    if r.get("site_irr_pct") is not None else "—"
                                ),
                                "Yield on cost": (
                                    f"{float(r['site_all_in_yield_on_cost_pct']):,.1f}%"
                                    if r.get("site_all_in_yield_on_cost_pct") is not None else "—"
                                ),
                                "Good": "✔" if r.get("is_good_candidate") else "",
                                "Would build": str(r.get("investment_thesis") or "—").replace("_", " "),
                                "Built": (
                                    str(int(r["existing_year_built"]))
                                    if r.get("existing_year_built") is not None else "—"
                                ),
                                "Storeys": (
                                    f"{int(r['existing_num_storeys'])} → {int(r['hbu_floors'])}"
                                    if r.get("existing_num_storeys") is not None
                                    and r.get("hbu_floors") is not None
                                    else "—"
                                ),
                                "Standing": r.get("existing_dominant_use_description") or "—",
                                "Zone": r.get("grid_zone") or "—",
                            }
                            for r in _top_sites
                        ],
                        "overview_top_sites",
                    )
                    st.caption(
                        "Ask the chat for the rest: *top_site_opportunities* "
                        "lists any thesis to any length, and the "
                        "**Opportunities** layer draws them."
                    )

    with rules_tab:
        selection = st.session_state.selected_lot
        zone_only = None if selection else st.session_state.selected_zone

        if selection:
            _render_lot_documents(selection, caps=caps)
        elif zone_only:
            st.markdown(f"### By-laws for zone {zone_only['zone']}")
            st.caption(
                f"{zone_only.get('neighborhood')} · snapshot "
                f"{zone_only.get('scrape_date')} · no lot at that point"
            )
            _render_zoning_grid(zone_only, has_chunks=caps.chunks)
        else:
            st.info(
                "Click a lot on the map and the by-law documents governing it "
                "appear here — the *grille des spécifications* for each zone "
                "covering it, to read, open or download. A click that lands on "
                "no lot resolves the zone under it instead."
            )

        st.divider()
        st.markdown("#### Retrieved passages")

        buffer = state.RagBuffer
        if not buffer.get("hits"):
            st.caption(
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
