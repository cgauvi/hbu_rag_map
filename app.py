"""
app.py — Streamlit front end: an interactive zoning map with a chat panel.

Two inputs, one state. The map is directly interactive — pan and zoom load lots
and buildings by viewport, and clicking a lot selects it — while the chat panel
reaches the same data through tools and can move the map back. Both write the
selection to the same place, so "what can I build here" asked after a click
means the lot that was clicked.

The click is resolved **server-side**, from its coordinates rather than from
whatever shape the browser reports being hit. A click near a boundary, on a lot
the viewport limit left undrawn, or on a simplified edge all still land on the
right parcel that way.

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

from src.utils import basemap, queries, state  # noqa: E402

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
    "layers": {"lots": True, "buildings": True, "zones": False},
    "filters": {"min_area_m2": None, "max_area_m2": None},
    "neighborhood": None,
    "scrape_date": None,
    "agent_note": None,
    "last_click": None,
}

for _key, _value in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value.copy() if isinstance(_value, (dict, list)) else _value


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

    if connected:
        present = [n for n in ("lots", "buildings", "features", "chunks") if getattr(caps, n)]
        st.success(f"Connected · {', '.join(present) or 'no tables yet'}")
        missing = caps.missing()
        if missing:
            st.caption("Not loaded: " + ", ".join(missing))
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
        f"Missing: {', '.join(caps.missing())}"
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

    lots = buildings = zones = None
    notes: list[str] = []

    if bounds:
        key = _cache_key(bounds)
        scrape, hood = st.session_state.scrape_date, st.session_state.neighborhood

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

    fmap = basemap.build_map(
        center=center,
        zoom=zoom,
        lots=lots,
        buildings=buildings,
        zones=zones,
        selected=st.session_state.selected_lot,
        fit_bounds=st.session_state.fit_bounds,
    )
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
        # Rerun once when the map first reports itself, or when a pan or zoom
        # changed what should be drawn. Comparing against the values this render
        # actually used — at ~100 m precision — is what keeps it from looping.
        moved = bounds is None or _cache_key(new_bounds, 3) != _cache_key(bounds, 3)
        if moved or new_zoom != zoom:
            st.session_state.viewport = new_bounds
            st.session_state.map_zoom = new_zoom
            if new_center:
                st.session_state.map_center = new_center
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
    drawn = ", ".join(
        f"{fs.count} {fs.layer}" for fs in (lots, buildings, zones) if fs and fs.count
    )
    st.caption(f"Drawn: {drawn or 'nothing in view'} · zoom {zoom}")

# ---------------------------------------------------------------------------
# Right panel
# ---------------------------------------------------------------------------

with side_col:
    lot_tab, rules_tab, chat_tab = st.tabs(["📍 Lot & zoning", "📖 Regulations", "💬 Chat"])

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
