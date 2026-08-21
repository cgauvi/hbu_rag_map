"""
state.py — The side-channel between the agent's tools and the map.

The chat panel and the map are the same application but not the same control
flow: a tool runs inside a LangGraph stream, and Streamlit renders after that
stream finishes. So a tool that wants to move the map cannot call Streamlit —
it writes here, and app.py reads and applies it once the turn is over. Same
arrangement as ``VizBuffer`` in ebird-llm, with two buffers instead of one
because a turn can change *where the map is looking* and *what the corpus said*
independently.

``MapCommand`` is deliberately a request rather than the state itself. The map's
real state — centre, zoom, bounds — belongs to the browser and arrives back
through ``st_folium``; if a tool wrote to it directly, every rerun would fight
the user's last pan. A command is consumed once and cleared.
"""

from __future__ import annotations

from typing import Any

#: What a tool asks the map to do. `pending` is the flag app.py checks and
#: clears; everything else is only meaningful while it is set.
MapCommand: dict[str, Any] = {
    "pending": False,
    "center": None,       # [lat, lon]
    "zoom": None,         # int
    "fit_bounds": None,   # [[south, west], [north, east]]
    "select_lot": None,   # lot_number — app.py loads and highlights it
    "layers": None,       # {"lots": bool, "buildings": bool, "zones": bool}
    "filters": None,      # {"min_area_m2": float|None, "max_area_m2": float|None}
    "note": None,         # one line shown under the map: what the agent did
}

#: Where the map is currently looking, as ``(west, south, east, north)``. This
#: one flows the other way — the browser reports it through ``st_folium`` and
#: app.py stashes it here so a tool can scope "in view" to what the user sees.
#: It is not part of MapCommand for exactly that reason: clearing a command
#: must not forget where the map is.
Viewport: dict[str, Any] = {"bounds": None, "zoom": None, "center": None}


#: The retrieval the last turn performed, so the Regulations pane shows what
#: the answer was actually built from rather than a second, different search.
RagBuffer: dict[str, Any] = {
    "query": None,
    "hits": None,      # list[dict] from rag.search_* / search_corpus
    "scope": None,     # "lot" | "near" | "corpus"
    "lot_number": None,
}


def clear_map_command() -> None:
    for key in MapCommand:
        MapCommand[key] = False if key == "pending" else None


def clear_rag_buffer() -> None:
    for key in RagBuffer:
        RagBuffer[key] = None


def request_map(**changes: Any) -> None:
    """Merge a tool's request into the pending command.

    Merging rather than replacing so one turn can both select a lot and toggle
    a layer without the second call discarding the first.
    """
    for key, value in changes.items():
        if key not in MapCommand:
            raise KeyError(f"{key!r} is not a MapCommand field")
        if isinstance(value, dict) and isinstance(MapCommand.get(key), dict):
            MapCommand[key] = {**MapCommand[key], **value}
        else:
            MapCommand[key] = value
    MapCommand["pending"] = True


def take_map_command() -> dict | None:
    """The pending command, cleared. None when there is nothing to apply."""
    if not MapCommand["pending"]:
        return None
    command = {k: v for k, v in MapCommand.items() if k != "pending"}
    clear_map_command()
    return command


def set_viewport(
    bounds: tuple[float, float, float, float] | None,
    zoom: int | None = None,
    center: tuple[float, float] | None = None,
) -> None:
    """Record what the map is showing, after every rerun."""
    Viewport.update({"bounds": bounds, "zoom": zoom, "center": center})


def get_viewport() -> tuple[float, float, float, float] | None:
    """``(west, south, east, north)``, or None before the map has reported."""
    return Viewport.get("bounds")


def get_viewport_zoom(default: int = 16) -> int:
    zoom = Viewport.get("zoom")
    return int(zoom) if zoom else default


def set_rag_result(query: str, hits: list[dict], scope: str, lot_number: str | None = None) -> None:
    RagBuffer.update(
        {"query": query, "hits": hits, "scope": scope, "lot_number": lot_number}
    )


# ---------------------------------------------------------------------------
# Turn tracking
# ---------------------------------------------------------------------------
#
# Lets a tool tell "the user asked about this lot in this turn" from "a lot is
# still selected from three turns ago". Without it an agent that skips the
# lookup can answer about whatever the map happens to be showing, which reads
# as a correct answer to a question nobody asked.

_current_turn: int = 0
_context_turn: int = -1

#: What the agent last resolved: the lot it is talking about and where it is.
#: Read by tools that take no coordinates, so "what can I build here" works
#: after a click without the LLM having to restate the lot number.
SelectedLot: dict[str, Any] = {
    "lot_number": None,
    "lon": None,
    "lat": None,
    "neighborhood": None,
}


def start_new_turn() -> int:
    global _current_turn
    _current_turn += 1
    return _current_turn


def mark_context_current() -> None:
    global _context_turn
    _context_turn = _current_turn


def context_is_current_turn() -> bool:
    return _context_turn == _current_turn


def set_selected_lot(
    lot_number: str | None,
    lon: float | None = None,
    lat: float | None = None,
    neighborhood: str | None = None,
) -> None:
    SelectedLot.update(
        {
            "lot_number": lot_number,
            "lon": lon,
            "lat": lat,
            "neighborhood": neighborhood,
        }
    )
    mark_context_current()


def get_selected_lot() -> dict:
    return dict(SelectedLot)


def clear_selected_lot() -> None:
    set_selected_lot(None)
