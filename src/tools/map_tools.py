"""
map_tools.py — Letting the chat move the map.

These tools write a request into ``src.utils.state.MapCommand``; app.py applies
it after the agent's turn and clears it. They deliberately do *not* read the
map's current position back — that belongs to the browser, and a tool that
wrote centre and zoom directly would undo the user's last pan on every rerun.

Keeping map control in tools rather than parsing it out of the answer is what
makes "show me the lots over 500 m² around Jarry" a single turn: the model
filters, then moves, and the panel it moved is the one the user is looking at.
"""

from __future__ import annotations

import logging

from langchain.tools import tool
from langchain_core.tools import ToolException

from src.utils import basemap, queries, state

logger = logging.getLogger(__name__)

#: Beyond this the map is off the loaded borough and shows an empty basemap.
_LAT_RANGE = (44.0, 47.0)
_LON_RANGE = (-75.0, -72.0)


@tool
def focus_map(lat: float, lon: float, zoom: int = 17) -> str:
    """Move the map to a coordinate.

    Use this when the user names a place you have coordinates for, or after a
    search that found something worth looking at. To centre on a lot, prefer
    find_lot — it fits the frame to the parcel's real extent instead of
    guessing a zoom.

    Args:
        lat: Latitude, between 44 and 47 for the Montreal region.
        lon: Longitude, between -75 and -72.
        zoom: 13 shows a borough, 15 a neighbourhood, 17 a street, 19 a parcel.

    Returns:
        Confirmation of where the map moved.
    """
    if not _LAT_RANGE[0] <= lat <= _LAT_RANGE[1] or not _LON_RANGE[0] <= lon <= _LON_RANGE[1]:
        raise ToolException(
            f"({lat}, {lon}) is outside the Montreal region this data covers. "
            f"Check you have not swapped latitude and longitude."
        )
    zoom = max(11, min(int(zoom), 19))
    state.request_map(center=[float(lat), float(lon)], zoom=zoom,
                      note=f"Moved to {lat:.5f}, {lon:.5f}")
    return f"Map centred on {lat:.5f}, {lon:.5f} at zoom {zoom}."


#: Which capability each layer needs, where the two are not spelled the same.
#: A tuple where a layer needs more than one table: land use reads today's
#: class off the gap table and the proposed one off the HBU table, and a
#: database with one and not the other cannot draw the layer either way.
_CAPABILITY_OF = {
    "zones": "features",
    "opportunities": "investment_opportunities",
    "land_use": ("redevelopment_gap", "highest_best_use"),
}


def _has_capability(caps, layer: str) -> bool:
    needed = _CAPABILITY_OF.get(layer, layer)
    if isinstance(needed, str):
        needed = (needed,)
    return all(getattr(caps, name, False) for name in needed)


@tool
def set_map_layers(
    lots: bool | None = None,
    buildings: bool | None = None,
    zones: bool | None = None,
    opportunities: bool | None = None,
    land_use: bool | None = None,
) -> str:
    """Turn map layers on or off.

    Use this when the user asks to see or hide something — "show the zoning",
    "hide the buildings", "just the lots", "show me the opportunities",
    "colour the lots by use". Omitted layers keep their current setting.

    Args:
        lots: Cadastral parcels from Infolot.
        buildings: Building footprints.
        zones: Zoning polygons, the layer carrying the link to each grid PDF.
        opportunities: The lots filed under a site thesis - brownfield,
            teardown, infill or improvement - coloured by which.
        land_use: Every lot coloured by what it is used for - residential,
            commercial, industrial, mixed or none - on the roll today or as
            the solver proposes; the sidebar picks which side.

    Returns:
        Which layers were changed.
    """
    changes = {
        name: value
        for name, value in (
            ("lots", lots),
            ("buildings", buildings),
            ("zones", zones),
            ("opportunities", opportunities),
            ("land_use", land_use),
        )
        if value is not None
    }
    if not changes:
        raise ToolException(
            "No layer given — pass at least one of lots, buildings, zones, "
            "opportunities, land_use."
        )

    caps = queries.capabilities()
    unavailable = [
        name
        for name, wanted in changes.items()
        if wanted and not _has_capability(caps, name)
    ]
    if unavailable:
        raise ToolException(
            f"Cannot show {', '.join(unavailable)} — the table behind that "
            f"layer is not loaded in this database. Tell the user rather than "
            f"retrying."
        )

    state.request_map(
        layers=changes,
        note="Layers: " + ", ".join(f"{k} {'on' if v else 'off'}" for k, v in changes.items()),
    )
    return "Layers updated: " + ", ".join(
        f"{name} {'on' if value else 'off'}" for name, value in changes.items()
    )


@tool
def filter_lots_on_map(
    min_area_m2: float | None = None, max_area_m2: float | None = None
) -> str:
    """Restrict the lots drawn on the map to a size range.

    Use this when the user wants to *see* which parcels qualify rather than
    read a list — "highlight the lots over 1000 m²". Pass both as null to clear
    the filter.

    Args:
        min_area_m2: Draw only lots at least this large.
        max_area_m2: Draw only lots at most this large.

    Returns:
        A description of the filter now applied.
    """
    state.request_map(
        filters={"min_area_m2": min_area_m2, "max_area_m2": max_area_m2},
        note="Lot size filter applied",
    )
    if min_area_m2 is None and max_area_m2 is None:
        return "Lot size filter cleared — all lots in view are drawn."
    bounds = []
    if min_area_m2 is not None:
        bounds.append(f"at least {float(min_area_m2):,.0f} m²")
    if max_area_m2 is not None:
        bounds.append(f"at most {float(max_area_m2):,.0f} m²")
    return f"Map now draws only lots {' and '.join(bounds)}."


@tool
def show_lot_on_map(lot_number: str) -> str:
    """Select a lot and fit the map to it.

    A thinner find_lot: use this when you already know the lot exists and only
    want the map to go there — for instance after listing several and the user
    picks one.

    Args:
        lot_number: The Infolot lot number; separators are ignored.

    Returns:
        Confirmation, or an error if no such lot is loaded.
    """
    lot = queries.lot_by_number(lot_number)
    if not lot:
        raise ToolException(f"No lot numbered {lot_number!r} in the loaded snapshots.")

    bounds = basemap.bounds_of(lot.get("geometry"))
    state.set_selected_lot(
        lot["lot_number"], lot.get("lon"), lot.get("lat"), lot.get("neighborhood")
    )
    state.request_map(
        select_lot=lot["lot_number"],
        fit_bounds=basemap.pad_bounds(bounds) if bounds else None,
        note=f"Lot {lot['lot_number']} selected",
    )
    return f"Lot {lot['lot_number']} selected and framed on the map."


MAP_TOOLS = [focus_map, set_map_layers, filter_lots_on_map, show_lot_on_map]
