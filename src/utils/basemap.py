"""
basemap.py — Building the folium map the Streamlit pane renders.

One function assembles every layer, because the layers have to agree about
draw order and about what a click means. Zones are drawn first and lots on top
of them: a click lands on the smallest shape under the cursor, and the lot is
what the user is after. Buildings sit on top of lots as fills, since a
footprint is read as a mass rather than as an outline.

Nothing here queries. The caller passes ``FeatureSet``s it has already fetched
and cached, so panning re-renders without re-deciding what to fetch.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Montreal, Villeray–Saint-Michel–Parc-Extension. Where the map opens when
#: nothing has been selected and the database has no extent to offer.
DEFAULT_CENTER = (45.5535, -73.6200)
DEFAULT_ZOOM = 15

#: Below these, the layer is not drawn at all. A lot is sub-pixel at zoom 13
#: and a borough's worth of them is a solid grey rectangle that costs a second
#: of browser time to produce.
MIN_LOT_ZOOM = 15
MIN_BUILDING_ZOOM = 16

_LOT_STYLE = {
    "color": "#3d5a80",
    "weight": 1,
    "fillColor": "#98c1d9",
    "fillOpacity": 0.25,
}
_LOT_HIGHLIGHT = {"weight": 3, "color": "#ee6c4d", "fillOpacity": 0.45}

_BUILDING_STYLE = {
    "color": "#5c5470",
    "weight": 0.5,
    "fillColor": "#5c5470",
    "fillOpacity": 0.55,
}

_ZONE_STYLE = {
    "color": "#e07a5f",
    "weight": 2,
    "fillColor": "#f2cc8f",
    "fillOpacity": 0.12,
    "dashArray": "4,3",
}

_SELECTED_STYLE = {
    "color": "#d62828",
    "weight": 4,
    "fillColor": "#f77f00",
    "fillOpacity": 0.35,
}


def build_map(
    *,
    center: tuple[float, float] = DEFAULT_CENTER,
    zoom: int = DEFAULT_ZOOM,
    lots: Any = None,
    buildings: Any = None,
    zones: Any = None,
    selected: dict | None = None,
    fit_bounds: list | None = None,
):
    """Assemble the map. ``lots``/``buildings``/``zones`` are ``FeatureSet``s."""
    import folium  # noqa: PLC0415

    fmap = folium.Map(
        location=list(center),
        zoom_start=zoom,
        tiles="CartoDB positron",
        control_scale=True,
        # The zoom-gated layers make a hard-zoomed-out view meaningless, and
        # the corpus only covers one borough anyway.
        min_zoom=11,
        max_zoom=19,
        prefer_canvas=True,
    )

    if zones is not None and zones.features:
        folium.GeoJson(
            zones.collection(),
            name=f"Zonage ({zones.count})",
            style_function=lambda _: dict(_ZONE_STYLE),
            highlight_function=lambda _: {"weight": 3, "fillOpacity": 0.25},
            tooltip=folium.GeoJsonTooltip(
                fields=["zone_label"],
                aliases=["Zone"],
                sticky=True,
            ),
            # Zones sit underneath so a click reaches the lot on top of them.
            control=True,
        ).add_to(fmap)

    if lots is not None and lots.features:
        folium.GeoJson(
            lots.collection(),
            name=f"Lots ({lots.count})",
            style_function=lambda _: dict(_LOT_STYLE),
            highlight_function=lambda _: dict(_LOT_HIGHLIGHT),
            tooltip=folium.GeoJsonTooltip(
                fields=["lot_number", "area_label"],
                aliases=["Lot", "Superficie"],
                sticky=True,
            ),
            control=True,
        ).add_to(fmap)

    if buildings is not None and buildings.features:
        folium.GeoJson(
            buildings.collection(),
            name=f"Bâtiments ({buildings.count})",
            style_function=lambda _: dict(_BUILDING_STYLE),
            highlight_function=lambda _: {"fillOpacity": 0.8},
            tooltip=folium.GeoJsonTooltip(
                fields=["area_label"],
                aliases=["Empreinte"],
                sticky=True,
            ),
            control=True,
        ).add_to(fmap)

    if selected and selected.get("geometry"):
        folium.GeoJson(
            {
                "type": "Feature",
                "geometry": selected["geometry"],
                "properties": {"lot_number": selected.get("lot_number", "")},
            },
            name="Lot sélectionné",
            style_function=lambda _: dict(_SELECTED_STYLE),
            tooltip=folium.GeoJsonTooltip(fields=["lot_number"], aliases=["Lot"]),
            control=False,
        ).add_to(fmap)
        if selected.get("lat") is not None and selected.get("lon") is not None:
            folium.Marker(
                location=[selected["lat"], selected["lon"]],
                tooltip=f"Lot {selected.get('lot_number', '')}",
                icon=folium.Icon(color="red", icon="info-sign"),
            ).add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)

    if fit_bounds:
        fmap.fit_bounds(fit_bounds)

    return fmap


def decorate(feature_set, layer: str) -> None:
    """Add the display-only properties the tooltips read.

    Folium's ``GeoJsonTooltip`` names fields by key and renders whatever is
    there, so formatting a number for a human has to happen before the map is
    built rather than in a style callback.
    """
    for feature in feature_set.features:
        props = feature["properties"]
        area = props.get("area_m2")
        props["area_label"] = f"{float(area):,.0f} m²" if area else "—"
        attributes = props.get("attributes") or {}
        if layer == "zones":
            props["zone_label"] = (
                attributes.get("NUMERO_COMPLET") or props.get("feature_id") or "—"
            )
        # The raw attribute bag is embedded verbatim in the page by folium, and
        # Infolot carries two dozen columns per lot. Two thousand lots' worth of
        # them is megabytes of HTML nothing on the map reads — the panes query
        # the row again by id when they need it.
        props.pop("attributes", None)


def bounds_of(geometry: dict | None) -> list | None:
    """``[[south, west], [north, east]]`` for a GeoJSON geometry.

    folium's ``fit_bounds`` wants latitude first; GeoJSON stores longitude
    first, which is the swap this exists to get right in one place.
    """
    if not geometry:
        return None
    lons: list[float] = []
    lats: list[float] = []

    def walk(coords) -> None:
        if not coords:
            return
        first = coords[0]
        if isinstance(first, (int, float)):
            lons.append(float(coords[0]))
            lats.append(float(coords[1]))
            return
        for item in coords:
            walk(item)

    walk(geometry.get("coordinates"))
    if not lons:
        return None
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


def pad_bounds(bounds: list, factor: float = 0.35) -> list:
    """Widen a bounding box so a fitted lot is not flush against the frame."""
    (south, west), (north, east) = bounds
    dy = max((north - south) * factor, 1e-4)
    dx = max((east - west) * factor, 1e-4)
    return [[south - dy, west - dx], [north + dy, east + dx]]
