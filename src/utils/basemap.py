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
import os
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
#: A massing is a building-sized rectangle, so it earns the same gate as a
#: footprint. It is also the layer read *against* the footprints - the proposal
#: over what stands - and showing one without the other would be half the
#: comparison.
MIN_MASSING_ZOOM = 16

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

#: The proposal, and the one layer whose colour carries a *finding* rather than
#: an identity. A massing the solver's footprint fits into is drawn in the
#: green; one that had to be shrunk to fit its own setback envelope is drawn in
#: the amber, because that lot is the interesting one - the solved footprint
#: has no shape the parcel can take, and the whole reason to put this layer on
#: a map is to see those without querying for them.
#:
#: Green and amber rather than green and red: a shrunk massing is a fact about
#: the parcel worth looking at, not an error, and red on a map is read as one.
_MASSING_FITTED_STYLE = {
    "color": "#1b512d",
    "weight": 1.5,
    "fillColor": "#40916c",
    "fillOpacity": 0.55,
}
_MASSING_SHRUNK_STYLE = {
    "color": "#9c6412",
    "weight": 1.5,
    "fillColor": "#e9a13b",
    "fillOpacity": 0.55,
    "dashArray": "5,3",
}


def _massing_style(feature: dict) -> dict:
    """Green where the footprint fits, amber where it had to be shrunk."""
    status = (feature.get("properties") or {}).get("massing_status")
    base = _MASSING_SHRUNK_STYLE if status == "shrunk" else _MASSING_FITTED_STYLE
    return dict(base)


#: Utilisation is a lot-sized question, so it takes the lot gate rather than
#: the building one: the shading is read across a block at a glance, and at
#: zoom 16 too little of the block is on screen for the comparison to mean
#: anything.
MIN_CAPACITY_ZOOM = MIN_LOT_ZOOM

#: How much of the permitted floor is standing, banded. Sequential rather than
#: categorical, because the underlying quantity is continuous and ordered - a
#: reader should be able to see "emptier" without consulting a legend.
#:
#: The bands are not even fifths. The interesting end is the low one, where a
#: parcel holds a fraction of what it is zoned for, so the classes are narrow
#: there and widen as they approach capacity; a lot at 70% and one at 85% are
#: the same finding for this map's purpose and do not earn separate colours.
#:
#: `over` is deliberately its own colour rather than the top of the ramp. A
#: building larger than today's zoning would allow is not the maximally
#: efficient case - it is a legal non-conformity, very common in a borough
#: whose housing predates its by-law, and reading it as "best" would invert the
#: map. Purple sits outside the ramp so it cannot be mistaken for one end of it.
_CAPACITY_BANDS = (
    (25.0, "#08519c", "moins de 25 %"),
    (50.0, "#3182bd", "25 – 50 %"),
    (75.0, "#6baed6", "50 – 75 %"),
    (95.0, "#bdd7e7", "75 – 95 %"),
    (float("inf"), "#eff3ff", "95 – 100 %"),
)
_CAPACITY_OVER_COLOR = "#7b3294"
#: No solved programme, so no denominator and no finding. Grey, and it means
#: "not answered" rather than "not used" - see gold.lot_highest_best_use's
#: hbu_status for the five reasons a lot lands here.
_CAPACITY_NONE_COLOR = "#bfbfbf"


def _capacity_style(feature: dict) -> dict:
    """Shade a lot by the share of its permitted floor that is standing."""
    props = feature.get("properties") or {}
    used = props.get("used_pct")
    if used is None:
        fill, opacity = _CAPACITY_NONE_COLOR, 0.30
    elif float(used) > 100.0:
        fill, opacity = _CAPACITY_OVER_COLOR, 0.55
    else:
        fill, opacity = _CAPACITY_NONE_COLOR, 0.65
        for upper, color, _ in _CAPACITY_BANDS:
            if float(used) < upper:
                fill = color
                break
    return {
        "color": "#4a4a4a",
        "weight": 0.6,
        "fillColor": fill,
        "fillOpacity": opacity,
    }


def capacity_legend_rows() -> list[tuple[str, str]]:
    """(colour, label) for the pane that draws the legend beside the map.

    Lives here rather than in `app.py` so the swatches and the style callback
    cannot drift apart - they read the same tuple.
    """
    rows = [(color, label) for _, color, label in _CAPACITY_BANDS]
    rows.append((_CAPACITY_OVER_COLOR, "plus que le zonage permet"))
    rows.append((_CAPACITY_NONE_COLOR, "aucun programme calculé"))
    return rows


_SELECTED_STYLE = {
    "color": "#d62828",
    "weight": 4,
    "fillColor": "#f77f00",
    "fillOpacity": 0.35,
}


def _mapbox_token() -> str | None:
    """The configured Mapbox token, or None.

    The deployed task always injects ``MAPBOX_TOKEN``; an unset secret arrives
    as the literal ``PLACEHOLDER`` Terraform wrote. Both mean "no token" — the
    same convention ``auth.py`` uses for the access password.
    """
    token = os.getenv("MAPBOX_TOKEN", "").strip()
    if not token or token == "PLACEHOLDER":
        return None
    return token


def _use_mapbox() -> bool:
    provider = os.getenv("MAP_TILE_PROVIDER", "auto").strip().lower()
    if provider == "mapbox":
        return True
    if provider in {"osm", "openstreetmap"}:
        return False
    return _mapbox_token() is not None  # "auto"


def _add_base_tiles(fmap) -> None:
    """Add the basemap the vector layers are drawn over.

    Mapbox when a token is configured: its ``light-v11`` style is the pale,
    low-contrast background parcel lines and footprints read best against, and
    it is what replaced ``CartoDB positron`` — which now needs a Carto account.
    Plain OpenStreetMap otherwise, so a local run needs no key at all.
    """
    import folium  # noqa: PLC0415

    token = _mapbox_token()
    if _use_mapbox() and token:
        style = os.getenv("MAPBOX_STYLE", "mapbox/light-v11").strip("/")
        folium.TileLayer(
            tiles=(
                f"https://api.mapbox.com/styles/v1/{style}/tiles/512/"
                "{z}/{x}/{y}@2x?access_token=" + token
            ),
            attr="© Mapbox © OpenStreetMap",
            name="Mapbox",
            # 512-px retina tiles align with Leaflet's 256 grid only with this
            # offset; without it every label sits half a zoom too large.
            tile_size=512,
            zoom_offset=-1,
            max_zoom=19,
            overlay=False,
            control=True,
        ).add_to(fmap)
        folium.TileLayer(
            tiles=(
                "https://api.mapbox.com/v4/mapbox.satellite/"
                "{z}/{x}/{y}@2x.jpg90?access_token=" + token
            ),
            attr="© Mapbox © Maxar",
            name="Satellite",
            max_zoom=19,
            overlay=False,
            control=True,
            show=False,
        ).add_to(fmap)
    else:
        folium.TileLayer("OpenStreetMap", overlay=False, control=True).add_to(fmap)


def build_map(
    *,
    center: tuple[float, float] = DEFAULT_CENTER,
    zoom: int = DEFAULT_ZOOM,
    lots: Any = None,
    buildings: Any = None,
    zones: Any = None,
    capacity: Any = None,
    massing: Any = None,
    selected: dict | None = None,
    fit_bounds: list | None = None,
):
    """Assemble the map. Every layer argument is a ``FeatureSet``.

    Draw order is the argument order below, and it is a decision: zones
    underneath, then lots, then the footprints standing today, then the
    proposed massing on top of them. The proposal goes last because it is
    what the map is being read for - a massing hidden under the building it
    would replace answers nothing.

    ``capacity`` shades the parcels themselves and so goes directly above the
    zones and below everything else: it is a property *of* the lot rather than
    an object standing on it, and a footprint drawn underneath its own lot's
    shading would be invisible.
    """
    import folium  # noqa: PLC0415

    fmap = folium.Map(
        location=list(center),
        zoom_start=zoom,
        tiles=None,  # added by _add_base_tiles so the provider is swappable
        control_scale=True,
        # The zoom-gated layers make a hard-zoomed-out view meaningless, and
        # the corpus only covers one borough anyway.
        min_zoom=11,
        max_zoom=19,
        prefer_canvas=True,
    )
    _add_base_tiles(fmap)

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

    if capacity is not None and capacity.features:
        folium.GeoJson(
            capacity.collection(),
            name=f"Utilisation ({capacity.count})",
            style_function=_capacity_style,
            highlight_function=lambda _: {"weight": 2.5, "color": "#ee6c4d"},
            tooltip=folium.GeoJsonTooltip(
                fields=["lot_number", "used_label", "headroom_label"],
                aliases=["Lot", "Utilisé", "Encore constructible"],
                sticky=True,
            ),
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

    if massing is not None and massing.features:
        folium.GeoJson(
            massing.collection(),
            name=f"Massing proposé ({massing.count})",
            style_function=_massing_style,
            highlight_function=lambda _: {"fillOpacity": 0.85, "weight": 2.5},
            tooltip=folium.GeoJsonTooltip(
                fields=["lot_number", "massing_label", "fit_label"],
                aliases=["Lot", "Proposé", "Empreinte"],
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
        if layer == "capacity":
            used = props.get("used_pct")
            status = props.get("hbu_status")
            if used is None:
                # Why there is no percentage, rather than a blank. The five
                # statuses are gold.lot_highest_best_use's own.
                props["used_label"] = {
                    "no_candidate_column": "zone sans usage valorisable",
                    # The former name of no_candidate_column, from when the
                    # solver priced dwellings alone; rows written before the
                    # rename still carry it.
                    "no_residential_column": "zone sans volet résidentiel",
                    "no_governing_column": "aucune colonne applicable",
                    "infeasible": "aucun programme réalisable",
                    "solver_error": "erreur de résolution",
                }.get(status, "non calculé")
            else:
                built = props.get("existing_floor_area_m2")
                permitted = props.get("hbu_floor_area_m2")
                shown = f"{float(used):,.0f} %"
                if permitted:
                    shown += (
                        f" ({float(built or 0):,.0f} / {float(permitted):,.0f} m²)"
                    )
                props["used_label"] = shown
            # The three classes summed, then the dwellings named separately:
            # "12 000 pi² de plus" and "14 logements de plus" are the two units
            # this question actually gets asked in.
            headroom = sum(
                float(props.get(key) or 0)
                for key in (
                    "residential_headroom_m2",
                    "commercial_headroom_m2",
                    "industrial_headroom_m2",
                )
            )
            if headroom <= 0:
                props["headroom_label"] = "—"
            else:
                parts = [f"{headroom:,.0f} m² ({headroom * 10.7639:,.0f} pi²)"]
                gap = props.get("dwelling_gap")
                if gap is None and props.get("hbu_num_dwellings") is not None:
                    gap = int(props["hbu_num_dwellings"]) - int(
                        props.get("existing_num_dwellings") or 0
                    )
                if gap and int(gap) > 0:
                    parts.append(f"{int(gap)} logements")
                props["headroom_label"] = " · ".join(parts)

        if layer == "massing":
            floors = props.get("floors")
            dwellings = props.get("num_dwellings")
            commercial = props.get("commercial_floors") or 0
            parts = []
            if floors:
                parts.append(f"{int(floors)} étages")
            if dwellings:
                parts.append(f"{int(dwellings)} logements")
            if commercial:
                parts.append(f"{int(commercial)} étages comm.")
            props["massing_label"] = " · ".join(parts) or "—"
            # The sanity check, in the tooltip: what was solved, what could be
            # drawn, and the share. A reader hovering a shrunk massing sees why
            # it is amber without opening the table.
            placed = props.get("placed_footprint_m2")
            fit = props.get("footprint_fit_pct")
            if placed is None:
                props["fit_label"] = "—"
            elif fit is not None and float(fit) < 99.5:
                props["fit_label"] = (
                    f"{float(placed):,.0f} m² — {float(fit):.0f} % du solvé"
                )
            else:
                props["fit_label"] = f"{float(placed):,.0f} m²"
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
