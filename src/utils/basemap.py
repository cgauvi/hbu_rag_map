"""
basemap.py — Building the folium map the Streamlit pane renders.

One function assembles every layer, because the layers have to agree about
draw order and about what a click means. Zones are drawn first and lots on top
of them: a click lands on the smallest shape under the cursor, and the lot is
what the user is after. Buildings sit on top of lots as fills, since a
footprint is read as a mass rather than as an outline.

**Two renderers draw that stack, and only one of them scales.**

*Vector tiles* — the default, and what `tile_layers` selects. Each layer is a
`L.vectorGrid.protobuf` pointed at this process's own tile server, so the page
carries six URLs instead of six collections of shapes and the browser fetches,
draws and discards geometry by the tileful as the user pans. Nothing about a
layer's *size* reaches Python at all.

*GeoJSON* — what `lots`, `buildings`, `zones`, `capacity`, `streets` and
`massing` still accept, kept for ``HBU_MAP_RENDERER=geojson`` and for the case
where the tile server could not bind its port. It embeds every coordinate in the document, so
it is bounded by ``HBU_MAP_FEATURE_LIMIT`` and it is the shape that could not
draw a borough.

The two are meant to look identical, and the way that is arranged is that they
share their constants rather than their code: one set of colours, one set of
zoom gates, one set of band thresholds, read by the Python style callbacks
below and *serialised into* the JavaScript ones. The one thing that is
genuinely written twice is the tooltip text — `decorate` builds it in Python
for the GeoJSON path and `_TOOLTIP_JS` builds it in the browser for the tile
path, because with tiles the feature only ever exists there. Change one and
change the other; they are adjacent in this file for that reason.

Nothing here queries. The GeoJSON path takes ``FeatureSet``s the caller has
already fetched and cached; the tile path takes URLs.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

# One of the two imports in this module that is not stdlib, and the reason is
# the zoom thresholds below: they are a property of the data rather than of the
# drawing, so `queries` owns them and this reads them. No cycle - `queries`
# imports only `db` - and nothing here calls it.
from src.utils import queries

logger = logging.getLogger(__name__)

#: Montreal, Villeray–Saint-Michel–Parc-Extension. Where the map opens when
#: nothing has been selected and the database has no extent to offer.
DEFAULT_CENTER = (45.5535, -73.6200)
DEFAULT_ZOOM = 15

#: Below these a layer stops drawing its own features. A lot is sub-pixel at
#: zoom 13 and a borough's worth of them is a solid grey rectangle that costs a
#: second of browser time to produce.
#:
#: **They are no longer where a layer stops.** Under the tile renderer, below
#: its detail zoom a layer is drawn from `gold.map_cell_aggregates` instead -
#: the same features, dissolved onto the tile grid, one shape per cell - so
#: what these now mark is the zoom where a *summary* becomes the parcels
#: themselves. The GeoJSON path has no aggregate to fall back to and still
#: treats them as a floor.
#:
#: Read from `queries` rather than declared here: they are a fact about how
#: dense the data is, and that module is the one that has to route on them.
#: Re-exported under these names because the notes and the legend on this
#: side have always called them this.
MIN_LOT_ZOOM = queries.MVT_DETAIL_ZOOM["lots"]
MIN_BUILDING_ZOOM = queries.MVT_DETAIL_ZOOM["buildings"]
#: A massing is a building-sized rectangle, so it earns the same gate as a
#: footprint. It is also the layer read *against* the footprints - the proposal
#: over what stands - and showing one without the other would be half the
#: comparison.
MIN_MASSING_ZOOM = queries.MVT_DETAIL_ZOOM["massing"]
#: Two zooms below the lots, and the reason is what this layer is for. A street
#: grid is the thing that says *where you are* before any parcel is legible, so
#: it earns a gate low enough to be on screen while the reader is still finding
#: the block. It is also cheap to draw at that zoom: a borough holds a few
#: thousand sides against Villeray's twenty-five thousand lots, and a line
#: quantised onto the tile grid is a handful of vertices.
MIN_STREET_ZOOM = queries.MVT_DETAIL_ZOOM["streets"]

#: How far out the map may be zoomed, and now also how far the aggregates have
#: to reach: every tile layer is requested from here up, so the dataplatform's
#: coarsest cell level has to cover this zoom. See `queries.AGGREGATE_CELL_ZOOMS`.
MAP_MIN_ZOOM = 11

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

#: The street sides. A teal deliberately outside every other ramp on this map
#: — the utilisation blues, the zone orange, the massing green and amber — so a
#: line crossing a shaded lot is never read as part of the shading.
#:
#: These are *sides*, so a street appears twice, once per curb, a few metres
#: apart. That doubling is the layer telling the truth about its own grain:
#: `silver.lot_frontage` joins a lot to one side, and a centre line could not
#: say which. The weight is kept at 2 so the pair stays legible as a pair
#: rather than merging into one fat stroke at zoom 15.
_STREET_STYLE = {
    "color": "#137a7f",
    "weight": 2,
    "opacity": 0.9,
    # The one layer here whose geometry is open. Leaflet fills a path by
    # closing it across its two ends, so a filled street side paints a wedge
    # across the block rather than a line along the curb — under either
    # renderer, which is why the flag lives in the shared style rather than in
    # one of the two callbacks.
    "fill": False,
}
_STREET_HIGHLIGHT = {"weight": 5, "color": "#ee6c4d", "opacity": 1.0}

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
    (25.0, "#08519c", "under 25%"),
    (50.0, "#3182bd", "25 – 50%"),
    (75.0, "#6baed6", "50 – 75%"),
    (95.0, "#bdd7e7", "75 – 95%"),
    (float("inf"), "#eff3ff", "95 – 100%"),
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
    rows.append((_CAPACITY_OVER_COLOR, "more than zoning permits"))
    rows.append((_CAPACITY_NONE_COLOR, "no programme solved"))
    return rows



# ---------------------------------------------------------------------------
# The low-zoom cells
#
# Below its detail zoom a layer is drawn from `gold.map_cell_aggregates`: one
# shape per tile-grid cell, carrying the count of features in it and one
# `value` whose meaning `value_kind` names. Two decisions here are worth
# stating, because both could reasonably have gone the other way.
#
# **A cell keeps its layer's colour.** The alternative is a ramp per layer,
# which is four more palettes to pick, four more legends to draw and four more
# things that can collide with the utilisation blues. Instead each layer shades
# its own fill by opacity, so a dense cell of lots is the lots' blue at full
# strength and a sparse one is the same blue faint - the layer is still
# identifiable at a glance, which at this zoom is most of what the colour is
# for.
#
# **Utilisation is the exception, and gets no new styling at all.** Its `value`
# *is* `used_pct`, on the same 0-100 scale the per-lot shading already uses, so
# the aggregate goes through `_CAPACITY_BANDS` unchanged. One palette, one
# legend, and a cell at 40% is the same blue as a lot at 40% - which is the
# property that makes zooming in feel like the same map rather than a different
# one.
# ---------------------------------------------------------------------------

#: The value each layer's ramp saturates at, in that layer's own units. A cell
#: at or above this is drawn at full opacity.
#:
#: Fixed rather than taken from the partition's own maximum, and that is the
#: whole point of writing them down: a scale computed per borough would make
#: two boroughs incomparable, and would make one borough change colour when a
#: single outlying cell appeared or went away. The numbers are judgement,
#: pitched a little above what a dense Villeray block actually reaches so that
#: saturating is a statement rather than the normal case:
#:
#: * ``lots`` - Villeray runs about 1 500 lots/km2 over its built blocks.
#: * ``buildings`` - the share of the ground under a footprint; a dense
#:   Montreal block sits near 50%, and above 60% there is no open space left.
#: * ``massing`` - proposed dwellings per hectare. A five-storey walk-up block
#:   is roughly 150; 300 is a decidedly denser proposal than this solver makes.
#: * ``streets`` - kilometres of street *side* per km2, so a grid counts twice.
_AGGREGATE_VALUE_MAX = {
    "lots": 2000.0,
    "buildings": 60.0,
    "massing": 300.0,
    "streets": 40.0,
}

#: The opacity a cell is drawn at, from a `value` of nothing to one at the
#: maximum above. The floor is not zero: a cell that *has* the layer in it but
#: barely any should still be visible as covered ground, because the thing a
#: reader is looking for at this zoom is often the gap - where the cadastre
#: stops, where the massing found nothing - and an invisible cell and an absent
#: one would look the same.
_AGGREGATE_MIN_OPACITY = 0.15
_AGGREGATE_MAX_OPACITY = 0.75

#: A cell whose `value` is NULL. It means "not answered here" - no feature of
#: its own, or a denominator of zero - and it is the same grey the per-lot
#: shading uses for a lot with no solved programme, for the same reason: it is
#: not a low value, it is the absence of one.
_AGGREGATE_NONE_COLOR = _CAPACITY_NONE_COLOR


#: What each layer's `value` is measured in. Said once, in the legend's
#: heading, rather than repeated on every swatch - which is both how a map
#: legend is normally written and the only way the top row reads properly:
#: "2 000 and over" under a heading of "lots/km\u00b2" says what "2 000 lots/km\u00b2
#: and over" has to fight its own word order to.
#:
#: The server says the same thing in `value_kind` on every row and the tooltip
#: reads it from there; this is the Python side of that one vocabulary.
_AGGREGATE_UNITS = {
    "lots": "lots per km\u00b2",
    "buildings": "% of the ground built on",
    "massing": "proposed dwellings per hectare",
    "streets": "km of street side per km\u00b2",
}


def aggregate_legend_units(layer: str) -> str:
    """What ``layer``'s cell numbers are in, for the legend's heading."""
    return _AGGREGATE_UNITS[layer]

#: How many steps the legend shows between nothing and the saturation point.
#: Four, because the ramp is continuous and any number here is a sampling of
#: it - few enough to read at a glance, enough to show that it *is* a ramp.
_AGGREGATE_LEGEND_STEPS = 4


def _rgba(hex_color: str, alpha: float) -> str:
    """``#rrggbb`` at ``alpha``, as the CSS the legend swatch needs.

    The map varies a cell's *opacity* rather than its hue, so a legend of flat
    hex swatches would draw four identical squares against four different
    numbers. This is what makes the swatch show what the map shows.
    """
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha:.2f})"


def aggregate_legend_rows(layer: str) -> list[tuple[str, str]]:
    """(CSS colour, label) for the cells of ``layer``, for the pane's legend.

    Sampled off the same two constants the style function interpolates
    between, so the swatches are the opacities the map actually draws rather
    than an approximation somebody chose to look right.

    The utilisation layer is deliberately absent: its cells are shaded by
    `_CAPACITY_BANDS` like its lots, so `capacity_legend_rows` already
    describes them, and a second legend saying the same thing in different
    words would be the one that goes stale.
    """
    if layer not in _AGGREGATE_VALUE_MAX:
        raise KeyError(f"{layer!r} has no aggregate ramp")
    fill = _AGGREGATE_COLOR[layer]
    top = _AGGREGATE_VALUE_MAX[layer]
    span = _AGGREGATE_MAX_OPACITY - _AGGREGATE_MIN_OPACITY

    rows: list[tuple[str, str]] = []
    for step in range(_AGGREGATE_LEGEND_STEPS):
        share = step / (_AGGREGATE_LEGEND_STEPS - 1)
        # Bare numbers: `aggregate_legend_units` names what they are in,
        # once, above them. A narrow no-break space groups the thousands,
        # which is the convention the rest of this pane already follows.
        label = f"{share * top:,.0f}".replace(",", "\u202f")
        if step == _AGGREGATE_LEGEND_STEPS - 1:
            label += " and over"
        rows.append((_rgba(fill, _AGGREGATE_MIN_OPACITY + share * span), label))
    # Not the bottom of the ramp - see the column comment in hbu_infra's
    # sql/023. A cell with no feature of its own is unanswered, not empty.
    rows.append((_rgba(_AGGREGATE_NONE_COLOR, 0.25), "not answered here"))
    return rows


#: The colour each layer's cells are drawn in - its own, so a layer is still
#: identifiable at a glance when it is a density surface rather than a set of
#: parcels. The streets take their *stroke* colour because that is the only
#: colour a line layer has.
_AGGREGATE_COLOR = {
    "lots": _LOT_STYLE["fillColor"],
    "buildings": _BUILDING_STYLE["fillColor"],
    "massing": _MASSING_FITTED_STYLE["fillColor"],
    "streets": _STREET_STYLE["color"],
}


def _aggregate_style_js(layer: str) -> str:
    """The style for one layer's cells, as JavaScript.

    Utilisation reuses the band function rather than the ramp - see the header
    above - so this is only ever called for the other four.

    **The streets branch is not a special case for tidiness.** A cell of that
    layer is the dissolved *linework* inside it, not a polygon covering it, and
    a fill on an open geometry paints nothing at all - so a shared fill-only
    style would draw an empty map and look exactly like a layer with no data.
    The same asymmetry `_STREET_STYLE`'s `fill: False` already carries, one
    zoom band further out.
    """
    color = _AGGREGATE_COLOR[layer]
    top = _AGGREGATE_VALUE_MAX[layer]
    span = _AGGREGATE_MAX_OPACITY - _AGGREGATE_MIN_OPACITY
    if layer == "streets":
        drawn = f"""{{
                fill: false,
                stroke: true,
                color: {_js(color)},
                // Thinner than the detail layer's 2. At this zoom the two
                // sides of a street are the same pixel, so a heavier stroke
                // would only make the grid bleed into a solid wash.
                weight: 1,
                opacity: {_AGGREGATE_MIN_OPACITY} + share * {span}
            }}"""
        blank = f"""{{
                fill: false, stroke: true,
                color: {_js(_AGGREGATE_NONE_COLOR)}, weight: 1, opacity: 0.25
            }}"""
    else:
        drawn = f"""{{
                fill: true,
                // No stroke, and it is not cosmetic: cells tile the ground
                // edge to edge, so any outline draws a grid over the borough
                // that reads as data rather than as the mesh it is.
                stroke: false,
                weight: 0,
                fillColor: {_js(color)},
                fillOpacity: {_AGGREGATE_MIN_OPACITY} + share * {span}
            }}"""
        blank = f"""{{
                fill: true, stroke: false, weight: 0,
                fillColor: {_js(_AGGREGATE_NONE_COLOR)}, fillOpacity: 0.25
            }}"""
    return f"""(function (properties) {{
            var value = properties.value;
            if (value === null || value === undefined) {{
                return {blank};
            }}
            var share = Math.max(0, Math.min(1, value / {top}));
            return {drawn};
        }})"""


# ---------------------------------------------------------------------------
# The tile renderer
#
# Everything below turns the constants above into the JavaScript Leaflet needs,
# and it is deliberately generated rather than written out. A colour that lives
# in `_CAPACITY_BANDS` and again in a hand-written style function is a colour
# that will disagree with itself the first time somebody changes one of them —
# and the way that failure shows up is a legend that no longer matches the map,
# which is worse than a broken map because it still looks like an answer.
# ---------------------------------------------------------------------------

#: Draw order, bottom to top, and the same order `build_map` adds the GeoJSON
#: layers in: zones underneath, then the shading, then lots, then what stands
#: today, then the proposal on top of it. The proposal goes last because it is
#: what the map is read for — a massing hidden under the building it would
#: replace answers nothing.
#:
#: Streets sit above the two area washes and below the cadastre, which is a
#: decision about *clicks* as much as about paint. Above the washes, because a
#: hairline under a 65 %-opaque utilisation band is not a line anybody can
#: follow. Below the lots, because a click on this map means "select the lot
#: under the cursor" — and an interactive line layer on top would swallow that
#: click along every frontage, which is the one place a reader is most likely
#: to aim.
TILE_LAYER_ORDER: tuple[str, ...] = (
    "zones",
    "capacity",
    "streets",
    "lots",
    "buildings",
    "massing",
)

#: The name each layer gets in the layer control, and the zoom it starts
#: drawing at. The gate is the same one the GeoJSON path applies in `app.py`,
#: moved into the browser: Leaflet simply does not request a tile below
#: ``minZoom``, so crossing the threshold costs no rerun and no query.
TILE_LAYER_NAMES = {
    "zones": "Zoning",
    "capacity": "Utilisation",
    # Named for the grain rather than for the thing: these are the two sides of
    # a street, not its centre line, and a reader who does not know that reads
    # the doubled lines as a rendering fault.
    "streets": "Street sides",
    "lots": "Lots",
    "buildings": "Buildings",
    "massing": "Proposed massing",
}

#: What Leaflet is told, which is no longer the same thing as the detail zoom
#: above. Every layer is now requested all the way down to the map's own floor,
#: because below its detail zoom the server answers with dissolved cells rather
#: than with nothing. A `minZoom` of 15 here would mean Leaflet never asked, and
#: the aggregates would sit in the table unread.
#:
#: The threshold has not gone away - it has moved to the one place that can act
#: on it, `queries.serves_aggregate`, which decides per request which of the two
#: tables answers. That keeps one Leaflet layer per map layer across the
#: boundary: one entry in the control, one visibility flag, and no remount when
#: the reader crosses it.
TILE_LAYER_MIN_ZOOM = {
    "zones": 0,
    "capacity": MAP_MIN_ZOOM,
    "streets": MAP_MIN_ZOOM,
    "lots": MAP_MIN_ZOOM,
    "buildings": MAP_MIN_ZOOM,
    "massing": MAP_MIN_ZOOM,
}

#: Which tile property identifies a feature. VectorGrid needs one to hold a
#: hover highlight, because unlike a GeoJSON layer it has no Leaflet object per
#: shape to restyle — it restyles by id and redraws the tile.
_TILE_FEATURE_ID = {
    "zones": "feature_id",
    "capacity": "lot_uid",
    # The publisher's own key for a street side, unique across the island, so
    # a hover holds the side it landed on rather than the whole street.
    "streets": "cote_rue_id",
    "lots": "lot_uid",
    "buildings": "building_uid",
    "massing": "lot_uid",
}



def _js(value: Any) -> str:
    """A Python value as the JavaScript literal for it."""
    return json.dumps(value)


def _capacity_bands_js() -> str:
    """`_CAPACITY_BANDS` as a JS array, with the open top edge as null.

    ``float('inf')`` has no JSON spelling, and the alternative — writing a
    large number — is the kind of thing that works until a percentage goes
    past it.
    """
    bands = [
        [None if upper == float("inf") else upper, color]
        for upper, color, _ in _CAPACITY_BANDS
    ]
    return _js(bands)


#: The style callbacks, as JavaScript. Each mirrors the Python function of the
#: same name above and reads the same constants, interpolated in.
#:
#: Every one of the five aggregated layers is wrapped by `_with_aggregate_js`
#: below, because one Leaflet layer now draws two kinds of feature: its own
#: below `TILE_LAYER_MIN_ZOOM`... and, below its *detail* zoom, the dissolved
#: cells the server substitutes. The branch is on `agg_level`, a property only
#: a cell carries.
def _style_js(layer: str) -> str:
    return _with_aggregate_js(layer, _detail_style_js(layer))


def _with_aggregate_js(layer: str, detail: str) -> str:
    """``detail`` guarded by the cell branch, for a layer that has cells.

    The cheapest possible discriminator - a property that is present or is not
    - rather than passing the zoom in. The style function is called per
    feature by VectorGrid and has no view state to consult, and a zoom
    threshold evaluated in two places is a threshold that will disagree with
    itself the first time one of them is tuned. The tile itself says which kind
    it is, which is the only answer that cannot be stale.
    """
    if layer not in queries.AGGREGATE_LAYERS:
        return detail
    # Utilisation shades its cells with the same bands as its lots, because its
    # `value` is `used_pct` on the same scale. So the cell is handed to the
    # detail function with that property filled in, and there is exactly one
    # capacity palette in this file.
    if layer == "capacity":
        cell = f"""({detail})(
                Object.assign({{}}, properties, {{used_pct: properties.value}})
            )"""
    else:
        cell = f"({_aggregate_style_js(layer)})(properties)"
    return f"""function (properties) {{
            if (properties.agg_level !== null
                && properties.agg_level !== undefined) {{
                return {cell};
            }}
            return ({detail})(properties);
        }}"""


def _detail_style_js(layer: str) -> str:
    if layer == "capacity":
        return f"""function (properties) {{
            var used = properties.used_pct;
            var fill = {_js(_CAPACITY_NONE_COLOR)};
            var opacity = 0.30;
            if (used !== null && used !== undefined) {{
                if (used > 100.0) {{
                    fill = {_js(_CAPACITY_OVER_COLOR)};
                    opacity = 0.55;
                }} else {{
                    opacity = 0.65;
                    var bands = {_capacity_bands_js()};
                    for (var i = 0; i < bands.length; i++) {{
                        if (bands[i][0] === null || used < bands[i][0]) {{
                            fill = bands[i][1];
                            break;
                        }}
                    }}
                }}
            }}
            return {{
                fill: true, color: "#4a4a4a", weight: 0.6,
                fillColor: fill, fillOpacity: opacity
            }};
        }}"""
    if layer == "massing":
        return f"""function (properties) {{
            var shrunk = properties.massing_status === "shrunk";
            var style = shrunk ? {_js(_MASSING_SHRUNK_STYLE)}
                               : {_js(_MASSING_FITTED_STYLE)};
            return Object.assign({{fill: true}}, style);
        }}"""
    base = {
        "zones": _ZONE_STYLE,
        "streets": _STREET_STYLE,
        "lots": _LOT_STYLE,
        "buildings": _BUILDING_STYLE,
    }[layer]
    # `fill` is true by default because five of the six layers are polygons.
    # The base style is assigned *over* that default rather than under it, so
    # the one layer that is a line — whose style carries `fill: false` — wins.
    return f"""function () {{
            return Object.assign({{fill: true}}, {_js(base)});
        }}"""


#: What a hover does to each layer, mirroring the Python `highlight_function`s.
_TILE_HIGHLIGHT = {
    "zones": {"weight": 3, "fillOpacity": 0.25},
    "capacity": {"weight": 2.5, "color": "#ee6c4d"},
    # A line has no fill to brighten, so the hover has to be the stroke itself.
    "streets": _STREET_HIGHLIGHT,
    "lots": _LOT_HIGHLIGHT,
    "buildings": {"fillOpacity": 0.8},
    "massing": {"fillOpacity": 0.85, "weight": 2.5},
}


#: The tooltip, in the browser.
#:
#: This is `decorate` written a second time, and the duplication is real rather
#: than accidental: with tiles the feature never exists in Python, so the only
#: place its label can be built is where it is drawn. The rules are the same
#: ones, including the five ``hbu_status`` reasons a lot has no percentage, and
#: the number formatting is `en-CA` rather than Python's ``,`` grouping —
#: the same grouping, in the locale the rest of these labels are written in.
#:
#: A raw string, so the ``\\uXXXX`` escapes below reach the browser as
#: JavaScript escapes rather than as the characters themselves. The labels are
#: English but the units are not ASCII — m², the em dash — this script
#: travels inside an ``srcdoc`` iframe that `streamlit_folium` builds, and
#: pure-ASCII source is the one form that cannot be mangled by a charset guess
#: anywhere on that path.
_TOOLTIP_JS = r"""
var hbuNumber = new Intl.NumberFormat('en-CA', {maximumFractionDigits: 0});

function hbuBlank(value) {
    return value === null || value === undefined || value === '';
}

function hbuArea(value) {
    return hbuBlank(value) ? '\u2014' : hbuNumber.format(value) + ' m\u00b2';
}

function hbuLength(value) {
    return hbuBlank(value) ? '\u2014' : hbuNumber.format(value) + ' m';
}

/* An unnamed service lane is a real street side, not a missing name, so it is
   labelled rather than blanked. */
function hbuStreetLabel(p) {
    return hbuBlank(p.street_name) ? 'unnamed lane' : p.street_name;
}

var HBU_HBU_STATUS = {
    'no_candidate_column': 'no use the solver prices is zoned here',
    'no_residential_column': 'no residential column',
    'no_governing_column': 'no governing column',
    'infeasible': 'no feasible programme',
    'solver_error': 'solver error'
};

function hbuUsedLabel(p) {
    if (hbuBlank(p.used_pct)) {
        return HBU_HBU_STATUS[p.hbu_status] || 'not solved';
    }
    var shown = hbuNumber.format(p.used_pct) + '%';
    if (!hbuBlank(p.hbu_floor_area_m2) && p.hbu_floor_area_m2) {
        shown += ' (' + hbuNumber.format(p.existing_floor_area_m2 || 0) + ' / '
              + hbuNumber.format(p.hbu_floor_area_m2) + ' m\u00b2)';
    }
    return shown;
}

function hbuHeadroomLabel(p) {
    var total = (p.residential_headroom_m2 || 0)
              + (p.commercial_headroom_m2 || 0)
              + (p.industrial_headroom_m2 || 0);
    if (total <= 0) { return '\u2014'; }
    var parts = [hbuNumber.format(total) + ' m\u00b2 ('
                 + hbuNumber.format(total * 10.7639) + ' sq ft)'];
    var gap = p.dwelling_gap;
    if (hbuBlank(gap) && !hbuBlank(p.hbu_num_dwellings)) {
        gap = p.hbu_num_dwellings - (p.existing_num_dwellings || 0);
    }
    if (!hbuBlank(gap) && gap > 0) { parts.push(Math.round(gap) + ' dwellings'); }
    return parts.join(' \u00b7 ');
}

function hbuMassingLabel(p) {
    var parts = [];
    if (p.floors) { parts.push(Math.round(p.floors) + ' storeys'); }
    if (p.num_dwellings) { parts.push(Math.round(p.num_dwellings) + ' dwellings'); }
    if (p.commercial_floors) {
        parts.push(Math.round(p.commercial_floors) + ' comm. storeys');
    }
    return parts.join(' \u00b7 ') || '\u2014';
}

function hbuFitLabel(p) {
    if (hbuBlank(p.placed_footprint_m2)) { return '\u2014'; }
    var area = hbuArea(p.placed_footprint_m2);
    if (!hbuBlank(p.footprint_fit_pct) && p.footprint_fit_pct < 99.5) {
        return area + ' \u2014 ' + Math.round(p.footprint_fit_pct)
             + '% of the solved footprint';
    }
    return area;
}

/* A cell of `gold.map_cell_aggregates` rather than one of the layer's own
   features. Everything below reads the same four columns whatever the layer
   is, plus that layer's own numbers out of `attributes` - which travels as
   text, because an MVT property is a scalar, and is parsed once per hover
   rather than once per drawn cell. */
function hbuCellAttributes(p) {
    if (!p.attributes) { return {}; }
    try { return JSON.parse(p.attributes) || {}; }
    catch (err) { return {}; }
}

/* What `value` means, spelled for a reader. The units are the server's -
   `value_kind` names them on every row - so this switch is the one place the
   two repositories have to agree on a vocabulary, and a kind it does not know
   is shown as a bare number rather than mislabelled. */
function hbuCellValue(p) {
    if (hbuBlank(p.value)) { return 'not answered here'; }
    var shown = hbuNumber.format(p.value);
    if (p.value_kind === 'lots_per_km2') { return shown + ' lots/km\u00b2'; }
    if (p.value_kind === 'built_coverage_pct') {
        return shown + '% of the ground built on';
    }
    if (p.value_kind === 'used_pct') { return shown + '% of permitted floor'; }
    if (p.value_kind === 'proposed_dwellings_per_ha') {
        return shown + ' dwellings/ha proposed';
    }
    if (p.value_kind === 'street_km_per_km2') {
        return shown + ' km of street side/km\u00b2';
    }
    return shown;
}

/* The rows a cell gets, on top of the two every cell gets. Each layer says the
   one thing its own numbers add that `value` does not: for utilisation that is
   the two floor areas the percentage is a ratio of, since a cell at 60% over
   four lots and one at 60% over forty are different findings. */
function hbuCellExtraRows(layer, p) {
    var a = hbuCellAttributes(p);
    if (layer === 'capacity') {
        var rows = [['Floor', hbuArea(a.existing_floor_area_m2) + ' of '
                     + hbuArea(a.hbu_floor_area_m2)]];
        if (a.dwelling_gap) {
            rows.push(['Dwelling gap', hbuNumber.format(a.dwelling_gap)]);
        }
        if (a.num_underbuilt) {
            rows.push(['Under-built', hbuNumber.format(a.num_underbuilt)
                       + ' of ' + hbuNumber.format(p.feature_count) + ' lots']);
        }
        return rows;
    }
    if (layer === 'lots') {
        return [['Lot area', hbuArea(a.lot_area_m2)]];
    }
    if (layer === 'buildings') {
        return [['Footprint', hbuArea(a.footprint_area_m2)]];
    }
    if (layer === 'massing') {
        var proposed = [];
        if (a.num_dwellings) {
            proposed.push(hbuNumber.format(a.num_dwellings) + ' dwellings');
        }
        if (a.placed_gross_floor_area_m2) {
            proposed.push(hbuArea(a.placed_gross_floor_area_m2));
        }
        var rows2 = [['Proposed', proposed.join(' \u00b7 ') || '\u2014']];
        /* The shrunk count travels because it is the finding the detail layer
           carries per lot: a cell where most massings had to be shrunk is one
           whose parcels cannot take the shape the solver costed. */
        if (a.num_shrunk) {
            rows2.push(['Shrunk to fit', hbuNumber.format(a.num_shrunk)
                        + ' of ' + hbuNumber.format(p.feature_count)]);
        }
        return rows2;
    }
    return [];
}

/* The label for what a cell counts. Plural nouns rather than "features",
   because "44 features" is the map describing its own implementation. */
var HBU_CELL_NOUN = {
    lots: 'lots',
    buildings: 'buildings',
    capacity: 'lots',
    massing: 'proposed buildings',
    streets: 'street sides'
};

function hbuCellRows(layer, p) {
    var noun = HBU_CELL_NOUN[layer] || 'features';
    var rows = [['Summary of', hbuNumber.format(p.feature_count || 0) + ' '
                 + noun]];
    rows = rows.concat(hbuCellExtraRows(layer, p));
    rows.push(['Density', hbuCellValue(p)]);
    /* Said out loud rather than left to be inferred from the zoom: a reader
       who does not know these are cells will read a shaded square as a parcel,
       and the numbers above as facts about it. */
    rows.push(['', 'zoom in for individual ' + noun]);
    return rows;
}

function hbuTooltipRows(layer, p) {
    /* A cell carries `agg_level` and a feature does not - the same
       discriminator the style functions branch on, for the same reason: the
       tile says which it is, and no view state has to be consulted. */
    if (!hbuBlank(p.agg_level)) {
        return hbuCellRows(layer, p);
    }
    if (layer === 'zones') {
        return [['Zone', p.zone_label]];
    }
    if (layer === 'lots') {
        return [['Lot', p.lot_number], ['Area', hbuArea(p.area_m2)]];
    }
    if (layer === 'buildings') {
        return [['Footprint', hbuArea(p.area_m2)]];
    }
    if (layer === 'streets') {
        return [['Street', hbuStreetLabel(p)],
                ['Length', hbuLength(p.length_m)]];
    }
    if (layer === 'capacity') {
        return [['Lot', p.lot_number],
                ['Used', hbuUsedLabel(p)],
                ['Still buildable', hbuHeadroomLabel(p)]];
    }
    if (layer === 'massing') {
        return [['Lot', p.lot_number],
                ['Proposed', hbuMassingLabel(p)],
                ['Footprint', hbuFitLabel(p)]];
    }
    return [];
}

function hbuTooltipHtml(layer, properties) {
    var rows = hbuTooltipRows(layer, properties || {});
    var html = '';
    for (var i = 0; i < rows.length; i++) {
        /* An empty label is a whole-width aside rather than a missing one -
           the cells' "zoom in for individual lots" uses it. Without this it
           would render as a bold colon with nothing before it. */
        if (!rows[i][0]) {
            html += '<div><i>' + (rows[i][1] || '') + '</i></div>';
            continue;
        }
        html += '<div><b>' + rows[i][0] + '</b>: ' + (rows[i][1] || '\u2014')
             + '</div>';
    }
    return html;
}
"""


def _vector_grid_class():
    """folium's VectorGrid plugin, pointed at our own copy of the library.

    The plugin ships an ``@latest`` unpkg URL, and neither half of that is
    survivable here.

    *Not ``@latest``*, because a map whose rendering changes when a CDN
    publishes a release overnight is a map nobody can bisect.

    *Not a CDN*, because of how `streamlit_folium` loads a plugin's
    JavaScript: it awaits every ``default_js`` URL before it renders anything,
    and catches nothing if one rejects. The map's own ``<div>`` is populated
    inside that promise's ``then``, so a script the browser cannot fetch does
    not cost the layers — it costs the whole map, and it does it with no error
    on the page. A third-party host is not a dependency this pane can afford;
    `tiles` serves the library instead, from the origin the tiles come from.

    Subclassed rather than mutated in place because ``default_js`` is a class
    attribute on the plugin: assigning to it would change the URL for anything
    else in the process that draws one, and a test that imported folium first
    would see a different map than one that did not.

    Read at call time rather than at import, because the URL depends on
    ``HBU_TILE_BASE_URL`` and the environment is not necessarily loaded by the
    time this module is.
    """
    from folium.plugins import VectorGridProtobuf  # noqa: PLC0415

    from src.utils import tiles  # noqa: PLC0415

    class _PinnedVectorGrid(VectorGridProtobuf):
        default_js = [("vectorGrid", tiles.vectorgrid_url())]

        def render(self, **kwargs):
            """Drop the `addTo` snippet left by a previous render.

            folium's `Layer.render` adds one child per render, keyed by the
            element's *current* name — and `streamlit_folium` rewrites that
            name to a stable ``div_N`` on its way through the tree. Rendered a
            second time the layer therefore holds two: the live one, and one
            still naming the id folium first stamped. Both are emitted, so the
            page carries
            ``vector_grid_protobuf_<32 hex>.addTo(map_div)`` for a variable
            that was never declared — an uncaught ReferenceError thrown before
            `initComponent`, which is a blank pane rather than an empty layer.

            The rewrite is only reversible on the first pass, because the map
            st_folium keeps from old id to new is rebuilt per render and by
            the second one the old id is not in it. So the stale child is
            dropped here rather than repaired downstream.
            """
            live = f"{self.get_name()}_add"
            for stale in [
                name
                for name in self._children
                if name.endswith("_add") and name != live
            ]:
                del self._children[stale]
            super().render(**kwargs)

    return _PinnedVectorGrid


def _tile_options(layer: str) -> str:
    """The options object for one layer, as the JS string folium passes through.

    A string rather than a dict because two of the six styles are *functions*
    of the feature — the shading band and the fitted/shrunk colour — and a dict
    can only carry data.
    """
    identifier = _TILE_FEATURE_ID[layer]
    return f"""{{
        rendererFactory: L.canvas.tile,
        interactive: true,
        minZoom: {TILE_LAYER_MIN_ZOOM[layer]},
        maxZoom: 19,
        getFeatureId: function (feature) {{
            return feature.properties[{_js(identifier)}];
        }},
        vectorTileLayerStyles: {{
            {_js(layer)}: {_style_js(layer)}
        }}
    }}"""


def _interaction_element(bindings: list[tuple[str, str]]):
    """The one script that gives every tile layer a tooltip and a click.

    ``bindings`` is ``(javascript variable, layer name)`` per layer, in the
    order they were added.

    Two things are going on here, and the second is the load-bearing one.

    **The tooltip** has to be built by hand because a vector tile has no
    Leaflet object per feature for ``bindTooltip`` to attach to; VectorGrid
    hands the properties to an event instead, so one tooltip is moved around
    the map rather than several being bound to shapes.

    **The click has to be forwarded.** With ``interactive: true`` VectorGrid
    calls ``L.DomEvent.fakeStop`` on a click that lands on a feature, which is
    precisely what stops Leaflet firing ``click`` on the map — and the map's
    click is what `streamlit_folium` reports back as ``last_clicked`` and what
    this app resolves into a selected lot. Without the re-fire below, clicking
    a lot would select nothing, and it would fail *only* on the lots: a click
    on empty ground would still work, which is the most confusing possible
    version of the bug.
    """
    from branca.element import MacroElement  # noqa: PLC0415
    from folium.template import Template  # noqa: PLC0415

    class _VectorGridInteraction(MacroElement):
        _template = Template(
            """
            {% macro script(this, kwargs) -%}
            """
            + _TOOLTIP_JS
            + """
            var hbuTip = L.tooltip({sticky: true, direction: 'auto'});

            function hbuBindVectorLayer(grid, layerName, map, highlight) {
                var held = null;
                grid.on('mouseover', function (e) {
                    var props = (e.layer && e.layer.properties) || {};
                    hbuTip.setContent(hbuTooltipHtml(layerName, props))
                          .setLatLng(e.latlng);
                    map.openTooltip(hbuTip);
                    var id = e.layer && e.layer.properties
                        ? grid.options.getFeatureId(e.layer) : null;
                    if (id !== null && id !== undefined) {
                        held = id;
                        grid.setFeatureStyle(id, highlight);
                    }
                });
                grid.on('mousemove', function (e) { hbuTip.setLatLng(e.latlng); });
                grid.on('mouseout', function () {
                    map.closeTooltip(hbuTip);
                    if (held !== null) { grid.resetFeatureStyle(held); held = null; }
                });
                grid.on('click', function (e) {
                    map.fire('click', {
                        latlng: e.latlng,
                        layerPoint: e.layerPoint,
                        containerPoint: e.containerPoint,
                        originalEvent: e.originalEvent
                    });
                });
            }
            {% for grid, layer_name, highlight in this.bindings %}
            hbuBindVectorLayer(
                {{ grid.get_name() }},
                {{ layer_name|tojson }},
                {{ this._parent.get_name() }},
                {{ highlight|tojson }}
            );
            {% endfor %}
            {%- endmacro %}
            """
        )

        def __init__(self, bindings) -> None:
            super().__init__()
            self._name = "VectorGridInteraction"
            self.bindings = bindings

    return _VectorGridInteraction(
        [(grid, layer, dict(_TILE_HIGHLIGHT[layer])) for grid, layer in bindings]
    )


def add_tile_layers(fmap, tile_layers: dict[str, str], visible: dict[str, bool] | None = None):
    """Add one `L.vectorGrid.protobuf` per entry, in draw order.

    ``tile_layers`` maps a layer name to the templated URL Leaflet fills in per
    tile — see `tiles.layer_url`. ``visible`` says which start ticked; a layer
    the user has turned off is still *added*, so the layer control can turn it
    back on without a rerun.
    """
    grid_class = _vector_grid_class()
    bindings: list[tuple[str, str]] = []

    for layer in TILE_LAYER_ORDER:
        url = tile_layers.get(layer)
        if not url:
            continue
        grid = grid_class(
            url,
            name=TILE_LAYER_NAMES[layer],
            options=_tile_options(layer),
            overlay=True,
            control=True,
            show=(visible or {}).get(layer, True),
        )
        grid.add_to(fmap)
        # The element, not its name. `get_name()` is read in the template
        # instead — see `_interaction_element`.
        bindings.append((grid, layer))

    if bindings:
        _interaction_element(bindings).add_to(fmap)
    return fmap


_SELECTED_STYLE = {
    "color": "#d62828",
    "weight": 4,
    "fillColor": "#f77f00",
    "fillOpacity": 0.35,
}


def _draw_selection(parent, selected: dict) -> None:
    """Draw the selected lot onto ``parent`` — a map, or a feature group."""
    import folium  # noqa: PLC0415

    folium.GeoJson(
        {
            "type": "Feature",
            "geometry": selected["geometry"],
            "properties": {"lot_number": selected.get("lot_number", "")},
        },
        name="Selected lot",
        style_function=lambda _: dict(_SELECTED_STYLE),
        tooltip=folium.GeoJsonTooltip(fields=["lot_number"], aliases=["Lot"]),
        control=False,
    ).add_to(parent)
    if selected.get("lat") is not None and selected.get("lon") is not None:
        folium.Marker(
            location=[selected["lat"], selected["lon"]],
            tooltip=f"Lot {selected.get('lot_number', '')}",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(parent)


def selection_layer(selected: dict | None):
    """The selected lot as a ``FeatureGroup``, or None when nothing is.

    Handed to ``st_folium(feature_group_to_add=...)`` rather than built into
    the map, and that is a decision about *reruns* rather than about drawing.
    `streamlit_folium` keys its component on a hash of the map's JavaScript,
    so a shape added to the map object makes a different map: a new key, a
    torn-down iframe, and every basemap and vector tile fetched again. A
    feature group is evaluated into the map already on screen instead, so a
    click paints the outline and moves nothing else.

    None is meaningful rather than merely empty — it is what removes the
    previous selection's group from the map.
    """
    import folium  # noqa: PLC0415

    if not selected or not selected.get("geometry"):
        return None
    group = folium.FeatureGroup(name="Selected lot", control=False)
    _draw_selection(group, selected)
    return group


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


_BASEMAP_STORAGE_KEY = "hbu-map-basemap"


def _basemap_memory(bases: list[tuple[Any, str]]):
    """The script that carries the chosen basemap across a remount.

    ``bases`` is ``(layer, name)`` per base tile layer, in the order they were
    added — the same shape `_interaction_element` takes, and held the same way:
    the element rather than its name, so the template reads the JavaScript
    variable `streamlit_folium` will have renamed by the time it renders.

    Ticking a layer in the sidebar rebuilds the map object, which changes the
    component's key, which throws the iframe away and starts Leaflet over —
    see the note above ``signature`` in `app.py`. Everything the rebuilt map is
    made of, Python knows; which *basemap* was showing, it does not. That is a
    click on Leaflet's own layer control, and it died with the iframe, so the
    satellite view snapped back to the pale street one every time a checkbox
    moved.

    So the browser remembers it instead. ``baselayerchange`` — fired by the
    layer control and by nothing else here, so switching a layer from this
    script cannot feed back into it — writes the name to ``localStorage``, and
    the next map reads it back and switches before it draws. The *name* is what
    is stored because it is the only thing that survives a rebuild: every
    variable in the document is regenerated per render, and the name is what
    the user picked from anyway.

    Storage can throw rather than merely be empty — a browser set to block site
    data — and a map that remembers nothing is exactly the map this was before,
    so every access is guarded and none of them is fatal.
    """
    from branca.element import MacroElement  # noqa: PLC0415
    from folium.template import Template  # noqa: PLC0415

    class _BasemapMemory(MacroElement):
        _template = Template(
            """
            {% macro script(this, kwargs) -%}
            var hbuMap = {{ this._parent.get_name() }};
            var hbuBasemapKey = {{ this.storage_key|tojson }};
            var hbuBasemaps = {
                {%- for layer, name in this.bases %}
                {{ name|tojson }}: {{ layer.get_name() }},
                {%- endfor %}
            };

            function hbuStoredBasemap() {
                try { return window.localStorage.getItem(hbuBasemapKey); }
                catch (e) { return null; }
            }

            // Synchronous, so nothing paints between removing one base and
            // adding the other: the layer being replaced costs a few tile
            // requests, not a visible flash.
            var hbuWanted = hbuStoredBasemap();
            if (hbuWanted && hbuBasemaps[hbuWanted]) {
                Object.keys(hbuBasemaps).forEach(function (name) {
                    var layer = hbuBasemaps[name];
                    if (name === hbuWanted) {
                        if (!hbuMap.hasLayer(layer)) { hbuMap.addLayer(layer); }
                    } else if (hbuMap.hasLayer(layer)) {
                        hbuMap.removeLayer(layer);
                    }
                });
            }

            hbuMap.on('baselayerchange', function (e) {
                if (!e || !hbuBasemaps[e.name]) { return; }
                try { window.localStorage.setItem(hbuBasemapKey, e.name); }
                catch (err) { /* storage blocked: the map simply forgets */ }
            });
            {%- endmacro %}
            """
        )

        def __init__(self, bases) -> None:
            super().__init__()
            self._name = "BasemapMemory"
            self.bases = bases
            self.storage_key = _BASEMAP_STORAGE_KEY

    return _BasemapMemory(bases)


def _add_base_tiles(fmap) -> None:
    """Add the basemap the vector layers are drawn over.

    Mapbox when a token is configured: its ``light-v11`` style is the pale,
    low-contrast background parcel lines and footprints read best against, and
    it is what replaced ``CartoDB positron`` — which now needs a Carto account.
    Plain OpenStreetMap otherwise, so a local run needs no key at all.

    With a token there are two of them and the choice between them is the
    user's, so `_basemap_memory` goes on here rather than in `build_map`: it
    names the two layers, and putting it beside them is what keeps it in step
    with which basemaps exist. Added at this point it also runs before the
    overlays and before the layer control, so the base it swaps in joins the
    tile pane ahead of the vector grids and the control is built already
    reading the right radio.
    """
    import folium  # noqa: PLC0415

    bases: list[tuple[Any, str]] = []
    token = _mapbox_token()
    if _use_mapbox() and token:
        style = os.getenv("MAPBOX_STYLE", "mapbox/light-v11").strip("/")
        streets = folium.TileLayer(
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
        )
        streets.add_to(fmap)
        satellite = folium.TileLayer(
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
        )
        satellite.add_to(fmap)
        # `show` above is still what a browser with nothing stored opens on;
        # the script below decides every visit after the first switch.
        bases = [(streets, "Mapbox"), (satellite, "Satellite")]
    else:
        folium.TileLayer("OpenStreetMap", overlay=False, control=True).add_to(fmap)

    # One basemap is nothing to remember.
    if len(bases) > 1:
        _basemap_memory(bases).add_to(fmap)


def build_map(
    *,
    center: tuple[float, float] = DEFAULT_CENTER,
    zoom: int = DEFAULT_ZOOM,
    lots: Any = None,
    buildings: Any = None,
    zones: Any = None,
    capacity: Any = None,
    streets: Any = None,
    massing: Any = None,
    tile_layers: dict[str, str] | None = None,
    tile_visibility: dict[str, bool] | None = None,
    selected: dict | None = None,
    fit_bounds: list | None = None,
):
    """Assemble the map.

    ``tile_layers`` is the vector-tile renderer: a mapping of layer name to
    the templated URL Leaflet fills in per tile. Given, it draws all six
    layers and the six ``FeatureSet`` arguments are ignored — the caller has
    nothing to fetch, which is the whole point.

    ``lots``/``buildings``/``zones``/``capacity``/``streets``/``massing`` are
    the GeoJSON renderer, each a ``FeatureSet`` the caller has already fetched.

    Draw order is the same under both, and it is a decision: zones underneath,
    then the shading, then the street sides, then lots, then the footprints
    standing today, then the proposed massing on top of them. The proposal goes last because it is what the map is being
    read for - a massing hidden under the building it would replace answers
    nothing.

    ``capacity`` shades the parcels themselves and so goes directly above the
    zones and below everything else: it is a property *of* the lot rather than
    an object standing on it, and a footprint drawn underneath its own lot's
    shading would be invisible.

    ``selected`` draws one lot as GeoJSON, above every tile layer regardless
    of which of them is on. `app.py` does not use it: it passes the same shape
    to ``st_folium`` as a feature group instead, so that a click does not
    rebuild the map — see `selection_layer`. The argument is kept because a
    map built here is otherwise a complete map, and a caller rendering one
    outside Streamlit has nowhere else to put the selection.
    """
    import folium  # noqa: PLC0415

    fmap = folium.Map(
        location=list(center),
        zoom_start=zoom,
        tiles=None,  # added by _add_base_tiles so the provider is swappable
        control_scale=True,
        # `MAP_MIN_ZOOM`, so the floor Leaflet enforces and the floor the
        # aggregates are built to reach are one number. The corpus only covers
        # one borough anyway, and below this a cell would summarise more
        # ground than the borough has.
        min_zoom=MAP_MIN_ZOOM,
        max_zoom=19,
        prefer_canvas=True,
    )
    _add_base_tiles(fmap)

    if tile_layers:
        add_tile_layers(fmap, tile_layers, tile_visibility)
        zones = capacity = streets = lots = buildings = massing = None

    if zones is not None and zones.features:
        folium.GeoJson(
            zones.collection(),
            name=f"Zoning ({zones.count})",
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
                aliases=["Lot", "Used", "Still buildable"],
                sticky=True,
            ),
            control=True,
        ).add_to(fmap)

    if streets is not None and streets.features:
        folium.GeoJson(
            streets.collection(),
            name=f"Street sides ({streets.count})",
            style_function=lambda _: dict(_STREET_STYLE),
            highlight_function=lambda _: dict(_STREET_HIGHLIGHT),
            tooltip=folium.GeoJsonTooltip(
                fields=["street_label", "length_label"],
                aliases=["Street", "Length"],
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
                aliases=["Lot", "Area"],
                sticky=True,
            ),
            control=True,
        ).add_to(fmap)

    if buildings is not None and buildings.features:
        folium.GeoJson(
            buildings.collection(),
            name=f"Buildings ({buildings.count})",
            style_function=lambda _: dict(_BUILDING_STYLE),
            highlight_function=lambda _: {"fillOpacity": 0.8},
            tooltip=folium.GeoJsonTooltip(
                fields=["area_label"],
                aliases=["Footprint"],
                sticky=True,
            ),
            control=True,
        ).add_to(fmap)

    if massing is not None and massing.features:
        folium.GeoJson(
            massing.collection(),
            name=f"Proposed massing ({massing.count})",
            style_function=_massing_style,
            highlight_function=lambda _: {"fillOpacity": 0.85, "weight": 2.5},
            tooltip=folium.GeoJsonTooltip(
                fields=["lot_number", "massing_label", "fit_label"],
                aliases=["Lot", "Proposed", "Footprint"],
                sticky=True,
            ),
            control=True,
        ).add_to(fmap)

    if selected and selected.get("geometry"):
        _draw_selection(fmap, selected)

    folium.LayerControl(collapsed=True).add_to(fmap)

    if fit_bounds:
        fmap.fit_bounds(fit_bounds)

    return fmap


def decorate(feature_set, layer: str) -> None:
    """Add the display-only properties the tooltips read. GeoJSON renderer only.

    Folium's ``GeoJsonTooltip`` names fields by key and renders whatever is
    there, so formatting a number for a human has to happen before the map is
    built rather than in a style callback.

    `_TOOLTIP_JS` above is this function again, in JavaScript, for the tile
    renderer — where the feature reaches the browser and never reaches Python.
    The two are kept in step by hand. Anything changed here belongs there too.
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
        if layer == "streets":
            # An unnamed service lane is a real street side, not a missing
            # name - the same rule `hbuStreetLabel` applies in the browser.
            props["street_label"] = props.get("street_name") or "unnamed lane"
            length = props.get("length_m")
            props["length_label"] = (
                f"{float(length):,.0f} m" if length else "—"
            )

        if layer == "capacity":
            used = props.get("used_pct")
            status = props.get("hbu_status")
            if used is None:
                # Why there is no percentage, rather than a blank. The five
                # statuses are gold.lot_highest_best_use's own.
                props["used_label"] = {
                    "no_candidate_column": "no use the solver prices is zoned here",
                    # The former name of no_candidate_column, from when the
                    # solver priced dwellings alone; rows written before the
                    # rename still carry it.
                    "no_residential_column": "no residential column",
                    "no_governing_column": "no governing column",
                    "infeasible": "no feasible programme",
                    "solver_error": "solver error",
                }.get(status, "not solved")
            else:
                built = props.get("existing_floor_area_m2")
                permitted = props.get("hbu_floor_area_m2")
                shown = f"{float(used):,.0f}%"
                if permitted:
                    shown += (
                        f" ({float(built or 0):,.0f} / {float(permitted):,.0f} m²)"
                    )
                props["used_label"] = shown
            # The three classes summed, then the dwellings named separately:
            # "12,000 more sq ft" and "14 more dwellings" are the two units
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
                parts = [f"{headroom:,.0f} m² ({headroom * 10.7639:,.0f} sq ft)"]
                gap = props.get("dwelling_gap")
                if gap is None and props.get("hbu_num_dwellings") is not None:
                    gap = int(props["hbu_num_dwellings"]) - int(
                        props.get("existing_num_dwellings") or 0
                    )
                if gap and int(gap) > 0:
                    parts.append(f"{int(gap)} dwellings")
                props["headroom_label"] = " · ".join(parts)

        if layer == "massing":
            floors = props.get("floors")
            dwellings = props.get("num_dwellings")
            commercial = props.get("commercial_floors") or 0
            parts = []
            if floors:
                parts.append(f"{int(floors)} storeys")
            if dwellings:
                parts.append(f"{int(dwellings)} dwellings")
            if commercial:
                parts.append(f"{int(commercial)} comm. storeys")
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
                    f"{float(placed):,.0f} m² — {float(fit):.0f}% of the "
                    f"solved footprint"
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
