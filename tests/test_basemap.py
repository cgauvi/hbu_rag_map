"""Map assembly: the coordinate-order swap and the payload trim.

``bounds_of`` exists because GeoJSON stores longitude first and folium's
``fit_bounds`` wants latitude first. Getting that backwards puts a Montreal lot
in the Indian Ocean, silently, so it is tested rather than reviewed.
"""

from __future__ import annotations

import re

from src.utils import basemap, queries
from src.utils.queries import FeatureSet

_POLYGON = {
    "type": "Polygon",
    "coordinates": [[[-73.62, 45.54], [-73.62, 45.541],
                     [-73.619, 45.541], [-73.619, 45.54], [-73.62, 45.54]]],
}

#: A RESTful WMTS template, in the order the REST binding puts the segments
#: in: matrix set, matrix, *row*, *column* - so {z}/{y}/{x}, not {z}/{x}/{y}.
WMTS_URL = (
    "https://example.gouv.qc.ca/erdas-iws/ogc/wmts/Imagerie_Continue/"
    "Imagerie_GQ/default/GoogleMapsCompatibleExt2:epsg:3857/{z}/{y}/{x}.jpg"
)


def test_bounds_are_lat_first_for_folium():
    (south, west), (north, east) = basemap.bounds_of(_POLYGON)
    assert 45.0 < south < 46.0 and 45.0 < north < 46.0
    assert -74.0 < west < -73.0 and -74.0 < east < -73.0
    assert south <= north and west <= east


def test_bounds_walk_a_multipolygon():
    multi = {"type": "MultiPolygon", "coordinates": [_POLYGON["coordinates"]]}
    assert basemap.bounds_of(multi) == basemap.bounds_of(_POLYGON)


def test_bounds_of_nothing_is_none():
    assert basemap.bounds_of(None) is None
    assert basemap.bounds_of({"type": "Polygon", "coordinates": []}) is None


def test_padding_widens_in_both_directions():
    padded = basemap.pad_bounds(basemap.bounds_of(_POLYGON), factor=0.5)
    (south, west), (north, east) = padded
    original = basemap.bounds_of(_POLYGON)
    assert south < original[0][0] and west < original[0][1]
    assert north > original[1][0] and east > original[1][1]


def test_padding_a_degenerate_box_still_produces_extent():
    """A point-sized lot must not fit_bounds to a zero-area rectangle."""
    (south, west), (north, east) = basemap.pad_bounds([[45.5, -73.6], [45.5, -73.6]])
    assert north > south and east > west


def _feature(**properties):
    return {"type": "Feature", "geometry": _POLYGON, "properties": properties}


def test_decorate_formats_an_area_label():
    features = FeatureSet([_feature(area_m2=1234.6, attributes={})], layer="lots")
    basemap.decorate(features, "lots")
    assert features.features[0]["properties"]["area_label"] == "1,235 m²"


def test_decorate_handles_a_missing_area():
    features = FeatureSet([_feature(area_m2=None, attributes={})], layer="lots")
    basemap.decorate(features, "lots")
    assert features.features[0]["properties"]["area_label"] == "—"


def test_decorate_labels_a_zone_from_its_attributes():
    features = FeatureSet(
        [_feature(feature_id="C01-001", attributes={"NUMERO_COMPLET": "C01-001"})],
        layer="zones",
    )
    basemap.decorate(features, "zones")
    assert features.features[0]["properties"]["zone_label"] == "C01-001"


def test_decorate_drops_the_attribute_bag():
    """Folium embeds properties verbatim; Infolot carries two dozen columns."""
    features = FeatureSet(
        [_feature(area_m2=100.0, attributes={"CO_STATT_LOT": "AC", "NO_LOT": "2 170 935"})],
        layer="lots",
    )
    basemap.decorate(features, "lots")
    assert "attributes" not in features.features[0]["properties"]


def test_build_map_survives_every_layer_being_empty():
    fmap = basemap.build_map()
    assert fmap.location == list(basemap.DEFAULT_CENTER)


def test_basemap_is_openstreetmap_without_a_mapbox_token(monkeypatch):
    monkeypatch.delenv("MAPBOX_TOKEN", raising=False)
    monkeypatch.delenv("MAP_TILE_PROVIDER", raising=False)
    rendered = basemap.build_map().get_root().render()
    assert "tile.openstreetmap.org" in rendered
    assert "api.mapbox.com" not in rendered


def test_basemap_uses_mapbox_when_a_token_is_set(monkeypatch):
    monkeypatch.setenv("MAPBOX_TOKEN", "pk.test_token")
    monkeypatch.delenv("MAP_TILE_PROVIDER", raising=False)
    rendered = basemap.build_map().get_root().render()
    assert "api.mapbox.com/styles/v1/mapbox/light-v11" in rendered
    assert "access_token=pk.test_token" in rendered
    assert "mapbox.satellite" in rendered  # offered as a toggle layer


def test_the_chosen_basemap_survives_a_rebuild(monkeypatch):
    """Ticking a layer rebuilds the map; the satellite view must not reset.

    A layer toggle changes the map's JavaScript, which changes the component's
    key, which remounts the iframe — so the basemap showing at the time is a
    piece of state Python never had. The browser keeps it instead, and what is
    asserted here is the wiring: both bases named, both reachable from the
    stored name, and the listener that records the choice.
    """
    monkeypatch.setenv("MAPBOX_TOKEN", "pk.test_token")
    monkeypatch.delenv("MAP_TILE_PROVIDER", raising=False)
    rendered = basemap.build_map().get_root().render()

    assert "baselayerchange" in rendered
    assert basemap._BASEMAP_STORAGE_KEY in rendered
    # The names the layer control reports, and the variables it holds — a
    # basemap missing from either half is one the map cannot switch back to.
    for name in ("Mapbox", "Satellite"):
        match = re.search(rf'"{name}": (\w+),', rendered)
        assert match, f"{name} is not in the remembered basemaps"
        assert f"var {match.group(1)} = L.tileLayer" in rendered
        assert f'"{name}" : {match.group(1)}' in rendered  # in the layer control


def test_the_basemap_memory_lasts_a_tab_not_a_lifetime(monkeypatch):
    """A new visit opens on Mapbox, whatever was picked last time.

    The choice survives a remount (session storage lives as long as the tab)
    but not a visit: with local storage one click on Satellite made every
    later visit open on imagery, and the pale street view stopped being the
    default in practice.
    """
    monkeypatch.setenv("MAPBOX_TOKEN", "pk.test_token")
    monkeypatch.delenv("MAP_TILE_PROVIDER", raising=False)
    rendered = basemap.build_map().get_root().render()
    assert "window.sessionStorage.getItem" in rendered
    assert "window.sessionStorage.setItem" in rendered
    assert "localStorage" not in rendered
    # The first visit's base: the Mapbox streets layer is the one added to the
    # map in the document itself, before any stored choice is consulted.
    match = re.search(r'"Mapbox": (\w+),', rendered)
    assert match
    assert re.search(rf"{match.group(1)}\.addTo\(\w+\);", rendered)


def test_one_basemap_is_nothing_to_remember(monkeypatch):
    """No token, one base tile layer, and no script to carry a choice across.

    Both the Mapbox pair and the orthophoto have to be off to get there, which
    is the point: with either of them the map has a choice worth keeping.
    """
    monkeypatch.delenv("MAPBOX_TOKEN", raising=False)
    monkeypatch.delenv("MAP_TILE_PROVIDER", raising=False)
    monkeypatch.setenv("MAP_WMTS_URL", "")
    rendered = basemap.build_map().get_root().render()
    assert "baselayerchange" not in rendered


def test_the_orthophoto_is_on_without_any_configuration(monkeypatch):
    """It needs no account, so unset is the service rather than no service."""
    monkeypatch.delenv("MAP_WMTS_URL", raising=False)
    monkeypatch.delenv("MAPBOX_TOKEN", raising=False)
    rendered = basemap.build_map().get_root().render()
    assert "servicesmatriciels.mern.gouv.qc.ca/erdas-iws" in rendered
    assert "epsg:3857/{z}/{y}/{x}.jpg" in rendered


def test_an_empty_wmts_url_switches_the_orthophoto_off(monkeypatch):
    """The knob declared and left blank: off, not the default."""
    monkeypatch.setenv("MAP_WMTS_URL", "")
    monkeypatch.setenv("MAPBOX_TOKEN", "pk.test_token")
    rendered = basemap.build_map().get_root().render()
    assert "erdas-iws" not in rendered


def test_a_configured_wmts_is_a_third_basemap(monkeypatch):
    """It joins the radio, keeps its path order, and is not shown first."""
    monkeypatch.setenv("MAPBOX_TOKEN", "pk.test_token")
    monkeypatch.delenv("MAP_TILE_PROVIDER", raising=False)
    monkeypatch.setenv("MAP_WMTS_URL", WMTS_URL)
    monkeypatch.setenv("MAP_WMTS_NAME", "Orthophoto QC")
    rendered = basemap.build_map().get_root().render()

    # The row/column order of a RESTful WMTS, which is not Leaflet's: written
    # {z}/{x}/{y} the tiles would load transposed rather than fail.
    assert "epsg:3857/{z}/{y}/{x}.jpg" in rendered
    assert '"Orthophoto QC" : ' in rendered  # in the layer control
    match = re.search(r'"Orthophoto QC": (\w+),', rendered)
    assert match, "the WMTS base is not remembered"
    assert f"var {match.group(1)} = L.tileLayer" in rendered
    # The pale street base is still what a first visit opens on.
    assert re.search(rf"{match.group(1)}\.addTo\(\w+\);", rendered) is None


def test_a_wmts_alongside_openstreetmap_is_remembered(monkeypatch):
    """Without a Mapbox token there are still two bases to choose between."""
    monkeypatch.delenv("MAPBOX_TOKEN", raising=False)
    monkeypatch.delenv("MAP_TILE_PROVIDER", raising=False)
    monkeypatch.delenv("MAP_WMTS_NAME", raising=False)
    monkeypatch.setenv("MAP_WMTS_URL", WMTS_URL)
    rendered = basemap.build_map().get_root().render()

    assert "tile.openstreetmap.org" in rendered
    assert "baselayerchange" in rendered
    assert f'"{basemap._WMTS_NAME_DEFAULT}": ' in rendered


def test_a_placeholder_wmts_url_is_off_rather_than_the_default(monkeypatch):
    """What Terraform writes for a variable nobody configured."""
    monkeypatch.delenv("MAPBOX_TOKEN", raising=False)
    monkeypatch.setenv("MAP_WMTS_URL", "PLACEHOLDER")
    rendered = basemap.build_map().get_root().render()
    assert "erdas-iws" not in rendered
    assert "baselayerchange" not in rendered


def test_placeholder_token_is_treated_as_unset(monkeypatch):
    """The deployed task injects the secret; unset, it arrives as PLACEHOLDER."""
    monkeypatch.setenv("MAPBOX_TOKEN", "PLACEHOLDER")
    monkeypatch.delenv("MAP_TILE_PROVIDER", raising=False)
    rendered = basemap.build_map().get_root().render()
    assert "tile.openstreetmap.org" in rendered
    assert "api.mapbox.com" not in rendered


def test_provider_override_forces_openstreetmap(monkeypatch):
    monkeypatch.setenv("MAPBOX_TOKEN", "pk.test_token")
    monkeypatch.setenv("MAP_TILE_PROVIDER", "osm")
    rendered = basemap.build_map().get_root().render()
    assert "tile.openstreetmap.org" in rendered
    assert "api.mapbox.com" not in rendered


def test_build_map_draws_a_selection():
    fmap = basemap.build_map(
        selected={"geometry": _POLYGON, "lot_number": "2 170 935",
                  "lat": 45.5405, "lon": -73.6195}
    )
    rendered = fmap.get_root().render()
    assert "2 170 935" in rendered


def test_lots_are_gated_above_buildings():
    """Buildings are denser than lots, so they may not appear sooner."""
    assert basemap.MIN_BUILDING_ZOOM >= basemap.MIN_LOT_ZOOM


def test_every_default_layer_draws_something_where_the_map_opens():
    """A layer on by default may not open *blank*, and the reason is the pane.

    The map opens at `DEFAULT_ZOOM`, far enough out to hold every registered
    city in one frame, which is below the detail zoom of all but `zones`. That
    is fine for the five layers `queries.AGGREGATE_LAYERS` names: below their
    own zoom the server substitutes `gold.map_cell_aggregates`, so what draws
    is the dissolved picture of where the data is, which is the whole point of
    opening out here.

    It is not fine for the layers with no cells behind them — `land_use`,
    `opportunities` and `surface_parking`, whose detail zoom is a floor rather
    than a handover. Ticking one of those on by default would open a map with
    the layer switched on and nothing drawn, which a reader reads as a fault
    rather than as a zoom. This is what ties `DEFAULT_ZOOM` to
    `DEFAULT_LAYERS`.

    It used to assert the stronger thing — that no default layer opened on its
    aggregate at all — which was right while the map opened over one borough
    and is not the map that exists now.
    """
    for layer, on in basemap.DEFAULT_LAYERS.items():
        if not on:
            continue
        draws_itself = basemap.DEFAULT_ZOOM >= queries.MVT_DETAIL_ZOOM[layer]
        assert draws_itself or layer in queries.AGGREGATE_LAYERS, (
            f"{layer} is on by default but has nothing to draw at zoom "
            f"{basemap.DEFAULT_ZOOM}: it draws itself from "
            f"{queries.MVT_DETAIL_ZOOM[layer]} and has no aggregate to fall "
            f"back to"
        )


# ---------------------------------------------------------------------------
# The massing layer
# ---------------------------------------------------------------------------


def _massing(**overrides):
    props = {
        "lot_number": "2 170 935",
        "massing_status": "fitted",
        "floors": 5,
        "num_dwellings": 11,
        "commercial_floors": 1,
        "footprint_m2": 116.0,
        "placed_footprint_m2": 116.0,
        "footprint_fit_pct": 100.0,
        "attributes": {},
    }
    props.update(overrides)
    return FeatureSet([_feature(**props)], layer="massing")


def test_massing_label_reads_the_programme():
    features = _massing()
    basemap.decorate(features, "massing")
    label = features.features[0]["properties"]["massing_label"]
    assert "5 storeys" in label and "11 dwellings" in label


def test_a_fitted_massing_reports_its_footprint_plainly():
    features = _massing()
    basemap.decorate(features, "massing")
    assert features.features[0]["properties"]["fit_label"] == "116 m²"


def test_a_shrunk_massing_says_how_much_of_the_solved_footprint_fits():
    """The sanity check, in the tooltip - see urban_rag.massing."""
    features = _massing(
        massing_status="shrunk", placed_footprint_m2=90.2,
        footprint_m2=148.0, footprint_fit_pct=60.9,
    )
    basemap.decorate(features, "massing")
    label = features.features[0]["properties"]["fit_label"]
    assert "90 m²" in label and "61%" in label


def test_every_massing_is_drawn_in_the_same_colour():
    """The fit is a tooltip, not a hue - see `_MASSING_STYLE`.

    The layer's colours were only ever named by the low-zoom cell legend, so
    for most of the zoom range a shrunk massing was a second colour with
    nothing on screen to read it by.
    """
    fitted = basemap._massing_style(_feature(massing_status="fitted"))
    shrunk = basemap._massing_style(_feature(massing_status="shrunk"))
    assert fitted == shrunk


def test_the_fit_survives_the_colour_it_used_to_be_carried_in():
    """Dropping the amber must not drop the finding: it moves to the label."""
    features = _massing(
        massing_status="shrunk", placed_footprint_m2=90.2,
        footprint_m2=148.0, footprint_fit_pct=60.9,
    )
    basemap.decorate(features, "massing")
    assert features.features[0]["properties"]["fit_label"]


def test_massing_style_is_a_copy_so_folium_cannot_mutate_the_constant():
    style = basemap._massing_style(_feature(massing_status="fitted"))
    style["fillOpacity"] = 0.01
    assert basemap._MASSING_STYLE["fillOpacity"] != 0.01


def test_massing_is_gated_with_the_buildings_it_is_read_against():
    """A proposal shown without the standing footprint is half the comparison."""
    assert basemap.MIN_MASSING_ZOOM == basemap.MIN_BUILDING_ZOOM


def test_build_map_draws_the_massing_last():
    """It is what the map is being read for; hiding it under a footprint
    answers nothing."""
    features = _massing()
    basemap.decorate(features, "massing")
    rendered = basemap.build_map(massing=features).get_root().render()
    assert "Proposed massing" in rendered


# ---------------------------------------------------------------------------
# The overlay ticks, across a rebuild
# ---------------------------------------------------------------------------

_TILE_URLS = {
    "zones": ["http://tiles/zones.pmtiles"],
    "capacity": ["http://tiles/capacity.pmtiles"],
    "lots": ["http://tiles/lots.pmtiles"],
}


def _tiled(**visibility):
    return basemap.build_map(
        tile_layers=_TILE_URLS, tile_visibility=visibility
    ).get_root().render()


def test_an_overlay_ticked_on_the_map_survives_a_rebuild():
    """The bug: the layer control checking and unchecking its own boxes.

    A vector layer has two switches — the sidebar, which Python owns, and
    Leaflet's control, which the browser owns — and the control's half is
    never reported back. Every rebuild redrew it from `st.session_state`
    alone, so a tick made on the map was put back the moment anything
    remounted the iframe, which a sidebar tick does on its own.

    Asserted here is the wiring that ends that: every overlay named, each
    reachable from the variable the control also holds, and the listener that
    records a click on it.
    """
    rendered = _tiled(zones=False, capacity=True, lots=True)

    assert basemap._LAYERS_STORAGE_KEY in rendered
    assert "overlayadd overlayremove" in rendered
    for name in ("Zoning", "Utilisation", "Lots"):
        match = re.search(rf'\{{name: "{name}",\s*layer: (\w+),', rendered)
        assert match, f"{name} is not in the remembered overlays"
        assert f"var {match.group(1)} = L.vectorGrid.pmtiles" in match.string
        assert f'"{name}" : {match.group(1)}' in rendered  # in the layer control


def test_the_remembered_tick_is_stamped_with_what_python_asked_for():
    """`from` is what lets the sidebar win when the sidebar is what moved.

    Without it the stored tick would outrank Python for ever, and a layer
    switched off on the map could never be switched back on from the sidebar.
    """
    rendered = _tiled(zones=False, capacity=True, lots=True)

    assert re.search(r'name: "Zoning",\s*layer: \w+,\s*show: false', rendered)
    assert re.search(r'name: "Lots",\s*layer: \w+,\s*show: true', rendered)
    assert "stored.from === entry.show" in rendered


def test_the_overlay_memory_runs_before_the_layer_control_is_built():
    """Order is the whole of why this does not flicker.

    Applied first, the boxes are drawn once and already right. Applied after,
    they would be drawn from `show` and then corrected — and the correction on
    screen is the checking and unchecking this fixes. It is also what keeps
    `overlayadd` honest: the control does not exist yet to fire it, so the
    listener records the user's clicks and none of this script's own.
    """
    rendered = _tiled(zones=False, capacity=True, lots=True)

    assert (
        rendered.index("hbuOverlays.forEach")
        < rendered.index("L.control.layers(")
    )


def test_the_control_reports_its_ticks_back_to_python():
    """The other half of the same bug: the sidebar showing the opposite.

    Remembering a tick keeps the *map* right across a rebuild. It said nothing
    to Python, so the sidebar box for a layer switched on in the map's own
    control stayed unticked and the pane's "Vector tiles:" line went on naming
    what Python had last asked for — two controls over one layer, disagreeing
    on screen.

    `selected_layers` is `st_folium`'s own return value and the channel back.
    Asserted here is that the state is written into it, in the shape its
    frontend already puts WMS layers there in, and from both the places the
    state can change.
    """
    rendered = _tiled(zones=False, capacity=True, lots=True)

    assert "window.__GLOBAL_DATA__" in rendered
    assert "data.selected_layers[entry.name] = {" in rendered
    assert "name: entry.name, url: entry.url" in rendered
    # Once for the mount, once per click on the control.
    assert rendered.count("hbuReportLayers();") == 2


def test_the_reported_entry_carries_the_layer_url():
    """`{name, url}`, because that is the shape st_folium writes itself.

    Python reads the name; the url is there so a reader of
    ``result["selected_layers"]`` finds the same fields whichever half of the
    frontend put the entry in.
    """
    rendered = _tiled(zones=False, lots=True)

    assert re.search(
        r'name: "Lots",\s*layer: \w+,\s*show: true,\s*'
        r'url: "http://tiles/lots\.pmtiles"',
        rendered,
    )


def test_the_first_report_is_made_before_the_control_is_built():
    """So the component's opening value is the truth, not `show`.

    `st_folium` sends its first value at the end of the render. A report made
    after the control existed would be one interaction late on a mount whose
    remembered ticks disagree with Python — which is exactly the mount where
    the sidebar is wrong and nothing yet has happened to correct it.
    """
    rendered = _tiled(zones=False, capacity=True, lots=True)

    assert (
        rendered.index("hbuReportLayers();")
        < rendered.index("L.control.layers(")
    )


def test_every_layer_is_drawn_in_the_control_under_its_declared_name():
    """The other end of the label the sidebar reads.

    `app.py` draws its checkboxes from `TILE_LAYER_NAMES` rather than typing
    the names out, so the two agree by construction — but only while this end
    stays the source. Asserted for every layer rather than the three the
    memory test happens to switch on, so a layer added with a name written
    straight into the control cannot slip past.
    """
    urls = {
        layer: [f"http://tiles/{layer}.pmtiles"]
        for layer in basemap.TILE_LAYER_ORDER
    }
    rendered = basemap.build_map(tile_layers=urls).get_root().render()

    for layer in basemap.TILE_LAYER_ORDER:
        name = basemap.TILE_LAYER_NAMES[layer]
        assert f'"{name}" :' in rendered, f"{layer} is not in the layer control"


def test_the_geojson_renderer_has_no_overlay_memory():
    """An unticked layer is not fetched there, so it is not on the map to
    remember — and those names carry feature counts, which would key the
    store to the viewport."""
    features = FeatureSet(
        [_feature(area_m2=500.0, lot_number="1 234 567", attributes={})],
        layer="lots",
    )
    basemap.decorate(features, "lots")
    rendered = basemap.build_map(lots=features).get_root().render()
    assert "Lots (1)" in rendered  # the count in the name, as described above
    assert basemap._LAYERS_STORAGE_KEY not in rendered
    # And nothing reports either, which is why `app.py` reconciles only under
    # the tile renderer: an empty `selected_layers` here means "this map has
    # no control to ask", not "every layer is off".
    assert "hbuReportLayers" not in rendered


# ---------------------------------------------------------------------------
# The surface parking layer
# ---------------------------------------------------------------------------


def _parking(**overrides):
    props = {
        "lot_number": "2 170 935",
        "parking_status": "fitted",
        "surface_stalls": 4,
        "placed_surface_stalls": 4.0,
        "surface_parking_area_m2": 111.5,
        "placed_surface_parking_m2": 111.5,
        "surface_parking_fit_pct": 100.0,
        "num_parking_bays": 1,
        "attributes": {},
    }
    props.update(overrides)
    return FeatureSet([_feature(**props)], layer="surface_parking")


def test_a_fitted_parking_lot_reports_its_asphalt_plainly():
    features = _parking()
    basemap.decorate(features, "surface_parking")
    label = features.features[0]["properties"]["parking_label"]
    assert "112 m²" in label and "4 stalls" in label
    # One bay is the ordinary case and saying so is noise.
    assert "bays" not in label


def test_two_bays_are_named_because_one_is_not():
    """A front yard and a rear yard is the answer, not a compromise."""
    features = _parking(num_parking_bays=2)
    basemap.decorate(features, "surface_parking")
    assert "2 bays" in features.features[0]["properties"]["parking_label"]


def test_a_shrunk_parking_lot_says_how_much_of_the_yard_took_it():
    """The sanity check applied to the ground - see urban_rag.massing."""
    features = _parking(
        parking_status="shrunk", placed_surface_parking_m2=55.0,
        placed_surface_stalls=1.0, surface_parking_fit_pct=49.3,
    )
    basemap.decorate(features, "surface_parking")
    label = features.features[0]["properties"]["parking_label"]
    assert "55 m²" in label and "49%" in label


def test_every_parking_bay_is_drawn_in_the_same_colour():
    """The fit is a tooltip, not a hue - the choice `_MASSING_STYLE` makes."""
    fitted = basemap._parking_style(_feature(parking_status="fitted"))
    shrunk = basemap._parking_style(_feature(parking_status="shrunk"))
    assert fitted == shrunk


def test_the_parking_does_not_read_as_a_shade_of_the_massing():
    """Two shapes of one proposal, and emphatically not one kind of thing.

    A building has storeys and a height; asphalt is ground with cars on it. If
    the two were a light and a dark of the same hue a reader would take the
    parking for part of the building, which is the one reading this whole
    separation exists to prevent.
    """
    assert basemap._PARKING_STYLE["fillColor"] != basemap._MASSING_STYLE["fillColor"]
    assert basemap._PARKING_STYLE["color"] != basemap._MASSING_STYLE["color"]


def test_the_parking_is_drawn_under_the_massing():
    """The building is what the map is read for; the asphalt is context."""
    order = basemap.TILE_LAYER_ORDER
    assert order.index("surface_parking") < order.index("massing")
