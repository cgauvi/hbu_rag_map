"""Map assembly: the coordinate-order swap and the payload trim.

``bounds_of`` exists because GeoJSON stores longitude first and folium's
``fit_bounds`` wants latitude first. Getting that backwards puts a Montreal lot
in the Indian Ocean, silently, so it is tested rather than reviewed.
"""

from __future__ import annotations

import re

from src.utils import basemap
from src.utils.queries import FeatureSet

_POLYGON = {
    "type": "Polygon",
    "coordinates": [[[-73.62, 45.54], [-73.62, 45.541],
                     [-73.619, 45.541], [-73.619, 45.54], [-73.62, 45.54]]],
}


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


def test_one_basemap_is_nothing_to_remember(monkeypatch):
    """No token, one base tile layer, and no script to carry a choice across."""
    monkeypatch.delenv("MAPBOX_TOKEN", raising=False)
    monkeypatch.delenv("MAP_TILE_PROVIDER", raising=False)
    rendered = basemap.build_map().get_root().render()
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


def test_a_shrunk_massing_is_drawn_in_the_warning_colour():
    """Colour carries the finding, so the amber ones are findable by eye."""
    fitted = basemap._massing_style(_feature(massing_status="fitted"))
    shrunk = basemap._massing_style(_feature(massing_status="shrunk"))
    assert fitted["fillColor"] != shrunk["fillColor"]
    assert "dashArray" in shrunk and "dashArray" not in fitted


def test_massing_style_is_a_copy_so_folium_cannot_mutate_the_constant():
    style = basemap._massing_style(_feature(massing_status="fitted"))
    style["fillOpacity"] = 0.01
    assert basemap._MASSING_FITTED_STYLE["fillOpacity"] != 0.01


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
