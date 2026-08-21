"""Map assembly: the coordinate-order swap and the payload trim.

``bounds_of`` exists because GeoJSON stores longitude first and folium's
``fit_bounds`` wants latitude first. Getting that backwards puts a Montreal lot
in the Indian Ocean, silently, so it is tested rather than reviewed.
"""

from __future__ import annotations

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
