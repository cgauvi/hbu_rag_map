"""The Land use layer: what each lot is for, today and as proposed.

Two gold tables carry the two sides - the gap table the roll's dominant
class, the HBU table the solver's - and this is the layer, the viewport read,
the colours and the tool that surface them together, with the floor,
footprint and dwellings on each side beside the class. What these pin: the
join is on the cadastral number (the gap table has been found stranded on an
old `lot_uid` generation once already), the side travels in the URL and an
unknown one draws the roll rather than nothing, the measured footprint comes
off the silver clip and is NULL rather than computed where that clip is
absent, and the legend, the style and both tooltips read the same classes.
"""

from __future__ import annotations

from datetime import date

import pytest
from langchain_core.tools import ToolException

from src.tools import map_tools, parcel_tools
from src.utils import basemap, queries, tiles


@pytest.fixture
def captured(monkeypatch):
    calls: list[tuple[str, object]] = []
    rows: list[dict] = []

    def fake_query(sql, params=None):
        calls.append((sql, params))
        return list(rows)

    monkeypatch.setattr(queries, "query", fake_query)
    monkeypatch.setattr(queries, "_building_lots_available", lambda: True)
    return calls, rows


@pytest.fixture
def captured_scalar(monkeypatch):
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        queries, "scalar", lambda sql, params=None: calls.append((sql, params)) or b""
    )
    monkeypatch.setattr(queries, "_building_lots_available", lambda: True)
    return calls


def land_use_row(**overrides) -> dict:
    row = {
        "lot_uid": 4211,
        "lot_number": "2 214 147",
        "neighborhood": "VSMPE",
        "scrape_date": date(2026, 9, 1),
        "area_m2": 312.0,
        "hbu_status": "solved",
        "existing_num_assessment_units": 1,
        "existing_use": "commercial",
        "existing_dominant_use_description": "Immeuble commercial",
        "hbu_use": "residential",
        "use_class": "commercial",
        "existing_floor_area_m2": 460.0,
        "hbu_floor_area_m2": 1380.0,
        "existing_footprint_m2": 164.0,
        "hbu_footprint_m2": 210.0,
        "existing_num_dwellings": 0,
        "hbu_num_dwellings": 9,
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0]]]},
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# The viewport read
# ---------------------------------------------------------------------------


def test_land_use_joins_the_lots_on_the_cadastral_number(captured):
    calls, _ = captured
    queries.land_use_in_bbox((-73.7, 45.5, -73.6, 45.6))
    sql = calls[0][0]
    assert "g.lot_number   = l.lot_number" in sql
    # The two gold tables share a generation and join to each other on the
    # surrogate; only the join to the cadastre goes through the number.
    assert "h.lot_uid      = g.lot_uid" in sql
    assert "lot_highest_best_use h" in sql


def test_the_side_is_chosen_in_sql_and_both_classes_travel(captured):
    calls, _ = captured
    queries.land_use_in_bbox((-73.7, 45.5, -73.6, 45.6), use_side="hbu")
    sql, params = calls[0]
    assert params["use_side"] == "hbu"
    assert "AS use_class" in sql
    assert "AS existing_use" in sql
    assert "AS hbu_use" in sql
    assert "%(use_side)s::text = 'hbu'" in sql


def test_the_side_defaults_to_the_roll(captured):
    calls, _ = captured
    queries.land_use_in_bbox((-73.7, 45.5, -73.6, 45.6))
    assert calls[0][1]["use_side"] == "existing"


def test_the_measured_footprint_is_the_piece_own(captured):
    """Off `silver.lot_zone_pieces`, not summed over the lot.

    It used to be a lateral sum of `silver.building_lot_intersections` at the
    lot grain, which was the same number while a lot was one feature. It is
    not on a split parcel: the pieces table clips those same footprints one
    cut further, so each piece carries the ground built on *it*, and the
    lateral would have painted the whole parcel's onto both.
    """
    calls, _ = captured
    queries.land_use_in_bbox((-73.7, 45.5, -73.6, 45.6))
    sql = calls[0][0]
    assert "l.existing_footprint_m2" in sql
    assert "building_lot_intersections" not in sql


def test_land_use_features_carry_what_the_tooltip_and_the_colour_read(captured):
    _, rows = captured
    rows.append(land_use_row())
    feature_set = queries.land_use_in_bbox((-73.7, 45.5, -73.6, 45.6))
    assert feature_set.layer == "land_use"
    props = feature_set.features[0]["properties"]
    for key in (
        "use_class", "existing_use", "hbu_use", "existing_dominant_use_description",
        "existing_floor_area_m2", "hbu_floor_area_m2",
        "existing_footprint_m2", "hbu_footprint_m2",
        "existing_num_dwellings", "hbu_num_dwellings", "hbu_status",
    ):
        assert key in props, key


# ---------------------------------------------------------------------------
# The tile
# ---------------------------------------------------------------------------


def test_the_land_use_tile_joins_on_the_cadastral_number_and_the_zone(
    captured_scalar,
):
    """Both halves of the key. The zone matters as much as the number now: the
    tile draws one feature per piece, and joining the gap on the lot alone
    would give a split parcel each of its two answers twice."""
    queries.mvt_tile("land_use", 15, 9700, 11800)
    sql = captured_scalar[0][0]
    assert "g.lot_number   = l.lot_number" in sql
    assert "g.feature_id   = l.feature_id" in sql
    assert "l.existing_footprint_m2" in sql


def test_the_land_use_tile_draws_the_piece(captured_scalar):
    """The geometry is the clip, so the layer needs no join to `rag.lots` -
    which is also what makes it immune to the reload that mints a new
    `lot_uid`."""
    queries.mvt_tile("land_use", 15, 9700, 11800)
    sql = captured_scalar[0][0]
    assert "lot_zone_pieces l" in sql
    assert "l.num_lot_zones" in sql


def test_land_use_has_no_fallback_spec_any_more():
    """There is nothing left to fall back to: the layer draws the pieces
    table's own geometry, so without that table it has no rows either way."""
    assert "land_use" not in queries._MVT_FALLBACK_LAYERS
    assert queries.MVT_LAYER_NAMES.count("land_use") == 1


def test_the_side_reaches_the_tile(captured_scalar):
    queries.mvt_tile("land_use", 15, 9700, 11800, use_side="hbu")
    assert captured_scalar[0][1]["use_side"] == "hbu"


def test_the_side_is_null_when_not_asked_which_the_sql_reads_as_the_roll(captured_scalar):
    queries.mvt_tile("land_use", 15, 9700, 11800)
    assert captured_scalar[0][1]["use_side"] is None


def test_the_layer_has_a_detail_zoom_and_no_aggregate():
    assert "land_use" in queries.MVT_LAYER_NAMES
    assert queries.MVT_DETAIL_ZOOM["land_use"] == queries.MVT_DETAIL_ZOOM["lots"]
    assert "land_use" not in queries.AGGREGATE_LAYERS
    assert not queries.serves_aggregate("land_use", 5)


def test_a_tile_url_names_a_side_the_layer_can_draw():
    assert tiles._tile_arguments({"use_side": ["hbu"]}) == {"use_side": "hbu"}
    assert tiles._tile_arguments({"use_side": ["HBU"]}) == {"use_side": "hbu"}
    assert tiles._tile_arguments({"use_side": ["existing"]}) == {"use_side": "existing"}
    # A value the layer cannot draw is today's side, not a blank map.
    assert tiles._tile_arguments({"use_side": ["proposed"]}) == {}


def test_the_partition_probe_knows_the_layer():
    schema, table, asset = queries.MAP_PARTITION_TABLES["land_use"]
    assert schema == queries.GOLD_SCHEMA
    assert table == asset == "lot_redevelopment_gap"


def test_the_lot_row_carries_the_class_the_pane_and_the_agent_read(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        queries, "query_one", lambda sql, params=None: calls.append(sql) or None
    )
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: queries.Capabilities(postgis=True, redevelopment_gap=True),
    )
    queries.lot_capacity(4211)
    assert "g.existing_dominant_income_class" in calls[0]


# ---------------------------------------------------------------------------
# The colours, the legend and the map
# ---------------------------------------------------------------------------


def test_every_class_has_a_colour_and_a_legend_row():
    rows = dict(basemap.land_use_legend_rows())
    for use in queries.LAND_USE_CLASSES:
        assert basemap.LAND_USE_COLORS[use] in rows
    assert len(rows) == len(queries.LAND_USE_CLASSES) + 1


def test_the_style_reads_the_side_the_server_chose():
    on_roll = basemap._land_use_style(
        {"properties": {"use_class": "commercial", "hbu_use": "residential"}}
    )
    assert on_roll["fillColor"] == basemap.LAND_USE_COLORS["commercial"]
    on_solve = basemap._land_use_style(
        {"properties": {"use_class": "residential", "existing_use": "commercial"}}
    )
    assert on_solve["fillColor"] == basemap.LAND_USE_COLORS["residential"]


def test_a_lot_with_no_class_on_that_side_is_grey_and_faint():
    style = basemap._land_use_style({"properties": {"use_class": None}})
    assert style["fillColor"] == basemap._LAND_USE_UNKNOWN_COLOR
    assert style["fillOpacity"] < basemap._land_use_style(
        {"properties": {"use_class": "residential"}}
    )["fillOpacity"]


def test_the_tile_style_reads_the_same_colours():
    js = basemap._style_js("land_use")
    assert "use_class" in js
    assert basemap.LAND_USE_COLORS["industrial"] in js
    assert basemap._LAND_USE_UNKNOWN_COLOR in js


def test_the_layer_sits_under_the_shading_and_is_gated_at_its_detail_zoom():
    order = basemap.TILE_LAYER_ORDER
    assert order.index("land_use") == order.index("zones") + 1
    assert order.index("land_use") < order.index("capacity")
    assert order == queries.MVT_LAYER_NAMES
    assert basemap.TILE_LAYER_MIN_ZOOM["land_use"] == queries.MVT_DETAIL_ZOOM["land_use"]
    assert basemap.MIN_LAND_USE_ZOOM == queries.MVT_DETAIL_ZOOM["land_use"]
    assert basemap.TILE_LAYER_NAMES["land_use"] == "Land use"
    assert basemap.DEFAULT_LAYERS["land_use"] is False


def test_the_tooltip_js_says_both_sides():
    js = basemap._TOOLTIP_JS
    assert "layer === 'land_use'" in js
    for name in ("hbuUseTodayLabel", "hbuUseProposedLabel", "hbuUseFloorLabel",
                 "hbuUseFootprintLabel", "hbuUseDwellingsLabel"):
        assert f"function {name}(" in js


def test_decorate_labels_both_sides_the_way_the_tooltip_does():
    feature_set = queries._as_feature_set(
        [land_use_row()], layer="land_use", id_key="lot_number", limit=10
    )
    basemap.decorate(feature_set, "land_use")
    props = feature_set.features[0]["properties"]
    assert props["use_today_label"] == "commercial · Immeuble commercial"
    assert props["use_proposed_label"] == "residential · changes use"
    assert props["floor_label"] == "460 m² → 1,380 m²"
    assert props["footprint_label"] == "164 m² → 210 m²"
    assert props["dwellings_label"] == "0 → 9"


def test_decorate_says_why_a_proposed_side_is_blank():
    feature_set = queries._as_feature_set(
        [land_use_row(hbu_use=None, hbu_status="infeasible", hbu_floor_area_m2=None,
                      hbu_footprint_m2=None, hbu_num_dwellings=None)],
        layer="land_use", id_key="lot_number", limit=10,
    )
    basemap.decorate(feature_set, "land_use")
    props = feature_set.features[0]["properties"]
    assert props["use_proposed_label"] == "no feasible programme"
    assert props["floor_label"] == "460 m² → —"
    assert props["dwellings_label"] == "0 → —"


def test_decorate_keeps_an_unreported_floor_apart_from_nothing_built():
    feature_set = queries._as_feature_set(
        [land_use_row(existing_floor_area_m2=None, existing_num_assessment_units=1)],
        layer="land_use", id_key="lot_number", limit=10,
    )
    basemap.decorate(feature_set, "land_use")
    props = feature_set.features[0]["properties"]
    assert props["floor_label"] == "not reported → 1,380 m²"


def test_decorate_does_not_call_the_first_use_of_vacant_ground_a_change():
    feature_set = queries._as_feature_set(
        [land_use_row(existing_use="none", existing_dominant_use_description=None)],
        layer="land_use", id_key="lot_number", limit=10,
    )
    basemap.decorate(feature_set, "land_use")
    props = feature_set.features[0]["properties"]
    assert props["use_today_label"] == "none"
    assert props["use_proposed_label"] == "residential"


def test_build_map_draws_the_layer_from_geojson():
    feature_set = queries._as_feature_set(
        [land_use_row()], layer="land_use", id_key="lot_number", limit=10
    )
    basemap.decorate(feature_set, "land_use")
    fmap = basemap.build_map(land_use=feature_set)
    names = [
        child.layer_name for child in fmap._children.values()
        if hasattr(child, "layer_name")
    ]
    assert any(name.startswith("Land use (1)") for name in names)


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------


def _caps(**flags):
    return queries.Capabilities(**{"postgis": True, "pgvector": True, **flags})


def test_the_land_use_layer_needs_both_tables_to_be_switched_on(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: _caps(redevelopment_gap=True, highest_best_use=False),
    )
    with pytest.raises(ToolException, match="not loaded"):
        map_tools.set_map_layers.invoke({"land_use": True})
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: _caps(redevelopment_gap=True, highest_best_use=True),
    )
    assert "land_use on" in map_tools.set_map_layers.invoke({"land_use": True})


def test_lot_efficiency_says_the_use_on_both_sides(monkeypatch):
    monkeypatch.setattr(
        queries, "lot_by_number",
        lambda number: {"lot_uid": 4211, "lot_number": "2 214 147",
                        "scrape_date": date(2026, 9, 1), "neighborhood": "VSMPE",
                        "area_m2": 312.0},
    )
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: _caps(lots=True, redevelopment_gap=True, highest_best_use=True),
    )
    row = {
        "lot_number": "2 214 147", "lot_area_m2": 312.0, "hbu_status": "solved",
        "existing_num_assessment_units": 1,
        "existing_floor_area_m2": 460.0, "hbu_floor_area_m2": 1380.0,
        "used_pct": 33.3, "hbu_num_dwellings": 9, "existing_num_dwellings": 0,
        "existing_dominant_income_class": "commercial",
        "existing_dominant_use_description": "Immeuble commercial",
        "hbu_dominant_use": "residential",
    }
    monkeypatch.setattr(queries, "lot_capacity", lambda *a, **k: row)
    monkeypatch.setattr(queries, "lot_opportunity", lambda *a, **k: None, raising=False)
    answer = parcel_tools.lot_efficiency.invoke({"lot_number": "2 214 147"})
    assert "Use: commercial (Immeuble commercial) today, residential proposed." in answer
    assert "The use changes." in answer
