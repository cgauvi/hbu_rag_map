"""The SQL layer, with the database stubbed.

These assert the *shape* of what is sent and what comes back — that a viewport
query asks for one row more than the limit so truncation is detectable, that
GeoJSON is assembled correctly, that the vector literal is what pgvector
parses. Whether the SQL runs is what the ``integration`` tests are for.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.utils import queries


@pytest.fixture
def captured(monkeypatch):
    """Capture the (sql, params) of every query, returning canned rows."""
    calls: list[tuple[str, object]] = []
    rows: list[dict] = []

    def fake_query(sql, params=None):
        calls.append((sql, params))
        return list(rows)

    monkeypatch.setattr(queries, "query", fake_query)
    return calls, rows


@pytest.fixture
def silver(monkeypatch):
    """Say which silver joins exist, without probing a database.

    `buildings_on_lot` and `zoning_for_lot` each choose between a precomputed
    table and an `ST_Intersection` fallback by asking `capabilities()`, which
    otherwise resolves a real connection.
    """

    def present(**flags):
        monkeypatch.setattr(
            queries, "capabilities", lambda: queries.Capabilities(**flags)
        )

    return present


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def test_capabilities_maps_every_column(monkeypatch):
    monkeypatch.setattr(
        queries, "query_one",
        lambda *_a, **_k: {
            "postgis": True, "pgvector": True, "lots": True, "buildings": False,
            "building_lots": False, "lot_features": False, "features": True,
            "chunks": False, "search_at_lot": False, "search_near": False,
        },
    )
    caps = queries.capabilities()

    assert caps.can_map
    assert not caps.can_retrieve
    assert f"{queries.SCHEMA}.buildings" in caps.missing()
    assert f"{queries.SCHEMA}.lots" not in caps.missing()
    # The silver joins are reported to the operator, so they can see the
    # pipeline has not run over this borough ...
    assert f"{queries.SILVER_SCHEMA}.lot_features" in caps.missing()
    # ... and withheld from anything a user reads, where "missing" would claim
    # a fault that did not happen: the answer still arrives, more slowly.
    user_facing = caps.missing(include_advisory=False)
    assert f"{queries.SILVER_SCHEMA}.lot_features" not in user_facing
    assert f"{queries.SILVER_SCHEMA}.building_lot_intersections" not in user_facing
    assert f"{queries.SCHEMA}.buildings" in user_facing


def test_a_missing_silver_join_does_not_stop_the_map_or_retrieval():
    """They only decide whether a click is answered fast or the slow way."""
    caps = queries.Capabilities(
        postgis=True, pgvector=True, lots=True, features=True, chunks=True
    )
    assert caps.can_map
    assert caps.can_retrieve


def test_capabilities_is_empty_when_the_probe_returns_nothing(monkeypatch):
    monkeypatch.setattr(queries, "query_one", lambda *_a, **_k: None)
    caps = queries.capabilities()
    assert not caps.can_map and not caps.can_retrieve


def test_can_map_needs_postgis_even_with_tables():
    assert not queries.Capabilities(postgis=False, lots=True).can_map
    assert queries.Capabilities(postgis=True, lots=True).can_map


# ---------------------------------------------------------------------------
# Viewport queries
# ---------------------------------------------------------------------------


def test_viewport_query_asks_for_one_row_past_the_limit(captured):
    """That extra row is how truncation is detected without a second count(*)."""
    calls, _ = captured
    queries.lots_in_bbox((-73.7, 45.5, -73.6, 45.6), limit=10)

    _sql, params = calls[0]
    assert params["limit"] == 11


def test_truncation_is_reported_and_the_extra_row_dropped(captured):
    calls, rows = captured
    rows.extend(
        {
            "lot_number": f"{n}",
            "neighborhood": "VSMPE",
            "scrape_date": date(2026, 8, 20),
            "area_m2": 100.0 + n,
            "attributes": {},
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0]]]},
        }
        for n in range(4)
    )

    found = queries.lots_in_bbox((-73.7, 45.5, -73.6, 45.6), limit=3)

    assert found.truncated
    assert found.count == 3
    assert found.layer == "lots"


def test_features_carry_an_id_and_a_layer(captured):
    _calls, rows = captured
    rows.append({
        "lot_number": "2 170 935",
        "neighborhood": "VSMPE",
        "scrape_date": date(2026, 8, 20),
        "area_m2": 267.9,
        "attributes": {"CO_STATT_LOT": "AC"},
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0]]]},
    })

    found = queries.lots_in_bbox((-73.7, 45.5, -73.6, 45.6))
    properties = found.features[0]["properties"]

    assert found.collection()["type"] == "FeatureCollection"
    assert properties["id"] == "2 170 935"
    assert properties["layer"] == "lots"
    # Dates must survive folium's JSON serialiser.
    assert properties["scrape_date"] == "2026-08-20"
    assert properties["attributes"] == {"CO_STATT_LOT": "AC"}


def test_a_row_with_no_geometry_is_skipped(captured):
    _calls, rows = captured
    rows.append({"lot_number": "1", "geometry": None, "attributes": {}})
    assert queries.lots_in_bbox((-73.7, 45.5, -73.6, 45.6)).count == 0


def test_size_filters_reach_the_query(captured):
    calls, _ = captured
    queries.lots_in_bbox((-73.7, 45.5, -73.6, 45.6), min_area_m2=500, max_area_m2=900)
    _sql, params = calls[0]
    assert params["min_area"] == 500
    assert params["max_area"] == 900


# ---------------------------------------------------------------------------
# The buildings viewport read is the intersection, not the footprints
# ---------------------------------------------------------------------------
#
# `mvt_tile`'s buildings layer written for the GeoJSON renderer, and it has to
# keep meaning the same thing: BDOI draws a terrace as one outline across every
# party wall, so an unclipped footprint overstates every parcel it crosses.


@pytest.fixture
def building_lots(monkeypatch):
    """Say whether the silver clip exists, without probing a database."""

    def present(exists: bool):
        monkeypatch.setattr(queries, "_building_lots_available", lambda: exists)

    return present


def test_buildings_in_view_read_the_precomputed_clip(captured, building_lots):
    building_lots(True)
    calls, _ = captured
    queries.buildings_in_bbox((-73.7, 45.5, -73.6, 45.6))
    sql, _params = calls[0]

    assert f"FROM {queries.SILVER_SCHEMA}.building_lot_intersections bl" in sql
    assert "bl.intersection_area_m2 AS area_m2" in sql
    assert "bl.geom && ST_MakeEnvelope" in sql
    # The unclipped footprint is the number this read exists not to report.
    assert "ST_Area(b.geom::geography)" not in sql


def test_buildings_in_view_fall_back_to_computing_the_clip(captured, building_lots):
    """A borough loaded this morning still draws clipped footprints."""
    building_lots(False)
    calls, _ = captured
    queries.buildings_in_bbox((-73.7, 45.5, -73.6, 45.6))
    sql, _params = calls[0]

    assert f"{queries.SILVER_SCHEMA}.building_lot_intersections" not in sql
    assert "ST_Intersection(b.geom, l.geom)" in sql
    assert "ST_Area(clip.geom::geography) AS area_m2" in sql
    # Indexed on the footprint, drawn from the clip - the same split the tile
    # makes, and for the same reason.
    assert "b.geom && ST_MakeEnvelope" in sql
    assert "clip.geom && ST_MakeEnvelope" not in sql
    assert "l.scrape_date  = b.scrape_date" in sql
    # A party wall intersects and clips to a line, not to a building.
    assert "NOT ST_IsEmpty(clip.geom)" in sql
    assert "ST_Dimension(clip.geom) = 2" in sql


def test_buildings_in_view_are_identified_by_the_pair(captured, building_lots):
    """One feature per (building, lot), so `building_uid` alone is not a key."""
    building_lots(True)
    _calls, rows = captured
    rows.append({
        "building_uid": 41,
        "building_lot_key": "41:7",
        "lot_number": "2 170 935",
        "neighborhood": "VSMPE",
        "scrape_date": date(2026, 8, 20),
        "area_m2": 164.0,
        "attributes": {},
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0]]]},
    })

    found = queries.buildings_in_bbox((-73.7, 45.5, -73.6, 45.6))
    properties = found.features[0]["properties"]

    assert properties["id"] == "41:7"
    assert properties["layer"] == "buildings"
    assert properties["area_m2"] == 164.0


def test_the_zoning_layer_is_selected_by_its_slug(captured):
    calls, _ = captured
    queries.zones_in_bbox((-73.7, 45.5, -73.6, 45.6))
    _sql, params = calls[0]
    assert params["source_table"] == "Reglement_urbanisme__VSP_REG_ZONE"
    assert params["url_attribute"] == "LIEN_GRILLE"


def test_the_use_attributes_are_the_permitted_columns_and_not_the_excluded_one():
    """The Lot pane lists uses off these; listing the wrong column would say a
    zone permits exactly what it forbids.

    Derived from `ZONING_FIELDS` rather than written down, so this also pins
    the derivation: a label reworded to something that no longer starts with
    "Permitted uses" silently empties the tuple, and the pane would report
    every zone as stating no use at all rather than raising anything.
    """
    assert queries.ZONING_USE_ATTRIBUTES == ("USAGE", "USAGE_AUT")
    assert "USAGE_EXC" not in queries.ZONING_USE_ATTRIBUTES
    keys = {key for key, _label in queries.ZONING_FIELDS}
    assert set(queries.ZONING_USE_ATTRIBUTES) <= keys


# ---------------------------------------------------------------------------
# The massing layer
# ---------------------------------------------------------------------------


def massing_row(**overrides) -> dict:
    row = {
        "lot_uid": 4211,
        "lot_number": "2 170 935",
        "feature_id": "H03-126",
        "neighborhood": "VSMPE",
        "scrape_date": date(2026, 8, 20),
        "massing_status": "fitted",
        "footprint_m2": 116.0,
        "placed_footprint_m2": 116.0,
        "footprint_fit_pct": 100.0,
        "aspect_ratio": 3.0,
        "width_m": 18.7,
        "depth_m": 6.2,
        "floors": 5,
        "height_m": 15.0,
        "num_dwellings": 11,
        "commercial_floors": 1,
        "placed_gross_floor_area_m2": 580.0,
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0]]]},
    }
    row.update(overrides)
    return row


def test_massing_reads_the_gold_schema(captured):
    """The one layer on this map that is an answer rather than a scrape."""
    calls, _ = captured
    queries.massing_in_bbox((-73.7, 45.5, -73.6, 45.6))
    sql, _params = calls[0]
    assert f"{queries.GOLD_SCHEMA}.lot_building_massing" in sql


def test_massing_features_carry_what_the_tooltip_reads(captured):
    _calls, rows = captured
    rows.append(massing_row())

    found = queries.massing_in_bbox((-73.7, 45.5, -73.6, 45.6))
    properties = found.features[0]["properties"]

    assert found.layer == "massing"
    # Keyed on the piece: a lot two zones cut in two gets two rectangles, and
    # a lot number alone would collide them into one feature.
    assert properties["id"] == "2 170 935@H03-126"
    assert properties["massing_status"] == "fitted"
    assert properties["floors"] == 5
    assert properties["num_dwellings"] == 11
    assert properties["scrape_date"] == "2026-08-20"


def test_massing_is_bbox_and_limit_bounded_like_every_other_layer(captured):
    calls, _ = captured
    queries.massing_in_bbox((-73.7, 45.5, -73.6, 45.6), limit=25)
    sql, params = calls[0]
    assert params["limit"] == 26
    assert "ST_MakeEnvelope" in sql and "ST_Intersects" in sql


def test_under_built_filter_is_off_unless_asked_for(captured):
    """Default is every drawn massing; the screen is opt-in."""
    calls, _ = captured
    queries.massing_in_bbox((-73.7, 45.5, -73.6, 45.6))
    _sql, params = calls[0]
    assert params["only_underbuilt"] is False

    queries.massing_in_bbox((-73.7, 45.5, -73.6, 45.6), only_underbuilt=True)
    sql, params = calls[1]
    assert params["only_underbuilt"] is True
    # A semi-join rather than a second layer the caller has to assemble.
    assert f"{queries.GOLD_SCHEMA}.lot_redevelopment_gap" in sql
    assert "is_underbuilt" in sql


def test_a_missing_massing_table_is_advisory_not_fatal(monkeypatch):
    """The map still answers without it, so it is the operator's note alone."""
    monkeypatch.setattr(
        queries, "query_one",
        lambda *_a, **_k: {
            "postgis": True, "pgvector": True, "lots": True, "buildings": True,
            "building_lots": True, "lot_features": True, "features": True,
            "massing": False, "chunks": True, "search_at_lot": True,
            "search_near": True,
        },
    )
    caps = queries.capabilities()

    assert caps.can_map
    assert not caps.massing
    table = f"{queries.GOLD_SCHEMA}.lot_building_massing"
    assert table in caps.missing()
    assert table not in caps.missing(include_advisory=False)


def test_streets_read_the_silver_schema(captured):
    """The one layer drawn straight off a silver table rather than a scrape."""
    calls, _ = captured
    queries.streets_in_bbox((-73.7, 45.5, -73.6, 45.6))
    sql, _params = calls[0]
    assert f"{queries.SILVER_SCHEMA}.neighborhood_streets" in sql


def test_street_features_carry_what_the_tooltip_reads(captured):
    _calls, rows = captured
    rows.append(
        {
            "cote_rue_id": "1234567",
            "neighborhood": "VSMPE",
            "scrape_date": date(2026, 8, 20),
            "street_name": "Rue Jarry Est",
            "length_m": 82.4,
            "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
        }
    )

    found = queries.streets_in_bbox((-73.7, 45.5, -73.6, 45.6))
    properties = found.features[0]["properties"]

    assert found.layer == "streets"
    # The publisher's own key, unique across the island - not a surrogate a
    # reload would mint again.
    assert properties["id"] == "1234567"
    assert properties["street_name"] == "Rue Jarry Est"
    assert properties["length_m"] == 82.4


def test_streets_are_bbox_and_limit_bounded_like_every_other_layer(captured):
    calls, _ = captured
    queries.streets_in_bbox((-73.7, 45.5, -73.6, 45.6), limit=25)
    sql, params = calls[0]
    assert params["limit"] == 26
    assert "ST_MakeEnvelope" in sql and "ST_Intersects" in sql


def test_a_missing_streets_table_is_advisory_not_fatal(monkeypatch):
    """The map still answers without it, so it is the operator's note alone."""
    monkeypatch.setattr(
        queries, "query_one",
        lambda *_a, **_k: {
            "postgis": True, "pgvector": True, "lots": True, "buildings": True,
            "building_lots": True, "lot_features": True, "features": True,
            "streets": False, "chunks": True, "search_at_lot": True,
            "search_near": True,
        },
    )
    caps = queries.capabilities()

    assert caps.can_map
    assert not caps.streets
    table = f"{queries.SILVER_SCHEMA}.neighborhood_streets"
    assert table in caps.missing()
    assert table not in caps.missing(include_advisory=False)


def test_every_map_partition_table_is_asked_for_in_its_own_schema():
    """Two of the three are gold answers and the third is a silver scrape; the
    only thing they share is the (scrape_date, neighborhood) pair asked about.
    A schema assumed rather than carried is how the streets check would end up
    probing `gold.neighborhood_streets`."""
    schemas = {
        layer: schema
        for layer, (schema, _table, _asset) in queries.MAP_PARTITION_TABLES.items()
    }
    assert schemas["massing"] == queries.GOLD_SCHEMA
    assert schemas["capacity"] == queries.GOLD_SCHEMA
    assert schemas["streets"] == queries.SILVER_SCHEMA


def test_simplify_tolerance_shrinks_as_zoom_grows():
    """Simplification has to track the pixel, or a zoomed-in lot loses corners."""
    assert queries.simplify_tolerance(19) < queries.simplify_tolerance(15)
    assert queries.simplify_tolerance(15) < queries.simplify_tolerance(12)
    # A pixel at zoom 19 is well under a metre; a metre is ~1e-5 degrees.
    assert queries.simplify_tolerance(19) < 1e-4


# ---------------------------------------------------------------------------
# One lot
# ---------------------------------------------------------------------------


def test_lot_by_number_ignores_separators(monkeypatch):
    captured = {}

    def fake_query_one(sql, params=None):
        captured.update(sql=sql, params=params)
        return None

    monkeypatch.setattr(queries, "query_one", fake_query_one)
    queries.lot_by_number("2170935")

    # Both sides are stripped of non-digits in SQL, so the caller may pass
    # either spelling.
    assert captured["sql"].count("regexp_replace") == 2
    assert captured["params"]["lot_number"] == "2170935"


def test_zoning_for_lot_orders_by_real_overlap(monkeypatch, silver):
    """The fallback, for a borough the pipeline has not joined yet."""
    silver(lot_features=False)
    captured = {}
    monkeypatch.setattr(
        queries, "query",
        lambda sql, params=None: captured.update(sql=sql, params=params) or [],
    )
    queries.zoning_for_lot("2 170 935")

    # geography, so the overlap is square metres and comparable across latitudes.
    assert "ST_Area(ST_Intersection(f.geom, lot.geom)::geography)" in captured["sql"]
    assert "ORDER BY overlap_m2 DESC" in captured["sql"]


def test_zoning_for_lot_returns_one_row_per_zone(monkeypatch, silver):
    """A zone is (neighborhood, feature_id), and it recurs once per snapshot.

    Without the DISTINCT ON, a caller that passes no ``scrape_date`` gets the
    same zone back once per date in the database, and the Lot pane reports a
    lot as straddling two zones that are one zone twice.
    """
    silver(lot_features=True)
    captured = {}
    monkeypatch.setattr(
        queries, "query",
        lambda sql, params=None: captured.update(sql=sql, params=params) or [{}],
    )
    queries.zoning_for_lot("2 170 935")

    assert "DISTINCT ON (lf.neighborhood, lf.feature_id)" in captured["sql"]
    # The row kept is the newest snapshot's - the one the rest of the pane
    # is reading.
    assert "lf.scrape_date DESC" in captured["sql"]


def test_zoning_for_lot_drops_a_sliver_of_the_zone_next_door(monkeypatch, silver):
    """A square metre is the two publishers disagreeing, not a second zone."""
    silver(lot_features=True)
    captured = {}
    monkeypatch.setattr(
        queries, "query",
        lambda sql, params=None: captured.update(sql=sql, params=params) or [{}],
    )
    queries.zoning_for_lot("2 170 935")

    assert "lf.overlap_area_m2 >= %(min_overlap_m2)s" in captured["sql"]
    assert captured["params"]["min_overlap_m2"] == queries.MIN_ZONE_OVERLAP_M2

    # And on the fallback, filtered on the same clip it reports rather than a
    # second ST_Intersection.
    silver(lot_features=False)
    queries.zoning_for_lot("2 170 935")
    clip = "ST_Area(ST_Intersection(f.geom, lot.geom)::geography)"
    assert captured["sql"].count(clip) == 1
    assert "WHERE overlap_m2 >= %(min_overlap_m2)s" in captured["sql"]


def test_zoning_for_lot_drops_a_zone_covering_under_one_per_cent(
    monkeypatch, silver
):
    """The sliver a square metre lets through, on both paths.

    Lot 6 291 714 is the case: 1.19 m2 of C03-130 on a 438 m2 parcel otherwise
    entirely in H03-126. That clears `MIN_ZONE_OVERLAP_M2` and is a quarter of
    a per cent of the lot, so the absolute cutoff alone left the pane offering
    a commercial zone beside the residential one and rounding its share to 0%.
    """
    silver(lot_features=True)
    captured = {}
    monkeypatch.setattr(
        queries, "query",
        lambda sql, params=None: captured.update(sql=sql, params=params) or [{}],
    )
    queries.zoning_for_lot("6 291 714")

    assert "lf.pct_of_lot >= %(min_pct_of_lot)s" in captured["sql"]
    assert captured["params"]["min_pct_of_lot"] == queries.MIN_ZONE_PCT_OF_LOT

    # The fallback has no precomputed pct_of_lot to read, so it divides the
    # clip it already reports rather than clipping a second time.
    silver(lot_features=False)
    queries.zoning_for_lot("6 291 714")
    clip = "ST_Area(ST_Intersection(f.geom, lot.geom)::geography)"
    assert captured["sql"].count(clip) == 1
    assert "100.0 * overlap_m2 / lot_area_m2 >= %(min_pct_of_lot)s" in captured["sql"]
    assert captured["params"]["min_pct_of_lot"] == queries.MIN_ZONE_PCT_OF_LOT


def test_the_fallback_keeps_a_zone_on_a_lot_of_no_area(monkeypatch, silver):
    """A percentage of zero is not a rejection, it is an unanswerable question.

    The precomputed path stores `pct_of_lot` as 0 for such a lot and the row
    goes; the fallback divides in SQL, and dividing by zero there would take
    out the whole query rather than the row. It keeps whatever cleared the
    absolute cutoff instead.
    """
    silver(lot_features=False)
    captured = {}
    monkeypatch.setattr(
        queries, "query",
        lambda sql, params=None: captured.update(sql=sql, params=params) or [],
    )
    queries.zoning_for_lot("2 170 935")

    assert "lot_area_m2 IS NULL OR lot_area_m2 <= 0" in captured["sql"]


def test_zoning_at_point_returns_one_row_per_zone(monkeypatch):
    """The same DISTINCT ON: the newest snapshot's zone, once."""
    captured = {}
    monkeypatch.setattr(
        queries, "query",
        lambda sql, params=None: captured.update(sql=sql, params=params) or [],
    )
    queries.zoning_at_point(-73.62, 45.54)

    assert "DISTINCT ON (f.neighborhood, f.feature_id)" in captured["sql"]
    # No area threshold here: a point is in a zone or it is not, and there is
    # no lot for a sliver to be a sliver of.
    assert "min_overlap_m2" not in captured["sql"]


def test_zoning_for_lot_reads_the_precomputed_join_when_it_is_there(monkeypatch, silver):
    """One index lookup instead of an ST_Intersection per zone per click."""
    silver(lot_features=True)
    sent = []
    monkeypatch.setattr(
        queries, "query",
        lambda sql, params=None: sent.append(sql) or [{"zone": "C01-001"}],
    )
    rows = queries.zoning_for_lot("2 170 935")

    assert rows == [{"zone": "C01-001"}]
    assert len(sent) == 1, "the fallback ran even though the join was available"
    assert f"FROM {queries.SILVER_SCHEMA}.lot_features" in sent[0]
    # attributes is not on the silver row, and the Lot pane renders it.
    assert f"JOIN {queries.SCHEMA}.features f" in sent[0]
    assert "ST_Intersection" not in sent[0]


def test_zoning_for_lot_falls_back_when_the_partition_has_no_rows(monkeypatch, silver):
    """The table exists but this borough-day is not in it yet.

    The two paths return the same column names, so a caller cannot tell which
    one answered — which is the property that lets the fallback stay silent.
    """
    silver(lot_features=True)
    sent = []
    monkeypatch.setattr(
        queries, "query", lambda sql, params=None: sent.append(sql) or []
    )
    queries.zoning_for_lot("2 170 935")

    assert len(sent) == 2
    assert "ST_Intersection" in sent[1]


def test_buildings_on_lot_reads_the_precomputed_join_by_lot_number(monkeypatch, silver):
    """Keyed on the lot number, not on a join back to rag.lots.

    `lot_uid` is a bigserial the pipeline mints again on every reload; the
    number is what survives one, and the silver table carries it for that
    reason.
    """
    silver(building_lots=True)
    sent = []
    monkeypatch.setattr(
        queries, "query",
        lambda sql, params=None: sent.append((sql, params)) or [{"building_uid": 1}],
    )
    queries.buildings_on_lot("2 170 935")

    sql, params = sent[0]
    assert f"FROM {queries.SILVER_SCHEMA}.building_lot_intersections" in sql
    assert "bl.lot_number = %(lot_number)s" in sql
    assert f"JOIN {queries.SCHEMA}.lots" not in sql
    assert params["lot_number"] == "2 170 935"


def test_buildings_on_lot_falls_back_to_the_intersection(monkeypatch, silver):
    silver(building_lots=False)
    sent = []
    monkeypatch.setattr(
        queries, "query", lambda sql, params=None: sent.append(sql) or []
    )
    queries.buildings_on_lot("2 170 935")

    assert len(sent) == 1
    assert "ST_Intersection" in sent[0]


def test_the_buildings_on_lot_fallback_screens_the_neighbours_wall(
    monkeypatch, silver
):
    """The screen the pipeline applies, on the path that computes its own clip.

    Lot 3 791 059 is the case: one 155 m2 house standing on it, plus 4.15 m2 of
    the house next door and 0.93 m2 of a shed clipping the corner, all three
    reported as buildings. The fallback had the dimension test and not this
    one, so the same lot answered two different counts depending on whether the
    pipeline had reached this borough.
    """
    silver(building_lots=False)
    sent = []
    monkeypatch.setattr(
        queries, "query",
        lambda sql, params=None: sent.append((sql, params)) or [],
    )
    queries.buildings_on_lot("3 791 059")

    sql, params = sent[0]
    assert queries._BUILDING_CLIP_SCREEN in sql
    assert params["min_building_overlap_m2"] == queries.MIN_BUILDING_OVERLAP_M2
    assert (
        params["min_building_pct_of_building"]
        == queries.MIN_BUILDING_PCT_OF_BUILDING
    )
    # The percentage half is computed here rather than left NULL, because it is
    # half of the screen above and the pane reads the column either way.
    assert "AS pct_of_building" in sql
    assert "NULL::float8 AS pct_of_building" not in sql


def test_the_building_screen_is_an_or_not_an_and():
    """The one thing about this screen that would be catastrophic reversed.

    A townhouse standing wholly on its own parcel is a small *percentage* of
    the block-long outline BDOI digitised it inside - over VSMPE the median
    clip between 3 and 10 per cent of its building is about 100 m2, a whole
    house. Requiring both cutoffs the way the zone cutoffs are required would
    delete 9 224 of 31 815 rows, most of them real buildings.
    """
    screen = queries._BUILDING_CLIP_SCREEN

    absolute = screen.index("%(min_building_overlap_m2)s")
    share = screen.index("%(min_building_pct_of_building)s")
    between = screen[absolute:share]

    # The only `AND` between the two cutoffs is the one guarding the division
    # inside the second, so the `OR` has to come first for the two to be
    # alternatives rather than requirements.
    assert " OR " in between
    assert between.index(" OR ") < between.index(" AND ")


def test_buildings_on_lot_counts_a_building_once(monkeypatch, silver):
    """The table's grain is the intersection; the pane's is the footprint.

    One row per (building, lot) means a caller counting rows counts
    intersections, and both the footprint count and the covered-area sum the
    Lot pane adds up were multiplied by every repeat.
    """
    silver(building_lots=True)
    sent = []
    monkeypatch.setattr(
        queries, "query",
        lambda sql, params=None: sent.append(sql) or [{"building_uid": 1}],
    )
    queries.buildings_on_lot("2 170 935")

    sql = sent[0]
    assert "DISTINCT ON (building_uid)" in sql
    assert "ORDER BY building_uid, overlap_m2 DESC" in sql


def test_buildings_on_lot_answers_from_one_snapshot(monkeypatch, silver):
    """`building_uid` is reminted on every load, so no distinct can do this.

    Without the screen a lot in a database holding two dates reports every
    footprint twice, under two ids that look like two buildings.
    """
    silver(building_lots=True)
    sent = []
    monkeypatch.setattr(
        queries, "query",
        lambda sql, params=None: sent.append(sql) or [{"building_uid": 1}],
    )
    queries.buildings_on_lot("2 170 935")

    assert "WHERE scrape_date = (SELECT max(scrape_date) FROM matched)" in sent[0]


def test_the_buildings_on_lot_fallback_answers_from_one_snapshot(monkeypatch, silver):
    """The same screen on the slow path: one lot geometry, one date of shapes."""
    silver(building_lots=False)
    sent = []
    monkeypatch.setattr(
        queries, "query", lambda sql, params=None: sent.append(sql) or []
    )
    queries.buildings_on_lot("2 170 935")

    assert "ST_Intersection" in sent[0]
    assert "WHERE scrape_date = (SELECT max(scrape_date) FROM matched)" in sent[0]


# ---------------------------------------------------------------------------
# Lot coverage — the measured taux d'implantation
#
# The bug these were written for: lot 2 165 628 is 311.6 m² with one building
# on it, 164.5 m² of which falls inside the lot — 53% covered. The Lot pane
# reported "2 footprint(s), 329 m²", 106% of a parcel that cannot be more than
# 100% covered, because the pane summed `buildings_on_lot` rows across every
# snapshot in the database and the borough had been loaded twice.
# ---------------------------------------------------------------------------

#: The lot the bug was reported on, and its true numbers in VSMPE 2026-09-01.
BUG_LOT = "2 165 628"
BUG_LOT_AREA_M2 = 311.6195007413626
BUG_COVERED_M2 = 164.48598719830625


@pytest.fixture
def one_row(monkeypatch):
    """Capture every `query_one`, answering each call from a queue of rows."""
    calls: list[tuple[str, object]] = []
    replies: list[dict | None] = []

    def fake_query_one(sql, params=None):
        calls.append((sql, params))
        return replies.pop(0) if replies else None

    monkeypatch.setattr(queries, "query_one", fake_query_one)
    return calls, replies


def coverage_row(**overrides) -> dict:
    row = {
        "lot_number": BUG_LOT,
        "lot_uid": 167933,
        "neighborhood": "VSMPE",
        "scrape_date": date(2026, 9, 1),
        "lot_area_m2": BUG_LOT_AREA_M2,
        "num_footprints": 1,
        "covered_area_m2": BUG_COVERED_M2,
    }
    row.update(overrides)
    return row


def test_coverage_pct_has_no_denominator_without_a_lot_area():
    """A lot with no recorded area has no coverage — 0/0 is not "0% built"."""
    assert queries.coverage_pct(120.0, None) is None
    assert queries.coverage_pct(120.0, 0) is None


def test_coverage_pct_does_not_clamp_an_impossible_share():
    """Over 100% is a signal, and a clamp would hide the day it reappears."""
    assert queries.coverage_pct(329.0, 311.6195007413626) > 100


def test_lot_coverage_reports_one_building_on_the_lot_it_was_reported_on(
    one_row, silver
):
    """The regression: 1 footprint over 311.6 m², not 2 over 329 m².

    The numbers are lot 2 165 628's own, and the assertion that matters is the
    last one — covered ground cannot exceed the ground there is.
    """
    silver(building_lots=True)
    _, replies = one_row
    replies.append(coverage_row())

    coverage = queries.lot_coverage(BUG_LOT, scrape_date=date(2026, 9, 1))

    assert coverage["num_footprints"] == 1
    assert coverage["covered_area_m2"] == pytest.approx(BUG_COVERED_M2)
    assert coverage["coverage_pct"] == pytest.approx(52.784, abs=0.001)
    assert coverage["covered_area_m2"] <= coverage["lot_area_m2"]


def test_lot_coverage_pairs_the_footprints_with_the_lots_own_snapshot(one_row, silver):
    """The date is the join key, not something screened for afterwards.

    This is the whole bug in one assertion. A coverage read across every load
    of the borough counts one footprint once per load, and the sum passes the
    lot's own area without anything noticing.
    """
    silver(building_lots=True)
    calls, replies = one_row
    replies.append(coverage_row())

    queries.lot_coverage(BUG_LOT, scrape_date=date(2026, 9, 1))

    sql, params = calls[0]
    assert "bl.scrape_date  = lot.scrape_date" in sql
    assert "bl.neighborhood = lot.neighborhood" in sql
    assert "max(scrape_date)" not in sql
    assert params["scrape_date"] == date(2026, 9, 1)


def test_lot_coverage_unions_the_clipped_shapes_rather_than_summing_them(
    one_row, silver
):
    """Ground under two overlapping footprints is covered once.

    Summing `intersection_area_m2` reports lot 2 249 834 as 186% built; the
    union of the same shapes reports 93%, which is what is on the ground.
    """
    silver(building_lots=True)
    calls, replies = one_row
    replies.append(coverage_row())

    queries.lot_coverage(BUG_LOT)

    sql, _ = calls[0]
    assert "ST_Area(ST_Union(clipped.geom)::geography)" in sql
    assert "sum(" not in sql.lower()
    # The silver grain is the (building, lot) intersection, so rows are not
    # footprints even inside one snapshot.
    assert "count(DISTINCT clipped.building_uid)" in sql


def test_lot_coverage_falls_back_when_the_silver_join_is_absent(one_row, silver):
    silver(building_lots=False)
    calls, replies = one_row
    replies.append(coverage_row())

    queries.lot_coverage(BUG_LOT)

    assert len(calls) == 1
    sql, params = calls[0]
    assert "ST_Intersection" in sql
    assert queries.SILVER_SCHEMA not in sql
    # The pipeline's own rule, both halves of it: a footprint sharing an edge
    # with the lot line intersects it and covers none of it, and a footprint
    # crossing the line by a hand's breadth covers a sliver that is not a
    # building here either.
    assert queries._BUILDING_CLIP_SCREEN in sql
    assert params["min_building_overlap_m2"] == queries.MIN_BUILDING_OVERLAP_M2
    assert (
        params["min_building_pct_of_building"]
        == queries.MIN_BUILDING_PCT_OF_BUILDING
    )


def test_lot_coverage_re_asks_the_slow_way_when_the_fast_path_finds_nothing(
    one_row, silver
):
    """The join table exists but this borough-day is not in it yet."""
    silver(building_lots=True)
    calls, replies = one_row
    replies.append(coverage_row(num_footprints=0, covered_area_m2=0.0))
    replies.append(coverage_row())

    coverage = queries.lot_coverage(BUG_LOT)

    assert len(calls) == 2
    assert "ST_Intersection" in calls[1][0]
    assert coverage["num_footprints"] == 1


def test_a_vacant_lot_keeps_its_area_after_both_paths_answer_empty(one_row, silver):
    """Nothing built is an answer about the lot, not an absent lot."""
    silver(building_lots=True)
    _, replies = one_row
    replies.append(coverage_row(num_footprints=0, covered_area_m2=0.0))
    replies.append(coverage_row(num_footprints=0, covered_area_m2=0.0))

    coverage = queries.lot_coverage(BUG_LOT)

    assert coverage["num_footprints"] == 0
    assert coverage["coverage_pct"] == 0.0
    assert coverage["lot_area_m2"] == pytest.approx(BUG_LOT_AREA_M2)


def test_lot_coverage_is_none_when_no_lot_carries_the_number(one_row, silver):
    silver(building_lots=True)
    assert queries.lot_coverage("9 999 999") is None


def test_lot_coverage_matches_a_lot_number_typed_without_its_spaces(one_row, silver):
    """The same normalisation `lot_by_number` does, for the same reason."""
    silver(building_lots=True)
    calls, replies = one_row
    replies.append(coverage_row())

    queries.lot_coverage("2165628")

    sql, params = calls[0]
    assert "regexp_replace(l.lot_number, '\\D', '', 'g')" in sql
    assert params["lot_number"] == "2165628"


def test_lot_documents_is_keyed_on_the_lot_uid(monkeypatch):
    """One lot_uid is one lot in one snapshot, so no date is passed with it."""
    sent = []
    monkeypatch.setattr(
        queries, "query",
        lambda sql, params=None: sent.append((sql, params)) or [],
    )
    queries.lot_documents(4242)

    sql, params = sent[0]
    assert f"FROM {queries.SCHEMA}.lot_documents d" in sql
    assert params["lot_uid"] == 4242
    assert "scrape_date" not in params


def test_lot_documents_returns_one_row_per_document(monkeypatch):
    """A sheet cited by both zones of a split lot is one PDF, not two.

    The view is one row per (lot, feature, document); offering the reader the
    same grid twice would read as two sets of rules.
    """
    sent = []
    monkeypatch.setattr(
        queries, "query",
        lambda sql, params=None: sent.append((sql, params)) or [],
    )
    queries.lot_documents(4242)

    sql, _params = sent[0]
    assert "GROUP BY doc_id, url, title" in sql
    assert "array_agg(feature_id" in sql
    # Deduplicated to one row per (document, feature) before the sum, so a
    # document whose chunks disagree about feature_ids is not counted twice.
    assert "DISTINCT ON (d.doc_id, d.feature_id)" in sql


def test_lot_documents_drops_a_sliver_of_the_zone_next_door(monkeypatch):
    """The same two cutoffs the Lot pane applies, applied to the sheets.

    Both, because a sheet reaches a lot through a zone and a zone can be over
    one cutoff and under the other - which is how the block next door's grid
    used to arrive beside the parcel's own.
    """
    sent = []
    monkeypatch.setattr(
        queries, "query",
        lambda sql, params=None: sent.append((sql, params)) or [],
    )
    queries.lot_documents(4242)

    sql, params = sent[0]
    assert params["min_overlap_m2"] == queries.MIN_ZONE_OVERLAP_M2
    assert params["min_pct_of_lot"] == queries.MIN_ZONE_PCT_OF_LOT
    assert "d.overlap_area_m2 >= %(min_overlap_m2)s" in sql
    assert "d.pct_of_lot >= %(min_pct_of_lot)s" in sql


def test_lot_documents_can_be_narrowed_to_one_layer(monkeypatch):
    sent = []
    monkeypatch.setattr(
        queries, "query",
        lambda sql, params=None: sent.append((sql, params)) or [],
    )
    queries.lot_documents(4242, source_table=queries.ZONING_SOURCE_TABLE)

    sql, params = sent[0]
    assert params["source_table"] == queries.ZONING_SOURCE_TABLE
    # NULL means every layer rather than none, so the filter is optional in SQL.
    assert "%(source_table)s::text IS NULL" in sql


def test_the_lot_document_join_is_advisory(monkeypatch):
    """Missing, it costs the reader documents no attribute links - not the pane.

    `_documents_for_lot` falls back to the zoning rows' own LIEN_GRILLE, so a
    user-facing message must not report a fault that did not happen.
    """
    monkeypatch.setattr(
        queries, "query_one",
        lambda *_a, **_k: {
            "postgis": True, "pgvector": True, "lots": True, "buildings": True,
            "building_lots": True, "lot_features": True, "features": True,
            "chunks": True, "lot_documents": False,
            "search_at_lot": True, "search_near": True,
        },
    )
    caps = queries.capabilities()

    assert not caps.lot_documents
    assert caps.can_retrieve
    assert f"{queries.SCHEMA}.lot_documents" in caps.missing()
    assert f"{queries.SCHEMA}.lot_documents" not in caps.missing(include_advisory=False)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def test_vector_literal_is_pgvector_input_format():
    assert queries._vector_literal([0.5, -0.25, 0.125]) == "[0.5,-0.25,0.125]"


def test_vector_literal_accepts_integers_and_numpy_like_floats():
    assert queries._vector_literal([1, 2]) == "[1,2]"


def test_search_at_lot_passes_lon_before_lat(monkeypatch):
    """The function signature is (embedding, lon, lat, k) — swapping is silent."""
    captured = {}
    monkeypatch.setattr(
        queries, "query",
        lambda sql, params=None: captured.update(sql=sql, params=params) or [],
    )
    queries.search_at_lot([0.1, 0.2], lon=-73.6, lat=45.5, match_count=3)

    _embedding, lon, lat, count = captured["params"]
    assert lon == -73.6
    assert lat == 45.5
    assert count == 3


def test_search_near_forwards_the_radius_and_date(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        queries, "query",
        lambda sql, params=None: captured.update(params=params) or [],
    )
    queries.search_near([0.1], -73.6, 45.5, radius_m=250, match_count=4,
                        on_scrape_date=date(2026, 8, 20))

    assert captured["params"][3] == 250
    assert captured["params"][5] == date(2026, 8, 20)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_table_name_interpolation_is_allowlisted():
    """These names reach SQL as text rather than as a bound parameter."""
    assert queries._safe("lots") == "lots"
    with pytest.raises(ValueError):
        queries._safe("lots; DROP TABLE rag.chunks")


def test_jsonable_converts_dates():
    assert queries._jsonable(date(2026, 8, 20)) == "2026-08-20"
    assert queries._jsonable(None) is None
    assert queries._jsonable(3) == 3


# ---------------------------------------------------------------------------
# search_corpus — two things that only fail against a real server
# ---------------------------------------------------------------------------


def test_ef_search_is_inlined_as_an_int_inside_a_transaction(monkeypatch):
    """`SET` takes no bound parameters, and `SET LOCAL` needs a transaction.

    The pool's connections are autocommit, so without an explicit transaction
    this setting would be a no-op Postgres only warns about — and passing the
    value as a parameter is a syntax error rather than a subtle one.
    """
    statements: list[str] = []

    class _Cursor:
        def execute(self, sql, params=None):
            statements.append(sql)

        def fetchall(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    class _Transaction:
        entered = False

        def __enter__(self):
            _Transaction.entered = True
            return self

        def __exit__(self, *_exc):
            return False

    class _Connection:
        def transaction(self):
            return _Transaction()

        def cursor(self, **_kwargs):
            return _Cursor()

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr("src.utils.db.connection", lambda: _Connection())

    queries.search_corpus([0.1, 0.2], match_count=5)

    assert _Transaction.entered, "SET LOCAL outside a transaction is a no-op"
    set_statement = next(s for s in statements if s.startswith("SET LOCAL"))
    assert set_statement == "SET LOCAL hnsw.ef_search = 100"
    assert "%s" not in set_statement


def test_ef_search_widens_with_the_match_count():
    """The index returns candidates and the WHERE is applied to them after, so
    filtering is a reason to ask for more of them, not fewer."""
    assert int(max(100, 4 * 5)) == 100
    assert int(max(100, 4 * 50)) == 200


# ---------------------------------------------------------------------------
# The surface parking layer
# ---------------------------------------------------------------------------
#
# The massing's other polygon, and a layer of its own because it is another
# kind of thing: a surface stall has no floor area, no storey and no height, so
# it is not part of the building and a map that extruded it would raise a solid
# where there is asphalt.


def parking_row(**overrides) -> dict:
    row = {
        "lot_uid": 4211,
        "lot_number": "2 170 935",
        "feature_id": "H03-126",
        "neighborhood": "VSMPE",
        "scrape_date": date(2026, 8, 20),
        "parking_status": "fitted",
        "surface_stalls": 4,
        "placed_surface_stalls": 4.0,
        "surface_parking_area_m2": 111.5,
        "placed_surface_parking_m2": 111.5,
        "surface_parking_fit_pct": 100.0,
        "parking_width_m": 20.3,
        "parking_depth_m": 5.5,
        "num_parking_bays": 1,
        "yard_area_m2": 184.0,
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0]]]},
    }
    row.update(overrides)
    return row


def test_surface_parking_reads_its_own_table(captured):
    """Its own table, not a second geometry column on the massing's.

    The two come apart: a lot whose building fits and whose parking does not
    belongs in one and not the other, and a lot that parks underground belongs
    in neither this table nor a hole in that one.
    """
    calls, _ = captured
    queries.surface_parking_in_bbox((-73.7, 45.5, -73.6, 45.6))
    sql, _params = calls[0]
    assert f"{queries.GOLD_SCHEMA}.lot_surface_parking" in sql
    assert f"{queries.GOLD_SCHEMA}.lot_building_massing" not in sql


def test_surface_parking_features_carry_what_the_tooltip_reads(captured):
    _calls, rows = captured
    rows.append(parking_row())

    found = queries.surface_parking_in_bbox((-73.7, 45.5, -73.6, 45.6))
    properties = found.features[0]["properties"]

    assert found.layer == "surface_parking"
    assert properties["id"] == "2 170 935@H03-126"
    assert properties["parking_status"] == "fitted"
    assert properties["placed_surface_stalls"] == 4.0
    assert properties["num_parking_bays"] == 1
    assert properties["scrape_date"] == "2026-08-20"


def test_surface_parking_is_bbox_and_limit_bounded_like_every_other_layer(captured):
    calls, _ = captured
    queries.surface_parking_in_bbox((-73.7, 45.5, -73.6, 45.6), limit=25)
    sql, params = calls[0]
    assert params["limit"] == 26
    assert "ST_MakeEnvelope" in sql and "ST_Intersects" in sql


def test_surface_parking_takes_the_same_under_built_screen(captured):
    """The two layers are one answer, so one filter covers both.

    A screen that hid the building and left its parking on the map would be
    drawing half a proposal.
    """
    calls, _ = captured
    queries.surface_parking_in_bbox((-73.7, 45.5, -73.6, 45.6))
    _sql, params = calls[0]
    assert params["only_underbuilt"] is False

    queries.surface_parking_in_bbox(
        (-73.7, 45.5, -73.6, 45.6), only_underbuilt=True
    )
    sql, params = calls[1]
    assert params["only_underbuilt"] is True
    assert f"{queries.GOLD_SCHEMA}.lot_redevelopment_gap" in sql
    assert "is_underbuilt" in sql


def test_a_missing_parking_table_is_advisory_and_leaves_the_massing_alone(
    monkeypatch,
):
    """Probed apart from the massing, because the two genuinely come apart.

    A borough materialized before sql/024 existed has every building and no
    asphalt. Greying out the one toggle is a truer thing to show than an empty
    layer, which reads as a borough that parks nowhere.
    """
    monkeypatch.setattr(
        queries, "query_one",
        lambda *_a, **_k: {
            "postgis": True, "pgvector": True, "lots": True, "buildings": True,
            "building_lots": True, "lot_features": True, "features": True,
            "massing": True, "surface_parking": False, "chunks": True,
            "search_at_lot": True, "search_near": True,
        },
    )
    caps = queries.capabilities()

    assert caps.can_map
    assert caps.massing
    assert not caps.surface_parking
    assert f"{queries.GOLD_SCHEMA}.lot_surface_parking" in caps.missing()
    # Advisory, so it is the operator's note and never a user-facing fault.
    assert (
        f"{queries.GOLD_SCHEMA}.lot_surface_parking"
        not in caps.missing(include_advisory=False)
    )
