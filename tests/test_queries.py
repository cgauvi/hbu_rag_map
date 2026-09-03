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


def test_the_zoning_layer_is_selected_by_its_slug(captured):
    calls, _ = captured
    queries.zones_in_bbox((-73.7, 45.5, -73.6, 45.6))
    _sql, params = calls[0]
    assert params["source_table"] == "Reglement_urbanisme__VSP_REG_ZONE"
    assert params["url_attribute"] == "LIEN_GRILLE"


# ---------------------------------------------------------------------------
# The massing layer
# ---------------------------------------------------------------------------


def massing_row(**overrides) -> dict:
    row = {
        "lot_uid": 4211,
        "lot_number": "2 170 935",
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
    assert properties["id"] == 4211
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
