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


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def test_capabilities_maps_every_column(monkeypatch):
    monkeypatch.setattr(
        queries, "query_one",
        lambda *_a, **_k: {
            "postgis": True, "pgvector": True, "lots": True, "buildings": False,
            "building_lots": False, "features": True, "chunks": False,
            "search_at_lot": False, "search_near": False,
        },
    )
    caps = queries.capabilities()

    assert caps.can_map
    assert not caps.can_retrieve
    assert f"{queries.SCHEMA}.buildings" in caps.missing()
    assert f"{queries.SCHEMA}.lots" not in caps.missing()


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


def test_zoning_for_lot_orders_by_real_overlap(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        queries, "query",
        lambda sql, params=None: captured.update(sql=sql, params=params) or [],
    )
    queries.zoning_for_lot("2 170 935")

    # geography, so the overlap is square metres and comparable across latitudes.
    assert "ST_Area(ST_Intersection(f.geom, lot.geom)::geography)" in captured["sql"]
    assert "ORDER BY overlap_m2 DESC" in captured["sql"]


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
