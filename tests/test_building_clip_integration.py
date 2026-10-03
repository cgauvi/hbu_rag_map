"""The buildings layer against a real database.

Marked ``integration``, and for the reason `test_coverage_integration` gives:
the unit tests pin the SQL's *text*, and every way this change can go wrong
lives in what the statement returns. Two of those are only visible against real
geometry — whether the clip actually splits a contiguous footprint between the
parcels it crosses, and whether the branch that computes the clip agrees with
the branch that reads it.

    make db-up && make test-integration

There is a third thing these catch that no stubbed suite can: whether the SQL
parses at all. The fallback tile is a lateral join assembled by string
interpolation, and a unit test asserting a substring is in it will pass on a
statement PostgreSQL refuses.
"""

from __future__ import annotations

import math
import time

import pytest

from src.utils import queries

pytestmark = pytest.mark.integration

#: Square metres of slack, for the reason `test_coverage_integration` gives: a
#: geography area comes off a spheroid in double precision, and the same shape
#: measured two ways lands within a few square centimetres.
TOLERANCE_M2 = 0.5

#: The zoom the buildings layer draws itself at. Below it a tile is an
#: aggregate and none of this applies.
DETAIL_ZOOM = queries.MVT_DETAIL_ZOOM["buildings"]


@pytest.fixture(autouse=True)
def _database_url(monkeypatch, database_url):
    if not database_url:
        pytest.skip("set DATABASE_URL (or HBU_TEST_DATABASE_URL) to run these")
    monkeypatch.setenv("DATABASE_URL", database_url)
    from src.utils.db import close_pool

    close_pool()


@pytest.fixture(autouse=True)
def _needs_both_tables():
    caps = queries.capabilities()
    if not (caps.lots and caps.buildings):
        pytest.skip("this database has no lots or no building footprints")


def _needs_silver():
    if not queries.capabilities().building_lots:
        pytest.skip("silver.building_lot_intersections is not in this database")


def _use_silver(present: bool) -> None:
    """Force one branch of the buildings layer, without touching the schema."""
    queries._building_lots_probe = (
        time.monotonic() + queries.TILE_CAPABILITY_TTL_S,
        present,
    )


def _tile_at(lon: float, lat: float, zoom: int) -> tuple[int, int, int]:
    """The slippy-map tile holding a point, in the grid Leaflet asks in."""
    n = 1 << zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return zoom, x, y


def _a_building_point() -> dict | None:
    """A point on some footprint in the newest snapshot, with its partition."""
    rows = queries.query(
        f"""
        SELECT ST_X(ST_PointOnSurface(b.geom)) AS lon,
               ST_Y(ST_PointOnSurface(b.geom)) AS lat,
               b.scrape_date,
               b.neighborhood
          FROM {queries.SCHEMA}.buildings b
         WHERE b.scrape_date = (
                   SELECT max(scrape_date) FROM {queries.SCHEMA}.buildings
               )
         ORDER BY b.building_uid
         LIMIT 1
        """
    )
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# The SQL runs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("silver", [True, False], ids=["precomputed", "computed"])
def test_the_buildings_tile_runs(silver):
    """Both branches, against PostGIS rather than against a substring check."""
    if silver:
        _needs_silver()
    point = _a_building_point()
    if point is None:
        pytest.skip("this database holds no footprints")

    _use_silver(silver)
    z, x, y = _tile_at(point["lon"], point["lat"], DETAIL_ZOOM)
    body = queries.mvt_tile(
        "buildings",
        z,
        x,
        y,
        scrape_date=point["scrape_date"],
        neighborhood=point["neighborhood"],
    )

    # Empty is a real answer over the river; on a tile centred on a footprint
    # it would mean the clip dropped everything.
    assert isinstance(body, bytes)
    assert len(body) > 0


@pytest.mark.parametrize("silver", [True, False], ids=["precomputed", "computed"])
def test_the_buildings_viewport_read_runs(silver):
    if silver:
        _needs_silver()
    point = _a_building_point()
    if point is None:
        pytest.skip("this database holds no footprints")

    _use_silver(silver)
    pad = 0.002
    found = queries.buildings_in_bbox(
        (
            point["lon"] - pad,
            point["lat"] - pad,
            point["lon"] + pad,
            point["lat"] + pad,
        ),
        scrape_date=point["scrape_date"],
        neighborhood=point["neighborhood"],
    )

    assert found.layer == "buildings"
    assert found.count > 0
    for feature in found.features:
        assert feature["properties"]["id"] is not None
        assert float(feature["properties"]["area_m2"]) > 0


# ---------------------------------------------------------------------------
# What the clip is for
# ---------------------------------------------------------------------------


def test_a_contiguous_footprint_is_split_between_the_parcels_it_crosses():
    """The overestimation this layer exists to stop.

    BDOI digitises a terrace as one outline across every party wall. Drawn
    whole, each of its parcels reports the *block's* area; clipped, each
    reports its own share, and the shares add back up to the block.
    """
    _needs_silver()
    rows = queries.query(
        f"""
        SELECT bl.building_uid,
               max(bl.building_area_m2)     AS whole_m2,
               count(*)                     AS parcels,
               sum(bl.intersection_area_m2) AS clipped_total_m2,
               max(bl.intersection_area_m2) AS largest_slice_m2
          FROM {queries.SILVER_SCHEMA}.building_lot_intersections bl
         WHERE bl.scrape_date = (
                   SELECT max(scrape_date)
                     FROM {queries.SILVER_SCHEMA}.building_lot_intersections
               )
         GROUP BY bl.building_uid
        HAVING count(*) > 1
         ORDER BY count(*) DESC, bl.building_uid
         LIMIT 5
        """
    )
    if not rows:
        pytest.skip("no footprint in this snapshot crosses a lot line")

    for row in rows:
        whole = float(row["whole_m2"])
        # Every slice is a *part* of the footprint. This is the assertion the
        # unclipped layer failed: it reported `whole` on each of them.
        assert float(row["largest_slice_m2"]) < whole, row["building_uid"]
        # And the parts are no more than the whole: the lots tile the building,
        # so nothing is added between them.
        assert float(row["clipped_total_m2"]) <= whole + TOLERANCE_M2, (
            row["building_uid"]
        )


def test_no_slice_is_larger_than_the_footprint_it_came_from():
    """The invariant, over the whole table rather than a sampled few."""
    _needs_silver()
    row = queries.query(
        f"""
        SELECT count(*) AS impossible
          FROM {queries.SILVER_SCHEMA}.building_lot_intersections bl
         WHERE bl.intersection_area_m2 > bl.building_area_m2 + %(tolerance)s
        """,
        {"tolerance": TOLERANCE_M2},
    )[0]
    assert int(row["impossible"]) == 0


def test_the_two_branches_of_the_layer_agree():
    """Read and computed are the same measurement.

    A borough whose silver join has not been built yet takes the fallback, and
    a map that meant one thing before the pipeline caught up and another after
    is the failure this whole change exists to avoid.
    """
    _needs_silver()
    point = _a_building_point()
    if point is None:
        pytest.skip("this database holds no footprints")

    pad = 0.001
    bounds = (
        point["lon"] - pad,
        point["lat"] - pad,
        point["lon"] + pad,
        point["lat"] + pad,
    )

    def areas() -> dict[str, float]:
        found = queries.buildings_in_bbox(
            bounds,
            # Full precision. The two branches simplify identically, but a
            # tolerance is one more thing standing between them and equality.
            zoom=22,
            scrape_date=point["scrape_date"],
            neighborhood=point["neighborhood"],
        )
        return {
            f["properties"]["building_lot_key"]: float(f["properties"]["area_m2"])
            for f in found.features
        }

    _use_silver(True)
    precomputed = areas()
    _use_silver(False)
    computed = areas()

    assert precomputed, "no footprints in this rectangle to compare"
    assert set(precomputed) == set(computed)
    for key, area in precomputed.items():
        assert computed[key] == pytest.approx(area, abs=TOLERANCE_M2), key
