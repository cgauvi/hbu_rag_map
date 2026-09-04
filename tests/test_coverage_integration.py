"""Lot coverage against a real database.

Marked ``integration``: the bug these exist for was invisible to a stubbed
query layer. `lot_coverage` is a few lines of Python around one statement, and
every way it went wrong lived in the statement — which snapshot the footprints
were taken from, and whether overlapping shapes were added up or unioned. The
unit tests in `test_queries.py` pin the SQL's text; these check what it
*returns* about ground that exists.

    make db-up && make test-integration

The expected values are re-derived from the database rather than written down,
because the two things being checked are invariants — a lot cannot be more than
fully covered, and a borough loaded twice does not grow buildings — and both
have to hold for whatever partitions happen to be loaded.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.utils import queries

pytestmark = pytest.mark.integration

#: The lot the coverage bug was reported on, in VSMPE. Reported as "2
#: footprint(s), 329 m²" on a 312 m² parcel — one building counted once per
#: snapshot loaded — where one building stands, covering 164 m² of it.
REPORTED_LOT = "2 165 628"

#: Square metres of slack on an area comparison. PostGIS answers a geography
#: area in double precision from a spheroid, and the same shape measured two
#: ways lands within a few square centimetres.
TOLERANCE_M2 = 0.5


@pytest.fixture(autouse=True)
def _database_url(monkeypatch, database_url):
    if not database_url:
        pytest.skip("set DATABASE_URL (or HBU_TEST_DATABASE_URL) to run these")
    monkeypatch.setenv("DATABASE_URL", database_url)
    from src.utils.db import close_pool

    close_pool()


@pytest.fixture(autouse=True)
def _needs_buildings():
    caps = queries.capabilities()
    if not (caps.lots and caps.buildings):
        pytest.skip("this database has no lots or no building footprints")


def _one(sql: str, params: dict | None = None) -> dict | None:
    rows = queries.query(sql, params or {})
    return rows[0] if rows else None


def test_the_reported_lot_carries_one_building_over_half_its_ground():
    """The regression, on the parcel it was reported on.

    Numbers are asserted as relations rather than as constants — one footprint,
    covering between a third and two thirds of the lot — so a re-scrape that
    moves a boundary by a metre does not fail this, while counting the same
    building once per snapshot still does.
    """
    coverage = queries.lot_coverage(REPORTED_LOT)
    if coverage is None:
        pytest.skip(f"lot {REPORTED_LOT} is not in this database")

    assert coverage["num_footprints"] == 1
    assert coverage["lot_area_m2"] == pytest.approx(311.6, abs=1.0)
    assert coverage["covered_area_m2"] == pytest.approx(164.5, abs=1.0)
    assert 33 < coverage["coverage_pct"] < 67


def test_a_lot_loaded_twice_is_not_covered_twice():
    """The bug itself: the same footprint, once per snapshot in the database.

    Finds a lot whose number carries footprints in more than one snapshot —
    which is every lot in a borough that has been re-scraped — and checks that
    the coverage reported is one snapshot's and not their sum.
    """
    row = _one(
        f"""
        SELECT bl.lot_number,
               count(DISTINCT bl.scrape_date)               AS snapshots,
               sum(bl.intersection_area_m2)                 AS every_snapshot_m2
          FROM {queries.SILVER_SCHEMA}.building_lot_intersections bl
         GROUP BY bl.lot_number
        HAVING count(DISTINCT bl.scrape_date) > 1
         ORDER BY bl.lot_number
         LIMIT 1
        """
    )
    if row is None:
        pytest.skip("this database holds only one snapshot of the footprints")

    coverage = queries.lot_coverage(row["lot_number"])

    assert coverage is not None
    assert coverage["covered_area_m2"] < float(row["every_snapshot_m2"])
    assert coverage["covered_area_m2"] <= coverage["lot_area_m2"] + TOLERANCE_M2


def test_no_lot_is_reported_as_more_than_fully_covered():
    """Ground covered cannot exceed the ground there is.

    Checked on the lots that break the arithmetic this replaced: the ones whose
    footprints overlap each other, where adding the clipped areas up passes the
    lot's own area. `lot_coverage` unions them, so those lots come back under
    100% — one of them read as 186% built before.
    """
    suspects = queries.query(
        f"""
        WITH per_lot AS (
            SELECT bl.lot_number,
                   bl.scrape_date,
                   sum(bl.intersection_area_m2)              AS summed_m2,
                   ST_Area(ST_Union(bl.geom)::geography)     AS union_m2
              FROM {queries.SILVER_SCHEMA}.building_lot_intersections bl
             WHERE bl.scrape_date = (
                       SELECT max(scrape_date)
                         FROM {queries.SILVER_SCHEMA}.building_lot_intersections
                   )
             GROUP BY bl.lot_number, bl.scrape_date
            HAVING count(*) > 1
        )
        SELECT lot_number, summed_m2, union_m2
          FROM per_lot
         WHERE summed_m2 > union_m2 * 1.001
         ORDER BY summed_m2 - union_m2 DESC
         LIMIT 5
        """
    )
    if not suspects:
        pytest.skip("no lot in this snapshot has footprints overlapping each other")

    for suspect in suspects:
        coverage = queries.lot_coverage(suspect["lot_number"])
        assert coverage is not None, suspect["lot_number"]
        assert coverage["covered_area_m2"] == pytest.approx(
            float(suspect["union_m2"]), abs=TOLERANCE_M2
        ), suspect["lot_number"]
        assert coverage["coverage_pct"] <= 100.0, suspect["lot_number"]


def test_the_two_paths_agree_on_what_covers_a_lot(monkeypatch):
    """The precomputed join and the live clip are the same measurement.

    A borough whose silver join has not been built yet takes the fallback and
    must get the same answer, or the pane reports a different coverage
    depending on how far the pipeline has run.
    """
    caps = queries.capabilities()
    if not caps.building_lots:
        pytest.skip("silver.building_lot_intersections is not in this database")

    row = _one(
        f"""
        SELECT bl.lot_number
          FROM {queries.SILVER_SCHEMA}.building_lot_intersections bl
         WHERE bl.scrape_date = (
                   SELECT max(scrape_date)
                     FROM {queries.SILVER_SCHEMA}.building_lot_intersections
               )
         ORDER BY bl.intersection_area_m2 DESC
         LIMIT 1
        """
    )
    fast = queries.lot_coverage(row["lot_number"])

    without_the_join = replace(caps, building_lots=False)
    monkeypatch.setattr(queries, "capabilities", lambda: without_the_join)
    slow = queries.lot_coverage(row["lot_number"])

    assert slow["num_footprints"] == fast["num_footprints"]
    assert slow["covered_area_m2"] == pytest.approx(
        fast["covered_area_m2"], abs=TOLERANCE_M2
    )


def test_coverage_is_measured_in_the_same_snapshot_as_the_lot():
    """Everything the Lot pane shows is one load of the cadastre, this included."""
    row = _one(
        f"""
        SELECT l.lot_number, l.scrape_date
          FROM {queries.SCHEMA}.lots l
         WHERE l.scrape_date < (SELECT max(scrape_date) FROM {queries.SCHEMA}.lots)
         ORDER BY l.lot_number
         LIMIT 1
        """
    )
    if row is None:
        pytest.skip("this database holds only one snapshot of the cadastre")

    coverage = queries.lot_coverage(row["lot_number"], scrape_date=row["scrape_date"])

    assert coverage["scrape_date"] == row["scrape_date"]
    assert coverage["covered_area_m2"] <= coverage["lot_area_m2"] + TOLERANCE_M2
