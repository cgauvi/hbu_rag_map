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


#: A parcel a zoning boundary crosses, in VSMPE: 70 095 m², 51 263 of it in
#: E04-064 and 18 806 in E04-065, with a 14 830 m² building all but entirely
#: on the first. The Lot pane's E04-065 side used to report the parcel's
#: footprint against a plate proposed for a piece that carries 83 m² of it.
SPLIT_LOT = "3 237 014"


@pytest.fixture
def split_lot_pieces():
    caps = queries.capabilities()
    if not (caps.building_lots and caps.lot_features):
        pytest.skip("this database has no zone pieces or no silver clip")
    pieces = queries.query(
        f"""
        SELECT p.lot_number, p.feature_id, p.scrape_date, p.neighborhood,
               p.piece_area_m2, p.footprint_share
          FROM {queries.SILVER_SCHEMA}.lot_zone_pieces p
         WHERE regexp_replace(p.lot_number, '\\D', '', 'g')
             = regexp_replace(%(lot)s, '\\D', '', 'g')
           AND p.scrape_date = (
                   SELECT max(scrape_date)
                     FROM {queries.SILVER_SCHEMA}.lot_zone_pieces
                    WHERE regexp_replace(lot_number, '\\D', '', 'g')
                        = regexp_replace(%(lot)s, '\\D', '', 'g')
               )
         ORDER BY p.zone_rank
        """,
        {"lot": SPLIT_LOT},
    )
    if len(pieces) < 2:
        pytest.skip(f"lot {SPLIT_LOT} is not a split lot in this database")
    return pieces


def _piece_coverages(pieces) -> dict[str, dict]:
    out = {}
    for piece in pieces:
        coverage = queries.piece_coverage(
            piece["lot_number"], piece["feature_id"],
            scrape_date=piece["scrape_date"], neighborhood=piece["neighborhood"],
        )
        assert coverage is not None, piece["feature_id"]
        out[piece["feature_id"]] = coverage
    return out


def test_a_split_lots_pieces_carry_the_parcels_ground_between_them(split_lot_pieces):
    """The footprint clipped to each piece sums to the footprint clipped to
    the lot, and no piece is more covered than it is large."""
    first = split_lot_pieces[0]
    parcel = queries.lot_coverage(first["lot_number"], scrape_date=first["scrape_date"])
    coverages = _piece_coverages(split_lot_pieces)

    for coverage in coverages.values():
        assert coverage["covered_area_m2"] <= coverage["piece_area_m2"] + TOLERANCE_M2
        assert coverage["scrape_date"] == first["scrape_date"]
    assert sum(c["covered_area_m2"] for c in coverages.values()) == pytest.approx(
        parcel["covered_area_m2"], abs=2 * TOLERANCE_M2
    )


def test_the_building_on_the_split_lot_is_on_one_piece_and_not_the_other(
    split_lot_pieces,
):
    """The regression, on the parcel it was reported on: the E04-065 side of
    the pane reads the ground the E04-065 piece carries, which is next to
    nothing, and the pieces agree with the shares the gap table divides by."""
    first = split_lot_pieces[0]
    parcel = queries.lot_coverage(first["lot_number"], scrape_date=first["scrape_date"])
    coverages = _piece_coverages(split_lot_pieces)
    if not {"E04-064", "E04-065"} <= coverages.keys():
        pytest.skip(f"lot {SPLIT_LOT} is no longer cut by E04-064 and E04-065")

    assert coverages["E04-065"]["covered_area_m2"] < 0.01 * parcel["covered_area_m2"]
    assert coverages["E04-064"]["covered_area_m2"] > 0.99 * parcel["covered_area_m2"]
    assert coverages["E04-065"]["coverage_pct"] < 1.0
    for piece in split_lot_pieces:
        share = piece.get("footprint_share")
        if share is None:
            continue
        measured = coverages[piece["feature_id"]]["covered_area_m2"] / parcel["covered_area_m2"]
        assert measured == pytest.approx(float(share), abs=0.01)


def test_the_rolls_units_on_a_split_lot_are_counted_once_between_its_pieces(
    split_lot_pieces,
):
    """Each record is on exactly one piece - the one its address falls in."""
    if not queries.capabilities().assessment_units:
        pytest.skip("this database has no assessment units")
    first = split_lot_pieces[0]
    parcel = queries.lot_roll_units(first["lot_number"], scrape_date=first["scrape_date"])
    if not parcel or not parcel.get("roll_loaded"):
        pytest.skip("the roll is not loaded for this snapshot")

    per_piece = [
        queries.lot_roll_units(
            piece["lot_number"], scrape_date=piece["scrape_date"],
            feature_id=piece["feature_id"], neighborhood=piece["neighborhood"],
        )
        for piece in split_lot_pieces
    ]
    assert sum(int(r["num_units"]) for r in per_piece) == int(parcel["num_units"])
    assert sum(int(r["num_nonresidential_units"]) for r in per_piece) == int(
        parcel["num_nonresidential_units"]
    )
