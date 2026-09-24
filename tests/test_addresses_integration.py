"""Address lookup against a real database.

Marked ``integration``. The unit tests in `test_queries.py` pin what
`lots_by_address` sends; these check what it returns about addresses that
exist. The one thing no stub can say is whether the folding written in SQL
(``_STREET_KEY_SQL``) and the folding written in Python (`street_key`) agree,
and they are written twice on purpose - the query has to fold the stored name
where the row is, and the tool has to fold what was typed before it knows
whether to ask at all.

The addresses are read off the table rather than written down, so the tests
hold for whichever borough happens to be loaded.
"""

from __future__ import annotations

import unicodedata

import pytest

from src.utils import queries

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _database_url(monkeypatch, database_url):
    if not database_url:
        pytest.skip("set DATABASE_URL (or HBU_TEST_DATABASE_URL) to run these")
    monkeypatch.setenv("DATABASE_URL", database_url)
    from src.utils.db import close_pool

    close_pool()


@pytest.fixture(autouse=True)
def _needs_addresses():
    caps = queries.capabilities()
    if not (caps.lots and caps.lot_addresses):
        pytest.skip("this database has no lots or no lot addresses")


def _one(sql: str, params: dict | None = None) -> dict | None:
    rows = queries.query(sql, params or {})
    return rows[0] if rows else None


@pytest.fixture
def a_door() -> dict:
    """One door on a street printed with its type - accented if the borough has one."""
    row = _one(
        f"""
        SELECT lot_number, neighborhood, civic_number, street_name, civic_address
          FROM {queries.SILVER_SCHEMA}.lot_addresses
         WHERE is_primary_address AND civic_suffix IS NULL
           AND street_name ~ '^(Rue|Avenue|Boulevard) '
         ORDER BY (street_name ~ '[àâäéèêëîïôöùûüç]') DESC, neighborhood, lot_number
         LIMIT 1
        """
    )
    if not row:
        pytest.skip("no primary address on a typed street")
    return row


def test_an_address_as_the_layer_prints_it_finds_its_lot(a_door):
    rows = queries.lots_by_address(
        a_door["street_name"], a_door["civic_number"], neighborhood=a_door["neighborhood"]
    )
    assert rows, a_door
    assert rows[0]["lot_number"] == a_door["lot_number"]
    assert rows[0]["exact_name"] is True
    assert rows[0]["civic_address"] == a_door["civic_address"]
    assert rows[0]["num_civic_addresses"] >= 1
    assert rows[0]["num_lot_addresses"] >= rows[0]["num_addresses"]


def test_the_type_and_the_accents_may_be_left_out(a_door):
    """"LAJEUNESSE" for "Rue Lajeunesse", as a person types it."""
    bare = a_door["street_name"].split(" ", 1)[1]
    typed = "".join(
        c for c in unicodedata.normalize("NFKD", bare) if not unicodedata.combining(c)
    ).upper()

    rows = queries.lots_by_address(
        typed, a_door["civic_number"], neighborhood=a_door["neighborhood"]
    )

    assert rows and rows[0]["lot_number"] == a_door["lot_number"]


def test_the_lot_the_address_names_is_in_the_cadastre(a_door):
    """The tool selects through `lot_by_number`, so the join has to close."""
    lot = queries.lot_by_number(a_door["lot_number"])
    assert lot and lot["neighborhood"] == a_door["neighborhood"]


def test_the_street_alone_spans_the_door(a_door):
    summary = queries.street_summary(a_door["street_name"], neighborhood=a_door["neighborhood"])
    mine = [s for s in summary if s["street_name"] == a_door["street_name"]]
    assert len(mine) == 1
    assert mine[0]["civic_min"] <= a_door["civic_number"] <= mine[0]["civic_max"]
    assert mine[0]["exact_name"] is True


def test_a_street_nobody_named_matches_nothing():
    assert queries.lots_by_address("Zzyzx Qwertyuiop", 1) == []
    assert queries.street_summary("Zzyzx Qwertyuiop") == []


def test_python_and_sql_fold_every_loaded_street_the_same_way():
    """The two foldings are written twice; this is where they are compared."""
    rows = queries.query(
        f"""
        SELECT DISTINCT a.street_name, {queries._STREET_CORE_SQL} AS core
          FROM {queries.SILVER_SCHEMA}.lot_addresses a
        """,
        {"type_prefix": queries.STREET_TYPE_PREFIX_RE},
    )
    assert rows
    mismatched = [
        (r["street_name"], r["core"], queries.street_key(r["street_name"]))
        for r in rows
        if queries.street_key(r["street_name"]) != r["core"]
    ]
    assert not mismatched, mismatched[:10]


def test_coverage_names_every_borough_with_addresses():
    coverage = queries.address_coverage()
    assert coverage
    for row in coverage:
        assert row["num_addresses"] >= row["num_lots"] >= 1
