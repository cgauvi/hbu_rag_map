"""Shared fixtures.

The unit suite never opens a socket: the database, the HuggingFace endpoint and
the city's web server are all stubbed. Anything that genuinely needs a live
database is marked ``integration`` and deselected by pyproject's addopts.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def database_url() -> str | None:
    """The URL as it was before ``_clean_environment`` scrubbed it.

    Session-scoped so it is captured once, at collection, ahead of the autouse
    fixture that empties the environment for the unit tests.
    """
    return os.environ.get("HBU_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    """Start every test from a known environment.

    ``src.utils.db`` reads four different environment contracts, and a stray
    ``DATABASE_URL`` in the developer's shell would otherwise silently change
    which branch a resolution test exercises.

    ``HBU_APP_PASSWORD`` is here for a sharper reason than a stray shell, and
    it is the one variable in this list that a *test* puts in the environment
    rather than the developer. `app.py` calls ``load_dotenv()`` above the
    access gate, so the first integration test to run the script publishes
    every name in ``.env`` into this process — and a filled-in ``.env`` has a
    password in it. Nothing restores it, because monkeypatch never set it.

    What that costs is not the auth tests, which set the variable themselves.
    It is the *tile server* tests: `tiles.tile_key` reads this at request time
    and derives the key from it, so every keyless URL those tests build starts
    coming back ``403``, in a suite that never configured a gate. The failure
    lands nowhere near its cause, appears only when the integration tests run
    in the same process, and reads as a broken tile server.
    """
    for name in (
        "DATABASE_URL",
        "URBAN_RAG_PG_DSN",
        "URBAN_RAG_PG_HOST",
        "URBAN_RAG_PG_PORT",
        "URBAN_RAG_PG_DATABASE",
        "URBAN_RAG_PG_USER",
        "URBAN_RAG_PG_PASSWORD",
        "URBAN_RAG_PG_SECRET_ID",
        "URBAN_RAG_PG_IAM_AUTH",
        "URBAN_RAG_PG_SSLMODE",
        "URBAN_RAG_PG_SSLROOTCERT",
        "URBAN_RAG_EMBEDDING_MODEL",
        "HUGGINGFACE_API_TOKEN",
        "HF_MODEL_ID",
        # `auth.PASSWORD_ENV`, spelled out: importing the app to read the name
        # would run the script this fixture exists to isolate.
        "HBU_APP_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture(autouse=True)
def _reset_resolution():
    """Drop `db.resolve`'s memo between tests.

    The memo is keyed on the environment, so a test that changes a variable
    already re-resolves. This covers the other half: a test that monkeypatches
    `_from_ssm` or `_secret_password` leaves the environment identical to the
    last one's, and would otherwise be served that test's answer.
    """
    from src.utils import db

    db.clear_resolved()
    yield
    db.clear_resolved()


@pytest.fixture(autouse=True)
def _reset_tile_capability_probe():
    """Drop `queries._building_lots_available`'s memo between tests.

    It is a module-level cache with a five-minute TTL, so without this the
    first test to answer "is the silver join there" answers it for the rest of
    the session — and the two branches of the buildings tile are chosen on
    exactly that answer.
    """
    from src.utils import queries

    queries._building_lots_probe = None
    yield
    queries._building_lots_probe = None


@pytest.fixture(autouse=True)
def _reset_state():
    """Clear the module-level buffers between tests."""
    from src.utils import state

    state.clear_map_command()
    state.clear_rag_buffer()
    state.clear_selected_lot()
    state.set_viewport(None, None, None)
    yield
    state.clear_map_command()
    state.clear_rag_buffer()


@pytest.fixture
def lot_row() -> dict:
    """One row shaped like queries.lot_by_number returns."""
    return {
        "lot_number": "2 170 935",
        "neighborhood": "VSMPE",
        "scrape_date": "2026-08-20",
        "area_m2": 267.9227,
        "attributes": {"CO_STATT_LOT": "AC"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-73.62, 45.54], [-73.62, 45.541],
                             [-73.619, 45.541], [-73.619, 45.54],
                             [-73.62, 45.54]]],
        },
        "lon": -73.6195,
        "lat": 45.5405,
        "bbox": [-73.62, 45.54, -73.619, 45.541],
    }


@pytest.fixture
def zone_row() -> dict:
    """One row shaped like queries.zoning_for_lot returns."""
    return {
        "zone": "C01-001",
        "source_table": "Reglement_urbanisme__VSP_REG_ZONE",
        "neighborhood": "VSMPE",
        "scrape_date": "2026-08-20",
        "attributes": {
            "NUMERO_COMPLET": "C01-001",
            "USAGE": "C.4;H",
            "ETAGE_MIN": "2",
            "ETAGE_MAX": "6",
            "METRE_MAX": "23",
            "TAUX_IMP_MIN": "50",
            "TAUX_IMP_MAX": "70",
            "LIEN_GRILLE": "http://www1.ville.montreal.qc.ca/CartesInteractives/"
                           "villeray/doc/zone/C01-001.pdf",
        },
        "zoning_pdf_url": "http://www1.ville.montreal.qc.ca/CartesInteractives/"
                          "villeray/doc/zone/C01-001.pdf",
        "overlap_m2": 267.0,
        "lot_area_m2": 267.9,
    }


@pytest.fixture
def hf_token(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "hf_test_token")
    return "hf_test_token"


def pytest_configure(config):
    os.environ.setdefault("APP_ENV", "test")
