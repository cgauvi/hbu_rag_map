"""
queries.py — Every statement this app sends, in one file.

Two reasons it is not spread across the tools that call it. The first is that
the schema belongs to two other repos — ``rag.features``, ``rag.lots`` and the
search functions to `hbu_infra`, ``rag.chunks`` to `hbu_dataplatform` — so the
set of assumptions this app makes about their shape is worth being able to read
in one sitting. The second is that the map and the agent ask the *same*
questions: clicking a lot and asking "what applies here" both end at
``rag.search_at_lot``, and duplicating that in two layers is how they drift.

Geometry comes back as GeoJSON, already simplified, because the only consumer
is folium. Simplifying server-side means the wire carries the vertices that get
drawn rather than the vertices Infolot recorded — a borough's worth of lots is
several megabytes of coordinates at full precision and a few hundred kilobytes
at screen precision.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date

from src.utils.db import SCHEMA, query, query_one, scalar

logger = logging.getLogger(__name__)

#: The scraped layer whose rows carry a link to a zoning grid PDF, and the
#: attribute holding it. Mirrors ``DOCUMENT_SOURCES`` in the dataplatform's
#: rag/documents.py — the same registry, read from the other end. Values in
#: ``rag.features.source_table`` are the file *slug* the scrape writes, not the
#: Spectrum path, because that is what ``rag.chunks.source_table`` holds and
#: the two are joined on it.
ZONING_SOURCE_TABLE = os.environ.get(
    "HBU_ZONING_SOURCE_TABLE", "Reglement_urbanisme__VSP_REG_ZONE"
)
ZONING_URL_ATTRIBUTE = os.environ.get("HBU_ZONING_URL_ATTRIBUTE", "LIEN_GRILLE")

#: The zoning-grid attributes worth showing next to a lot, in reading order,
#: with the label the grid itself uses. Anything not listed still reaches the
#: agent through the raw attributes; this is what the Lot pane renders.
ZONING_FIELDS: tuple[tuple[str, str], ...] = (
    ("NUMERO_COMPLET", "Zone"),
    ("USAGE", "Usages autorisés"),
    ("USAGE_AUT", "Usages autorisés (suite)"),
    ("USAGE_EXC", "Usages exclus"),
    ("ETAGE_MIN", "Étages min"),
    ("ETAGE_MAX", "Étages max"),
    ("METRE_MIN", "Hauteur min (m)"),
    ("METRE_MAX", "Hauteur max (m)"),
    ("TAUX_IMP_MIN", "Taux d'implantation min (%)"),
    ("TAUX_IMP_MAX", "Taux d'implantation max (%)"),
    ("COS_MIN", "COS min"),
    ("COS_MAX", "COS max"),
    ("IMPLANTATION", "Mode d'implantation"),
    ("RDC_COMMERCIAL", "RDC commercial"),
    ("SECTEUR_PAT", "Secteur patrimonial"),
    ("SECTEUR_PIIA", "Secteur PIIA"),
    ("CAT_AFFICHAGE", "Catégorie d'affichage"),
)

#: A viewport query returns at most this many shapes. Past it the map is a
#: solid block of outlines and the browser is the bottleneck, not the database.
DEFAULT_FEATURE_LIMIT = int(os.environ.get("HBU_MAP_FEATURE_LIMIT", 2000))


# ---------------------------------------------------------------------------
# What the database actually has
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Capabilities:
    """Which halves of the schema are present.

    Checked once per session rather than assumed, because the three repos land
    in a database independently: `hbu_infra` creates the geometry tables, the
    dataplatform's ``document_index`` asset creates ``rag.chunks`` on its first
    load, and ``rag.buildings`` arrives with whoever loads BDOI. A map that
    silently draws nothing is worse than one that says which table is missing.
    """

    postgis: bool = False
    pgvector: bool = False
    lots: bool = False
    buildings: bool = False
    building_lots: bool = False
    features: bool = False
    chunks: bool = False
    search_at_lot: bool = False
    search_near: bool = False

    @property
    def can_map(self) -> bool:
        return self.postgis and (self.lots or self.buildings or self.features)

    @property
    def can_retrieve(self) -> bool:
        return self.pgvector and self.chunks

    def missing(self) -> list[str]:
        """Human-readable names of what is absent, for the status pane."""
        checks = {
            "postgis extension": self.postgis,
            "vector extension": self.pgvector,
            f"{SCHEMA}.lots": self.lots,
            f"{SCHEMA}.buildings": self.buildings,
            f"{SCHEMA}.building_lots": self.building_lots,
            f"{SCHEMA}.features": self.features,
            f"{SCHEMA}.chunks": self.chunks,
            f"{SCHEMA}.search_at_lot()": self.search_at_lot,
            f"{SCHEMA}.search_near()": self.search_near,
        }
        return [name for name, present in checks.items() if not present]


def capabilities() -> Capabilities:
    """One round trip that answers every "does this exist" question."""
    row = query_one(
        """
        SELECT
          (SELECT count(*) FROM pg_extension WHERE extname = 'postgis')  > 0 AS postgis,
          (SELECT count(*) FROM pg_extension WHERE extname = 'vector')   > 0 AS pgvector,
          to_regclass(%(schema)s || '.lots')     IS NOT NULL AS lots,
          to_regclass(%(schema)s || '.buildings') IS NOT NULL AS buildings,
          to_regclass(%(schema)s || '.building_lots') IS NOT NULL AS building_lots,
          to_regclass(%(schema)s || '.features') IS NOT NULL AS features,
          to_regclass(%(schema)s || '.chunks')   IS NOT NULL AS chunks,
          (SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE n.nspname = %(schema)s AND p.proname = 'search_at_lot') > 0 AS search_at_lot,
          (SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE n.nspname = %(schema)s AND p.proname = 'search_near') > 0 AS search_near
        """,
        {"schema": SCHEMA},
    )
    return Capabilities(**row) if row else Capabilities()


def corpus_dimension() -> int | None:
    """The vector width the corpus was built at, from ``rag.chunks_meta``.

    Querying a 1024-wide corpus with a 384-wide encoder does not error, it
    returns confident nonsense — so the width is checked before the first
    search rather than after a user disbelieves an answer.
    """
    if scalar(f"SELECT to_regclass('{SCHEMA}.chunks_meta') IS NOT NULL") is not True:
        return None
    value = scalar(
        f"SELECT value FROM {SCHEMA}.chunks_meta WHERE key = 'dimension'"
    )
    return int(value) if value is not None else None


def corpus_model() -> str | None:
    """The encoder the corpus was built with, so a query can match it."""
    if scalar(f"SELECT to_regclass('{SCHEMA}.chunks_meta') IS NOT NULL") is not True:
        return None
    return scalar(f"SELECT value FROM {SCHEMA}.chunks_meta WHERE key = 'embedding_model'")


def corpus_status() -> list[dict]:
    """Per neighborhood and scrape date: documents, chunks, features loaded."""
    if scalar(f"SELECT to_regclass('{SCHEMA}.corpus_status') IS NOT NULL") is True:
        return query(f"SELECT * FROM {SCHEMA}.corpus_status")
    # The view lives in 003_spatial_search.sql, which is skipped until
    # rag.chunks exists. Fall back to counting the table directly.
    return query(
        f"""
        SELECT neighborhood, scrape_date,
               count(DISTINCT doc_id) AS documents,
               count(*)               AS chunks
          FROM {SCHEMA}.chunks
         GROUP BY neighborhood, scrape_date
         ORDER BY scrape_date DESC, neighborhood
        """
    )


# ---------------------------------------------------------------------------
# Partitions — which snapshot the map is looking at
# ---------------------------------------------------------------------------


def neighborhoods(table: str = "lots") -> list[str]:
    return [
        r["neighborhood"]
        for r in query(
            f"SELECT DISTINCT neighborhood FROM {SCHEMA}.{_safe(table)} ORDER BY 1"
        )
    ]


def scrape_dates(table: str = "lots", neighborhood: str | None = None) -> list[date]:
    """Snapshot dates present, newest first."""
    return [
        r["scrape_date"]
        for r in query(
            f"""
            SELECT DISTINCT scrape_date
              FROM {SCHEMA}.{_safe(table)}
             WHERE (%s::text IS NULL OR neighborhood = %s)
             ORDER BY scrape_date DESC
            """,
            (neighborhood, neighborhood),
        )
    ]


def latest_scrape_date(table: str = "lots", neighborhood: str | None = None) -> date | None:
    dates = scrape_dates(table, neighborhood)
    return dates[0] if dates else None


# ---------------------------------------------------------------------------
# Viewport queries
# ---------------------------------------------------------------------------


@dataclass
class FeatureSet:
    """A GeoJSON FeatureCollection plus what the limit did to it."""

    features: list[dict] = field(default_factory=list)
    truncated: bool = False
    layer: str = ""

    @property
    def count(self) -> int:
        return len(self.features)

    def collection(self) -> dict:
        return {"type": "FeatureCollection", "features": self.features}


def _bbox_params(bounds: tuple[float, float, float, float]) -> dict:
    west, south, east, north = bounds
    return {"west": west, "south": south, "east": east, "north": north}


def simplify_tolerance(zoom: int) -> float:
    """Degrees of simplification appropriate to a zoom level.

    A screen pixel is roughly ``360 / (256 * 2**zoom)`` degrees of longitude, so
    collapsing vertices below that is invisible by construction. Doubling it
    trades a pixel of fidelity for roughly half the coordinates.
    """
    return 360.0 / (256.0 * (2 ** max(zoom, 1))) * 2.0


def lots_in_bbox(
    bounds: tuple[float, float, float, float],
    *,
    zoom: int = 16,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
    min_area_m2: float | None = None,
    max_area_m2: float | None = None,
    lot_numbers: list[str] | None = None,
    limit: int = DEFAULT_FEATURE_LIMIT,
) -> FeatureSet:
    """Cadastral lots intersecting the visible rectangle.

    ``geom && ST_MakeEnvelope`` is the part the GiST index answers; the
    ``ST_Intersects`` after it is what turns the index's bounding-box answer
    into a real one. Both are needed — the operator alone returns lots whose
    envelope overlaps the viewport while their shape does not.
    """
    params = _bbox_params(bounds)
    params.update(
        {
            "tolerance": simplify_tolerance(zoom),
            "scrape_date": scrape_date,
            "neighborhood": neighborhood,
            "min_area": min_area_m2,
            "max_area": max_area_m2,
            "lot_numbers": lot_numbers,
            "limit": limit + 1,
        }
    )
    rows = query(
        f"""
        SELECT l.lot_uid,
               l.lot_number,
               l.neighborhood,
               l.scrape_date,
               COALESCE(l.area_m2, ST_Area(l.geom::geography)) AS area_m2,
               l.attributes,
               ST_AsGeoJSON(
                   ST_SimplifyPreserveTopology(l.geom, %(tolerance)s)
               )::json AS geometry
          FROM {SCHEMA}.lots l
         WHERE l.geom && ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326)
           AND ST_Intersects(l.geom,
                   ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326))
           AND (%(scrape_date)s::date IS NULL OR l.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR l.neighborhood = %(neighborhood)s)
           AND (%(min_area)s::float8 IS NULL
                OR COALESCE(l.area_m2, ST_Area(l.geom::geography)) >= %(min_area)s)
           AND (%(max_area)s::float8 IS NULL
                OR COALESCE(l.area_m2, ST_Area(l.geom::geography)) <= %(max_area)s)
           AND (%(lot_numbers)s::text[] IS NULL OR l.lot_number = ANY(%(lot_numbers)s))
         LIMIT %(limit)s
        """,
        params,
    )
    return _as_feature_set(rows, layer="lots", id_key="lot_number", limit=limit)


def buildings_in_bbox(
    bounds: tuple[float, float, float, float],
    *,
    zoom: int = 16,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
    limit: int = DEFAULT_FEATURE_LIMIT,
) -> FeatureSet:
    """Building footprints intersecting the visible rectangle.

    Same shape as :func:`lots_in_bbox`, with one difference worth knowing:
    ``rag.buildings`` has no natural key. BDOI carries no id that survives an
    extract, so a load replaces a whole ``(neighborhood, scrape_date)``
    partition and the surrogate ``building_uid`` is the only handle a footprint
    has — which means it is stable within a snapshot and meaningless across two.
    """
    params = _bbox_params(bounds)
    params.update(
        {
            "tolerance": simplify_tolerance(zoom),
            "scrape_date": scrape_date,
            "neighborhood": neighborhood,
            "limit": limit + 1,
        }
    )
    rows = query(
        f"""
        SELECT b.building_uid,
               b.neighborhood,
               b.scrape_date,
               COALESCE(b.area_m2, ST_Area(b.geom::geography)) AS area_m2,
               b.attributes,
               ST_AsGeoJSON(
                   ST_SimplifyPreserveTopology(b.geom, %(tolerance)s)
               )::json AS geometry
          FROM {SCHEMA}.buildings b
         WHERE b.geom && ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326)
           AND ST_Intersects(b.geom,
                   ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326))
           AND (%(scrape_date)s::date IS NULL OR b.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR b.neighborhood = %(neighborhood)s)
         LIMIT %(limit)s
        """,
        params,
    )
    return _as_feature_set(rows, layer="buildings", id_key="building_uid", limit=limit)


def zones_in_bbox(
    bounds: tuple[float, float, float, float],
    *,
    zoom: int = 14,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
    limit: int = DEFAULT_FEATURE_LIMIT,
) -> FeatureSet:
    """Zoning polygons — the layer that carries the link to the grid PDF.

    Far coarser than lots (633 against 24,953 for one borough), so this is the
    layer worth drawing when zoomed out past the point where a lot is a pixel.
    """
    params = _bbox_params(bounds)
    params.update(
        {
            "tolerance": simplify_tolerance(zoom),
            "scrape_date": scrape_date,
            "neighborhood": neighborhood,
            "source_table": ZONING_SOURCE_TABLE,
            "url_attribute": ZONING_URL_ATTRIBUTE,
            "limit": limit + 1,
        }
    )
    rows = query(
        f"""
        SELECT f.feature_id,
               f.neighborhood,
               f.scrape_date,
               f.attributes,
               f.attributes ->> %(url_attribute)s AS zoning_pdf_url,
               ST_AsGeoJSON(
                   ST_SimplifyPreserveTopology(f.geom, %(tolerance)s)
               )::json AS geometry
          FROM {SCHEMA}.features f
         WHERE f.source_table = %(source_table)s
           AND f.geom && ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326)
           AND ST_Intersects(f.geom,
                   ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326))
           AND (%(scrape_date)s::date IS NULL OR f.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR f.neighborhood = %(neighborhood)s)
         LIMIT %(limit)s
        """,
        params,
    )
    return _as_feature_set(rows, layer="zones", id_key="feature_id", limit=limit)


def _as_feature_set(rows: list[dict], *, layer: str, id_key: str, limit: int) -> FeatureSet:
    """Turn query rows into GeoJSON features, noting whether the limit bit.

    Each query asks for ``limit + 1`` rows; getting them back is how the caller
    learns the viewport holds more than it is being shown, without a second
    ``count(*)`` over the same predicate.
    """
    truncated = len(rows) > limit
    features = []
    for row in rows[:limit]:
        geometry = row.pop("geometry", None)
        if geometry is None:
            continue
        attributes = row.pop("attributes", None) or {}
        properties = {k: _jsonable(v) for k, v in row.items()}
        properties["layer"] = layer
        properties["id"] = properties.get(id_key)
        properties["attributes"] = attributes
        features.append(
            {"type": "Feature", "geometry": geometry, "properties": properties}
        )
    if truncated:
        logger.info("%s: %d shapes in view, showing %d", layer, len(rows), limit)
    return FeatureSet(features=features, truncated=truncated, layer=layer)


def _jsonable(value):
    """Dates and Decimals through folium's JSON serialiser unharmed."""
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "__float__") and not isinstance(value, (int, float, bool)):
        return float(value)
    return value


# ---------------------------------------------------------------------------
# One lot
# ---------------------------------------------------------------------------


def lot_at_point(lon: float, lat: float, *, scrape_date: date | None = None) -> dict | None:
    """The lot a map click fell in.

    Resolved server-side from the click's coordinates rather than from whatever
    shape the browser thinks was hit: a click near a boundary, on a lot the
    viewport limit left undrawn, or on a simplified edge all still land on the
    right parcel this way. Newest snapshot wins when several are loaded — a
    cadastre changes rarely, but the current shape is the one that matters.
    """
    return query_one(
        f"""
        SELECT l.lot_uid,
               l.lot_number,
               l.neighborhood,
               l.scrape_date,
               COALESCE(l.area_m2, ST_Area(l.geom::geography)) AS area_m2,
               l.attributes,
               ST_AsGeoJSON(l.geom)::json AS geometry,
               ST_X(ST_PointOnSurface(l.geom)) AS lon,
               ST_Y(ST_PointOnSurface(l.geom)) AS lat,
               ARRAY[ST_XMin(l.geom), ST_YMin(l.geom),
                     ST_XMax(l.geom), ST_YMax(l.geom)]::float8[] AS bbox
          FROM {SCHEMA}.lots l
         WHERE ST_Intersects(l.geom, ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326))
           AND (%(scrape_date)s::date IS NULL OR l.scrape_date = %(scrape_date)s)
         ORDER BY l.scrape_date DESC
         LIMIT 1
        """,
        {"lon": lon, "lat": lat, "scrape_date": scrape_date},
    )


def lot_by_number(lot_number: str, *, scrape_date: date | None = None) -> dict | None:
    """A lot by its cadastral number.

    Infolot spells it with thin spaces — ``"2 170 935"`` — and nobody types it
    that way, so digits are compared with the separators stripped from both
    sides.
    """
    return query_one(
        f"""
        SELECT l.lot_uid,
               l.lot_number,
               l.neighborhood,
               l.scrape_date,
               COALESCE(l.area_m2, ST_Area(l.geom::geography)) AS area_m2,
               l.attributes,
               ST_AsGeoJSON(l.geom)::json AS geometry,
               ST_X(ST_PointOnSurface(l.geom)) AS lon,
               ST_Y(ST_PointOnSurface(l.geom)) AS lat,
               ARRAY[ST_XMin(l.geom), ST_YMin(l.geom),
                     ST_XMax(l.geom), ST_YMax(l.geom)]::float8[] AS bbox
          FROM {SCHEMA}.lots l
         WHERE regexp_replace(l.lot_number, '\\D', '', 'g')
             = regexp_replace(%(lot_number)s, '\\D', '', 'g')
           AND (%(scrape_date)s::date IS NULL OR l.scrape_date = %(scrape_date)s)
         ORDER BY l.scrape_date DESC
         LIMIT 1
        """,
        {"lot_number": lot_number, "scrape_date": scrape_date},
    )


def buildings_on_lot(lot_number: str, *, scrape_date: date | None = None) -> list[dict]:
    """Footprints standing on a lot, largest overlap first.

    Reads ``rag.building_lots`` when it is there. That table already holds each
    footprint clipped to each lot it falls in, with the clipped area and the
    share of the building it represents, so this is an index lookup on
    ``lot_uid`` rather than an ``ST_Intersection`` per query — and it gets the
    hard case right for free: a school or a tower spanning several parcels has
    one row per lot, each carrying only the portion inside it, with no notion
    of a "primary" lot to guess at.

    Falls back to computing the intersection when the join table has not been
    built for this partition yet, so a freshly loaded borough still answers.
    """
    if capabilities().building_lots:
        rows = query(
            f"""
            SELECT bl.building_uid,
                   bl.building_area_m2       AS area_m2,
                   bl.intersection_area_m2   AS overlap_m2,
                   bl.pct_of_building,
                   b.attributes
              FROM {SCHEMA}.building_lots bl
              JOIN {SCHEMA}.lots l ON l.lot_uid = bl.lot_uid
              JOIN {SCHEMA}.buildings b ON b.building_uid = bl.building_uid
             WHERE l.lot_number = %(lot_number)s
               AND (%(scrape_date)s::date IS NULL OR bl.scrape_date = %(scrape_date)s)
             ORDER BY bl.intersection_area_m2 DESC
             LIMIT 50
            """,
            {"lot_number": lot_number, "scrape_date": scrape_date},
        )
        if rows:
            return rows

    return query(
        f"""
        WITH lot AS (
            SELECT geom FROM {SCHEMA}.lots
             WHERE lot_number = %(lot_number)s
             ORDER BY scrape_date DESC
             LIMIT 1
        )
        SELECT b.building_uid,
               COALESCE(b.area_m2, ST_Area(b.geom::geography)) AS area_m2,
               ST_Area(ST_Intersection(b.geom, lot.geom)::geography) AS overlap_m2,
               NULL::float8 AS pct_of_building,
               b.attributes
          FROM {SCHEMA}.buildings b, lot
         WHERE b.geom && lot.geom
           AND ST_Intersects(b.geom, lot.geom)
           AND (%(scrape_date)s::date IS NULL OR b.scrape_date = %(scrape_date)s)
         ORDER BY overlap_m2 DESC
         LIMIT 50
        """,
        {"lot_number": lot_number, "scrape_date": scrape_date},
    )


def zoning_for_lot(lot_number: str, *, scrape_date: date | None = None) -> list[dict]:
    """The zoning polygons covering a lot, and the grid PDF each links to.

    Ordered by how much of the lot each zone actually covers, because a lot on
    a zone boundary intersects both and only one of them is the answer. The
    overlap is computed on ``geography`` so the number is square metres rather
    than square degrees.
    """
    return query(
        f"""
        WITH lot AS (
            SELECT geom FROM {SCHEMA}.lots
             WHERE lot_number = %(lot_number)s
             ORDER BY scrape_date DESC
             LIMIT 1
        )
        SELECT f.feature_id                          AS zone,
               f.source_table,
               f.neighborhood,
               f.scrape_date,
               f.attributes,
               f.attributes ->> %(url_attribute)s    AS zoning_pdf_url,
               ST_Area(ST_Intersection(f.geom, lot.geom)::geography) AS overlap_m2,
               ST_Area(lot.geom::geography)                          AS lot_area_m2
          FROM {SCHEMA}.features f, lot
         WHERE f.source_table = %(source_table)s
           AND f.geom && lot.geom
           AND ST_Intersects(f.geom, lot.geom)
           AND (%(scrape_date)s::date IS NULL OR f.scrape_date = %(scrape_date)s)
         ORDER BY overlap_m2 DESC
        """,
        {
            "lot_number": lot_number,
            "scrape_date": scrape_date,
            "source_table": ZONING_SOURCE_TABLE,
            "url_attribute": ZONING_URL_ATTRIBUTE,
        },
    )


def zoning_at_point(lon: float, lat: float, *, scrape_date: date | None = None) -> list[dict]:
    """The same, for a click that did not land on any lot."""
    return query(
        f"""
        SELECT f.feature_id                       AS zone,
               f.source_table,
               f.neighborhood,
               f.scrape_date,
               f.attributes,
               f.attributes ->> %(url_attribute)s AS zoning_pdf_url,
               NULL::float8                       AS overlap_m2,
               NULL::float8                       AS lot_area_m2
          FROM {SCHEMA}.features f
         WHERE f.source_table = %(source_table)s
           AND ST_Intersects(f.geom, ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326))
           AND (%(scrape_date)s::date IS NULL OR f.scrape_date = %(scrape_date)s)
         ORDER BY f.scrape_date DESC
        """,
        {
            "lon": lon,
            "lat": lat,
            "scrape_date": scrape_date,
            "source_table": ZONING_SOURCE_TABLE,
            "url_attribute": ZONING_URL_ATTRIBUTE,
        },
    )


def zoning_pdf_url_fallback(zone: str) -> str | None:
    """The grid PDF for a zone, via the corpus rather than via the attributes.

    ``rag.chunks`` records the URL it embedded and the feature ids that cited
    it, so a zone whose ``attributes`` lost its link — an older scrape, a layer
    the city reshaped — can still be resolved from the document side.
    """
    return scalar(
        f"""
        SELECT c.url
          FROM {SCHEMA}.chunks c
         WHERE c.source_table = %(source_table)s
           AND c.feature_ids ? %(zone)s
         ORDER BY c.scrape_date DESC
         LIMIT 1
        """,
        {"source_table": ZONING_SOURCE_TABLE, "zone": zone},
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _vector_literal(embedding: list[float]) -> str:
    """pgvector's text input format.

    Passed as a string with an explicit ``::vector`` cast rather than through
    pgvector's psycopg adapter, so retrieval needs no type registration on a
    connection borrowed from the pool.
    """
    return "[" + ",".join(f"{float(v):.7g}" for v in embedding) + "]"


def search_at_lot(
    embedding: list[float], lon: float, lat: float, *, match_count: int = 5
) -> list[dict]:
    """Everything the corpus says about the lot a point falls in.

    ``rag.search_at_lot`` — containment rather than proximity, so there is no
    radius to pick and no neighbouring zone bleeding into the answer.
    """
    return query(
        f"SELECT * FROM {SCHEMA}.search_at_lot(%s::vector, %s, %s, %s)",
        (_vector_literal(embedding), lon, lat, match_count),
    )


def search_near(
    embedding: list[float],
    lon: float,
    lat: float,
    *,
    radius_m: float = 500,
    match_count: int = 5,
    on_scrape_date: date | None = None,
) -> list[dict]:
    """Vector search narrowed to a radius around a point (``rag.search_near``).

    The spatial filter runs first and is selective, so the planner scans the
    surviving chunks rather than using the HNSW index — which makes recall here
    exact rather than approximate.
    """
    return query(
        f"SELECT * FROM {SCHEMA}.search_near(%s::vector, %s, %s, %s, %s, %s)",
        (_vector_literal(embedding), lon, lat, radius_m, match_count, on_scrape_date),
    )


def search_corpus(
    embedding: list[float],
    *,
    match_count: int = 5,
    neighborhood: str | None = None,
    scrape_date: date | None = None,
) -> list[dict]:
    """Unfiltered vector search — the question with no place attached.

    ``hnsw.ef_search`` is widened because the index returns its candidates and
    the ``WHERE`` clause is applied to them afterwards: filtering by
    neighborhood is a reason to ask for more candidates, not fewer.
    """
    from psycopg.rows import dict_row  # noqa: PLC0415

    from src.utils.db import connection  # noqa: PLC0415

    # SET takes no bound parameters, so the value is inlined — as an int, which
    # is where the coercion belongs rather than in a format string.
    ef_search = int(max(100, 4 * match_count))
    # SET LOCAL is scoped to a transaction, and the pool's connections are
    # autocommit, so without an explicit one this would be a no-op that
    # Postgres only warns about. The explicit block also returns the setting to
    # its default when the connection goes back to the pool.
    with connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
        cur.execute(
            f"""
            SELECT c.chunk_id, c.doc_id, c.url, c.title, c.source_table,
                   c.neighborhood, c.scrape_date, c.text AS chunk_text,
                   1 - (c.embedding <=> %(embedding)s::vector) AS similarity
              FROM {SCHEMA}.chunks c
             WHERE (%(neighborhood)s::text IS NULL OR c.neighborhood = %(neighborhood)s)
               AND (%(scrape_date)s::date IS NULL OR c.scrape_date = %(scrape_date)s)
             ORDER BY c.embedding <=> %(embedding)s::vector
             LIMIT %(match_count)s
            """,
            {
                "embedding": _vector_literal(embedding),
                "neighborhood": neighborhood,
                "scrape_date": scrape_date,
                "match_count": match_count,
            },
        )
        return list(cur.fetchall())


def _safe(identifier: str) -> str:
    """Guard the two table names that reach SQL as text rather than as a param."""
    allowed = {"lots", "buildings", "features", "chunks"}
    if identifier not in allowed:
        raise ValueError(f"{identifier!r} is not one of {sorted(allowed)}")
    return identifier
