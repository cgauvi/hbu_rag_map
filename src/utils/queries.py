"""
queries.py — Every statement this app sends, in one file.

Two reasons it is not spread across the tools that call it. The first is that
the schema belongs to two other repos — ``rag.features``, ``rag.lots``, the
``silver``/``gold`` tables and the search functions to `hbu_infra`,
``rag.chunks`` to `hbu_dataplatform` — so the set of assumptions this app makes
about their shape is worth being able to read in one sitting. The second is
that the map and the agent ask the *same* questions: clicking a lot and asking
"what applies here" both end at ``rag.search_at_lot``, and duplicating that in
two layers is how they drift.

**Two schemas, and the difference matters when reading a query below.**
``rag`` holds what the scrape loaded — the lots, the buildings, the map
features, the corpus — and is queried live. ``silver`` holds the joins the
pipeline has *already computed* between them, one table per asset, partitioned
by ``(neighborhood, scrape_date)``. Two of the reads here have a fast path off a
silver table and a fallback that computes the same thing with
``ST_Intersection``: `buildings_on_lot` and `zoning_for_lot`. The fallback is
not dead code — a borough loaded this morning has its ``rag`` rows before the
silver assets have run over them — so both paths have to keep returning the
same column names, and `capabilities()` is what chooses between them.

**Geometry leaves this file two ways, and the difference is what the map can
draw.** `mvt_tile` cuts a layer to one Mapbox Vector Tile, which is what the
map asks for now: a tile is bounded by its own area, so panning a borough
costs a constant amount of browser rather than a growing one. The
``*_in_bbox`` reads return the same five layers as GeoJSON for a whole
viewport, capped at ``DEFAULT_FEATURE_LIMIT`` — still what the agent's
`find_lots_in_view` tool wants, and still what ``HBU_MAP_RENDERER=geojson``
selects, but no longer how the map is drawn. See the tile section below for
why the cap was never the fix.

Either way the wire carries the vertices that get *drawn* rather than the
vertices Infolot recorded — a borough's worth of lots is several megabytes of
coordinates at full precision and a few hundred kilobytes at screen precision.
The viewport reads get there with ``ST_SimplifyPreserveTopology`` and a
tolerance in degrees; a tile gets there by quantising onto its own grid.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date

from src.utils.db import (
    GOLD_SCHEMA,
    SCHEMA,
    SILVER_SCHEMA,
    query,
    query_one,
    scalar,
)

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

#: The attribute the zoning layer's own number lives in - the label the map
#: puts on a zone, and the id every grid join is keyed on. Read off
#: ``ZONING_FIELDS`` rather than written down again, so the Lot pane and the
#: map cannot disagree about which attribute that is.
ZONE_LABEL_ATTRIBUTE = ZONING_FIELDS[0][0]

#: A viewport query returns at most this many shapes. Past it the map is a
#: solid block of outlines and the browser is the bottleneck, not the database.
#:
#: This is the *GeoJSON* renderer's cap, and it is why that renderer is no
#: longer the default: a cap is not a fix. Two thousand lots is still two
#: thousand full coordinate lists embedded in the page on every rerun, and the
#: browser gives out well below a borough. The tile renderer needs no such
#: number, because a tile is bounded by its own area - see `mvt_tile`.
DEFAULT_FEATURE_LIMIT = int(os.environ.get("HBU_MAP_FEATURE_LIMIT", 2000))


# ---------------------------------------------------------------------------
# What the database actually has
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Capabilities:
    """Which halves of the schema are present.

    Checked once per session rather than assumed, because the three repos land
    in a database independently: `hbu_infra` creates the tables, the
    dataplatform's ``document_index`` asset creates ``rag.chunks`` on its first
    load, and ``rag.buildings`` arrives with whoever loads BDOI. A map that
    silently draws nothing is worse than one that says which table is missing.

    ``building_lots`` and ``lot_features`` are the two silver joins, and they
    are a different kind of absent from the rest. Everything else here missing
    means a question cannot be answered; these two missing only mean it is
    answered the slow way, by computing the intersection per click. So neither
    `can_map` nor `can_retrieve` turns on them, and `missing()` takes an
    argument for which of the two readings the caller wants.
    """

    postgis: bool = False
    #: ST_AsMVT and the five-argument ST_TileEnvelope - PostGIS 3.1 and later.
    #: Everything about the map's *geometry* rides on this: without it there
    #: are no vector tiles and the renderer falls back to GeoJSON, which is
    #: the shape that could not draw a borough.
    mvt: bool = False
    pgvector: bool = False
    lots: bool = False
    buildings: bool = False
    building_lots: bool = False
    lot_features: bool = False
    features: bool = False
    massing: bool = False
    highest_best_use: bool = False
    redevelopment_gap: bool = False
    chunks: bool = False
    search_at_lot: bool = False
    search_near: bool = False

    @property
    def can_map(self) -> bool:
        return self.postgis and (self.lots or self.buildings or self.features)

    @property
    def can_retrieve(self) -> bool:
        return self.pgvector and self.chunks

    def missing(self, *, include_advisory: bool = True) -> list[str]:
        """Human-readable names of what is absent.

        ``include_advisory=True`` is the operator's reading — the sidebar pane
        and `scripts/doctor.py` — where "the pipeline has not joined this
        borough yet" is worth seeing.

        ``include_advisory=False`` is what a *user-facing* message wants, and
        what the agent's tools pass. Telling someone that
        `silver.lot_features` is missing when the answer arrived anyway, just
        by the slower route, is reporting a fault that did not happen.
        """
        checks = {
            "postgis extension": (self.postgis, True),
            # Advisory: a PostGIS too old for tiles still answers every other
            # query on this map, and the renderer says so itself.
            "PostGIS 3.1+ (ST_AsMVT)": (self.mvt, False),
            "vector extension": (self.pgvector, True),
            f"{SCHEMA}.lots": (self.lots, True),
            f"{SCHEMA}.buildings": (self.buildings, True),
            f"{SILVER_SCHEMA}.building_lot_intersections": (self.building_lots, False),
            f"{SILVER_SCHEMA}.lot_features": (self.lot_features, False),
            f"{SCHEMA}.features": (self.features, True),
            f"{GOLD_SCHEMA}.lot_building_massing": (self.massing, False),
            f"{GOLD_SCHEMA}.lot_highest_best_use": (self.highest_best_use, False),
            f"{GOLD_SCHEMA}.lot_redevelopment_gap": (self.redevelopment_gap, False),
            f"{SCHEMA}.chunks": (self.chunks, True),
            f"{SCHEMA}.search_at_lot()": (self.search_at_lot, True),
            f"{SCHEMA}.search_near()": (self.search_near, True),
        }
        return [
            name
            for name, (present, required) in checks.items()
            if not present and (required or include_advisory)
        ]


def capabilities() -> Capabilities:
    """One round trip that answers every "does this exist" question."""
    row = query_one(
        """
        SELECT
          (SELECT count(*) FROM pg_extension WHERE extname = 'postgis')  > 0 AS postgis,
          -- The margin argument arrived with PostGIS 3.1, and ST_AsMVT with
          -- it in every build that has one; asking for the five-argument
          -- form is therefore one test for both halves of the tile path.
          (SELECT count(*) FROM pg_proc
            WHERE proname = 'st_tileenvelope' AND pronargs >= 5) > 0 AS mvt,
          (SELECT count(*) FROM pg_extension WHERE extname = 'vector')   > 0 AS pgvector,
          to_regclass(%(schema)s || '.lots')     IS NOT NULL AS lots,
          to_regclass(%(schema)s || '.buildings') IS NOT NULL AS buildings,
          to_regclass(%(silver)s || '.building_lot_intersections')
            IS NOT NULL AS building_lots,
          to_regclass(%(silver)s || '.lot_features') IS NOT NULL AS lot_features,
          to_regclass(%(schema)s || '.features') IS NOT NULL AS features,
          to_regclass(%(gold)s || '.lot_building_massing')
            IS NOT NULL AS massing,
          to_regclass(%(gold)s || '.lot_highest_best_use')
            IS NOT NULL AS highest_best_use,
          to_regclass(%(gold)s || '.lot_redevelopment_gap')
            IS NOT NULL AS redevelopment_gap,
          to_regclass(%(schema)s || '.chunks')   IS NOT NULL AS chunks,
          (SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE n.nspname = %(schema)s AND p.proname = 'search_at_lot') > 0 AS search_at_lot,
          (SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE n.nspname = %(schema)s AND p.proname = 'search_near') > 0 AS search_near
        """,
        {"schema": SCHEMA, "silver": SILVER_SCHEMA, "gold": GOLD_SCHEMA},
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


#: The gold tables the map can draw but the pipeline may not have written yet,
#: and the asset an operator has to run for each. Both are *advisory* — the map
#: works without either — so the pane says which asset is missing rather than
#: leaving the layer silently blank, which reads as "nothing can be built here"
#: and as "every lot is fully used" respectively.
GOLD_MAP_TABLES = {
    "massing": ("lot_building_massing", "lot_building_massing"),
    "capacity": ("lot_redevelopment_gap", "lot_redevelopment_gap"),
}


def partition_has_rows(
    layer: str,
    *,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
) -> bool:
    """Does this borough-snapshot hold any rows of ``layer``'s gold table?

    A partition-level fact rather than a viewport one, which is what makes it
    worth a query of its own: under the tile renderer nothing on the Python
    side ever sees a feature, so "the massing asset has not run for this
    borough" has to be asked directly instead of inferred from an empty layer.
    ``EXISTS`` stops at the first row, so it costs an index probe rather than
    a count over the partition.
    """
    table, _asset = GOLD_MAP_TABLES[layer]
    return bool(
        scalar(
            f"""
            SELECT EXISTS (
                SELECT 1
                  FROM {GOLD_SCHEMA}.{table}
                 WHERE (%(scrape_date)s::date IS NULL
                        OR scrape_date = %(scrape_date)s)
                   AND (%(neighborhood)s::text IS NULL
                        OR neighborhood = %(neighborhood)s)
            )
            """,
            {"scrape_date": scrape_date, "neighborhood": neighborhood},
        )
    )


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


def massing_in_bbox(
    bounds: tuple[float, float, float, float],
    *,
    zoom: int = 16,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
    only_underbuilt: bool = False,
    limit: int = DEFAULT_FEATURE_LIMIT,
) -> FeatureSet:
    """The proposed building of each lot — the one layer that is not a scrape.

    Every other layer on this map is something a publisher drew: a cadastral
    lot, a BDOI footprint, a zoning polygon. This one is an *answer* — the
    highest-and-best-use program the dataplatform solved for the lot, drawn as
    a rectangle inside that lot's setback envelope so the zone's four margins
    are respected by the shape itself. Which is why it is the only read here
    that reaches into `gold`.

    ``ST_SimplifyPreserveTopology`` is applied for symmetry with the layers
    above and does nothing: a massing is a rectangle and has four corners at
    any tolerance. It stays so the shape of this function does not have to be
    remembered as the exceptional one.

    ``only_underbuilt`` narrows to the proposals that are bigger than what
    stands on the lot today — the screen `gold.lot_redevelopment_gap` exists
    for, applied here as a semi-join rather than a second layer, so the map can
    show "where redevelopment is worth drawing" without the caller assembling
    two feature sets and hiding one.

    Rows with no rectangle are not in the table at all: a lot whose footprint
    could not be drawn has no geometry, and `urban_rag.warehouse` skips a
    geometry-less row on the way into a spatial table. So this returns the
    drawn massings and nothing else, and a lot missing from the layer is not a
    lot missing from the answer — see that asset's `massing_status`.
    """
    params = _bbox_params(bounds)
    params.update(
        {
            "tolerance": simplify_tolerance(zoom),
            "scrape_date": scrape_date,
            "neighborhood": neighborhood,
            "only_underbuilt": only_underbuilt,
            "limit": limit + 1,
        }
    )
    rows = query(
        f"""
        SELECT m.lot_uid,
               m.lot_number,
               m.neighborhood,
               m.scrape_date,
               m.massing_status,
               m.footprint_m2,
               m.placed_footprint_m2,
               m.footprint_fit_pct,
               m.aspect_ratio,
               m.width_m,
               m.depth_m,
               m.floors,
               m.height_m,
               m.num_dwellings,
               m.commercial_floors,
               m.placed_gross_floor_area_m2,
               ST_AsGeoJSON(
                   ST_SimplifyPreserveTopology(m.geom, %(tolerance)s)
               )::json AS geometry
          FROM {GOLD_SCHEMA}.lot_building_massing m
         WHERE m.geom && ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326)
           AND ST_Intersects(m.geom,
                   ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326))
           AND (%(scrape_date)s::date IS NULL OR m.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR m.neighborhood = %(neighborhood)s)
           AND (NOT %(only_underbuilt)s::boolean OR EXISTS (
                   SELECT 1
                     FROM {GOLD_SCHEMA}.lot_redevelopment_gap g
                    WHERE g.scrape_date = m.scrape_date
                      AND g.neighborhood = m.neighborhood
                      AND g.lot_uid = m.lot_uid
                      AND g.is_underbuilt
               ))
         LIMIT %(limit)s
        """,
        params,
    )
    return _as_feature_set(rows, layer="massing", id_key="lot_uid", limit=limit)


#: "What could still be added here", per class, as a SQL expression over
#: ``gold.lot_redevelopment_gap`` aliased ``g``.
#:
#: That table already carries ``{class}_floor_area_gap_m2``, and this does not
#: use it, for one reason worth stating: the gap column is NULL wherever
#: *either* side is, and the side that is missing is almost always the existing
#: one — a lot the assessment roll never reached is usually a lot with nothing
#: standing on it. Summing the published gap would therefore drop exactly the
#: vacant parcels, which are the ones carrying the most headroom, and would do
#: it silently.
#:
#: So a missing existing floor is read as zero here. That is not a new
#: assumption: it is the rule ``is_underbuilt`` is documented to follow, and
#: this keeps the two consistent. ``GREATEST`` also clamps at zero, so a lot
#: built past what today's zoning would allow contributes nothing rather than
#: cancelling a neighbour's headroom — those lots are counted separately, and
#: the signed total is reported beside this one by `capacity_totals`.
#:
#: On an unsolved lot every ``hbu_*`` column is NULL, and PostgreSQL's
#: ``GREATEST`` ignores NULL arguments rather than propagating them, so the
#: expression yields 0 and the lot adds nothing to a sum. That is the intended
#: reading — "no answer" is not "no room", and `num_solved` is what says how
#: much of the borough the totals actually speak for.
def _headroom_m2(cls: str) -> str:
    return (
        f"GREATEST(g.hbu_{cls}_floor_area_m2"
        f" - COALESCE(g.existing_{cls}_floor_area_m2, 0), 0)"
    )


#: How much of what the zoning would permit is standing today, as a percentage.
#: NULL on a lot with no solved programme — there is no denominator — which is
#: what the map draws in its "no programme" colour rather than as 0%.
_USED_PCT = (
    "100.0 * COALESCE(g.existing_floor_area_m2, 0)"
    " / NULLIF(g.hbu_floor_area_m2, 0)"
)


def capacity_in_bbox(
    bounds: tuple[float, float, float, float],
    *,
    zoom: int = 15,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
    only_underbuilt: bool = False,
    limit: int = DEFAULT_FEATURE_LIMIT,
) -> FeatureSet:
    """Each lot shaded by how much of its permitted floor is actually built.

    The question "is this lot used efficiently" asked of every parcel in view
    at once. ``gold.lot_redevelopment_gap`` holds the subtraction but carries
    no geometry — it is keyed on ``lot_uid`` and nothing else — so the shape
    comes from ``rag.lots`` and the finding from the join.

    Joined on the whole partition triple rather than on ``lot_uid`` alone. The
    key is ``(scrape_date, neighborhood, lot_uid)`` on the gold side and
    ``lot_uid`` is a bigserial that a reload mints again, so a two-snapshot
    database joined on the surrogate alone would cross the snapshots and shade
    this year's parcels with last year's answer.

    ``only_underbuilt`` narrows to the lots the gap table flags — the same
    screen the massing layer takes, applied to the same rows, so turning both
    layers on with the filter set cannot show a proposal on a lot this layer
    has hidden.
    """
    params = _bbox_params(bounds)
    params.update(
        {
            "tolerance": simplify_tolerance(zoom),
            "scrape_date": scrape_date,
            "neighborhood": neighborhood,
            "only_underbuilt": only_underbuilt,
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
               g.hbu_status,
               g.has_assessment,
               g.is_underbuilt,
               g.existing_floor_area_m2,
               g.hbu_floor_area_m2,
               g.floor_area_gap_m2,
               g.existing_num_dwellings,
               g.hbu_num_dwellings,
               g.dwelling_gap,
               {_USED_PCT} AS used_pct,
               {_headroom_m2("residential")} AS residential_headroom_m2,
               {_headroom_m2("commercial")}  AS commercial_headroom_m2,
               {_headroom_m2("industrial")}  AS industrial_headroom_m2,
               ST_AsGeoJSON(
                   ST_SimplifyPreserveTopology(l.geom, %(tolerance)s)
               )::json AS geometry
          FROM {SCHEMA}.lots l
          JOIN {GOLD_SCHEMA}.lot_redevelopment_gap g
            ON g.lot_uid      = l.lot_uid
           AND g.neighborhood = l.neighborhood
           AND g.scrape_date  = l.scrape_date
         WHERE l.geom && ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326)
           AND ST_Intersects(l.geom,
                   ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326))
           AND (%(scrape_date)s::date IS NULL OR l.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR l.neighborhood = %(neighborhood)s)
           AND (NOT %(only_underbuilt)s::boolean OR g.is_underbuilt)
         LIMIT %(limit)s
        """,
        params,
    )
    return _as_feature_set(rows, layer="capacity", id_key="lot_number", limit=limit)


# ---------------------------------------------------------------------------
# Vector tiles
#
# The same five layers as above, cut to a tile instead of to a viewport, and
# the reason this app can draw a borough at all.
#
# A viewport read has no natural bound: pan to zoom 15 over Villeray and the
# honest answer is twenty-five thousand lots, which is why the functions above
# take a `limit` and why that limit is not a fix. The cap keeps the *database*
# out of trouble and leaves the browser exactly where it was - folium embeds
# every returned coordinate in the page, Streamlit ships the whole document
# down the websocket on every rerun, and the tab dies somewhere in the low
# thousands of polygons.
#
# A tile is bounded by construction. `ST_AsMVTGeom` clips each shape to the
# tile, quantises its coordinates onto a 4096-step grid - so a vertex finer
# than a screen pixel costs nothing, which is the trade `simplify_tolerance`
# already makes, done exactly rather than in degrees - and drops whatever
# falls outside. The browser then holds the tiles on screen and discards the
# rest by itself, which is the part no server-side cap can do for it.
#
# Three details worth knowing, because each replaces something the viewport
# reads have to do explicitly:
#
#   * There is no `ST_Intersects`. The `&&` is the index's prefilter and the
#     exact test is the clip itself: a shape whose envelope overlaps the tile
#     while its geometry does not comes back NULL and is filtered out below.
#   * There is no `ST_SimplifyPreserveTopology`. Quantising onto the extent
#     grid *is* the simplification, and unlike a tolerance in degrees it
#     cannot produce an invalid ring.
#   * There is no per-layer `limit`, only the fuse below.
# ---------------------------------------------------------------------------

#: Coordinate steps across a tile. 4096 is the Mapbox default and what every
#: renderer assumes when a tile does not say otherwise; at zoom 15 one step is
#: about 30 mm on the ground, four orders of magnitude finer than the cadastre
#: was surveyed to.
MVT_EXTENT = 4096

#: How far past the tile edge to keep geometry, in extent units. Without it a
#: lot straddling two tiles is cut exactly at the seam and its outline is drawn
#: along that seam in both - a grid of hairlines over the whole map. 64 is a
#: quarter of a 256-pixel tile's worth of slack, wider than any stroke here.
MVT_BUFFER = 64

#: The same slack as the fraction of the envelope `ST_TileEnvelope` wants, so
#: the rows *selected* cover the same ground as the rows kept. Getting this
#: wrong is invisible until a polygon centred in the next tile stops drawing
#: its edge into this one.
MVT_MARGIN = MVT_BUFFER / MVT_EXTENT

#: A fuse, not a policy. Every layer is either zoom-gated or coarse enough that
#: no tile comes near this; it is here so a mistake upstream - a borough loaded
#: into one scrape_date twice, a filter that stops filtering - costs a slow
#: tile rather than the task's memory.
MVT_FEATURE_FUSE = int(os.environ.get("HBU_TILE_FEATURE_FUSE", 20_000))


#: What each layer selects, as the body of the tile CTE. ``geom`` is appended
#: by `mvt_tile` so no layer can get the clip wrong, and every other column
#: becomes a tile property the browser reads - which is why the lists are
#: short. The viewport reads above return whole rows because a pane may want
#: any of them; a tile carries what the style function and the tooltip need
#: and nothing else, because it carries it once per feature per tile.
#:
#: ``attributes`` is deliberately absent from all five. Infolot puts two dozen
#: columns on every lot, and a tile is the one place where paying for them
#: again on every pan would be permanent - the panes query the row by id when
#: they actually need it.
_MVT_LAYERS: dict[str, dict[str, str]] = {}


def _mvt_layer(name: str, *, source: str, columns: str, where: str) -> None:
    _MVT_LAYERS[name] = {"source": source, "columns": columns, "where": where}


def _register_mvt_layers() -> None:
    """The five layers, in the order `basemap` draws them.

    A function rather than a literal because the schema names are resolved
    from the environment at import, and reading them in one place is what
    keeps a review copy - ``URBAN_RAG_PG_SCHEMA`` and its two siblings -
    working for tiles as it already does for the viewport reads.
    """
    _mvt_layer(
        "zones",
        source=f"{SCHEMA}.features f",
        columns=f"""
               f.feature_id,
               COALESCE(NULLIF(f.attributes ->> '{ZONE_LABEL_ATTRIBUTE}', ''),
                        f.feature_id) AS zone_label,
               f.attributes ->> %(url_attribute)s AS zoning_pdf_url""",
        where="""
           f.source_table = %(source_table)s
           AND (%(scrape_date)s::date IS NULL OR f.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL
                OR f.neighborhood = %(neighborhood)s)""",
    )

    # The lot's shape carrying the gap table's finding. Joined on the whole
    # partition triple rather than on lot_uid alone, for the reason
    # `capacity_in_bbox` gives: lot_uid is a bigserial a reload mints again,
    # so a two-snapshot database joined on the surrogate would shade this
    # year's parcels with last year's answer.
    _mvt_layer(
        "capacity",
        source=f"""{SCHEMA}.lots l
          JOIN {GOLD_SCHEMA}.lot_redevelopment_gap g
            ON g.lot_uid      = l.lot_uid
           AND g.neighborhood = l.neighborhood
           AND g.scrape_date  = l.scrape_date""",
        columns=f"""
               l.lot_uid,
               l.lot_number,
               g.hbu_status,
               g.is_underbuilt,
               g.existing_floor_area_m2,
               g.hbu_floor_area_m2,
               g.existing_num_dwellings,
               g.hbu_num_dwellings,
               g.dwelling_gap,
               {_USED_PCT} AS used_pct,
               {_headroom_m2("residential")} AS residential_headroom_m2,
               {_headroom_m2("commercial")}  AS commercial_headroom_m2,
               {_headroom_m2("industrial")}  AS industrial_headroom_m2""",
        where="""
           (%(scrape_date)s::date IS NULL OR l.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL
                OR l.neighborhood = %(neighborhood)s)
           AND (NOT %(only_underbuilt)s::boolean OR g.is_underbuilt)""",
    )

    _mvt_layer(
        "lots",
        source=f"{SCHEMA}.lots l",
        columns="""
               l.lot_uid,
               l.lot_number,
               COALESCE(l.area_m2, ST_Area(l.geom::geography)) AS area_m2""",
        where="""
           (%(scrape_date)s::date IS NULL OR l.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL
                OR l.neighborhood = %(neighborhood)s)
           AND (%(min_area)s::float8 IS NULL
                OR COALESCE(l.area_m2, ST_Area(l.geom::geography)) >= %(min_area)s)
           AND (%(max_area)s::float8 IS NULL
                OR COALESCE(l.area_m2, ST_Area(l.geom::geography)) <= %(max_area)s)""",
    )

    _mvt_layer(
        "buildings",
        source=f"{SCHEMA}.buildings b",
        columns="""
               b.building_uid,
               COALESCE(b.area_m2, ST_Area(b.geom::geography)) AS area_m2""",
        where="""
           (%(scrape_date)s::date IS NULL OR b.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL
                OR b.neighborhood = %(neighborhood)s)""",
    )

    # The proposal, and the same `only_underbuilt` screen the capacity layer
    # takes - applied to the same rows, so turning both layers on with the
    # filter set cannot show a massing on a lot the shading has hidden.
    _mvt_layer(
        "massing",
        source=f"{GOLD_SCHEMA}.lot_building_massing m",
        columns="""
               m.lot_uid,
               m.lot_number,
               m.massing_status,
               m.floors,
               m.num_dwellings,
               m.commercial_floors,
               m.placed_footprint_m2,
               m.footprint_fit_pct""",
        where=f"""
           (%(scrape_date)s::date IS NULL OR m.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL
                OR m.neighborhood = %(neighborhood)s)
           AND (NOT %(only_underbuilt)s::boolean OR EXISTS (
                   SELECT 1
                     FROM {GOLD_SCHEMA}.lot_redevelopment_gap g
                    WHERE g.scrape_date  = m.scrape_date
                      AND g.neighborhood = m.neighborhood
                      AND g.lot_uid      = m.lot_uid
                      AND g.is_underbuilt
               ))""",
    )


_register_mvt_layers()

#: The alias each layer's geometry hangs off, so the clip can name it. One
#: letter per source table, the same aliases the viewport reads use.
_MVT_GEOM = {
    "lots": "l.geom",
    "buildings": "b.geom",
    "zones": "f.geom",
    "capacity": "l.geom",
    "massing": "m.geom",
}

#: The layers a tile may be asked for, in draw order. `tiles.py` validates the
#: path against this and `basemap` builds one Leaflet layer per entry, so a
#: layer added here reaches both without a third list to keep in step.
MVT_LAYER_NAMES: tuple[str, ...] = tuple(_MVT_LAYERS)


def mvt_tile(
    layer: str,
    z: int,
    x: int,
    y: int,
    *,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
    min_area_m2: float | None = None,
    max_area_m2: float | None = None,
    only_underbuilt: bool = False,
) -> bytes:
    """One Mapbox Vector Tile of ``layer``, as the protobuf bytes to serve.

    Empty is a real answer rather than an error: a tile over the river holds
    no lots, and the right thing to send back is a zero-length body the
    browser caches like any other. ``ST_AsMVT`` over no rows returns NULL,
    which is what the ``COALESCE`` turns into those zero bytes.

    ``z``/``x``/``y`` are the slippy-map tile the browser asked for, in the
    Web Mercator grid Leaflet uses. The envelope is built twice and the two
    are not interchangeable: the 3857 one is what `ST_AsMVTGeom` measures
    against, and a 4326 copy of it is what the `&&` tests - because the GiST
    indexes on all five tables are on the 4326 column, and comparing against a
    projected envelope would drop the index and scan the borough instead.
    """
    if layer not in _MVT_LAYERS:
        raise ValueError(f"unknown tile layer {layer!r}")
    spec = _MVT_LAYERS[layer]

    params = {
        "z": z,
        "x": x,
        "y": y,
        "layer": layer,
        "extent": MVT_EXTENT,
        "buffer": MVT_BUFFER,
        "margin": MVT_MARGIN,
        "fuse": MVT_FEATURE_FUSE,
        "scrape_date": scrape_date,
        "neighborhood": neighborhood,
        "min_area": min_area_m2,
        "max_area": max_area_m2,
        "only_underbuilt": only_underbuilt,
        "source_table": ZONING_SOURCE_TABLE,
        "url_attribute": ZONING_URL_ATTRIBUTE,
    }
    body = scalar(
        f"""
        WITH envelope AS (
            SELECT ST_TileEnvelope(%(z)s, %(x)s, %(y)s) AS mercator,
                   ST_Transform(
                       ST_TileEnvelope(%(z)s, %(x)s, %(y)s, margin => %(margin)s),
                       4326
                   ) AS lonlat
        ),
        tile AS (
            SELECT {spec["columns"]},
                   ST_AsMVTGeom(
                       ST_Transform({_MVT_GEOM[layer]}, 3857),
                       envelope.mercator,
                       %(extent)s,
                       %(buffer)s,
                       true
                   ) AS geom
              FROM {spec["source"]}, envelope
             WHERE {_MVT_GEOM[layer]} && envelope.lonlat
               AND {spec["where"]}
             LIMIT %(fuse)s
        )
        SELECT COALESCE(
                   ST_AsMVT(tile, %(layer)s, %(extent)s, 'geom'),
                   ''::bytea
               )
          FROM tile
         WHERE tile.geom IS NOT NULL
        """,
        params,
    )
    # psycopg hands bytea back as a memoryview, which is neither what an HTTP
    # response body wants nor what an equality check in a test wants.
    return bytes(body) if body is not None else b""



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

    Reads ``silver.building_lot_intersections`` when it is there. That table
    already holds each footprint clipped to each lot it falls in, with the
    clipped area and the share of the building it represents, so this is an
    index lookup rather than an ``ST_Intersection`` per query — and it gets the
    hard case right for free: a school or a tower spanning several parcels has
    one row per lot, each carrying only the portion inside it, with no notion
    of a "primary" lot to guess at.

    Looked up by ``lot_number`` directly rather than through a join to
    ``rag.lots``: the table carries the number itself, because ``lot_uid`` is a
    bigserial a reload mints again and the number is what survives one.

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
              FROM {SILVER_SCHEMA}.building_lot_intersections bl
              JOIN {SCHEMA}.buildings b ON b.building_uid = bl.building_uid
             WHERE bl.lot_number = %(lot_number)s
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
    overlap is in square metres, not square degrees, either way.

    Reads ``silver.lot_features`` when it is there — the same trade
    `buildings_on_lot` makes, and the same table the pipeline computes once per
    partition instead of once per click. The join to ``rag.features`` is still
    needed for ``attributes``, which the Lot pane renders and the silver table
    does not carry; it is an index lookup on the identity the two share.

    The fast path is also the more correct of the two when no ``scrape_date`` is
    given. The fallback intersects the *newest* lot geometry against features of
    every date; the precomputed rows pair each date's lot with that date's
    features, which is what they actually mean.
    """
    if capabilities().lot_features:
        rows = query(
            f"""
            SELECT lf.feature_id                       AS zone,
                   lf.source_table,
                   lf.neighborhood,
                   lf.scrape_date,
                   f.attributes,
                   f.attributes ->> %(url_attribute)s  AS zoning_pdf_url,
                   lf.overlap_area_m2                  AS overlap_m2,
                   lf.lot_area_m2
              FROM {SILVER_SCHEMA}.lot_features lf
              JOIN {SCHEMA}.features f
                ON f.source_table = lf.source_table
               AND f.feature_id   = lf.feature_id
               AND f.neighborhood = lf.neighborhood
               AND f.scrape_date  = lf.scrape_date
             WHERE lf.lot_number = %(lot_number)s
               AND lf.source_table = %(source_table)s
               AND (%(scrape_date)s::date IS NULL OR lf.scrape_date = %(scrape_date)s)
             ORDER BY lf.overlap_area_m2 DESC
            """,
            {
                "lot_number": lot_number,
                "scrape_date": scrape_date,
                "source_table": ZONING_SOURCE_TABLE,
                "url_attribute": ZONING_URL_ATTRIBUTE,
            },
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


def lot_capacity(
    lot_uid: int, *, scrape_date: date | None = None, neighborhood: str | None = None
) -> dict | None:
    """Whether one lot is used for what it is zoned for, and what else it holds.

    The two gold tables read together, because they answer two halves of one
    question and a pane showing either alone would mislead.
    ``gold.lot_redevelopment_gap`` is the subtraction — what stands against
    what the governing envelope would hold. ``gold.lot_highest_best_use`` is
    the programme behind the proposed side: the storeys, the height, the unit
    mix, and the zone whose grid authorised them.

    ``gold.lot_building_massing`` is joined only when it is there, and only for
    ``footprint_fit_pct``. That column is the one caveat this pane cannot
    honestly omit: the solver caps a footprint on the lesser of two *areas*,
    and an area is not a shape, so a fit below 100 means the proposed floor
    area is overstated for this parcel. Reporting a dwelling count without it
    would state a number the lot's own geometry refuses.

    Keyed on ``lot_uid`` because the gold tables are — a lot the roll never
    named has no ``lot_number`` and is exactly the under-built parcel worth
    finding, so a lookup by number would drop it.
    """
    caps = capabilities()
    if not caps.redevelopment_gap:
        return None

    massing_select, massing_join = "", ""
    if caps.massing:
        massing_select = """,
               m.massing_status,
               m.footprint_fit_pct,
               m.placed_gross_floor_area_m2"""
        massing_join = f"""
          LEFT JOIN {GOLD_SCHEMA}.lot_building_massing m
                 ON m.lot_uid      = g.lot_uid
                AND m.neighborhood = g.neighborhood
                AND m.scrape_date  = g.scrape_date"""

    hbu_select, hbu_join = "", ""
    if caps.highest_best_use:
        hbu_select = """,
               h.grid_zone,
               h.usages,
               h.permits_residential,
               h.permits_commercial,
               h.permits_industrial,
               h.hbu_dominant_use,
               h.buildable_area_m2,
               h.num_candidates,
               h.num_zones,
               h.units,
               h.floors,
               h.height_m,
               h.footprint_m2,
               h.residential_floors,
               h.commercial_floors,
               h.industrial_floors,
               h.total_stalls,
               h.total_capital_cost_cad,
               h.npv_cad,
               h.present_value_cad,
               h.annual_stabilised_noi_cad,
               h.binding"""
        hbu_join = f"""
          LEFT JOIN {GOLD_SCHEMA}.lot_highest_best_use h
                 ON h.lot_uid      = g.lot_uid
                AND h.neighborhood = g.neighborhood
                AND h.scrape_date  = g.scrape_date"""

    return query_one(
        f"""
        SELECT g.lot_uid,
               g.lot_number,
               g.neighborhood,
               g.scrape_date,
               g.lot_area_m2,
               g.primary_frontage_m,
               g.hbu_status,
               g.has_assessment,
               g.is_underbuilt,
               g.existing_floor_area_m2,
               g.hbu_floor_area_m2,
               g.floor_area_gap_m2,
               g.floor_area_gap_sqft,
               g.existing_residential_floor_area_m2,
               g.hbu_residential_floor_area_m2,
               g.existing_commercial_floor_area_m2,
               g.hbu_commercial_floor_area_m2,
               g.existing_industrial_floor_area_m2,
               g.hbu_industrial_floor_area_m2,
               g.existing_num_dwellings,
               g.hbu_num_dwellings,
               g.dwelling_gap,
               g.existing_dominant_use_code,
               g.existing_total_assessed_value,
               g.hbu_total_capital_cost_cad,
               g.annual_stabilised_noi_gap_cad,
               g.hbu_npv_cad,
               g.existing_present_value_cad,
               g.redevelopment_npv_gain_cad,
               {_USED_PCT} AS used_pct,
               {_headroom_m2("residential")} AS residential_headroom_m2,
               {_headroom_m2("commercial")}  AS commercial_headroom_m2,
               {_headroom_m2("industrial")}  AS industrial_headroom_m2{hbu_select}{massing_select}
          FROM {GOLD_SCHEMA}.lot_redevelopment_gap g{hbu_join}{massing_join}
         WHERE g.lot_uid = %(lot_uid)s
           AND (%(scrape_date)s::date IS NULL OR g.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR g.neighborhood = %(neighborhood)s)
         ORDER BY g.scrape_date DESC
         LIMIT 1
        """,
        {"lot_uid": lot_uid, "scrape_date": scrape_date, "neighborhood": neighborhood},
    )


def capacity_totals(
    *, neighborhood: str | None = None, scrape_date: date | None = None
) -> dict | None:
    """How much more the borough could hold, added up over every lot.

    One row. The headline figures are the *positive* headroom — what could be
    added on the lots that have room, with over-built lots contributing zero
    rather than a negative — because "how much more could we build" is a
    question about the parcels where building is possible, and letting a
    century-old six-storey walk-up on a now-three-storey zone cancel a vacant
    lot next door would answer a different question quietly.

    ``net_floor_area_gap_m2`` is that different question, kept beside it: the
    signed sum over the lots the roll actually reached, which is what the
    borough's floor area would become if every parcel were rebuilt exactly to
    today's zoning. In an old and dense borough it can be negative, and that is
    a real finding about the by-law rather than an error.

    The counts are not decoration. ``num_solved`` is how much of the borough
    these totals speak for at all; ``num_without_assessment`` is how many lots
    contributed their whole envelope because nothing is recorded standing on
    them; ``num_over_built`` is how many were clamped. A total read without
    them is a number with no error bar.
    """
    if not capabilities().redevelopment_gap:
        return None
    return query_one(
        f"""
        SELECT count(*)                                        AS num_lots,
               count(*) FILTER (WHERE g.hbu_status = 'solved')  AS num_solved,
               count(*) FILTER (WHERE g.is_underbuilt)          AS num_underbuilt,
               count(*) FILTER (WHERE NOT g.has_assessment)     AS num_without_assessment,
               count(*) FILTER (
                   WHERE g.existing_floor_area_m2 > g.hbu_floor_area_m2
               )                                               AS num_over_built,

               sum({_headroom_m2("residential")}) AS residential_headroom_m2,
               sum({_headroom_m2("commercial")})  AS commercial_headroom_m2,
               sum({_headroom_m2("industrial")})  AS industrial_headroom_m2,
               sum({_headroom_m2("residential")}
                   + {_headroom_m2("commercial")}
                   + {_headroom_m2("industrial")}) AS total_headroom_m2,

               sum(GREATEST(g.hbu_num_dwellings
                            - COALESCE(g.existing_num_dwellings, 0), 0))
                                                   AS additional_dwellings,

               -- How many solved lots were given *any* floor of each class.
               -- Zero headroom and zero proposals are different findings: the
               -- first says the envelopes are full, the second says nothing
               -- of that class was ever modelled, and a bare 0 m² reads as
               -- the first while usually meaning the second. The governing
               -- envelope is by construction a residential column, so its
               -- `permits_commercial` is false and `solve_program` caps
               -- commercial and industrial floors at zero — the pane needs to
               -- be able to say that rather than report "no room".
               count(*) FILTER (WHERE g.hbu_residential_floor_area_m2 > 0)
                                                   AS num_with_residential,
               count(*) FILTER (WHERE g.hbu_commercial_floor_area_m2 > 0)
                                                   AS num_with_commercial,
               count(*) FILTER (WHERE g.hbu_industrial_floor_area_m2 > 0)
                                                   AS num_with_industrial,

               sum(g.existing_floor_area_m2)       AS existing_floor_area_m2,
               sum(g.hbu_floor_area_m2)            AS hbu_floor_area_m2,
               sum(g.floor_area_gap_m2)            AS net_floor_area_gap_m2,
               sum(g.existing_num_dwellings)       AS existing_num_dwellings,
               sum(g.hbu_num_dwellings)            AS hbu_num_dwellings,

               -- The developer's verdict, summed where it is positive: what
               -- redeveloping every lot it pays to redevelop would be worth
               -- over keeping what stands, at the assumptions the solve ran
               -- with. The count says how many lots that is.
               sum(GREATEST(g.redevelopment_npv_gain_cad, 0))
                                                   AS redevelopment_npv_gain_cad,
               count(*) FILTER (WHERE g.redevelopment_npv_gain_cad > 0)
                                                   AS num_npv_gain_positive
          FROM {GOLD_SCHEMA}.lot_redevelopment_gap g
         WHERE (%(scrape_date)s::date IS NULL OR g.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR g.neighborhood = %(neighborhood)s)
        """,
        {"scrape_date": scrape_date, "neighborhood": neighborhood},
    )


def top_capacity_lots(
    *,
    neighborhood: str | None = None,
    scrape_date: date | None = None,
    limit: int = 10,
) -> list[dict]:
    """The lots contributing most of the borough's headroom, largest first.

    A borough total is an average of 25,000 parcels and a handful of enormous
    ones can carry a third of it. In Villeray the two largest contributors are
    a 159 ha and a 94 ha parcel — the scale of a park or a rail yard rather
    than a development site — and a reader given only the total has no way to
    see that, or to judge whether the sites driving it are ones anybody would
    build on.

    So this is not a "top opportunities" list. It is the total showing its own
    working, which is why it returns lot *area* beside the proposed dwellings:
    the parcels worth disbelieving are the ones whose area is implausible for a
    development site, and that is visible at a glance in the two columns
    together.
    """
    if not capabilities().redevelopment_gap:
        return []
    return query(
        f"""
        SELECT g.lot_number,
               g.lot_area_m2,
               g.existing_num_dwellings,
               g.hbu_num_dwellings,
               g.hbu_floor_area_m2,
               GREATEST(g.hbu_num_dwellings
                        - COALESCE(g.existing_num_dwellings, 0), 0)
                   AS additional_dwellings
          FROM {GOLD_SCHEMA}.lot_redevelopment_gap g
         WHERE g.hbu_status = 'solved'
           AND (%(scrape_date)s::date IS NULL OR g.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR g.neighborhood = %(neighborhood)s)
         ORDER BY additional_dwellings DESC NULLS LAST
         LIMIT %(limit)s
        """,
        {"scrape_date": scrape_date, "neighborhood": neighborhood, "limit": limit},
    )


def top_npv_gain_lots(
    *,
    neighborhood: str | None = None,
    scrape_date: date | None = None,
    limit: int = 10,
) -> list[dict]:
    """The lots where redevelopment beats holding by the most, largest first.

    The developer's shortlist, on the developer's number:
    ``redevelopment_npv_gain_cad`` is the discounted value of building the
    lot's highest and best use less the discounted value of keeping what
    stands, at the assumptions the solve ran with. `top_capacity_lots` beside
    this ranks by *room*; this ranks by *money*, and the two lists disagree
    exactly where a big envelope does not pencil.

    The join to ``lot_highest_best_use`` brings the one-word answer to "a
    building of what" (``hbu_dominant_use``) and the program's own economics,
    so the list reads as a shortlist rather than a column of dollars.
    """
    if not capabilities().redevelopment_gap:
        return []
    return query(
        f"""
        SELECT g.lot_number,
               g.lot_area_m2,
               g.redevelopment_npv_gain_cad,
               g.hbu_npv_cad,
               g.existing_present_value_cad,
               g.existing_num_dwellings,
               g.hbu_num_dwellings,
               h.hbu_dominant_use,
               h.floors,
               h.num_dwellings,
               h.commercial_area_m2,
               h.industrial_area_m2,
               h.total_capital_cost_cad
          FROM {GOLD_SCHEMA}.lot_redevelopment_gap g
          LEFT JOIN {GOLD_SCHEMA}.lot_highest_best_use h
                 ON h.lot_uid      = g.lot_uid
                AND h.neighborhood = g.neighborhood
                AND h.scrape_date  = g.scrape_date
         WHERE g.hbu_status = 'solved'
           AND g.redevelopment_npv_gain_cad > 0
           AND (%(scrape_date)s::date IS NULL OR g.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR g.neighborhood = %(neighborhood)s)
         ORDER BY g.redevelopment_npv_gain_cad DESC NULLS LAST
         LIMIT %(limit)s
        """,
        {"scrape_date": scrape_date, "neighborhood": neighborhood, "limit": limit},
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
