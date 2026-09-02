"""
rag_tools.py — Retrieval over the zoning corpus, in three scopes.

The corpus is the borough resolutions and the zoning grids, cut
into chunks and embedded by ``hbu_dataplatform``. What makes it worth putting
in Postgres next to the geometry is that a highest-and-best-use question is two
questions at once: *what do the rules say* is a vector search, and *which rules
apply here* is a spatial one. The three tools below are those two questions
combined in the three useful ways.

``regulations_at_lot``   containment — every rule citing a zone that covers this lot
``regulations_near``     proximity — rules within a radius of a point
``search_regulations``   neither — the corpus-wide question with no place attached

Prefer the narrowest scope the question allows. An unfiltered search over a
borough returns the chunk that reads most like the question, which for "how
tall can I build" is some other zone's height limit stated more fluently than
the right one.
"""

from __future__ import annotations

import logging

from langchain.tools import tool
from langchain_core.tools import ToolException

from src.utils import queries, state
from src.utils.embeddings import EmbeddingError, embed_query

logger = logging.getLogger(__name__)

#: Chunks are up to 512 tokens; more than a handful in one tool result buries
#: the answer and costs the context it needs to reason about them.
MAX_MATCHES = 8

#: How much of a chunk reaches the model. The grids are dense — a full chunk is
#: mostly table scaffolding — and the pane shows the untruncated text anyway.
CHUNK_PREVIEW_CHARS = 900


def _require_corpus() -> None:
    caps = queries.capabilities()
    if not caps.can_retrieve:
        raise ToolException(
            "The regulation corpus is not loaded in this database "
            f"({', '.join(caps.missing(include_advisory=False))} missing). "
            "The dataplatform's "
            "`document_index` asset creates and fills rag.chunks. Tell the "
            "user that rather than retrying."
        )


def _embed(question: str) -> list[float]:
    try:
        return embed_query(question)
    except EmbeddingError as exc:
        raise ToolException(str(exc)) from exc


def _render(hits: list[dict], *, header: str) -> str:
    if not hits:
        return (
            f"{header}\n\nNothing in the corpus matched. Either no document is "
            f"linked to this place, or the question is about something the "
            f"zoning grids do not cover."
        )
    lines = [header, ""]
    for index, hit in enumerate(hits, 1):
        text = (hit.get("chunk_text") or "").strip().replace("\n", " ")
        if len(text) > CHUNK_PREVIEW_CHARS:
            text = text[:CHUNK_PREVIEW_CHARS] + "…"
        provenance = [f"[{index}]"]
        if hit.get("similarity") is not None:
            provenance.append(f"similarity {float(hit['similarity']):.3f}")
        if hit.get("distance_m") is not None:
            provenance.append(f"{float(hit['distance_m']):.0f} m away")
        if hit.get("lot_number"):
            provenance.append(f"lot {hit['lot_number']}")
        if hit.get("url"):
            provenance.append(hit["url"])
        lines.append(f"{' · '.join(provenance)}\n{text}\n")
    lines.append(
        "Cite these by their bracketed number when you use them, and say when "
        "the passages do not answer the question."
    )
    return "\n".join(lines)


@tool
def regulations_at_lot(question: str, lot_number: str = "", match_count: int = 5) -> str:
    """Retrieve the regulation text that applies to one lot.

    THE tool for "what can I build here", "what usages are allowed on this
    lot", "how tall". It resolves the lot to its zones and searches only the
    documents those zones cite, so no neighbouring zone's rules can appear in
    the answer.

    Args:
        question: What to look for, in the user's own words. The corpus is in
            French — a French question retrieves better, but the encoder is
            multilingual so either works.
        lot_number: The lot to scope to. Leave empty to use the selected lot.
        match_count: How many passages to return, capped at 8.

    Returns:
        Numbered passages with their similarity and source URL.
    """
    _require_corpus()
    if not queries.capabilities().search_at_lot:
        raise ToolException(
            "rag.search_at_lot() does not exist yet — it is created by "
            "hbu_infra's sql/003_spatial_search.sql, which is skipped until "
            "rag.chunks exists. Use search_regulations instead."
        )

    selected = state.get_selected_lot()
    lot_number = lot_number or (selected.get("lot_number") or "")
    if not lot_number:
        raise ToolException(
            "No lot given and none selected. Ask the user to click a lot on "
            "the map or name one."
        )

    lot = queries.lot_by_number(lot_number)
    if not lot:
        raise ToolException(f"No lot numbered {lot_number!r} in the loaded snapshots.")

    hits = queries.search_at_lot(
        _embed(question),
        float(lot["lon"]),
        float(lot["lat"]),
        match_count=min(int(match_count), MAX_MATCHES),
    )
    state.set_rag_result(question, hits, scope="lot", lot_number=lot["lot_number"])
    state.set_selected_lot(
        lot["lot_number"], lot.get("lon"), lot.get("lat"), lot.get("neighborhood")
    )
    return _render(hits, header=f"Regulations applying to lot {lot['lot_number']}:")


@tool
def regulations_near(
    question: str,
    lat: float | None = None,
    lon: float | None = None,
    radius_m: float = 500,
    match_count: int = 5,
) -> str:
    """Retrieve regulation text near a point rather than on one lot.

    Use this when the question is about an area — "what is the character of
    this block", "what applies around here" — or when a point falls outside
    any lot. Prefer regulations_at_lot when the user means one parcel.

    Args:
        question: What to look for.
        lat: Latitude. Defaults to the selected lot, then to the map centre.
        lon: Longitude. Same defaults.
        radius_m: Search radius in metres. 500 is a few blocks.
        match_count: How many passages to return, capped at 8.

    Returns:
        Numbered passages with their distance from the point.
    """
    _require_corpus()
    if not queries.capabilities().search_near:
        raise ToolException(
            "rag.search_near() does not exist yet — hbu_infra's "
            "sql/003_spatial_search.sql creates it once rag.chunks exists."
        )

    if lat is None or lon is None:
        selected = state.get_selected_lot()
        lat = lat if lat is not None else selected.get("lat")
        lon = lon if lon is not None else selected.get("lon")
    if lat is None or lon is None:
        center = state.Viewport.get("center")
        if center:
            lat, lon = float(center[0]), float(center[1])
    if lat is None or lon is None:
        raise ToolException(
            "No coordinates given, no lot selected, and the map has not "
            "reported its centre. Ask the user to click on the map."
        )

    hits = queries.search_near(
        _embed(question),
        float(lon),
        float(lat),
        radius_m=float(radius_m),
        match_count=min(int(match_count), MAX_MATCHES),
    )
    state.set_rag_result(question, hits, scope="near")
    return _render(
        hits,
        header=f"Regulations within {radius_m:.0f} m of {lat:.5f}, {lon:.5f}:",
    )


@tool
def search_regulations(
    question: str, neighborhood: str | None = None, match_count: int = 5
) -> str:
    """Search the whole regulation corpus, with no place attached.

    Use this for general questions — "what does the by-law say about café
    terrasses", "how is taux d'implantation defined" — and when the spatial
    search functions are not available. For anything about a specific lot,
    regulations_at_lot is more accurate.

    Args:
        question: What to look for.
        neighborhood: Restrict to one borough code, e.g. "VSMPE".
        match_count: How many passages to return, capped at 8.

    Returns:
        Numbered passages with their similarity and source URL.
    """
    _require_corpus()
    hits = queries.search_corpus(
        _embed(question),
        match_count=min(int(match_count), MAX_MATCHES),
        neighborhood=neighborhood,
    )
    state.set_rag_result(question, hits, scope="corpus")
    scope = f" in {neighborhood}" if neighborhood else ""
    return _render(hits, header=f"Corpus search{scope} for {question!r}:")


RAG_TOOLS = [regulations_at_lot, regulations_near, search_regulations]
