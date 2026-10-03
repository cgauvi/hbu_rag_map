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
import re
from concurrent.futures import ThreadPoolExecutor

from langchain.tools import tool
from langchain_core.tools import ToolException

from src.utils import neighborhoods, queries, state
from src.utils.embeddings import EmbeddingError, embed_query

logger = logging.getLogger(__name__)

#: Chunks are up to 512 tokens. The old ceiling was 8, on the reasoning that
#: more would bury the answer - but that traded away recall to save a cost this
#: endpoint does not charge: a 10k-token prompt measured no slower than a 2k
#: one, because latency tracks tokens generated rather than tokens read.
#: Ranking is the real constraint, not how much is carried, so the right move
#: is to hand the model more of the corpus rather than less.
MAX_MATCHES = 16

#: How much of a chunk reaches the model. The grids are dense - a full chunk is
#: mostly table scaffolding - and the pane shows the untruncated text anyway;
#: but 900 chars cut most grid chunks before their norms table, which is the
#: part that answers the question. Raised for the same reason as MAX_MATCHES.
#: MAX_MATCHES x this is ~26 kB, which `agent._MAX_TOOL_OUTPUT_CHARS` has to
#: stay above or the widening is silently truncated there instead.
CHUNK_PREVIEW_CHARS = 1600


#: Zone codes, as the three cities write them. Montreal numbers a zone
#: ``C01-001``; Quebec City writes ``11004Mc`` and sometimes a bare ``70520``;
#: Saguenay follows Quebec's.
#:
#: These are lifted out of the question and matched against ``feature_ids``
#: rather than left to the encoder, because a zone code is the one thing in a
#: zoning question that similarity is worst at: "C01-001" and "C01-007" embed
#: almost identically, so the grid that phrases the question's words most
#: fluently wins over the grid that actually governs. The scrape already knows
#: which document belongs to which zone, so naming one is a lookup.
#:
#: The bare-digits arm is deliberately last and deliberately narrow: five
#: digits is a Quebec zone, but it is also a postal-ish number and a year, so
#: it only ever *narrows* a search, and `search_regulations` falls back to the
#: unnarrowed one when the code reaches no document.
_ZONE_CODE = re.compile(
    r"\b(?:[A-Z]{1,2}\d{2}-\d{3}"      # C01-001, H04-072
    r"|\d{4,5}[A-Za-z]{1,3}"           # 11004Mc, 14040Hb
    r"|\d{5})\b"                       # 70520
)


def zone_codes(question: str) -> list[str]:
    """The zone codes a question names, in the order it names them."""
    seen: list[str] = []
    for match in _ZONE_CODE.findall(question or ""):
        if match not in seen:
            seen.append(match)
    return seen



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


def _page_label(hit: dict) -> str:
    """Where in the document a passage came from, when that is known.

    Absent for anything indexed before the corpus carried page offsets, and
    for a chunk the chunker could not place in its document - so the label is
    omitted rather than guessed. A citation that names a page nobody can turn
    to is worse than one that names only the sheet.
    """
    first, last = hit.get("page_from"), hit.get("page_to")
    if not first:
        return ""
    if last and last != first:
        return f"p. {first}-{last}"
    return f"p. {first}"


def _render(hits: list[dict], *, header: str, query: str = "", scope: str = "") -> str:
    """Number the passages for citation and lay them out for the model.

    The numbering comes from ``state.record_citation`` rather than from
    ``enumerate``, so it continues across every retrieval in a turn. It used to
    restart at 1 per tool call, which meant a turn that searched twice handed
    the model two passages both called [1] and showed the pane only the second
    - a citation that looked checkable and was not.
    """
    if not hits:
        return (
            f"{header}\n\nNothing in the corpus matched. Either no document is "
            f"linked to this place, or the question is about something the "
            f"zoning grids do not cover."
        )
    lines = [header, ""]
    for hit in hits:
        index = state.record_citation(hit, query=query, scope=scope)
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
        page = _page_label(hit)
        if page:
            provenance.append(page)
        if hit.get("url"):
            provenance.append(hit["url"])
        lines.append(f"{' · '.join(provenance)}\n{text}\n")
    lines.append(
        "Cite these by their bracketed number when you use them, and say when "
        "the passages do not answer the question. The numbers run on across "
        "every search in this turn, so one number always means one passage."
    )
    return "\n".join(lines)


@tool
def regulations_at_lot(question: str, lot_number: str = "", match_count: int = 10) -> str:
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
        match_count: How many passages to return, capped at 16.

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
    return _render(
        hits,
        header=f"Regulations applying to lot {lot['lot_number']}:",
        query=question,
        scope="lot",
    )


@tool
def regulations_near(
    question: str,
    lat: float | None = None,
    lon: float | None = None,
    radius_m: float = 500,
    match_count: int = 10,
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
        match_count: How many passages to return, capped at 16.

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
        query=question,
        scope="near",
    )


@tool
def search_regulations(
    question: str, neighborhood: str | None = None, match_count: int = 10
) -> str:
    """Search the whole regulation corpus, with no place attached.

    Use this for general questions — "what does the by-law say about café
    terrasses", "how is taux d'implantation defined" — and when the spatial
    search functions are not available. For anything about a specific lot,
    regulations_at_lot is more accurate.

    Args:
        question: What to look for.
        neighborhood: Restrict to one borough code, e.g. "VSMPE".
        match_count: How many passages to return, capped at 16.

    Returns:
        Numbered passages with their similarity and source URL.
    """
    _require_corpus()
    match_count = min(int(match_count), MAX_MATCHES)
    embedding = _embed(question)

    # A question that names a zone is answered by that zone's own sheet or by
    # nothing - so try the narrow search first and fall back only when the code
    # reaches no document. Falling back matters: the user may have typed a zone
    # from a borough that is not loaded, or simply mistyped one, and a silent
    # empty answer reads as "the by-law says nothing".
    # The question itself goes to the lexical arm. That is what makes an exact
    # term - a by-law number, "PIIA", a term of art - outrank a sheet that
    # merely phrases the question well; see rag.search_corpus.
    codes = zone_codes(question)
    hits: list[dict] = []
    if codes:
        hits = queries.search_corpus(
            embedding,
            match_count=match_count,
            neighborhood=neighborhood,
            zones=codes,
            query_text=question,
        )
    if not hits:
        hits = queries.search_corpus(
            embedding,
            match_count=match_count,
            neighborhood=neighborhood,
            query_text=question,
        )
    state.set_rag_result(question, hits, scope="corpus")
    scope = f" in {neighborhoods.label(neighborhood)}" if neighborhood else ""
    return _render(
        hits,
        header=f"Corpus search{scope} for {question!r}:",
        query=question,
        scope="corpus",
    )


#: Lots one call will retrieve for. Past a handful the passages stop being
#: comparable and start being a wall, and each lot is a database round trip.
MAX_FAN_OUT = 5

#: Threads for those round trips. The read pool is small and the map competes
#: for it, so this stays well under it.
_FAN_OUT_WORKERS = 4


@tool
def regulations_for_lots(question: str, lot_numbers: str, match_count: int = 3) -> str:
    """Retrieve the by-law passages that apply to each of several lots.

    Use this when a question about the wording of the by-law covers more than
    one parcel — comparing what two zones permit, or checking a condition
    across a shortlist. For one lot, regulations_at_lot says more.

    This is the only fan-out the agent has, and it is a loop in Python rather
    than one tool call per lot: the question is embedded once and the lots are
    searched in parallel, so five lots cost one model turn instead of five.

    Args:
        question: What to look for, in the corpus's own French where you can.
        lot_numbers: The lots, comma-separated — "2 170 935, 1 740 794".
        match_count: Passages per lot. Three keeps five lots readable.

    Returns:
        Numbered passages grouped by lot. The numbering runs on across the
        whole turn, so a citation means one passage.
    """
    _require_corpus()
    if not queries.capabilities().search_at_lot:
        raise ToolException(
            "rag.search_at_lot() does not exist yet — it is created by "
            "hbu_infra's sql/003_spatial_search.sql. Use search_regulations "
            "instead."
        )

    numbers = [n.strip() for n in (lot_numbers or "").replace(";", ",").split(",")]
    numbers = [n for n in numbers if n]
    if not numbers:
        raise ToolException(
            "Name the lots, comma-separated — \"2 170 935, 1 740 794\"."
        )
    if len(numbers) > MAX_FAN_OUT:
        raise ToolException(
            f"{len(numbers)} lots is more than one call retrieves for. Narrow "
            f"to at most {MAX_FAN_OUT} — rank them first and take the top few."
        )

    # Embedded once: the question is the same for every lot, and the encoder is
    # a network round trip.
    embedding = _embed(question)

    def _for_lot(number: str):
        lot = queries.lot_by_number(number)
        if not lot:
            return number, None, []
        hits = queries.search_at_lot(
            embedding,
            float(lot["lon"]),
            float(lot["lat"]),
            match_count=min(int(match_count), MAX_MATCHES),
        )
        return number, lot, hits

    with ThreadPoolExecutor(max_workers=_FAN_OUT_WORKERS) as pool:
        results = list(pool.map(_for_lot, numbers))

    sections: list[str] = []
    missing: list[str] = []
    for number, lot, hits in results:
        if lot is None:
            missing.append(number)
            continue
        sections.append(
            _render(
                hits,
                header=f"— Lot {lot['lot_number']} —",
                query=question,
                scope="lot",
            )
        )

    if not sections:
        raise ToolException(
            "None of those lots is in the loaded snapshots: "
            + ", ".join(repr(n) for n in missing)
            + "."
        )

    answer = [f"Regulations for {len(sections)} lot(s), on {question!r}:", ""]
    answer.extend(sections)
    if missing:
        answer.append(
            "Not in the loaded snapshots, so nothing was searched for them: "
            + ", ".join(missing)
            + "."
        )
    return "\n".join(answer)


RAG_TOOLS = [
    regulations_at_lot,
    regulations_near,
    regulations_for_lots,
    search_regulations,
]
