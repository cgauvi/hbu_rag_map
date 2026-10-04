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

Two more read a second corpus, Quebec City's *conseils de quartier*: the
minutes of their assemblies and the fiches, sommaires décisionnels and
resolutions they trail to, which the dataplatform reads into one row per
planning item - a demolition, a zoning amendment, a dérogation mineure - with
its outcome, and puts on the ground by the lots and addresses it names
(``silver.council_planning_items``, ``silver.council_item_sites``).

``council_decisions_near``  what was decided within a radius of a place, filtered
                            by kind, outcome and date - structured, no embedding -
                            with the passages when a question is asked too
``search_council_minutes``  the council corpus by meaning, no place attached
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from langchain.tools import tool
from langchain_core.tools import ToolException

from src.utils import neighborhoods, places, queries, state
from src.utils.db import DbError
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
            "hbu_infra's sql/004_spatial_search.sql, which is skipped until "
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
            "sql/004_spatial_search.sql creates it once rag.chunks exists."
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
            "hbu_infra's sql/004_spatial_search.sql. Use search_regulations "
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


# ---------------------------------------------------------------------------
# The conseils de quartier
# ---------------------------------------------------------------------------

#: How much of an item's excerpt reaches the model: enough for the request and
#: the operative sentence, not the whole agenda item.
ITEM_PREVIEW_CHARS = 500

#: Items one call lists. Past this the list stops being readable; the caller
#: narrows by kind, outcome or date instead.
MAX_ITEMS = 20

#: What a caller may write for ``outcome`` and what each means.
_OUTCOME_FILTERS: dict[str, tuple[str, ...]] = {
    "approved": ("approved",),
    "refused": ("refused",),
    "rejected": ("refused",),
    "denied": ("refused",),
    "in_progress": ("in_progress",),
    "pending": ("in_progress",),
    "decided": ("approved", "refused"),
}

#: Plain words a caller may use for a kind, mapped onto the vocabulary.
_KIND_ALIASES: dict[str, str] = {
    "demolition": "demolition",
    "demolitions": "demolition",
    "démolition": "demolition",
    "zoning": "zoning_amendment",
    "zoning_amendment": "zoning_amendment",
    "amendment": "zoning_amendment",
    "rezoning": "zoning_amendment",
    "ppcmoi": "ppcmoi",
    "variance": "minor_variance",
    "minor_variance": "minor_variance",
    "dérogation": "minor_variance",
    "derogation": "minor_variance",
    "conditional_use": "conditional_use",
    "usage_conditionnel": "conditional_use",
    "planning": "planning",
    "heritage": "heritage",
    "patrimoine": "heritage",
    "housing": "housing",
    "logement": "housing",
    "other": "other",
}


def _require_council(*, search: bool = False) -> None:
    caps = queries.capabilities()
    if not caps.council_items:
        raise ToolException(
            "The conseils de quartier minutes are not loaded in this database: "
            f"{queries.SILVER_SCHEMA}.council_item_sites or rag.council_items_near() is "
            "missing. hbu_infra's sql/035_silver_council_item_sites.sql creates them "
            "and the dataplatform's `make council-minutes` fills them (Quebec City "
            "only). Tell the user that rather than retrying."
        )
    if search and not caps.council_search:
        raise ToolException(
            "rag.search_council_chunks() is not in this database - hbu_infra's "
            "sql/036_council_search.sql creates it once rag.chunks exists, and "
            "`make council-publish` loads the minutes into it. Use "
            "council_decisions_near without a question instead."
        )


def _kinds(item_kinds: str) -> list[str] | None:
    """The kinds a caller named, in the vocabulary, or None for all."""
    wanted: list[str] = []
    unknown: list[str] = []
    for raw in re.split(r"[,;/]", item_kinds or ""):
        word = raw.strip().lower().replace(" ", "_")
        if not word:
            continue
        kind = _KIND_ALIASES.get(word)
        if kind is None:
            unknown.append(raw.strip())
        elif kind not in wanted:
            wanted.append(kind)
    if unknown:
        raise ToolException(
            f"Unknown item kind(s) {', '.join(repr(u) for u in unknown)}. Use: "
            + ", ".join(queries.COUNCIL_ITEM_KINDS) + "."
        )
    return wanted or None


def _outcomes(outcome: str) -> list[str] | None:
    word = (outcome or "").strip().lower().replace(" ", "_")
    if not word or word in {"any", "all"}:
        return None
    try:
        return list(_OUTCOME_FILTERS[word])
    except KeyError:
        raise ToolException(
            f"Unknown outcome {outcome!r}. Use approved, refused, in_progress, or "
            "decided (approved or refused)."
        ) from None


def _date(value: str, name: str) -> date | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        raise ToolException(f"{name} must be an ISO date such as 2025-01-31, not {value!r}.") from None


def _window(since: str, until: str, last_months: int) -> tuple[date | None, date | None]:
    """The date range, from explicit dates or from "the last N months"."""
    start, end = _date(since, "since"), _date(until, "until")
    if last_months and start is None:
        months = max(1, int(last_months))
        today = date.today()
        # Calendar months back, clamped to a valid day.
        year, month = divmod(today.month - 1 - months, 12)
        start = date(today.year + year, month + 1, min(today.day, 28))
    if start and end and end < start:
        raise ToolException(f"until ({end}) is before since ({start}).")
    return start, end


def _place(address: str, lat: float | None, lon: float | None) -> tuple[float, float, str]:
    """``(lon, lat, label)`` for the place a question is about.

    Explicit coordinates first; then an address, resolved through the same
    civic-address join `find_lot_by_address` uses and selected on the map
    when it names one lot; then the lot the user has selected.
    """
    if lat is not None and lon is not None:
        return float(lon), float(lat), f"{lat:.5f}, {lon:.5f}"

    if (address or "").strip():
        numbers, suffix, street, _written = queries.split_civic_numbers(address)
        segments = places.places_from_address(queries.without_civic_list(address))
        keys = [c.key for c in places.resolve(segments[0], segments[1:])] if segments else None
        if not numbers or not queries.street_key(street):
            raise ToolException(
                f"Could not read a civic number and a street in {address!r}. Write it as "
                "'439 rue Jeanne-d'Arc, Québec', or call find_lot_by_address first."
            )
        rows = queries.lots_by_address(
            street, numbers[0], civic_suffix=suffix, municipalities=keys,
            bounds=state.get_viewport(), limit=5,
        )
        if not rows:
            raise ToolException(
                f"No loaded address matches {address!r}. Call find_lot_by_address to see "
                "what it might mean, or give lat/lon."
            )
        lot = queries.lot_by_number(rows[0]["lot_number"])
        if not lot:
            raise ToolException(f"The lot under {address!r} is not in the loaded snapshots.")
        state.set_selected_lot(lot["lot_number"], lot.get("lon"), lot.get("lat"), lot.get("neighborhood"))
        label = f"{address.strip()} (lot {lot['lot_number']})"
        return float(lot["lon"]), float(lot["lat"]), label

    selected = state.get_selected_lot()
    if selected.get("lat") is not None and selected.get("lon") is not None:
        label = f"lot {selected.get('lot_number')}" if selected.get("lot_number") else "the selected lot"
        return float(selected["lon"]), float(selected["lat"]), label

    raise ToolException(
        "No place given: pass an address, lat/lon, or ask the user to click a lot "
        "on the map."
    )


def _item_label(item: dict) -> str:
    """One line that says what an item is: date, kind, outcome, the body."""
    when = item.get("item_date") or item.get("meeting_date")
    kind = (item.get("item_kind") or "other").replace("_", " ")
    outcome = item.get("outcome") or "no decision read"
    parts = [str(when) if when else "undated", kind, outcome]
    if item.get("council_opinion"):
        parts.append(f"council {item['council_opinion'].replace('_', ' ')}")
    source = {
        "minutes": "minutes",
        "gpd": "resolution/sommaire",
        "fiche": "fiche",
        "consultation_file": "consultation",
        "council_file": "council file",
    }.get(item.get("source_kind") or "", item.get("source_kind") or "")
    if item.get("document_number"):
        source = f"{source} {item['document_number']}"
    parts.append(source)
    if item.get("council_name"):
        parts.append(f"conseil de quartier {item['council_name']}")
    return " · ".join(parts)


def _item_hit(item: dict) -> dict:
    """An item as a citable hit: its PDF and its excerpt, shaped like a chunk."""
    return {
        "chunk_id": None,
        "doc_id": item.get("doc_id"),
        "url": item.get("url"),
        "title": item.get("title"),
        "source_table": f"council_{item.get('source_kind') or 'minutes'}",
        "neighborhood": item.get("neighborhood"),
        "scrape_date": item.get("scrape_date"),
        "chunk_text": item.get("excerpt") or "",
        "distance_m": item.get("distance_m"),
        "item_kind": item.get("item_kind"),
        "outcome": item.get("outcome"),
        "item_date": item.get("item_date"),
        "lot_number": item.get("site_lot_number"),
    }


def _render_items(items: list[dict], *, header: str, query: str, filters: str) -> list[str]:
    lines = [header, ""]
    if not items:
        lines.append(
            "No planning item has a site in that radius"
            + (f" with {filters}" if filters else "")
            + ". The minutes only reach a decision that names an address or a lot the "
            "cadastre knows; a wider radius, a wider date range or no outcome filter may "
            "find more. 'no decision read' means the document states none that the "
            "parser recognises, not that nothing was decided."
        )
        return lines
    for item in items:
        index = state.record_citation(_item_hit(item), query=query, scope="council")
        head = [f"[{index}]", _item_label(item)]
        if item.get("distance_m") is not None:
            site = item.get("site_key") or ""
            basis = "about" if item.get("is_subject") else "names"
            head.append(f"{float(item['distance_m']):.0f} m away ({basis} {site})")
        lines.append(" · ".join(head))
        if item.get("title"):
            lines.append(f"  {str(item['title'])[:160]}")
        named = []
        if item.get("subject_addresses"):
            named.append("addresses: " + ", ".join(map(str, item["subject_addresses"][:4])))
        if item.get("lot_numbers"):
            named.append("lots: " + ", ".join(map(str, item["lot_numbers"][:6])))
        if item.get("subject_zone_codes"):
            named.append("zones: " + ", ".join(map(str, item["subject_zone_codes"][:4])))
        if item.get("project_dwellings") or item.get("max_dwellings_after"):
            named.append(
                f"dwellings: cap {item.get('max_dwellings_before') or '?'} → "
                f"{item.get('max_dwellings_after') or '?'}, project {item.get('project_dwellings') or '?'}"
            )
        if named:
            lines.append("  " + " · ".join(named))
        if item.get("council_opinion_excerpt"):
            lines.append(f"  council: “{str(item['council_opinion_excerpt'])[:240]}”")
        excerpt = (item.get("excerpt") or "").strip().replace("\n", " ")
        if excerpt:
            lines.append(f"  {excerpt[:ITEM_PREVIEW_CHARS]}{'…' if len(excerpt) > ITEM_PREVIEW_CHARS else ''}")
        if item.get("url"):
            lines.append(f"  {item['url']}")
        lines.append("")
    return lines


def _render_passages(hits: list[dict], *, query: str) -> list[str]:
    lines = ["Passages from the minutes and their documents:", ""]
    for hit in hits:
        index = state.record_citation(hit, query=query, scope="council")
        text = (hit.get("chunk_text") or "").strip().replace("\n", " ")
        if len(text) > CHUNK_PREVIEW_CHARS:
            text = text[:CHUNK_PREVIEW_CHARS] + "…"
        provenance = [f"[{index}]"]
        if hit.get("title"):
            provenance.append(str(hit["title"])[:120])
        if hit.get("item_date"):
            provenance.append(str(hit["item_date"]))
        if hit.get("item_kind"):
            provenance.append(str(hit["item_kind"]).replace("_", " "))
        if hit.get("outcome"):
            provenance.append(str(hit["outcome"]))
        if hit.get("similarity") is not None:
            provenance.append(f"similarity {float(hit['similarity']):.3f}")
        if hit.get("distance_m") is not None:
            provenance.append(f"{float(hit['distance_m']):.0f} m away")
        if hit.get("url"):
            provenance.append(hit["url"])
        lines.append(f"{' · '.join(provenance)}\n{text}\n")
    return lines


_COUNCIL_FOOTER = (
    "Cite items and passages by their bracketed number. 'approved'/'refused' is the "
    "arrondissement's decision as the document states it; 'in progress' is a notice "
    "of motion, a draft or a consultation; the council's own opinion is advice, not "
    "the decision. Everything here is read from a scrape of the minutes by pattern - "
    "say the city is the authority for anything that matters."
)


def _filters_label(kinds, outcomes, since, until) -> str:
    parts = []
    if kinds:
        parts.append("kind " + "/".join(k.replace("_", " ") for k in kinds))
    if outcomes:
        parts.append("outcome " + "/".join(outcomes))
    if since and until:
        parts.append(f"between {since} and {until}")
    elif since:
        parts.append(f"since {since}")
    elif until:
        parts.append(f"until {until}")
    return ", ".join(parts)


@tool
def council_decisions_near(
    question: str = "",
    address: str = "",
    lat: float | None = None,
    lon: float | None = None,
    radius_m: float = 500,
    item_kinds: str = "",
    outcome: str = "",
    since: str = "",
    until: str = "",
    last_months: int = 0,
    match_count: int = 10,
) -> str:
    """What Quebec City's conseils de quartier and the arrondissement decided
    near a place: demolitions, zoning amendments, dérogations mineures,
    PPCMOI, conditional uses — approved, refused or still in progress — with
    the minutes, sommaires and resolutions that say so.

    THE tool for "were any demolitions near 439 rue Jeanne-d'Arc approved or
    refused", "what zoning changes were decided around here last year",
    "has the council opposed anything on this street". It needs no
    embedding: the items are already read into columns, placed on the lots
    and addresses they name, and filtered in SQL by radius, kind, outcome
    and date. Add a ``question`` to also retrieve the passages that answer
    it, numbered for citation.

    Quebec City only (La Cité-Limoilou today). A Montreal address finds
    nothing, and the tool says so.

    Args:
        question: Optional. What to look for in the text — "démolition
            résidentielle", "hauteur", "stationnement". French retrieves
            best. Without it the decisions are listed from their fields.
        address: The place, as written — "439 rue Jeanne-d'Arc, Québec",
            "355 boulevard René-Lévesque Ouest, Montcalm". Resolved to its
            lot and selected on the map. Leave empty to use lat/lon or the
            lot the user has selected.
        lat: Latitude, when the place is a point rather than an address.
        lon: Longitude.
        radius_m: How far "near" reaches, in metres. 500 is a few blocks;
            use 150 for one block, 1500 for a quartier.
        item_kinds: Comma-separated kinds to keep: demolition,
            zoning_amendment, ppcmoi, minor_variance, conditional_use,
            planning, heritage, housing. Empty keeps every kind.
        outcome: approved, refused, in_progress, or decided (approved or
            refused). Empty keeps every outcome, including items where no
            decision could be read.
        since: ISO date, "2025-01-01": keep items decided or discussed on
            or after it.
        until: ISO date: on or before it.
        last_months: Shorthand for ``since``: 12 means the last year. Ignored
            when ``since`` is given.
        match_count: How many items to list, capped at 20; and how many
            passages, when a question is asked.

    Returns:
        The items nearest first, each numbered for citation with its date,
        kind, outcome, the council's opinion, what it names, how far it is
        and the PDF it was read from; then the matching passages when a
        question was given.
    """
    _require_council(search=bool((question or "").strip()))
    kinds = _kinds(item_kinds)
    outcomes = _outcomes(outcome)
    start, end = _window(since, until, last_months)
    count = max(1, min(int(match_count), MAX_ITEMS))

    try:
        x, y, label = _place(address, lat, lon)
        items = queries.council_items_near(
            x, y, radius_m=float(radius_m), item_kinds=kinds, outcomes=outcomes,
            since=start, until=end, limit=count,
        )
        hits: list[dict] = []
        if (question or "").strip():
            hits = queries.search_council_chunks(
                _embed(question), match_count=count, lon=x, lat=y, radius_m=float(radius_m),
                since=start, until=end, item_kinds=kinds, outcomes=outcomes,
            )
    except DbError as exc:
        raise ToolException(str(exc)) from exc

    filters = _filters_label(kinds, outcomes, start, end)
    header = f"Council planning items within {float(radius_m):.0f} m of {label}"
    header += f" ({filters})" if filters else ""
    header += f": {len(items)}" + (f" (the {count} nearest)" if len(items) >= count else "")
    lines = _render_items(items, header=header, query=question or label, filters=filters)
    if hits:
        lines.extend(_render_passages(hits, query=question))
    elif (question or "").strip() and items:
        lines.append("No passage of the minutes matched the question within that radius; the items above are from their parsed fields.")
        lines.append("")
    lines.append(_COUNCIL_FOOTER)
    state.set_rag_result(question or label, hits or [_item_hit(i) for i in items], scope="council")
    return "\n".join(lines)


@tool
def search_council_minutes(
    question: str,
    neighborhood: str | None = None,
    item_kinds: str = "",
    outcome: str = "",
    since: str = "",
    until: str = "",
    last_months: int = 0,
    match_count: int = 10,
) -> str:
    """Search the conseils de quartier corpus — the minutes, sommaires,
    resolutions and consultation reports — by meaning, with no place
    attached.

    Use it for "what have the councils said about short-term rentals",
    "which amendments raised a dwelling cap", "what was the consultation on
    the demolition by-law". For anything near a place, council_decisions_near
    is more accurate. Quebec City only.

    Args:
        question: What to look for. French retrieves best.
        neighborhood: Restrict to one borough code, e.g. "CIL".
        item_kinds: Comma-separated kinds to keep (see council_decisions_near).
        outcome: approved, refused, in_progress or decided.
        since: ISO date: items decided or discussed on or after it.
        until: ISO date: on or before it.
        last_months: Shorthand for ``since``.
        match_count: How many passages, capped at 16.

    Returns:
        Numbered passages with the item each belongs to — its date, kind
        and outcome — and the PDF it came from.
    """
    _require_council(search=True)
    if not (question or "").strip():
        raise ToolException("Give the question to search the minutes for.")
    kinds = _kinds(item_kinds)
    outcomes = _outcomes(outcome)
    start, end = _window(since, until, last_months)
    count = max(1, min(int(match_count), MAX_MATCHES))
    try:
        hits = queries.search_council_chunks(
            _embed(question), match_count=count, neighborhood=neighborhood,
            since=start, until=end, item_kinds=kinds, outcomes=outcomes,
        )
    except DbError as exc:
        raise ToolException(str(exc)) from exc

    filters = _filters_label(kinds, outcomes, start, end)
    scope = f" in {neighborhoods.label(neighborhood)}" if neighborhood else ""
    header = f"Council minutes search{scope} for {question!r}" + (f" ({filters})" if filters else "") + ":"
    if not hits:
        lines = [
            header, "",
            "Nothing in the council corpus matched"
            + (f" with {filters}" if filters else "")
            + ". Either the minutes are not loaded for that borough (`make council-publish`), "
            "or the councils never discussed it.",
        ]
    else:
        lines = [header, ""]
        lines.extend(_render_passages(hits, query=question))
    lines.append(_COUNCIL_FOOTER)
    state.set_rag_result(question, hits, scope="council")
    return "\n".join(lines)


RAG_TOOLS = [
    regulations_at_lot,
    regulations_near,
    regulations_for_lots,
    search_regulations,
    council_decisions_near,
    search_council_minutes,
]
