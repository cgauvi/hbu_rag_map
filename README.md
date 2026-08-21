# hbu_rag_map

An interactive zoning map for Montreal, over the Postgres that
[`hbu_infra`](../hbu_infra) provisions. Pan the map to load lots and building
footprints; click a lot to see the zoning grid that applies to it, including
the *grille des spécifications* PDF itself; ask the chat panel what may be
built there, and it answers from the by-law rather than from memory.

The point of the arrangement is that a highest-and-best-use question is two
questions at once. *What do the rules say* is a vector search over the
embedded resolutions; *which rules apply here* is a spatial one over the
cadastre. Both live in the same database, so both halves of the answer come
back in one query — and the map and the chat cannot disagree about which lot
is under discussion, because they read the same selection.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  app.py — Streamlit                                                      │
│                                                                          │
│  ┌── Map (folium / st_folium) ────────┐  ┌── Lot & zoning ────────────┐  │
│  │  lots · buildings · zoning         │  │  attributes, built area    │  │
│  │  loaded by viewport, capped        │  │  the grid's values         │  │
│  │  a click → lot, resolved in SQL ───┼──┼→ the grid PDF, rasterised  │  │
│  └────────────────────────────────────┘  ├── Regulations ─────────────┤  │
│                    ▲                     │  what the last turn cited  │  │
│                    │ MapCommand          ├── Chat ────────────────────┤  │
│                    └─────────────────────┤  LangGraph ReAct agent     │  │
│                       SelectedLot ───────┤  14 tools                  │  │
│                                          └────────────────────────────┘  │
│                                                                          │
│  src/utils/db.py ──► DATABASE_URL │ URBAN_RAG_PG_* │ SSM /hbu-<env>/db/* │
│  src/utils/embeddings.py ──► HuggingFace Inference API (BAAI/bge-m3)     │
│  src/utils/documents.py ──► the city's PDFs, cached, rendered to PNG     │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
              RDS PostgreSQL · postgis · pgvector
              rag.lots · rag.buildings · rag.building_lots · rag.features
              rag.chunks · rag.search_near() · rag.search_at_lot()
```

This repo **reads**. It creates no tables and loads no data: the geometry
tables and the search functions belong to `hbu_infra`, and `rag.chunks` to
[`hbu_dataplatform`](../hbu_dataplatform). `make check` reports which of them
are actually there and what to run for each that is not.

---

## Quick start

```bash
make install                        # .venv + deps, and a .env to fill in
make db-up                          # local postgis+pgvector, hbu_infra's schema applied
make check                          # what is loaded, and what is missing
make run                            # http://localhost:8501
```

`make db-up` builds a container from
[`docker/postgres.Dockerfile`](docker/postgres.Dockerfile) — PostGIS *and*
pgvector, because neither published image has both and this database's whole
point is asking one question of both. It then applies `../hbu_infra/sql/*.sql`,
which is the same schema RDS gets. Set `HBU_INFRA=` if that repo is elsewhere.

The container starts empty. Load a borough into it the way you load RDS — see
[Getting data in](#getting-data-in).

### Against the real database instead

```bash
cd ../hbu_infra && eval "$(make -s db-app-env ENV=dev)"   # then `make run` here
```

Or set nothing at all: with AWS credentials, the endpoint is discovered from
SSM `/hbu-dev/db/*` and the app-role password from Secrets Manager.

---

## Configuration

Copy `.env.example` to `.env`. Two things matter.

**`HUGGINGFACE_API_TOKEN`** — one token, two uses: the chat model and the query
encoder. The map works without it; the chat panel and the corpus search do not.
This is the same variable [`ebird-llm`](../ebird-llm) uses.

**The database** — resolved in this order, most explicit first:

| | |
|---|---|
| `DATABASE_URL` | A full URL. A local container, or an open tunnel. |
| `URBAN_RAG_PG_DSN` | A full libpq string. |
| `URBAN_RAG_PG_*` | The dataplatform's contract — host, port, database, user, plus a Secrets Manager id or an IAM auth flag. `make db-app-env` in `hbu_infra` sets exactly these. |
| *(nothing)* | SSM `/hbu-<env>/db/*` + the app-role secret, from `HBU_ENV` and AWS credentials. |

The order is the contract: a developer with a container running and AWS
credentials in the shell gets the container, and someone with nothing set gets
the SSM lookup rather than a confusing localhost refusal.

Credentials are resolved **per connection**, never cached, because an RDS IAM
auth token is signed for fifteen minutes — a pool that cached one would hand
out an expired token on its second hour.

### The encoder is not a free choice

The corpus is embedded with `BAAI/bge-m3` by the dataplatform. A query must be
embedded by the same model, and **the failure is silent**: a 1024-wide vector
from a different encoder is a perfectly valid argument to `rag.search_near`,
and returns rows, ranked, with plausible similarities, all of them meaningless.
So the app checks its configured model and vector width against
`rag.chunks_meta` and reports a mismatch rather than letting it through. That
check is also the first thing `make check` prints about retrieval.

Embedding happens through the **Inference API**, not locally: the alternative
is 2.2 GB of weights and a torch install to answer a question a Streamlit
process asks a few times a minute. It is why the runtime image is 223 MB.

---

## The map

Layers load **by viewport**. A borough is 24,953 lots; the visible rectangle is
a few hundred. Each query is `geom && ST_MakeEnvelope(...)` — the part the GiST
index answers — followed by `ST_Intersects`, which turns the index's
bounding-box answer into a real one.

Three things keep it responsive:

- **Geometry is simplified server-side**, to a tolerance derived from the zoom.
  A screen pixel is about `360 / (256 · 2^zoom)` degrees, so collapsing
  vertices below that is invisible by construction. The wire carries the
  vertices that get drawn rather than the ones Infolot recorded.
- **Layers are zoom-gated.** Lots draw from zoom 15, buildings from 16. Below
  that a lot is sub-pixel and a borough of them is a grey rectangle that costs
  a second of browser time to produce.
- **Queries are capped** at `HBU_MAP_FEATURE_LIMIT` (2000) per layer, and each
  asks for one row past the cap — which is how the pane can say "capped, zoom
  in for the rest" without a second `count(*)` over the same predicate.

Streamlit replays the whole script on every interaction and `st_folium` reports
a viewport on every one of them, so results are cached on a *rounded* bounding
box: a nudge redraws from memory, a real pan re-queries.

### Clicking a lot

The click is resolved **server-side, from its coordinates** —
`ST_Intersects(l.geom, ST_MakePoint(lng, lat))` — not from whatever shape the
browser reports being hit. A click near a boundary, on a lot the viewport cap
left undrawn, or on a simplified edge all still land on the right parcel that
way.

From the lot, the Lot pane assembles:

- its attributes and the footprints standing on it, from `rag.building_lots`
  when that table is populated (an index lookup on `lot_uid` rather than an
  `ST_Intersection` per click) and computed on the fly when it is not;
- the zoning polygons covering it, **ordered by how much of the lot each
  actually covers** — a lot on a zone boundary intersects both, and only one of
  them is the answer. When more than one applies, the pane says so and lets you
  pick;
- the *grille des spécifications* for the chosen zone: its values as a table,
  and the PDF itself.

### The zoning PDF

`Reglement_urbanisme__VSP_REG_ZONE` carries a link, not a description:

```
LIEN_GRILLE = http://www1.ville.montreal.qc.ca/CartesInteractives/villeray/doc/zone/C01-001.pdf
```

Pages are **rasterised**, not embedded. The links are `http://`, and a browser
on an `https://` page refuses to frame them; Chrome also blocks `data:` URIs in
an iframe for PDFs. Rendering to PNG with pypdfium2 sidesteps both, works when
the cache is warm and the network is not, and is the same bytes the download
button hands over.

The cache key is `sha256(url)[:16]` — **identical to the dataplatform's
`document_id`** — so pointing `HBU_PDF_CACHE_DIR` at
`../hbu_dataplatform/data/cache/pdf` reuses the PDFs the pipeline already
downloaded rather than pulling them from the city's web server again. A
published zoning grid does not change once issued, which is what makes a cache
with no expiry correct here rather than merely convenient.

A dead link fails its own document, not the pane: these are municipal URLs, and
some answer `200` with an HTML "page not found" body, so the content is checked
for a PDF header rather than trusted.

---

## The chat panel

A LangGraph ReAct agent over the HuggingFace Inference API. It reaches the same
data the map does, and can move the map back.

| | |
|---|---|
| `describe_selected_lot` | which lot the user clicked — called before asking them to repeat it |
| `find_lot`, `show_lot_on_map` | look a lot up by number, frame it |
| `list_lots` | the lots in the current view, optionally by size |
| `zoning_for_lot` | the grid's values, and its PDF |
| `read_zoning_grid` | the grid PDF's full text, when the values fall short |
| `buildings_on_lot` | the footprints, and how much of the lot they cover |
| `regulations_at_lot` | by-law passages for one parcel — `rag.search_at_lot` |
| `regulations_near` | by-law passages around a point — `rag.search_near` |
| `search_regulations` | the corpus with no place attached |
| `focus_map`, `set_map_layers`, `filter_lots_on_map` | move and filter the map |
| `data_status` | which boroughs, snapshots and corpus are loaded |

Tools return **compact text, never geometry**. Handing an LLM a polygon's
coordinates costs thousands of tokens and buys nothing, so a tool that finds a
shape sends it to the map through `src/utils/state.py` and tells the model only
what it found.

That module is a one-shot request, not the map's state. The map's real position
belongs to the browser and arrives back through `st_folium`; a tool that wrote
centre and zoom directly would undo the user's last pan on every rerun.

### Scope, and why it matters

Prefer the narrowest scope a question allows. `rag.search_at_lot` is
containment — the lot's own zones, no radius to pick, no neighbouring zone
bleeding in. An unfiltered corpus search for "how tall can I build" returns the
chunk that reads most like the question, which is often some *other* zone's
height limit stated more fluently than the right one.

Because the spatial filter runs first and is selective, the planner scans the
surviving chunks rather than using the HNSW index — which is the right plan,
and makes recall in the spatial searches **exact** rather than approximate.

### What the prompt insists on

The by-law and every grid are in French; the agent answers in the user's
language but quotes the regulation's own terms. Beyond that, four rules earn
their place:

- Never state a number that did not come from a tool result in this turn.
- A lot straddling zones is reported as straddling, not as its largest zone.
- **The grid states what is permitted; a footprint states what exists.** Never
  present one as the other. (A lot measured at 57% built under a 65% *taux
  d'implantation* is two different facts.)
- Snapshots carry a date, and this is a scrape of a by-law rather than the
  by-law — for anything with consequences, the borough is the authority.

---

## Getting data in

Nothing here loads data. The tables come from elsewhere, and `make check`
walks the chain in order and names the first thing missing:

```
$ make check

Database
  [ok] postgis extension
  [ok] rag.lots
  [--] rag.chunks
    hbu_dataplatform: make publish DATE=... NEIGHBORHOOD=...
  [--] rag.search_at_lot()
    hbu_infra sql/003_spatial_search.sql — skipped until rag.chunks exists,
    so re-run `make db-init` after the first publish
```

That last one is the ordering trap worth knowing: a SQL-language function body
is parsed at `CREATE` time, so the spatial search functions genuinely cannot be
created before `rag.chunks` exists. `hbu_infra`'s `db.py` skips the file with a
note; `make db-init` here reports it as skipped. Publish a partition from the
dataplatform, then run `db-init` once more.

The app degrades rather than breaks around each gap: a missing `rag.buildings`
greys out its layer, a missing corpus disables the Regulations pane and makes
the retrieval tools tell the model which asset creates the table — so it
reports the gap instead of retrying three times.

---

## Tests

```bash
make test              # 115 unit tests; no socket is opened
make test-integration  # 7 more, against DATABASE_URL
make docker-test       # the unit suite inside the image
```

The unit suite stubs the database, the HuggingFace endpoint and the city's web
server. The integration suite drives `app.py` through Streamlit's own
`AppTest`, with `st_folium` stubbed to play the browser's half of the
conversation — which is how a pan and a click are simulated at all. It checks
the things unit tests cannot: that the rerun loop **converges** (one rerun to
adopt the viewport, then stop), that a click resolves to a real lot, and that
the zoning pane reaches a real PDF.

---

## Files

| | |
|---|---|
| [`app.py`](app.py) | The Streamlit page: map, Lot pane, Regulations pane, chat |
| [`src/agent.py`](src/agent.py) | The ReAct agent, its prompt, and the streaming loop |
| [`src/config.py`](src/config.py) | The chat-model catalog and `build_llm()` |
| [`src/utils/db.py`](src/utils/db.py) | Four ways to find the database, and the pool |
| [`src/utils/queries.py`](src/utils/queries.py) | Every statement this app sends |
| [`src/utils/embeddings.py`](src/utils/embeddings.py) | Query embedding, and the encoder-mismatch guard |
| [`src/utils/documents.py`](src/utils/documents.py) | Fetching, caching and rasterising the grid PDFs |
| [`src/utils/basemap.py`](src/utils/basemap.py) | Assembling the folium map |
| [`src/utils/state.py`](src/utils/state.py) | The side-channel between tools and the map |
| [`src/tools/`](src/tools/) | Parcel, retrieval and map-control tools |
| [`scripts/doctor.py`](scripts/doctor.py) | `make check` |
| [`docker/postgres.Dockerfile`](docker/postgres.Dockerfile) | PostGIS + pgvector, for local work |

### Two things it does not do yet

**Sessions share the process-level buffers.** `src/utils/state.py` holds
module-level dicts, so the agent's view of "the selected lot" is per process,
not per browser session. `app.py` re-mirrors this session's selection and
viewport into it at the top of every run, which makes the common case correct;
two people using one server concurrently could still cross an agent turn. It is
the same shape as `ebird-llm`'s `VizBuffer`, and the fix is a session-keyed
store.

**There is no deployment.** No Terraform, no Cognito, no ECS. The `Dockerfile`
builds a runtime image and `docker-compose.yml` runs it against the local
database, but nothing publishes it — deliberately, since the database it reads
is itself not yet applied. `ebird-llm/infra` is the pattern to follow when it is.
