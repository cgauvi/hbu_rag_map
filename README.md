# hbu_rag_map

An interactive zoning map for Montreal, over the Postgres that
[`hbu_infra`](../hbu_infra) provisions. Pan across a borough's lots, building
footprints, street sides and proposed massings, drawn as vector tiles straight
out of PostGIS; click a lot to see the zoning grid that applies to it,
including the *grille des spécifications* PDF itself — as a hyperlink you can
keep and in a PDF viewer beside the map; ask the chat panel what
may be built there, and it answers from the by-law rather than from memory.

The point of the arrangement is that a highest-and-best-use question is two
questions at once. *What do the rules say* is a vector search over the
embedded resolutions; *which rules apply here* is a spatial one over the
cadastre. Both live in the same database, so both halves of the answer come
back in one query — and the map and the chat cannot disagree about which lot
is under discussion, because they read the same selection.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  serve.py ──► tile server (:8502)  +  app.py — Streamlit (:8501)         │
│                                                                          │
│  ┌── Map (folium / st_folium) ────────┐  ┌── Lot ─────────────────────┐  │
│  │  lots · buildings · zoning · rues  │  │  attributes, built area    │  │
│  │  drawn from vector tiles ──────┐   │  │  what else would fit       │  │
│  │  a click → lot, else the zone  ┼───┼──┼→ the grid's values         │  │
│  │                                │   │  ├── HBU ─────────────────────┤  │
│  │                                │   │  │  the proposed building     │  │
│  │                                │   │  ├── Overview ────────────────┤  │
│  │                                │   │  │  the borough's headroom    │  │
│  └────────────────────────────────┼───┘  ├── Regulations ─────────────┤  │
│                    ▲              │      │  its sheets, in pdf.js     │  │
│                    │ MapCommand   │      ├── Chat ────────────────────┤  │
│                    └──────────────┼──────┤  LangGraph ReAct agent     │  │
│                       SelectedLot─┼──────┤  16 tools                  │  │
│                                   │      └────────────────────────────┘  │
│    GET /tiles/<layer>/{z}/{x}/{y}.mvt   ·   /tiles/vendor/<library>.js   │
│    GET /tiles/grid/<doc_id>.pdf — the grille, on this app's own origin   │
│                                   │                                      │
│  src/utils/tiles.py ◄─────────────┘  ST_AsMVT, one query per tile        │
│  src/utils/db.py ──► DATABASE_URL │ URBAN_RAG_PG_* │ SSM /hbu-<env>/db/* │
│  src/utils/embeddings.py ──► HuggingFace Inference API (BAAI/bge-m3)     │
│  src/utils/documents.py ──► the city's PDFs, cached, published, to PNG   │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
              RDS PostgreSQL · postgis (3.1+) · pgvector

              rag.lots · rag.buildings · rag.features      what was scraped
              rag.chunks · rag.search_near() · …           the corpus
              silver.building_lot_intersections            joins already
              silver.lot_features                          computed
              silver.neighborhood_streets                  the street sides
              gold.lot_building_massing                    what could be built
              gold.lot_surface_parking                     and where it parks
              gold.lot_highest_best_use                    the programme
              gold.lot_redevelopment_gap                   what is missing
```

This repo **reads**. It creates no tables and loads no data: every table and
the search functions belong to `hbu_infra`, and the rows in them to
[`hbu_dataplatform`](../hbu_dataplatform). `make check` reports which of them
are actually there and what to run for each that is not.

### Three schemas, and why the app cares

`rag` holds what the scrape loaded and is queried live. `silver` holds joins
the pipeline has **already computed** between those tables — one table per
asset, partitioned by `(neighborhood, scrape_date)`. `gold` holds its
*answers*, and the app reads three of them: the massing it draws, the
programme behind it, and the subtraction against what stands today.

Two reads have a fast path off a silver table and a fallback that computes the
same thing with `ST_Intersection`:

| read | fast path | fallback |
|---|---|---|
| the footprints standing on a lot | `silver.building_lot_intersections` | clip `rag.buildings` against the lot |
| **the buildings layer on the map** | `silver.building_lot_intersections` | clip `rag.buildings` against `rag.lots` in the tile |
| the zones covering a lot | `silver.lot_features` | clip `rag.features` against the lot |

The fallback is not dead code. A borough loaded this morning has its `rag` rows
before the silver assets have run over them, and both paths return the same
column names so nothing above `queries.py` can tell which one answered.

That is also why a missing silver table is reported differently from a missing
`rag` one. The sidebar and `make check` show it — an operator should know the
pipeline has not caught up — but the agent's tools never mention it, because
"missing" would claim a fault when the answer arrived anyway, just more slowly.

### The map is drawn from vector tiles, and why that is not a detail

All six layers on this map are fetched by the browser as **Mapbox Vector
Tiles**, one HTTP request per 256-pixel square, off a small server this same
process runs on port 8502. It is worth a section because the alternative was tried
first and it does not work, and because the failure is one this repo's
structure invites.

The obvious thing to do in Streamlit is to query the shapes in the viewport,
turn them into GeoJSON, and hand the collection to folium. That is what
`lots_in_bbox` and its five siblings do, and what the map used to be built
from. It has a ceiling, and the ceiling is low: folium embeds every coordinate
in the map document, Streamlit ships that whole document down the websocket on
every rerun, and a rerun is what a pan *is*. Villeray holds about 25,000 lots.
At zoom 15 a screenful is thousands of polygons and tens of megabytes of HTML,
re-serialised on every nudge of the map, and the tab stops responding.

`HBU_MAP_FEATURE_LIMIT` was the first answer and it is why the reads are
capped at 2,000 shapes. **A cap is not a fix.** It keeps the database out of
trouble and leaves the browser exactly where it was — two thousand parcels is
still more geometry than a page can be rebuilt around several times a second —
and it buys that by *not drawing the borough*, which is the thing the map is
for.

A tile is bounded by construction instead of by decree:

| | GeoJSON by viewport | Vector tiles |
|---|---|---|
| what the page holds | every shape in view, inline | six URLs |
| what a pan costs | a query, a re-render, a full document over the websocket | the tiles newly on screen, fetched by the browser |
| how much can be drawn | `HBU_MAP_FEATURE_LIMIT`, then nothing | the whole borough |
| where the zoom gate lives | Python, one rerun behind | Leaflet, immediate |
| simplification | `ST_SimplifyPreserveTopology`, a tolerance in degrees | quantisation onto the tile's own 4096-step grid |

`ST_AsMVTGeom` does the clipping, the quantisation and the discarding in one
call, so a vertex finer than a screen pixel costs nothing and a shape outside
the tile costs nothing. The browser then keeps the tiles on screen and throws
the rest away by itself, which is the part no server-side cap can do for it.

Three consequences are visible in the code and worth knowing before reading it:

**The map object stops depending on the viewport, and is rebuilt every
rerun.** A tile URL mentions neither the centre, the zoom nor the bounding
box, so a pan changes nothing the map is built from. `app.py` therefore builds
a fresh folium map on every run rather than caching one — which sounds like it
would reload the iframe on every viewport report, and does not:
`streamlit_folium` keys its component on a hash that *strips* the `_<suffix>`
off every variable name, so two independently built maps with the same inputs
are the same component and the pane does not blink.

**With the same inputs** — and the centre and the zoom are inputs. They are
written into the map's script, so a map built where the browser last said it
was is a different map on every drag: a new key, a remounted iframe, Leaflet
thrown away and the basemap and every vector tile fetched again. That is what
the pane visibly redrawing itself during a pan actually was. So `app.py` keeps
two positions rather than one:

| | the anchor (`map_center`, `map_zoom`) | the live view (`viewport`, `view_center`, `view_zoom`) |
|---|---|---|
| what it is | where the folium object is built | where the browser actually is |
| who writes it | the app, and only when the map is being rebuilt anyway | Leaflet, at the end of every drag |
| who reads it | `basemap.build_map` — so the component's key | the notes under the map, and the agent's "in view" tools |

A pan therefore records a position and stops. The anchor moves onto the live
view only when something else has already changed the map's script — a layer,
a borough, a snapshot, a fit — so the remount that was going to happen anyway
lands on the view the reader was looking at. The selected lot is kept out of
the map object for the same reason and handed to `st_folium` as a feature
group, which is evaluated into the map already on screen: a click paints an
outline over tiles that were never refetched.

Caching the object instead is what the code used to do, and it could not: **a
folium map survives being rendered exactly once.** `st_folium` rewrites every
element's `_id` to a stable `div_N` as it walks the tree, and whatever holds a
*name* rather than an element does not follow — folium's own `Layer.render`
re-adds its `addTo` snippet under the new name and leaves the previous one
pointing at a variable that no longer exists. The second render ships
`vector_grid_protobuf_<32 hex>.addTo(map_div)`, which is an uncaught
`ReferenceError` in the map script, thrown before `initComponent` — so the
pane goes blank rather than the layer going empty. `basemap` drops that stale
child on re-render as well, so the object is safe either way.

**The tooltip moved into the browser.** With tiles a feature never exists in
Python, so `basemap.decorate` — which builds the labels for the GeoJSON path —
has a twin in JavaScript, `_TOOLTIP_JS`. They implement the same rules,
including the five `hbu_status` reasons a lot has no percentage, and they are
adjacent in the file so a change to one is a change to the other. Everything
else the two renderers share is shared as *constants* rather than as code: the
colour ramp, the bands, the zoom gates and the highlight styles are read by
the Python style callbacks and serialised into the JavaScript ones, so the
legend beside the map cannot disagree with the map.

**The click still resolves server-side, and it has to be repaired before it
can be forwarded.** A vector tile is drawn onto a canvas carrying Leaflet's
`_leaflet_disable_events`, so the map never sees the DOM event itself — the
plugin's `L.Canvas.Tile._onClick` is the only thing that turns it into the
`click` `streamlit_folium` reports back as `last_clicked`. Leaflet.VectorGrid
1.3.0 forked that method from a Leaflet that no longer exists, and on 1.9 its
copy fails both ways: it calls `L.DomEvent.fakeStop`, deleted in Leaflet 1.8,
so a click on a feature throws before firing anything; and it returns without
firing at all when the click hits no feature. Between them the map stops
responding to clicks entirely for as long as any vector layer is ticked —
while the hover tooltips keep working, which is what makes it read as an app
fault rather than a plugin one. `basemap._CANVAS_TILE_CLICK_FIX_JS` replaces
the method on the prototype at run time, so the vendored file stays
byte-identical to the release it is named after;
`basemap._interaction_element` then re-fires the feature's click on the map.

The GeoJSON path is still there, still tested, and selected by
`HBU_MAP_RENDERER=geojson`. The app also falls back to it on its own, saying
so in the sidebar, when PostGIS is older than 3.1 (no `ST_AsMVT`) or the tile
server could not take its port. `find_lots_in_view`, the agent's tool, reads
`lots_in_bbox` either way — a tool wants rows, not tiles.

### The tile endpoint is behind the same password the app is

The tiles are on their own port, so no login form stands in front of them.
Every tile URL therefore carries a key derived from `HBU_APP_PASSWORD` — an
HMAC of it, never the password — and the server refuses a request without one.
Unset the password and both gates are off together: there is no configuration
in which the map is reachable and the app is not.

The key is *derived* rather than random on purpose. Every task in a service
computes the same one, so a tile request may be answered by any of them and
the tile target group needs no stickiness — unlike the app's, whose session
state lives in one task's memory.

`/tiles/healthz` is the one path outside the check, because a load balancer's
health check carries no credentials. So is
`/tiles/vendor/leaflet-vectorgrid-1.3.0.js`, and for a sharper reason.

**The renderer's own library is served from here, not from a CDN.** It is
committed under [`src/utils/vendor/`](src/utils/vendor/) and handed out by the
tile server on the origin the tiles already come from. This is not tidiness.
`streamlit_folium` loads a folium plugin's JavaScript by *awaiting* every
`default_js` URL before it renders, and it catches nothing if one of them
rejects — and it populates the map's own `<div>` inside that promise. A script
the browser cannot fetch therefore does not cost the vector layers, it costs
the entire map: a blank pane, no error on the page, and a console message
about a promise. Pointing that fetch at a third-party host makes an unrelated
CDN a hard dependency of the map existing at all. Off this server it is
reachable on exactly the condition the tiles are, which is the condition the
vector renderer already requires.

The URL carries the version, so the response is cached for a year and an
upgrade is a new path rather than an argument with a browser about a stale
body.

**Two ports means two things to publish.** `make run` and `make docker-run`
handle it; a hand-rolled `docker run -p 8501:8501` does not, and the symptom
is a map that draws a basemap and nothing else with the only evidence in the
browser console. Deployed, `hbu_infra` routes `/tiles/*` to the second port on
the same listener, so the URLs come out relative and name no port at all.

**`serve.py`, not `streamlit run app.py`.** Streamlit runs the app script per
*session*, so a tile server started from `app.py` comes up on the first page
load. Behind the load balancer that is a deployment loop: a task nobody has
visited fails the tile health check, ECS replaces it, and the replacement is
never visited either. `serve.py` starts the tile server first and then hands
every argument it was given to `streamlit run`.

### The buildings layer is footprints clipped to lots

Not the footprints. BDOI digitises a terrace, a semi-detached pair or a
shopping strip as **one contiguous outline across every party wall**, so a
footprint drawn whole spills over its neighbours' parcels and the area in its
hover is the block's rather than the building's — five row houses reported five
times as one 900 m² mass.

So the layer is the *intersection* of `rag.buildings` with `rag.lots`, and two
things follow from that:

- **a feature is one (building, lot) pair**, not one footprint. A school or a
  tower across three parcels is three features, each carrying only the part
  standing on its own lot. The hover key is `building_lot_key` — the two
  surrogates paired — because keyed on the footprint's id alone, hovering one
  house would highlight the whole terrace while the tooltip reported one house;
- **the hover says `Footprint on lot`**, and the number is the ground that
  footprint covers on *that* parcel. It is the same measurement the Lot pane
  reports as the measured *taux d'implantation*, off the same table, which is
  what stops the map and the pane disagreeing about one building.

A footprint standing on no lot at all is not drawn: it has no intersection to
be. `silver.building_lot_intersections` answers when the pipeline has built it
and the tile computes the clip when it has not — with the same
`ST_Dimension(...) = 2` screen the pipeline applies, because a party wall on a
lot line intersects and clips to a *line*, and without the screen every terrace
would draw a zero-area thread down each of its neighbours.

**Zoomed out, below zoom 16, the cells are still the unclipped ones.** They
come from the dataplatform's `gold.map_cell_aggregates`, built over
`rag.buildings`. The shading is a *dissolved* coverage, so it does not
double-count what the footprints share — but the `Footprints` row on a cell's
hover is a sum of whole footprints, which is why it is labelled in the plural
and why it is not the same measurement as the row one zoom in.

### The one layer that is not a scrape

Every other layer on the map is something a publisher drew: a cadastral lot, a
BDOI footprint, a zoning polygon. **Proposed massing** is not — it is
`gold.lot_building_massing`, the highest-and-best-use programme the
dataplatform solved for each lot, drawn as a rectangle inside that lot's
setback envelope so the zone's four margins are respected by the shape itself.

It is off by default and gated to zoom 16, the same gate the footprints take —
the proposal is read *against* what stands today, and showing one without the
other is half the comparison.

**Every massing is drawn in the same colour, and the layer has no legend.** It
used to carry a finding in its hue — green where the solved footprint fitted,
amber where it had to be shrunk, because a solver that caps a footprint on the
lesser of two *areas* never asks whether a building of that area has a shape
the parcel can take. The finding was right; the hue was the wrong place for
it. The only legend that ever named this layer's colours was the low-zoom
cell one, which appears in a single zoom band and is driven by a `view_zoom`
one interaction behind the map — so it showed erratically, and when it showed
it was describing a density ramp rather than the fit. Two colours with nothing
on screen to read them by is worse than one colour, because it still looks
like an answer. The fit is still reported, per lot, in the hover and on the
parcel pane's *as drawn* block, where it can be said in words. Its low-zoom
cells are flat for the same reason: one colour wherever a proposal was solved,
no ramp and no legend.

**Surface parking** is the massing's other half, and a separate layer because
it is another kind of thing. A surface stall has no floor area, no storey and
no height, so it is not part of the building: folding it into the massing would
inflate the very footprint the fit percentage is checking, and extruding it
would raise a solid where there is asphalt. It is drawn grey and dashed rather
than as a shade of the massing green, for the same reason - a light and a dark
of one hue would read as one thing.

It is fitted onto the **parcel** less the building, not into the setback
envelope: a margin is what a *building* keeps, and a car in a side or rear yard
stands exactly where the margin said no building may go. It need not front the
street, and no access route is modelled - nothing here proves a car can get to
the stall it can stand on. A bay is at least one stall deep, and a programme
whose yard cannot take every stall it asked for says so in the hover and on the
parcel pane, the way a shrunk massing does.

It is the one layer with no aggregate behind it, so unlike the others it is
gated at its own zoom rather than requested all the way down: below zoom 16 it
is simply unavailable. A lot missing from it is usually a lot that parks
underground, on a deck or in a ground-floor bay rather than one that failed to
park - `parking_status` on `gold.lot_building_massing` tells the two apart.

*Under-built lots only* narrows to the proposals that hold more floor than the
roll says stands there today. It is indented under **Proposed massing** in the
sidebar because it is a sub-option of it rather than a layer of its own, and it
screens the parking with it so the two cannot disagree about which parcels are
in scope.

**Utilisation** is the other one, and it is the same finding read the other
way round. Where the massing draws what *could* stand, this shades each lot by
how much of its permitted floor area already *does* —
`gold.lot_redevelopment_gap`, joined to the cadastre for a shape, because that
table carries no geometry of its own. The join is on `lot_number` within the
partition and deliberately not on `lot_uid`: the uid is a bigserial minted
fresh on every load of `rag.lots`, so reloading a borough-day behind an
already-materialized gold partition renumbers every lot and the join stops
matching anything at all. That empties the layer at *every* zoom rather than
shading it wrongly, because the low-zoom cells in `gold.map_cell_aggregates`
are dissolved from the same join over in the dataplatform — which is why the
symptom reads as a renderer that has forgotten one layer. It takes the
lot gate rather than the building one: the shading is read across a block at a
glance, and at zoom 16 too little of the block is on screen for the comparison
to mean anything.

The ramp is sequential and darkens as a lot empties, so "there is room here"
reads without consulting the legend. Two colours sit outside the ramp
deliberately. **Purple** is a lot holding *more* floor than today's grid
permits — a legal non-conformity, ordinary in a borough whose housing predates
its by-law, and colouring it as the efficient end of the ramp would invert the
map. **Grey** is a lot with no solved programme at all, which is not the same
as a lot with no room: `hbu_status` says which of the five reasons applies, and
the tooltip repeats it rather than showing a blank percentage.

*Under-built lots only* narrows this layer and the massing together, so the two
cannot disagree about which parcels are in scope.

A database without either table disables its toggle and changes nothing else —
the same advisory treatment the two silver joins get, for the same reason.

### Street sides, not centre lines

**Street sides** draws `silver.neighborhood_streets`, and the layer is doubled on
purpose. The city publishes a *géobase double*: two rows per street, one per
curb, keyed on `COTE_RUE_ID` — and that is the grain the question this map
exists to ask is asked at. A lot fronts on one **side** of a street, and the
side is where the curb and the sidewalk limits are, which is why
`silver.lot_frontage` joins a lot to one of these rather than to a road. A
centre line could not say which side, so it could not carry a frontage.

That doubling is the layer telling the truth about itself, and the name says
so — because a reader who does not know it reads the pair of lines as a
rendering fault. It says so in both places the layer can be ticked: the
sidebar's boxes are drawn from `basemap.TILE_LAYER_NAMES`, which is also what
names the layer in Leaflet's own control, so the two cannot drift. They had —
the sidebar said *Streets* while the map said *Street sides*, and the half of
the name carrying the whole point was the half the sidebar dropped.

Three decisions about how it draws:

- **Zoom 14**, two below the lots. A street grid is what says *where you are*
  before any parcel is legible, so it is on screen while the reader is still
  finding the block — and it is cheap there, a few thousand sides against
  Villeray's twenty-five thousand lots.
- **Above the shading, below the cadastre.** Above, because a hairline under a
  65 %-opaque utilisation band is not a line anybody can follow. Below, because
  a click on this map means *select the lot under the cursor*, and an
  interactive line layer on top would swallow that click along every frontage —
  which is exactly where a reader aims.
- **Not filled.** Leaflet fills a path by closing it across its two ends, so a
  filled street side paints a wedge across the block instead of a line along
  the curb. This is the only layer here whose geometry is open, and its style
  carries `fill: false` under both renderers rather than in one callback.

The geometry is already clipped to its borough by the pipeline, so a side that
crosses a borough line is short here on purpose, and the length in the tooltip
is the surviving piece rather than the published one. An unnamed service lane
is labelled *voie sans nom* rather than blanked: it is a real street side.

### The subtraction, and the two ways to get it wrong

The Overview pane totals the same comparison over the whole partition: how much
more residential, commercial and industrial floor area the borough could hold,
and how many more dwellings. Two things about that sum are worth stating,
because both are invisible in the answer and wrong in a way that looks
plausible.

**A missing existing floor is read as zero, not as unknown.**
`gold.lot_redevelopment_gap` publishes a gap column per class, and this app does
not sum it. That column is NULL wherever either side is, and the side that is
missing is nearly always the existing one — a lot the assessment roll never
reached is usually a lot with nothing standing on it. Summing the published gap
would therefore drop exactly the vacant parcels, which are the ones carrying the
most headroom, and it would do it silently. The rule used instead is the one
`is_underbuilt` is already documented to follow, so the two cannot disagree.

**An over-built lot contributes zero rather than a negative.** "How much more
could we build" is a question about the parcels where building is possible, and
letting a six-storey walk-up on a now-three-storey zone cancel the vacant lot
next door answers a different question quietly. The signed total is that other
question, and the pane states it separately — in a dense borough it can be
negative, which is a finding about the by-law rather than an error.

Neither total means much without the counts beside it, so the pane always shows
them: how many lots have a solved programme at all, how many are under-built,
how many were clamped, and how many had no assessment to compare against.

**The programme behind the numbers is a developer's, not a planner's.** The
dataplatform's solver prices all three usage families — housing at CMHC's
surveyed rents with a stated new-build premium, commerce and industry at the
borough's resolved commercial rents — and picks, per lot, the governing zoning
envelope worth the most *discounted net profit*: stabilised NOI discounted over
a hold, a terminal sale, construction cost off the top. The Lot pane shows that
arithmetic (`npv`, construction cost, and whether rebuilding beats holding the
standing building) and the HBU pane shows the programme behind it whole, the
Overview pane totals the gain where it is positive, and
a class with no proposed floor anywhere is an economics finding — at the
assumed rents nothing pencils — rather than a statement about the zoning. Every
assumption travels in `program_assumptions` on the gold rows.

### The HBU pane, and why it is not a section of the Lot one

The Lot pane answers *is there room here* — a subtraction, three headroom
figures and a dwelling count — and the Overview pane adds that subtraction up
over a borough. Neither says what is actually being proposed, and by the time
the answer is a building rather than a number there are about thirty columns of
it. **HBU** is that pane: one lot, the whole of `gold.lot_highest_best_use`'s
chosen row, read through `queries.lot_program`.

What is on it, in the order it is read:

| | |
|---|---|
| the shape | storeys, height, footprint, gross floor area — and the plate as a share of the lot and of the area the setbacks leave |
| as drawn | the massing rectangle's width, depth and bearing, and the fit against the costed footprint |
| the stack | `floor_stack` — which use stands on which levels, at what plate, with the dwellings and stalls on each run |
| housing | dwellings proposed against today, and the mix by CMHC bedroom class |
| commerce and industry | floors and floor area of each, beside whether the governing column authorises it at all |
| parking | the stalls, split across the four places one can go |
| the money | construction by class, parking, the total, stabilised NOI, and the discounted net profit the envelope was chosen on |
| why not more | `binding` — the printed caps the answer is pressed against |
| the assumptions | `program_assumptions`, whole |

It is a pane rather than a section under **Lot** because every number on it is
conditional on one choice — the governing envelope the solver picked — and
reading them beside what stands today is exactly the confusion the Lot pane
already has to caption its way out of twice, once for footprint against floor
area and once for floor area against the roll. Here the only figures about the
standing building are the two labelled as such: the dwelling count it is
compared against, and the verdict on whether building beats holding. Both come
from the gap table through the same cached read the Lot pane makes, so the two
panes cannot disagree, and a database with the programme but not the
subtraction loses the comparison rather than the proposal.

Three things on it are worth stating, because each is a distinction the numbers
do not make on their own:

**Four places a stall can go, and they are not interchangeable.** A dug level
is built and paid for and sits outside the *superficie de plancher* (article
38 1° of by-law 01-283); a parking deck is a storey and floor area both; a
garage bay in the ground floor is floor area **without** being a storey, so
*Densité* counts it and *En étage* does not; a stall on the yard is not in a
building at all. They also cost an order of magnitude apart — which is why the
pane splits the total rather than reporting it, and why the two provisions that
are floor area are named as taking it from the dwellings.

**A class authorised and not built is a different finding from one not
authorised.** The commerce-and-industry table shows *none proposed* against
`—` for exactly that reason: the first says the solver priced the storey and
something outbid it at the borough's surveyed rents, which is an economics
finding; the second says the governing column never permitted it. The Overview
pane draws the same distinction over a whole borough.

**`binding` answers two different questions and takes two headings.** On a
solved row every name on it is a cap the programme *reached*, and the list
answers "why is it not bigger" — change one of those rows in the grid and the
answer moves, change one that is not on the list and it does not. On an
infeasible row it is a pair of printed rows contradicting each other, and the
list answers "why is there nothing"; calling that a cap the programme reached
would describe a building that was never solved. `nothing_pencils` is the third
case and the sharpest: the envelope is whatever the grid prints and what is
zero is the best programme inside it.

A lot with no solved programme keeps the pane rather than emptying it. The five
`hbu_status` reasons are shared with the Lot pane from one dict — a status the
dataplatform renames is otherwise half-updated in two places — and the candidate
counts, the zone and any `binding` are what there is to say about such a lot.
A lot the gap table files as `road_parcel` is refused here the way it is
refused there: the parcel is the public way itself, and every figure would be
arithmetic on an artefact.

A database without `gold.lot_highest_best_use` disables the pane and changes
nothing else — the same advisory treatment the two silver joins and the massing
get, for the same reason.

Set `URBAN_RAG_PG_SCHEMA` / `URBAN_RAG_PG_SILVER_SCHEMA` /
`URBAN_RAG_PG_GOLD_SCHEMA` to read a review copy of any of them; they default
to `rag`, `silver` and `gold`.

---

## Quick start

```bash
make install                        # .venv + deps, and a .env to fill in
make db-up                          # local postgis+pgvector, hbu_infra's schema applied
make check                          # what is loaded, and what is missing
make run                            # http://localhost:8501, tiles on 8502
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

A private RDS instance has no public endpoint, so from outside the VPC it is
reached through the SSM bastion tunnel. Leave it open in one terminal:

```bash
cd ../hbu_infra
make db-tunnel ENV=dev LOCAL_PORT=5433
```

Then, in another, either:

```bash
make run-tunnel                    # natively
make docker-run-tunnel TUNNEL_PORT=5433   # in the image
```

**Both targets hold `sslmode=verify-full`, and the thing that makes it
possible is keeping the certificate's name separate from the address.**
`docker-run-tunnel` maps the real RDS hostname onto Docker's host gateway, so
the name libpq checks still resolves. A native run cannot rewrite its own
resolver, so `run-tunnel` splits the two instead: `URBAN_RAG_PG_HOST` goes on
naming the RDS endpoint — which is what the hostname check matches against —
while `URBAN_RAG_PG_HOSTADDR` carries the `127.0.0.1` the socket actually
connects to. Either way `TUNNEL_DB_HOST` must stay the RDS endpoint and never
`localhost`, and both targets refuse to start if it is a loopback address.

Before `hostaddr` was wired through, the native target had to drop to
`require` — encrypted, but authenticating nothing, leaning on the SSM session
as the only authenticated hop. That trade is no longer necessary.

`run-tunnel` addresses the tunnel as `127.0.0.1`, never `localhost`, and that
is not a style choice. The Session Manager plugin binds IPv4 only; Windows
resolves `localhost` to `::1` first, and libpq spends the *entire*
`connect_timeout` on that dead address before falling back and succeeding — so
every connect takes exactly `connect_timeout` seconds and then works, which
reads as a slow tunnel rather than as a misresolution. The pool's own timeout
is 15 s, so through `localhost` it raises `PoolTimeout` while libpq is still
waiting.

Both targets pass `AWS_PROFILE`. The app-role secret can live in a different
account from the caller's default credentials, and without the profile the run
dies on a cross-account `secretsmanager:GetSecretValue` denial — a much less
obvious message than "wrong profile".

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
| `URBAN_RAG_PG_SCHEMA` | Where the corpus and the source geometry live. Default `rag`. |
| `URBAN_RAG_PG_SILVER_SCHEMA` | Where the pipeline's derived tables live — `building_lot_intersections` and the rest. Default `silver`. Two settings rather than one because they really are two schemas, and a review copy may rename only one. |
| *(nothing)* | SSM `/hbu-<env>/db/*` + the app-role secret, from `HBU_ENV` and AWS credentials. |

The order is the contract: a developer with a container running and AWS
credentials in the shell gets the container, and someone with nothing set gets
the SSM lookup rather than a confusing localhost refusal.

Resolution is **memoised** for `HBU_PG_RESOLVE_TTL` (default ten minutes),
keyed on every environment variable the four paths read — so changing one
re-resolves, and `get_pool`'s reopen-on-endpoint-change still holds.

The credential that genuinely expires is the RDS IAM auth token, signed for
fifteen minutes, and it is **not** in what `resolve()` returns:
`_fresh_connection` mints one per connect. What the memo holds is an endpoint
and, on the Secrets Manager path, a password that does not expire between
rotations. Resolving that per call was costing two `GetSecretValue` round trips
per query — one from `_Borrowed`, one from `get_pool` — which is ten AWS calls
per map pan, on the Streamlit script thread.

The pool validates a connection before lending it (`check`), and retires idle
ones after five minutes. Without that it hands out handles the server has
already closed — an RDS idle timeout, a NAT or firewall idle drop, a failover,
a restarted tunnel — and the borrower dies on `server closed the connection
unexpectedly` rather than on anything it did.

`sslmode=verify-full` needs a CA bundle, and the configured path is not
portable between run modes: the image carries Amazon's global bundle at
`/etc/ssl/certs/rds-global-bundle.pem` (the `Dockerfile` puts it there, and
`hbu_infra/ecs.tf` names that path), while a native run has it where `make
db-ca` writes it. One `.env` serves both, so a configured path that is absent
falls back to the default one with a warning naming both. Both files are the
same bundle; if neither is there, the connection is still refused.

AWS SDK calls use botocore's certificate bundle, not `SSL_CERT_FILE`. If
Secrets Manager or SSM fails with `CERTIFICATE_VERIFY_FAILED` behind a
TLS-inspecting proxy, set `AWS_CA_BUNDLE` or `URBAN_RAG_AWS_CA_BUNDLE` to a PEM
bundle that contains the proxy root. For Docker, pass `AWS_CA_BUNDLE=/host/path`
to `make docker-run`; the Makefile mounts it into the container and points
botocore at that container path.

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
- **Layers are zoom-gated.** Street sides draw from zoom 14, lots from 15,
  buildings and massings from 16. Below that a lot is sub-pixel and a borough
  of them is a grey rectangle that costs a second of browser time to produce.
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

- its attributes and the footprints standing on it, from
  `silver.building_lot_intersections` when that table is populated (an index
  lookup on `lot_number` rather than an `ST_Intersection` per click) and
  computed on the fly when it is not — **except on a parcel that is the
  street**. A lot whose gold row reads `hbu_status = road_parcel` is the public
  way itself, and the pane reports no coverage, no utilisation and no
  developer economics for it: a footprint overlapping a roadway is two layers
  meeting at the curb, and every number computed from it would be arithmetic
  on an artefact;
- the zoning polygons covering it, **ordered by how much of the lot each
  actually covers** — a lot on a zone boundary intersects both, and only one of
  them is the answer. When more than one applies, the pane says so and lets you
  pick. Read from `silver.lot_features` when it is populated, on the same terms
  as the footprints above, and clipped on the fly when it is not;
- the *grille des spécifications* for the chosen zone: its values as a table,
  and the PDF itself.

### The zoning PDF

`Reglement_urbanisme__VSP_REG_ZONE` carries a link, not a description:

```
LIEN_GRILLE = http://www1.ville.montreal.qc.ca/CartesInteractives/villeray/doc/zone/C01-001.pdf
```

**The sheet is served from this app's own origin**, at
`/tiles/grid/<doc_id>.pdf`, and that is what makes it a link a reader can
actually follow. A `LIEN_GRILLE` is an `http://` URL, and an `https://` page
will not open one without objecting to the downgrade — so the pane offers the
city's citable URL *and* this app's copy of the same bytes, which is
same-origin under either deployment shape.

**The viewer beside it is `st.pdf`, not an iframe.** It used to be an iframe
pointed at that route, and that is the thing Microsoft Edge paints *"This page
has been blocked by Microsoft Edge"* over: Streamlit renders every iframe it
declares with a `sandbox` attribute, Chromium will not start a PDF plugin
inside a sandboxed frame, and a browser's built-in PDF viewer is plugin
content. Edge draws its interstitial; Chrome draws nothing. No response header
fixes it, because the block is on the frame rather than on the response.
`st.pdf` is pdf.js drawing to a canvas in the page's own DOM — a CCv2
component, so not inside an iframe at all — and it keeps the text selection,
the search and the page zoom that were the reason for embedding a viewer in
the first place. It is handed the *bytes*, which go through Streamlit's media
file manager, so the pane shows the sheet whether or not the tile server took
its port. It needs the `streamlit-pdf` package; `st.pdf` raises without it.

The route takes an **id, never a URL**. Only a document this process has
already fetched for a zone somebody clicked resolves, so a PDF proxy is not
also an open one: there is no address in a request for the server to go and
get. It answers from a small in-process registry or from the disk cache below,
and it is behind the same key the tiles are.

Pages are **also rasterised**, with pypdfium2, and the pane keeps them under
*Pages as images* — open by default when the viewer above is not there, which
now means only a deployment missing `streamlit-pdf`. They work when the cache
is warm and the network is not, and they are the same bytes the download
button hands over.

The cache key is `sha256(url)[:16]` — **identical to the dataplatform's
`document_id`** — so pointing `HBU_PDF_CACHE_DIR` at
`../hbu_dataplatform/data/cache/pdf` reuses the PDFs the pipeline already
downloaded rather than pulling them from the city's web server again. A
published zoning grid does not change once issued, which is what makes a cache
with no expiry correct here rather than merely convenient.

A dead link fails its own document, not the pane: these are municipal URLs, and
some answer `200` with an HTML "page not found" body, so the content is checked
for a PDF header rather than trusted — and the route serves those bytes with
`X-Content-Type-Options: nosniff`, so a document that lied about its type is
refused by the browser rather than sniffed into whatever it actually is.

### A click that lands on no lot

Zoning covers ground the cadastre does not — a park, a right of way, the far
side of a rail cut — and the grid that applies there is a real answer rather
than an absence. So a click resolves the lot first, because that is the finer
answer and the one the rest of the pane is built around, and failing that
resolves the **zone**: the pane shows its values and its grille with no parcel
behind them. Selecting a lot clears a zone chosen that way, because two
selections disagreeing about which zone is under discussion is the one thing
that pane exists to prevent.

Turn **Zoning** on in the sidebar to see where the boundaries run; the layer
draws at every zoom.

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
| `buildings_on_lot` | the footprints, and how much *ground* they cover inside the lot — the measured taux d'implantation |
| `lot_efficiency` | how much of one lot's permitted *floor area* is used, and what else fits |
| `development_capacity` | the same subtraction, totalled over the borough |
| `top_redevelopment_lots` | the lots where rebuilding beats holding, by discounted gain |
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
  [!!] silver.lot_features
    hbu_infra sql/005_silver_lot_features.sql, filled by the same asset;
    without it the zoning a lot falls under is intersected per click
  [--] rag.chunks
    hbu_dataplatform: make publish DATE=... NEIGHBORHOOD=...
  [--] rag.search_at_lot()
    hbu_infra sql/003_spatial_search.sql — skipped until rag.chunks exists,
    so re-run `make db-init` after the first publish
```

Two markers, and the difference is the point. `[--]` is a fault: something
cannot be answered. `[!!]` is advisory — a silver join the pipeline has not
computed for this borough yet, so the answer is worked out per click instead of
looked up. Only the faults count toward the total at the bottom.

`rag.search_at_lot()` is the ordering trap worth knowing: a SQL-language
function body is parsed at `CREATE` time, so the spatial search functions
genuinely cannot be created before `rag.chunks` exists. `hbu_infra`'s `db.py`
skips the file with a note. Publish a partition from the dataplatform, then run
`db-init` once more.

The app degrades rather than breaks around each gap: a missing `rag.buildings`
greys out its layer, a missing corpus stops *retrieval* and makes the retrieval
tools tell the model which asset creates the table — so it reports the gap
instead of retrying three times. The Regulations pane keeps its top half
through that one: which sheets govern a lot is a join, and without
`rag.lot_documents` it is answered from the `LIEN_GRILLE` on the zoning rows,
which needs no corpus at all.

---

## Tests

```bash
make test              # 176 unit tests; no socket is opened
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

### The gate the integration suite has to get through

`auth.py` switches itself off when `HBU_APP_PASSWORD` is unset, and that is
what let this suite run unchanged — until a filled-in `.env` set it. `app.py`
calls `load_dotenv()` *above* the gate, so on any developer machine with a real
`.env` every test in `test_app.py` stopped at the password form. The symptom
named nothing: a bare `KeyError: 'Lots'` from a sidebar that was never drawn,
and `at.exception` empty, because stopping a script is not an error.

So `_app()` seeds the session flag a successful login sets, unconditionally —
assuming the gate is off is exactly how that failure comes back. What keeps the
shortcut honest is `tests/test_auth.py`, which drives the real form and asserts
from both directions: a correct password sets that key, and that key alone is
enough to get past. It also covers what the gate must refuse — a wrong
password, a non-ASCII attempt (which would raise `TypeError` rather than fail
if the comparison were made on `str`), and Terraform's `PLACEHOLDER`, which
draws no form at all because it is a password written down in a `.tf` file.

---

## Files

| | |
|---|---|
| [`serve.py`](serve.py) | The entrypoint: the tile server, then Streamlit |
| [`app.py`](app.py) | The Streamlit page: map, Lot pane, Regulations pane, chat |
| [`src/agent.py`](src/agent.py) | The ReAct agent, its prompt, and the streaming loop |
| [`src/config.py`](src/config.py) | The chat-model catalog and `build_llm()` |
| [`src/utils/db.py`](src/utils/db.py) | Four ways to find the database, and the pool |
| [`src/utils/queries.py`](src/utils/queries.py) | Every statement this app sends |
| [`src/utils/embeddings.py`](src/utils/embeddings.py) | Query embedding, and the encoder-mismatch guard |
| [`src/utils/documents.py`](src/utils/documents.py) | Fetching, caching and rasterising the grid PDFs |
| [`src/utils/basemap.py`](src/utils/basemap.py) | Assembling the folium map, under either renderer |
| [`src/utils/tiles.py`](src/utils/tiles.py) | The tile server, its cache, and the key that guards it |
| [`src/utils/vendor/`](src/utils/vendor/) | Leaflet.VectorGrid, committed — see the README there for why |
| [`src/utils/state.py`](src/utils/state.py) | The side-channel between tools and the map |
| [`src/utils/auth.py`](src/utils/auth.py) | The shared password, and everything it does not buy |
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
