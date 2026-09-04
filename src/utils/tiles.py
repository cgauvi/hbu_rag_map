"""
tiles.py — the HTTP side of the vector tiles, and why there is a second port.

`queries.mvt_tile` produces a tile; something has to serve it. Streamlit
cannot: it owns its own routes and has no extension point for another one, and
a tile has to arrive over HTTP because the thing asking for it is Leaflet
inside the map's iframe, not the Python that rendered the page.

So this module runs a small HTTP server on a second port **in the same
process**. Not a sidecar container and not a separate service, for three
reasons worth stating:

* it borrows the same `db` pool, so a tile is authenticated, TLS-verified and
  timed out by exactly the code every other query already goes through;
* one process means one deployment — the image, the task definition and
  `make run` all keep working with one port added rather than a second thing
  to start;
* tiles are I/O on a socket, so the GIL is released for the whole time a tile
  is being computed and the server thread does not compete with the Streamlit
  script thread for anything but the pool.

**What guards it.** The Streamlit UI sits behind ``HBU_APP_PASSWORD``; a tile
endpoint on a public listener would walk straight around that, and a cadastre
joined to a solved development programme is not something to publish by
accident. So every tile URL carries a key derived from that same password —
an HMAC of it, never the password itself — and a request without the key is
refused. Two consequences follow, both deliberate:

* with no password set the gate is off, exactly as `auth.py` is off, which is
  what keeps `make run` and the test suite working on a laptop;
* the key is derived rather than random, so every task in a service computes
  the same one and a tile request may land on any of them. The tile target
  group therefore needs no stickiness — only the Streamlit one does, because
  only it holds session state.

It is the same class of protection as the password: it authenticates access,
not people, and anyone holding a rendered page holds the key.

**It also serves the zoning grids.** ``/tiles/grid/<doc_id>.pdf`` hands back a
PDF `documents` has already fetched, and it is here rather than anywhere else
for the same reason the tiles are: it needs an origin the *browser* can reach.
A ``LIEN_GRILLE`` is an ``http://`` city URL, and a page served over ``https://``
will not open it without the browser complaining about a downgrade. Off this
server the same bytes are same-origin under either deployment shape, so the
pane's "Open the grid" button is a link that simply works.

It is no longer what the *inline* viewer reads. That was an iframe pointed
here, and Streamlit sandboxes every iframe it renders, which is a context
Chromium will not start a PDF plugin in - Edge paints "This page has been
blocked by Microsoft Edge" over the pane and Chrome paints nothing. The pane
now draws the sheet with `st.pdf`, from bytes, through Streamlit's own media
store; see `app._embed_pdf`.

The route takes an id, never a URL: see `documents.published`.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import threading
import urllib.parse
from collections import OrderedDict
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logger = logging.getLogger(__name__)

#: The port the tile server listens on inside the container. `hbu_infra`'s
#: ecs.tf maps it and routes ``/tiles/*`` to it; `make run` publishes it too.
DEFAULT_PORT = int(os.environ.get("HBU_TILE_PORT", 8502))

#: Every route this server answers begins here, and so does the ALB rule that
#: sends traffic to it. One prefix in both places or neither works.
PATH_PREFIX = "/tiles"

#: What the ALB polls. Deliberately outside the key check: a health check
#: carries no credentials, and a target group that cannot reach its own health
#: path drains the service on the first deploy.
HEALTH_PATH = f"{PATH_PREFIX}/healthz"

#: The browser-side library the vector renderer *is*, served from here rather
#: than from a CDN. `streamlit_folium` awaits every plugin script before it
#: draws anything, and catches no failure — so a script it cannot fetch does
#: not leave the map without its layers, it leaves the page without the map.
#: Off this server the library is reachable on exactly the condition the tiles
#: are, and that is the condition the vector renderer already turns on.
#:
#: Keyless, like the health path and unlike a tile: it is public MIT-licensed
#: code that carries no cadastre, and a browser allowed to keep it across
#: sessions is one that fetches it once.
VENDOR_PREFIX = f"{PATH_PREFIX}/vendor"

#: The version is in the *name*, so an upgrade is a new URL rather than a new
#: body at an old one — which is what lets the response be cached hard without
#: a deploy having to argue with a browser about it.
VECTORGRID_FILE = "leaflet-vectorgrid-1.3.0.js"

#: Committed under `src/utils/vendor`; that README carries the provenance and
#: the digest.
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"

#: Where a zoning grid is published. Keyed like a tile rather than public like
#: the vendored library: the document itself is a municipal publication, but
#: *which* grids this server holds is a trace of what has been looked at, and
#: the id is the only thing standing between a request and a file in the cache
#: directory.
GRID_PREFIX = f"{PATH_PREFIX}/grid"

#: A year. See `VECTORGRID_FILE` — the only way to get a different body is to
#: ask for a different URL.
VENDOR_CACHE_SECONDS = 31_536_000


#: Tiles are immutable for a given (layer, z, x, y, filters) until the
#: partition behind them is reloaded, which is a daily event at most. An hour
#: of browser cache turns a pan back over ground already seen into no requests
#: at all; the in-process cache below covers the first visit and the reloads.
CACHE_SECONDS = int(os.environ.get("HBU_TILE_CACHE_SECONDS", 3600))

#: How many tiles to keep in this process. A tile of lots is a few tens of
#: kilobytes, so the default is single-digit megabytes and covers roughly two
#: screens' worth of every layer at once.
CACHE_TILES = int(os.environ.get("HBU_TILE_CACHE_TILES", 512))

#: Leaflet opens as many tile requests as it has tiles to fill, which is
#: comfortably more than the pool has connections. Without a bound the pool's
#: 15-second borrow timeout is what a user meets on the *map* — the Streamlit
#: script thread waiting behind twenty tiles for a connection to answer the
#: click it is handling. Unset, the bound is one less than the pool, which
#: leaves that thread a slot.
_CONFIGURED_CONCURRENCY = int(os.environ.get("HBU_TILE_CONCURRENCY", 0))


def _concurrency() -> int:
    if _CONFIGURED_CONCURRENCY > 0:
        return _CONFIGURED_CONCURRENCY
    from src.utils.db import POOL_MAX_SIZE  # noqa: PLC0415

    return max(1, POOL_MAX_SIZE - 1)


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------

#: The string the HMAC is taken over. Fixed and public — what makes the key a
#: secret is the password it is keyed with, not this.
_KEY_MESSAGE = b"hbu-rag-map/tiles"


def tile_key() -> str | None:
    """The key every tile URL must carry, or None when the gate is off.

    Derived from ``HBU_APP_PASSWORD`` so the tiles are exactly as reachable as
    the app is and no more. Truncated to 32 hex characters because it travels
    in a query string on every tile request and the full digest buys nothing
    against a password this is only as strong as anyway.

    Returns None both when no password is set and when it is still Terraform's
    placeholder — `auth.py` refuses every login in that state, so there is no
    page to hold a key and nothing to let through.
    """
    from src.utils.auth import _PLACEHOLDER, PASSWORD_ENV  # noqa: PLC0415

    password = os.environ.get(PASSWORD_ENV, "").strip()
    if not password or password == _PLACEHOLDER:
        return None
    digest = hmac.new(password.encode("utf-8"), _KEY_MESSAGE, hashlib.sha256)
    return digest.hexdigest()[:32]


def _key_accepted(offered: str | None) -> bool:
    expected = tile_key()
    if expected is None:
        return True
    return offered is not None and hmac.compare_digest(offered, expected)


# ---------------------------------------------------------------------------
# Where the browser is told to look
# ---------------------------------------------------------------------------


def base_url() -> str:
    """The origin to prefix every tile URL with — possibly the empty string.

    Two shapes, and which one is right is a property of the *deployment*
    rather than of this code, which is why it is configuration:

    ``HBU_TILE_BASE_URL`` **unset** — a laptop. Streamlit is on 8501 and this
    server is on its own port, two different origins, so the URL has to be
    absolute and the responses carry CORS headers.

    ``HBU_TILE_BASE_URL`` set to ``""`` or ``same-origin`` — behind the load
    balancer. ``/tiles/*`` is a rule on the same listener the app is served
    from, so a relative URL is both correct and one less thing to keep in step
    with the DNS name. `hbu_infra`'s ecs.tf sets it.

    Anything else is used as given, for a CDN or a hostname in front.
    """
    configured = os.environ.get("HBU_TILE_BASE_URL")
    if configured is not None:
        value = configured.strip()
        if value in ("", "same-origin"):
            return ""
        return value.rstrip("/")
    host = os.environ.get("HBU_TILE_PUBLIC_HOST", "localhost")
    return f"http://{host}:{DEFAULT_PORT}"


def vectorgrid_url() -> str:
    """Where the page should fetch Leaflet.VectorGrid from.

    The same `base_url` the tiles use, so the library and the data it draws
    share an origin under every deployment shape: an absolute
    ``localhost:8502`` on a laptop, and a root-relative ``/tiles/...`` behind
    the load balancer, where it rides the ALB rule that is already there.

    No key. See `VENDOR_PREFIX` — this is a public library, and one a browser
    should be allowed to keep.
    """
    return f"{base_url()}{VENDOR_PREFIX}/{VECTORGRID_FILE}"


def grid_url(doc_id: str) -> str | None:
    """Where the page should point at an already-fetched zoning grid.

    None for anything not shaped like a ``document_id``, so a caller cannot
    build a URL this server would refuse anyway.

    Behind the load balancer `base_url` is empty and this comes out
    root-relative — which is the point: the link inherits the page's scheme and
    host, so it is ``https`` where the page is, and the browser frames it.
    """
    from src.utils.documents import is_document_id  # noqa: PLC0415

    if not is_document_id(doc_id):
        return None
    key = tile_key()
    suffix = f"?{urllib.parse.urlencode({'k': key})}" if key else ""
    return f"{base_url()}{GRID_PREFIX}/{doc_id}.pdf{suffix}"


def layer_url(layer: str, filters: dict[str, object] | None = None) -> str:
    """The templated URL Leaflet fills in, for one layer.

    ``{z}/{x}/{y}`` are left as literal braces on purpose: Leaflet substitutes
    them per tile, and the filters — which snapshot, which borough, which
    screen — are the same for every tile of one map and travel in the query
    string. Putting them there rather than in the path is what lets the
    browser's own cache key them, so toggling *under-built lots only* and
    toggling it back costs no requests at all.
    """
    query: dict[str, str] = {}
    for name, value in (filters or {}).items():
        if value is None or value is False or value == "":
            continue
        query[name] = value.isoformat() if isinstance(value, date) else str(value)
    key = tile_key()
    if key:
        query["k"] = key
    suffix = f"?{urllib.parse.urlencode(query)}" if query else ""
    return f"{base_url()}{PATH_PREFIX}/{layer}/{{z}}/{{x}}/{{y}}.mvt{suffix}"


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------


class _TileCache:
    """A bounded LRU over rendered tiles, shared by every server thread.

    Not `functools.lru_cache`: this is read from several threads at once and
    wants an explicit lock, and the eviction has to be by tile count rather
    than by call signature so the memory it can hold is a number an operator
    can reason about.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, capacity)
        self._entries: OrderedDict[tuple, bytes] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple) -> bytes | None:
        with self._lock:
            if key not in self._entries:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return self._entries[key]

    def put(self, key: tuple, body: bytes) -> None:
        with self._lock:
            self._entries[key] = body
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_cache = _TileCache(CACHE_TILES)


# ---------------------------------------------------------------------------
# Parsing a request
# ---------------------------------------------------------------------------


def parse_path(path: str) -> tuple[str, int, int, int] | None:
    """``/tiles/lots/15/9646/11732.mvt`` → ``("lots", 15, 9646, 11732)``.

    None for anything that is not a tile of a layer this app serves, which the
    handler turns into a 404. The layer is checked against
    `queries.MVT_LAYER_NAMES` rather than against a list kept here, so adding
    a layer to the SQL is the whole of adding it to the server.
    """
    from src.utils import queries  # noqa: PLC0415

    if not path.startswith(f"{PATH_PREFIX}/"):
        return None
    parts = path[len(PATH_PREFIX) + 1 :].split("/")
    if len(parts) != 4:
        return None
    layer, z, x, y = parts
    if layer not in queries.MVT_LAYER_NAMES:
        return None
    if not y.endswith((".mvt", ".pbf")):
        return None
    try:
        zoom, column, row = int(z), int(x), int(y.rsplit(".", 1)[0])
    except ValueError:
        return None
    # A zoom outside the grid makes ST_TileEnvelope raise rather than return
    # an empty tile, and x/y outside the zoom's range would be a request for a
    # tile that does not exist. Refusing here keeps a malformed URL from
    # costing a round trip to the database.
    if not 0 <= zoom <= 24:
        return None
    span = 1 << zoom
    if not (0 <= column < span and 0 <= row < span):
        return None
    return layer, zoom, column, row


def vendor_file(path: str) -> bytes | None:
    """The bytes of the vendored asset ``path`` addresses, or None.

    The name is matched against the one file this server publishes rather than
    joined onto `VENDOR_DIR`, so no request can name a path at all — there is
    nothing here for ``..`` to traverse, because nothing from the URL reaches
    the filesystem.
    """
    if path != f"{VENDOR_PREFIX}/{VECTORGRID_FILE}":
        return None
    try:
        return (VENDOR_DIR / VECTORGRID_FILE).read_bytes()
    except OSError as exc:
        # A checkout missing the file, which is a packaging fault rather than a
        # request fault. Logged loudly because the symptom downstream is the
        # blank pane this whole route exists to prevent.
        logger.error("Vendored %s is unreadable: %s", VECTORGRID_FILE, exc)
        return None


def parse_grid_path(path: str) -> str | None:
    """``/tiles/grid/784a0b4f710d1785.pdf`` -> ``"784a0b4f710d1785"``.

    None for anything else, which the handler turns into a 404. The id is
    checked for shape before it goes anywhere near the filesystem — see
    `documents.is_document_id`; sixteen hex characters cannot name a directory
    to climb out of or an extension to change, so nothing from the URL has to
    be sanitised further down.
    """
    from src.utils.documents import is_document_id  # noqa: PLC0415

    if not path.startswith(f"{GRID_PREFIX}/"):
        return None
    name = path[len(GRID_PREFIX) + 1 :]
    if not name.endswith(".pdf"):
        return None
    doc_id = name[: -len(".pdf")]
    return doc_id if is_document_id(doc_id) else None


def _tile_arguments(query: dict[str, list[str]]) -> dict[str, object]:
    """The filter half of a tile URL, as `queries.mvt_tile` keyword arguments.

    Everything is optional and anything unrecognised is ignored: the URL is
    written by `layer_url` and read here, but it is also a URL a browser has
    cached and may replay after a deploy that changed the parameter set.
    """

    def first(name: str) -> str | None:
        values = query.get(name) or []
        return values[0].strip() if values and values[0].strip() else None

    arguments: dict[str, object] = {}
    scrape_date = first("scrape_date")
    if scrape_date:
        try:
            arguments["scrape_date"] = date.fromisoformat(scrape_date)
        except ValueError:
            pass
    neighborhood = first("neighborhood")
    if neighborhood:
        arguments["neighborhood"] = neighborhood
    for name, target in (("min_area", "min_area_m2"), ("max_area", "max_area_m2")):
        raw = first(name)
        if raw:
            try:
                arguments[target] = float(raw)
            except ValueError:
                pass
    if (first("underbuilt") or "").lower() in {"1", "true", "yes", "on"}:
        arguments["only_underbuilt"] = True
    return arguments


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    # HTTP/1.1 so Leaflet's dozen-odd parallel tile requests reuse connections
    # instead of opening one apiece. Every response below sets Content-Length,
    # which is what makes keep-alive safe.
    protocol_version = "HTTP/1.1"
    server_version = "hbu-tiles"
    sys_version = ""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == HEALTH_PATH:
            self._respond(200, b"ok", "text/plain; charset=utf-8", cache=False)
            return

        asset = vendor_file(parsed.path)
        if asset is not None:
            self._respond(
                200,
                asset,
                "application/javascript; charset=utf-8",
                cache_control=f"public, max-age={VENDOR_CACHE_SECONDS}, immutable",
            )
            return

        doc_id = parse_grid_path(parsed.path)
        if doc_id is not None:
            self._serve_grid(doc_id, urllib.parse.parse_qs(parsed.query))
            return

        target = parse_path(parsed.path)
        if target is None:
            self._respond(404, b"not found", "text/plain; charset=utf-8", cache=False)
            return

        query = urllib.parse.parse_qs(parsed.query)
        offered = (query.get("k") or [None])[0]
        if not _key_accepted(offered):
            # 403 rather than 401: there is no challenge to answer and no
            # credential the browser could be asked to supply. The key comes
            # from the page, and a page is what the password gate hands out.
            self._respond(403, b"forbidden", "text/plain; charset=utf-8", cache=False)
            return

        layer, z, x, y = target
        arguments = _tile_arguments(query)
        key = (layer, z, x, y, tuple(sorted(arguments.items(), key=lambda kv: kv[0])))

        body = _cache.get(key)
        if body is None:
            from src.utils import queries  # noqa: PLC0415

            try:
                with _slots:
                    # Below a layer's detail zoom the tile comes from the
                    # dissolved cells instead of from the layer itself, and
                    # below `queries.AGGREGATE_OUTLINE_ZOOM` those cells come
                    # back as outlines - `mvt_aggregate_tile` reads the zoom
                    # for that second threshold itself, because it is the same
                    # query either way and only its columns and its geometry
                    # differ. The cache key already carries `z`, so the three
                    # kinds of tile never collide in it and crossing either
                    # threshold is a miss rather than a stale hit.
                    if queries.serves_aggregate(layer, z):
                        body = queries.mvt_aggregate_tile(
                            layer,
                            z,
                            x,
                            y,
                            # Only the partition filters. The lot area range
                            # and the under-built screen are properties of a
                            # lot, and a cell is not one - see
                            # `mvt_aggregate_tile` on why they are refused
                            # rather than ignored.
                            scrape_date=arguments.get("scrape_date"),
                            neighborhood=arguments.get("neighborhood"),
                        )
                    else:
                        body = queries.mvt_tile(layer, z, x, y, **arguments)
            except Exception as exc:  # noqa: BLE001 - one bad tile, not the map
                # A 500 here would make Leaflet retry and the console fill up.
                # The map is expected to survive a layer whose table is not
                # loaded, so the failure is logged once and the tile comes
                # back empty — which draws nothing, which is the truth.
                logger.warning("Tile %s/%d/%d/%d failed: %s", layer, z, x, y, exc)
                self._respond(
                    200, b"", "application/vnd.mapbox-vector-tile", cache=False
                )
                return
            _cache.put(key, body)

        self._respond(200, body, "application/vnd.mapbox-vector-tile")

    def _serve_grid(self, doc_id: str, query: dict[str, list[str]]) -> None:
        """One zoning grid, from what `documents` has already fetched.

        A 404 rather than a fetch when the id is unknown, and that is the whole
        security argument for this route: the only ids that resolve are ones
        this process has been handed a *URL* for by the database, so no request
        can name an address for the server to go and get.
        """
        if not _key_accepted((query.get("k") or [None])[0]):
            self._respond(403, b"forbidden", "text/plain; charset=utf-8", cache=False)
            return

        from src.utils import documents  # noqa: PLC0415

        body = documents.published(doc_id)
        if not body:
            # The grid the page linked to has aged out of the registry and was
            # never written to disk — a read-only filesystem, most likely. The
            # pane's rasterised pages and its download button both still work,
            # so this is a missing viewer rather than a missing document.
            logger.info("Grid %s is not published in this process", doc_id)
            self._respond(404, b"no such grid", "text/plain; charset=utf-8", cache=False)
            return

        self._respond(
            200,
            body,
            "application/pdf",
            # Immutable for the same reason the vendored library is: the id is
            # a digest of the URL, so different bytes are a different path.
            cache_control=f"private, max-age={VENDOR_CACHE_SECONDS}, immutable",
            extra_headers=(
                # `inline` so a link opens the browser's PDF viewer rather than
                # saving a file; the pane has a download button for the other.
                ("Content-Disposition", f'inline; filename="{doc_id}.pdf"'),
                # These bytes came off a municipal web server. `fetch` checks
                # them for a PDF header, but a document that lied about its
                # type should be refused by the browser rather than sniffed
                # into whatever it actually is.
                ("X-Content-Type-Options", "nosniff"),
            ),
        )

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
        # Leaflet does not preflight a plain GET, but a proxy or an extension
        # may; answering keeps a cross-origin laptop run from failing on it.
        self._respond(204, b"", "text/plain", cache=False)

    def _respond(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        cache: bool = True,
        cache_control: str | None = None,
        extra_headers: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in extra_headers:
            self.send_header(name, value)
        # The laptop shape of `base_url` is cross-origin by construction —
        # Streamlit on 8501, this on 8502 — so the tiles have to say they may
        # be read. Behind the load balancer the two share an origin and this
        # header is simply unused.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        if cache_control is not None:
            # The vendored library, whose URL carries its version — see
            # `VENDOR_CACHE_SECONDS`. Everything else takes the tile policy.
            self.send_header("Cache-Control", cache_control)
        elif cache and status == 200:
            self.send_header("Cache-Control", f"public, max-age={CACHE_SECONDS}")
        else:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        # BaseHTTPRequestHandler writes one line per request to stderr, which
        # in a container is one CloudWatch line per *tile*. The map draws
        # thirty per pan.
        logger.debug("tile server: " + fmt, *args)


_slots = threading.BoundedSemaphore(1)  # replaced by `start`, sized from the pool

_server: ThreadingHTTPServer | None = None
_server_lock = threading.Lock()
_server_port: int | None = None


def start(port: int | None = None) -> int | None:
    """Start the tile server once per process. Returns the port, or None.

    None means the port could not be bound — most often because something else
    already has it, which on a laptop is usually a previous `streamlit run`
    that has not exited. It is not fatal: `app.py` reads a None as "the tile
    renderer is unavailable" and says so rather than drawing an empty map,
    and ``HBU_MAP_RENDERER=geojson`` is the way back to a map in the meantime.
    """
    global _server, _server_port, _slots

    if os.environ.get("HBU_TILE_ENABLED", "1").strip().lower() in {"0", "false", "no"}:
        logger.info("Tile server disabled by HBU_TILE_ENABLED")
        return None

    with _server_lock:
        if _server is not None:
            return _server_port

        _slots = threading.BoundedSemaphore(_concurrency())
        bind = os.environ.get("HBU_TILE_BIND", "0.0.0.0")
        chosen = DEFAULT_PORT if port is None else port
        try:
            server = ThreadingHTTPServer((bind, chosen), _Handler)
        except OSError as exc:
            logger.warning("Tile server could not bind %s:%s — %s", bind, chosen, exc)
            return None

        server.daemon_threads = True
        thread = threading.Thread(
            target=server.serve_forever,
            name="hbu-tile-server",
            daemon=True,
        )
        thread.start()
        _server, _server_port = server, server.server_address[1]
        logger.info("Tile server listening on %s:%s%s", bind, _server_port, PATH_PREFIX)
        return _server_port


def stop() -> None:
    """Shut the server down. Used by the tests; nothing in the app calls it."""
    global _server, _server_port

    with _server_lock:
        if _server is not None:
            _server.shutdown()
            _server.server_close()
        _server, _server_port = None, None
    _cache.clear()


def port() -> int | None:
    """The port the server actually bound, or None if it is not running."""
    return _server_port


def cache_stats() -> tuple[int, int]:
    """``(hits, misses)`` since the process started, for the sidebar pane."""
    return _cache.hits, _cache.misses
