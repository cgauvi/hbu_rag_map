"""
tiles.py — where the map's tile archives are, and the small HTTP server that
still runs on a second port.

The map's nine layers are **PMTiles archives**: one file per layer per
``(scrape_date, neighborhood)`` partition, rendered by the dataplatform's
``map_tiles`` asset with the same ``ST_AsMVT`` SQL this process used to run
per request, and written into its tree - ``s3://<bucket>/<env>/gold/map_tiles/
<date>/<borough>/<layer>.pmtiles`` when the pipeline writes to S3. A PMTiles
file carries its own directory, so the browser fetches any one tile with a
byte-range request and nothing on this side of the wire is in the path: no
tile server, no database, no connection pool shared with the panes.

**What this process does instead** is tell the browser where the archives are,
in a form it can reach. `HBU_TILES_URL` names the root, and its scheme decides
how each archive URL is minted:

``s3://bucket/prefix``
    The deployed shape, and the laptop shape against the pipeline's bucket.
    The bucket is private - a cadastre joined to a solved development
    programme is not something to publish by accident - so every archive URL
    is **presigned** by this process with its own credentials (the task role
    on Fargate, the AWS profile on a laptop), for `PRESIGN_SECONDS`. That
    makes the tiles exactly as reachable as the app is and no more: a
    presigned URL comes from a page, and a page is what the password gate
    hands out. The bucket also needs a CORS rule that lets the page's origin
    read ranges; ``hbu_infra`` sets it.

``https://host/prefix``
    Used as given, for a CDN or a public bucket someone else has decided on.

a directory
    The dataplatform's local tree, for a laptop running both repositories
    with no bucket at all. The archives are served from here, off the second
    port, with the byte-range support the PMTiles reader needs - and behind
    the same key the grid PDFs are.

**What the second port still carries.** The vector renderer's own JavaScript -
Leaflet.VectorGrid and the PMTiles reader - is committed under
``src/utils/vendor/`` and served from here rather than from a CDN, because
`streamlit_folium` awaits every plugin script before it draws anything and
catches no failure: a script it cannot fetch does not leave the map without
its layers, it leaves the page without the map. The zoning grids the
Regulations pane has fetched are published here too, on an origin the browser
can open from an ``https://`` page. And ``/tiles/healthz`` is what the load
balancer polls. So the port, the ALB rule and the second target group all
stay; what left them is the geometry.

**What guards it.** The Streamlit UI sits behind ``HBU_APP_PASSWORD``. A
locally served archive or a grid PDF on a public listener would walk around
that, so those routes carry a key derived from the same password - an HMAC of
it, never the password itself - and a request without the key is refused.
With no password set the gate is off, exactly as `auth.py` is off, which is
what keeps `make run` and the test suite working on a laptop. The key is
derived rather than random so every task in a service computes the same one.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
import urllib.parse
from dataclasses import dataclass
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logger = logging.getLogger(__name__)

#: The port the server listens on inside the container. `hbu_infra`'s
#: ecs.tf maps it and routes ``/tiles/*`` to it; `make run` publishes it too.
DEFAULT_PORT = int(os.environ.get("HBU_TILE_PORT", 8502))

#: Every route this server answers begins here, and so does the ALB rule that
#: sends traffic to it. One prefix in both places or neither works.
PATH_PREFIX = "/tiles"

#: What the ALB polls. Deliberately outside the key check: a health check
#: carries no credentials, and a target group that cannot reach its own health
#: path drains the service on the first deploy.
HEALTH_PATH = f"{PATH_PREFIX}/healthz"

#: The browser-side libraries the vector renderer *is*, served from here
#: rather than from a CDN - see the module docstring. Keyless, like the health
#: path: public code that carries no cadastre, and a browser allowed to keep
#: it across sessions is one that fetches it once.
VENDOR_PREFIX = f"{PATH_PREFIX}/vendor"

#: The version is in each *name*, so an upgrade is a new URL rather than a
#: new body at an old one - which is what lets the response be cached hard
#: without a deploy having to argue with a browser about it.
VECTORGRID_FILE = "leaflet-vectorgrid-1.3.0.js"
PMTILES_FILE = "pmtiles-4.5.0.js"
VENDOR_FILES: tuple[str, ...] = (VECTORGRID_FILE, PMTILES_FILE)

#: Committed under `src/utils/vendor`; that README carries the provenance and
#: the digests.
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"

#: Where a zoning grid is published. Keyed like an archive rather than public
#: like the vendored libraries: the document itself is a municipal
#: publication, but *which* grids this server holds is a trace of what has
#: been looked at.
GRID_PREFIX = f"{PATH_PREFIX}/grid"

#: Where a *local* archive is served from - the directory shape of
#: `HBU_TILES_URL` only. Keyed, because it is the cadastre.
ARCHIVE_PREFIX = f"{PATH_PREFIX}/pmtiles"

#: A year. See `VECTORGRID_FILE` - the only way to get a different body is to
#: ask for a different URL.
VENDOR_CACHE_SECONDS = 31_536_000

#: How long a browser may keep a locally served archive's bytes. An archive
#: is immutable until its partition is re-materialised, which is at most a
#: daily event; the ETag below is what tells a browser it happened.
ARCHIVE_CACHE_SECONDS = int(os.environ.get("HBU_TILE_CACHE_SECONDS", 3600))

# ---------------------------------------------------------------------------
# The archives
# ---------------------------------------------------------------------------

#: The root the archives live under. See the module docstring for the three
#: shapes. Unset, the tile renderer is unavailable and `app.py` says so.
TILES_URL_ENV = "HBU_TILES_URL"

#: The names the dataplatform writes - `urban_rag.map_tiles.MANIFEST_FILE` and
#: `ARCHIVE_SUFFIX` over there. Mirrored, because the two repositories share
#: no code and the path is the contract.
MANIFEST_FILE = "map_tiles.json"
ARCHIVE_SUFFIX = ".pmtiles"

#: How long a presigned URL is good for. An hour is a long session; the URL is
#: reused for half of that (see `_presign`) so a rerun does not hand the
#: browser a new address for bytes it has already cached.
PRESIGN_SECONDS = int(os.environ.get("HBU_TILES_PRESIGN_SECONDS", 3600))

#: How long an answer to "which archives exist for this partition" is trusted.
#: A minute is what being wrong costs: a borough whose tiles land mid-session
#: appears on the next rerun after that.
MANIFEST_TTL_SECONDS = int(os.environ.get("HBU_TILES_MANIFEST_TTL", 60))

#: The region the bucket is in, which a presigned URL is signed for. The
#: pipeline's bucket sits beside the database.
REGION = (
    os.environ.get("HBU_TILES_REGION")
    or os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or "us-east-1"
)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NEIGHBORHOOD_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


@dataclass(frozen=True)
class TileSource:
    """Where the archives are, parsed once from `HBU_TILES_URL`.

    ``kind`` is ``"s3"``, ``"http"`` or ``"local"``; ``root`` the value as
    given, without a trailing slash; ``bucket`` and ``prefix`` the two halves
    of an S3 root and empty otherwise.
    """

    kind: str
    root: str
    bucket: str = ""
    prefix: str = ""

    def describe(self) -> str:
        """One line for the sidebar."""
        return f"{self.kind} · {self.root}"


def source() -> TileSource | None:
    """The configured archive root, or None when there is none.

    Read from the environment on every call rather than at import, because
    ``load_dotenv`` runs after this module is imported and a test sets it per
    case.
    """
    raw = (os.environ.get(TILES_URL_ENV) or "").strip()
    if not raw:
        return None
    if raw.startswith("s3://"):
        bucket, _, prefix = raw[len("s3://") :].partition("/")
        if not bucket:
            logger.warning("%s=%r names no bucket", TILES_URL_ENV, raw)
            return None
        prefix = prefix.strip("/")
        return TileSource(
            "s3", f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}",
            bucket=bucket, prefix=prefix,
        )
    root = raw.rstrip("/")
    if root.startswith(("http://", "https://")):
        return TileSource("http", root)
    return TileSource("local", str(Path(root).expanduser()))


def configured() -> bool:
    """Whether the tile renderer has archives to point at."""
    return source() is not None


def describe() -> str:
    """The sidebar's one line about where the tiles come from."""
    configured_source = source()
    return configured_source.describe() if configured_source else "not configured"


def _partition_key(scrape_date: date | str) -> str:
    return scrape_date.isoformat() if isinstance(scrape_date, date) else str(scrape_date)


def archive_key(scrape_date: date | str, neighborhood: str, layer: str) -> str:
    """``<date>/<borough>/<layer>.pmtiles`` - the path under the root."""
    return f"{_partition_key(scrape_date)}/{neighborhood}/{layer}{ARCHIVE_SUFFIX}"


def manifest_key(scrape_date: date | str, neighborhood: str) -> str:
    return f"{_partition_key(scrape_date)}/{neighborhood}/{MANIFEST_FILE}"


# -- the manifest ------------------------------------------------------------

_manifests: dict[tuple[str, str], tuple[float, dict | None]] = {}
_manifest_lock = threading.Lock()


def manifest(scrape_date: date | str, neighborhood: str) -> dict | None:
    """What the dataplatform built for one partition, or None if nothing.

    The ``map_tiles.json`` the asset writes beside its archives: which layers
    have a file, at which zooms, with what on them. Read once per
    `MANIFEST_TTL_SECONDS` per partition, because the sidebar asks for every
    layer on every rerun and a round trip per layer per rerun is what a pan
    used to cost.

    None is "no tiles for this partition", never an error: a missing manifest
    is the normal state of a borough whose ``map_tiles`` asset has not run,
    and the map's job is to say so rather than to fail.
    """
    configured_source = source()
    if configured_source is None:
        return None
    key = (_partition_key(scrape_date), neighborhood)
    now = time.monotonic()
    with _manifest_lock:
        cached = _manifests.get(key)
        if cached is not None and cached[0] > now:
            return cached[1]
    try:
        text = _read_text(configured_source, manifest_key(*key))
        loaded = json.loads(text) if text else None
        if loaded is not None and not isinstance(loaded, dict):
            loaded = None
    except Exception as exc:  # noqa: BLE001 - a note in the sidebar, never a failure
        logger.warning("Tile manifest %s/%s unreadable: %s", key[0], key[1], exc)
        loaded = None
    with _manifest_lock:
        _manifests[key] = (now + MANIFEST_TTL_SECONDS, loaded)
    return loaded


def forget_manifests() -> None:
    """Drop every cached manifest. Used by the tests; nothing in the app calls it."""
    with _manifest_lock:
        _manifests.clear()
    with _presign_lock:
        _presigned.clear()


def _read_text(configured_source: TileSource, key: str) -> str | None:
    if configured_source.kind == "s3":
        client = _s3_client()
        try:
            response = client.get_object(
                Bucket=configured_source.bucket, Key=_s3_key(configured_source, key)
            )
        except client.exceptions.NoSuchKey:
            return None
        except Exception as exc:  # noqa: BLE001
            # botocore raises a ClientError for a 404 on a bucket without
            # ListBucket, and that is the ordinary "not built yet".
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        return response["Body"].read().decode("utf-8")
    if configured_source.kind == "http":
        import requests  # noqa: PLC0415

        response = requests.get(f"{configured_source.root}/{key}", timeout=10)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.text
    path = Path(configured_source.root) / key
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _s3_key(configured_source: TileSource, key: str) -> str:
    return f"{configured_source.prefix}/{key}" if configured_source.prefix else key


def _s3_client():
    from src.utils.db import _boto  # noqa: PLC0415

    return _boto("s3", REGION)


# -- the URLs ----------------------------------------------------------------

_presigned: dict[str, tuple[float, str]] = {}
_presign_lock = threading.Lock()


def _presign(configured_source: TileSource, key: str) -> str:
    """A presigned GET for one object, reused while it has half its life left.

    Reused on purpose: `streamlit_folium` keys the map on its script, so a
    URL that changed on every rerun would remount the map on every rerun and
    the browser would refetch bytes it already holds. Half the life is the
    slack that keeps a URL handed out near the end of the window from
    expiring under a long session.
    """
    now = time.monotonic()
    with _presign_lock:
        cached = _presigned.get(key)
        if cached is not None and cached[0] > now:
            return cached[1]
    url = _s3_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": configured_source.bucket, "Key": _s3_key(configured_source, key)},
        ExpiresIn=PRESIGN_SECONDS,
    )
    with _presign_lock:
        _presigned[key] = (now + PRESIGN_SECONDS / 2, url)
    return url


def archive_url(scrape_date: date | str, neighborhood: str, layer: str) -> str | None:
    """Where the browser fetches one layer's archive from, or None if unbuilt.

    None whenever the partition's manifest does not list the layer: the
    dataplatform did not write that file, and a URL to a missing object would
    only turn into a console full of 403s.
    """
    configured_source = source()
    if configured_source is None:
        return None
    listed = manifest(scrape_date, neighborhood)
    if not listed or layer not in (listed.get("layers") or {}):
        return None
    key = archive_key(scrape_date, neighborhood, layer)
    if configured_source.kind == "s3":
        return _presign(configured_source, key)
    if configured_source.kind == "http":
        return f"{configured_source.root}/{key}"
    suffix = ""
    if (key_value := tile_key()) is not None:
        suffix = f"?{urllib.parse.urlencode({'k': key_value})}"
    return f"{base_url()}{ARCHIVE_PREFIX}/{key}{suffix}"


def layer_archives(
    layer: str, scrape_date: date | str | None, neighborhoods: list[str]
) -> list[str]:
    """Every archive of ``layer`` the map should draw, one per borough listed.

    One URL when a borough is selected; one per loaded borough for "all
    loaded", which the browser reads side by side and merges tile by tile.
    Empty when nothing has been built, which the caller turns into a note.
    """
    if scrape_date is None:
        return []
    urls = []
    for neighborhood in neighborhoods:
        url = archive_url(scrape_date, neighborhood, layer)
        if url:
            urls.append(url)
    return urls


def built_layers(scrape_date: date | str | None, neighborhoods: list[str]) -> set[str]:
    """The layers that have an archive in at least one of ``neighborhoods``."""
    if scrape_date is None:
        return set()
    layers: set[str] = set()
    for neighborhood in neighborhoods:
        listed = manifest(scrape_date, neighborhood)
        if listed:
            layers.update((listed.get("layers") or {}).keys())
    return layers


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------

#: The string the HMAC is taken over. Fixed and public - what makes the key a
#: secret is the password it is keyed with, not this.
_KEY_MESSAGE = b"hbu-rag-map/tiles"


def tile_key() -> str | None:
    """The key every keyed URL must carry, or None when the gate is off.

    Derived from ``HBU_APP_PASSWORD`` so the locally served archives and the
    grids are exactly as reachable as the app is and no more. Truncated to 32
    hex characters because it travels in a query string and the full digest
    buys nothing against a password this is only as strong as anyway.

    Returns None both when no password is set and when it is still Terraform's
    placeholder - `auth.py` refuses every login in that state, so there is no
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
# Where the browser is told to look for this server
# ---------------------------------------------------------------------------


def base_url() -> str:
    """The origin to prefix every URL of this server with - possibly empty.

    ``HBU_TILE_BASE_URL`` **unset** - a laptop. Streamlit is on 8501 and this
    server is on its own port, two different origins, so the URL has to be
    absolute and the responses carry CORS headers.

    ``HBU_TILE_BASE_URL`` set to ``""`` or ``same-origin`` - behind the load
    balancer. ``/tiles/*`` is a rule on the same listener the app is served
    from, so a relative URL is both correct and one less thing to keep in step
    with the DNS name. `hbu_infra`'s ecs.tf sets it.

    Anything else is used as given.
    """
    configured_base = os.environ.get("HBU_TILE_BASE_URL")
    if configured_base is not None:
        value = configured_base.strip()
        if value in ("", "same-origin"):
            return ""
        return value.rstrip("/")
    host = os.environ.get("HBU_TILE_PUBLIC_HOST", "localhost")
    return f"http://{host}:{DEFAULT_PORT}"


def vectorgrid_url() -> str:
    """Where the page should fetch Leaflet.VectorGrid from."""
    return f"{base_url()}{VENDOR_PREFIX}/{VECTORGRID_FILE}"


def pmtiles_url() -> str:
    """Where the page should fetch the PMTiles reader from - the same origin."""
    return f"{base_url()}{VENDOR_PREFIX}/{PMTILES_FILE}"


def grid_url(doc_id: str) -> str | None:
    """Where the page should point at an already-fetched zoning grid.

    None for anything not shaped like a ``document_id``, so a caller cannot
    build a URL this server would refuse anyway. Behind the load balancer
    `base_url` is empty and this comes out root-relative, so the link
    inherits the page's scheme and host.
    """
    from src.utils.documents import is_document_id  # noqa: PLC0415

    if not is_document_id(doc_id):
        return None
    key = tile_key()
    suffix = f"?{urllib.parse.urlencode({'k': key})}" if key else ""
    return f"{base_url()}{GRID_PREFIX}/{doc_id}.pdf{suffix}"


# ---------------------------------------------------------------------------
# Parsing a request
# ---------------------------------------------------------------------------


def vendor_file(path: str) -> bytes | None:
    """The bytes of the vendored asset ``path`` addresses, or None.

    The name is matched against the files this server publishes rather than
    joined onto `VENDOR_DIR`, so no request can name a path at all - there is
    nothing here for ``..`` to traverse, because nothing from the URL reaches
    the filesystem.
    """
    for name in VENDOR_FILES:
        if path == f"{VENDOR_PREFIX}/{name}":
            try:
                return (VENDOR_DIR / name).read_bytes()
            except OSError as exc:
                # A checkout missing the file, which is a packaging fault rather
                # than a request fault. Logged loudly because the symptom
                # downstream is the blank pane this whole route exists to prevent.
                logger.error("Vendored %s is unreadable: %s", name, exc)
                return None
    return None


def parse_grid_path(path: str) -> str | None:
    """``/tiles/grid/784a0b4f710d1785.pdf`` -> ``"784a0b4f710d1785"``.

    None for anything else. The id is checked for shape before it goes
    anywhere near the filesystem - sixteen hex characters cannot name a
    directory to climb out of or an extension to change.
    """
    from src.utils.documents import is_document_id  # noqa: PLC0415

    if not path.startswith(f"{GRID_PREFIX}/"):
        return None
    name = path[len(GRID_PREFIX) + 1 :]
    if not name.endswith(".pdf"):
        return None
    doc_id = name[: -len(".pdf")]
    return doc_id if is_document_id(doc_id) else None


def parse_archive_path(path: str) -> tuple[str, str, str] | None:
    """``/tiles/pmtiles/2026-09-01/VSMPE/lots.pmtiles`` -> ``(date, borough, layer)``.

    None for anything else, which the handler turns into a 404. Each part is
    checked for shape - a date, a borough key, a layer this app draws - so
    nothing from the URL reaches the filesystem that could name a path.
    """
    from src.utils import queries  # noqa: PLC0415

    if not path.startswith(f"{ARCHIVE_PREFIX}/"):
        return None
    parts = path[len(ARCHIVE_PREFIX) + 1 :].split("/")
    if len(parts) != 3:
        return None
    scrape_date, neighborhood, filename = parts
    if not filename.endswith(ARCHIVE_SUFFIX):
        return None
    layer = filename[: -len(ARCHIVE_SUFFIX)]
    if not _DATE_RE.match(scrape_date) or not _NEIGHBORHOOD_RE.match(neighborhood):
        return None
    if layer not in queries.MVT_LAYER_NAMES:
        return None
    return scrape_date, neighborhood, layer


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """``bytes=a-b`` -> ``(a, b)`` inclusive and clamped to ``size``, or None.

    None both for no header and for one this does not understand, and the
    handler then sends the whole file - which is what a plain GET means.
    An unsatisfiable range (a start past the end) is the one case a client
    must be *told* about, so it comes back as ``(size, size - 1)`` and the
    handler answers 416.
    """
    if not header or not header.startswith("bytes="):
        return None
    spec = header[len("bytes=") :].split(",")[0].strip()
    start_text, _, end_text = spec.partition("-")
    try:
        if start_text == "" and end_text:
            length = int(end_text)
            if length <= 0:
                return None
            return max(0, size - length), size - 1
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    except ValueError:
        return None
    if start < 0 or end < start:
        return None
    if start >= size:
        return size, size - 1
    return start, min(end, size - 1)


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    # HTTP/1.1 so the reader's range requests reuse a connection instead of
    # opening one apiece. Every response below sets Content-Length, which is
    # what makes keep-alive safe.
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

        archive = parse_archive_path(parsed.path)
        if archive is not None:
            self._serve_archive(archive, urllib.parse.parse_qs(parsed.query))
            return

        self._respond(404, b"not found", "text/plain; charset=utf-8", cache=False)

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
        # The PMTiles reader never sends one, but a proxy in front may.
        self.do_GET()

    def _serve_archive(self, archive: tuple[str, str, str], query: dict[str, list[str]]) -> None:
        """One locally held archive, whole or by byte range.

        Only the directory shape of `HBU_TILES_URL` answers here: an S3 or an
        HTTP root is fetched by the browser directly and this route is a 404,
        so a URL that happens to be shaped right cannot make this process
        proxy a bucket it was never asked to.
        """
        if not _key_accepted((query.get("k") or [None])[0]):
            # 403 rather than 401: there is no challenge to answer and no
            # credential the browser could be asked to supply. The key comes
            # from the page, and a page is what the password gate hands out.
            self._respond(403, b"forbidden", "text/plain; charset=utf-8", cache=False)
            return

        configured_source = source()
        if configured_source is None or configured_source.kind != "local":
            self._respond(404, b"no local archives", "text/plain; charset=utf-8", cache=False)
            return

        path = Path(configured_source.root) / archive_key(*archive)
        try:
            size = path.stat().st_size
            mtime = int(path.stat().st_mtime)
        except OSError:
            self._respond(404, b"no such archive", "text/plain; charset=utf-8", cache=False)
            return

        etag = f'"{size:x}-{mtime:x}"'
        extra = (
            ("Accept-Ranges", "bytes"),
            ("ETag", etag),
            # The reader checks the ETag and the Content-Range on every range
            # response, and a browser only lets a cross-origin page read the
            # headers a server names.
            ("Access-Control-Expose-Headers", "ETag, Content-Range, Content-Length, Accept-Ranges"),
        )
        wanted = parse_range(self.headers.get("Range"), size)
        if wanted is None:
            with path.open("rb") as handle:
                body = handle.read()
            self._respond(
                200,
                body,
                "application/vnd.pmtiles",
                cache_control=f"private, max-age={ARCHIVE_CACHE_SECONDS}",
                extra_headers=extra,
                head_only=self.command == "HEAD",
            )
            return
        start, end = wanted
        if start >= size:
            self._respond(
                416,
                b"",
                "text/plain; charset=utf-8",
                cache=False,
                extra_headers=(("Content-Range", f"bytes */{size}"), *extra),
            )
            return
        with path.open("rb") as handle:
            handle.seek(start)
            body = handle.read(end - start + 1)
        self._respond(
            206,
            body,
            "application/vnd.pmtiles",
            cache_control=f"private, max-age={ARCHIVE_CACHE_SECONDS}",
            extra_headers=(("Content-Range", f"bytes {start}-{end}/{size}"), *extra),
            head_only=self.command == "HEAD",
        )

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
            # never written to disk - a read-only filesystem, most likely. The
            # pane's rasterised pages and its download button both still work,
            # so this is a missing viewer rather than a missing document.
            logger.info("Grid %s is not published in this process", doc_id)
            self._respond(404, b"no such grid", "text/plain; charset=utf-8", cache=False)
            return

        self._respond(
            200,
            body,
            "application/pdf",
            # Immutable for the same reason the vendored libraries are: the id
            # is a digest of the URL, so different bytes are a different path.
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
        # A `Range` header is not on the browser's safelist, so a cross-origin
        # range request - the laptop shape, Streamlit on 8501 and this on 8502
        # - is preflighted. Answering it is what lets the reader in.
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
        head_only: bool = False,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in extra_headers:
            self.send_header(name, value)
        # The laptop shape of `base_url` is cross-origin by construction -
        # Streamlit on 8501, this on 8502 - so the responses have to say they
        # may be read. Behind the load balancer the two share an origin and
        # this header is simply unused.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        if cache_control is not None:
            self.send_header("Cache-Control", cache_control)
        elif cache and status == 200:
            self.send_header("Cache-Control", f"public, max-age={ARCHIVE_CACHE_SECONDS}")
        else:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body and not head_only:
            self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        # BaseHTTPRequestHandler writes one line per request to stderr, which
        # in a container is one CloudWatch line per range request.
        logger.debug("tile server: " + fmt, *args)


_server: ThreadingHTTPServer | None = None
_server_lock = threading.Lock()
_server_port: int | None = None


def start(port: int | None = None) -> int | None:
    """Start the server once per process. Returns the port, or None.

    None means the port could not be bound - most often because something else
    already has it, which on a laptop is usually a previous `streamlit run`
    that has not exited. It is not fatal: `app.py` reads a None as "the
    vector renderer's library cannot be served" and draws the GeoJSON map
    instead, saying so, and ``HBU_MAP_RENDERER=geojson`` is the way back to a
    map in the meantime.
    """
    global _server, _server_port

    if os.environ.get("HBU_TILE_ENABLED", "1").strip().lower() in {"0", "false", "no"}:
        logger.info("Tile server disabled by HBU_TILE_ENABLED")
        return None

    with _server_lock:
        if _server is not None:
            return _server_port

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


def port() -> int | None:
    """The port the server actually bound, or None if it is not running."""
    return _server_port
