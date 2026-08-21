"""
documents.py — The zoning grid PDF behind a lot.

``Reglement_urbanisme__VSP_REG_ZONE`` carries a link, not a description:

    LIEN_GRILLE = http://www1.ville.montreal.qc.ca/CartesInteractives/villeray/doc/zone/C01-001.pdf

That "grille des spécifications" is the document a highest-and-best-use
question is actually answered from — the usages, the storeys, the heights, the
implantation rates for one zone on one page. Clicking a lot on the map should
put it on screen, so this module fetches it, caches it, and rasterises it.

Three decisions worth knowing:

**The cache is keyed exactly as the dataplatform's is** — ``sha256(url)[:16]``,
which is its ``document_id`` — so pointing ``HBU_PDF_CACHE_DIR`` at
``hbu_dataplatform/data/cache/pdf`` reuses the PDFs the pipeline already
downloaded instead of pulling them from the city's web server again. A
published zoning grid does not change once issued, which is what makes a cache
with no expiry correct here rather than merely convenient.

**Pages are rasterised rather than embedded.** The links are ``http://``, and a
browser on an ``https://`` page refuses to frame them; Chrome also blocks
``data:`` URIs in an iframe for PDFs. Rendering to PNG server-side with
pypdfium2 sidesteps both, works when the cache is warm and the network is not,
and is the same bytes the download button hands over.

**A dead link fails its own document, not the pane.** These are municipal URLs
scraped months apart; some answer 200 with an HTML "page not found" body, which
is why the content is checked for a PDF header rather than trusted.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Same default as the dataplatform's, so the two agree when they share a tree.
DEFAULT_CACHE_DIR = Path(os.environ.get("HBU_PDF_CACHE_DIR", "data/cache/pdf"))

REQUEST_TIMEOUT_SECONDS = float(os.environ.get("HBU_PDF_TIMEOUT", 30))
REQUEST_DELAY_SECONDS = 0.25

#: How many pages the viewer rasterises before it stops. A grille is one or two
#: pages; the cap is there for the document that is not one.
MAX_RENDER_PAGES = int(os.environ.get("HBU_PDF_MAX_PAGES", 8))

#: Render scale. 2.0 is legible on a HiDPI screen without making a page a
#: megabyte of PNG.
RENDER_SCALE = float(os.environ.get("HBU_PDF_RENDER_SCALE", 2.0))


class DocumentError(RuntimeError):
    """A linked document could not be fetched or read."""


@dataclass(frozen=True)
class ZoningDocument:
    url: str
    doc_id: str
    content: bytes
    from_cache: bool

    @property
    def filename(self) -> str:
        tail = self.url.rstrip("/").rsplit("/", 1)[-1]
        return tail if tail.lower().endswith(".pdf") else f"{self.doc_id}.pdf"

    @property
    def num_bytes(self) -> int:
        return len(self.content)


def document_id(url: str) -> str:
    """Stable id for a link.

    Identical to ``rag.documents.document_id`` in the dataplatform — the same
    16 hex characters — which is what lets the two share a cache directory and
    what makes a ``doc_id`` from ``rag.chunks`` resolve to a file here.
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def cache_path(url: str, cache_dir: Path | str | None = None) -> Path:
    return Path(cache_dir or DEFAULT_CACHE_DIR) / f"{document_id(url)}.pdf"


def _session():
    import requests  # noqa: PLC0415
    from requests.adapters import HTTPAdapter  # noqa: PLC0415
    from urllib3.util.retry import Retry  # noqa: PLC0415

    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    bundle = os.environ.get("URBAN_RAG_CA_BUNDLE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if bundle:
        session.verify = bundle
    return session


def fetch(url: str, *, cache_dir: Path | str | None = None) -> ZoningDocument:
    """The PDF at *url*, from disk when it is already there."""
    if not url or not url.startswith(("http://", "https://")):
        raise DocumentError(f"not a fetchable link: {url!r}")

    cached = cache_path(url, cache_dir)
    if cached.exists() and cached.stat().st_size:
        return ZoningDocument(
            url=url, doc_id=document_id(url), content=cached.read_bytes(), from_cache=True
        )

    import requests  # noqa: PLC0415

    if REQUEST_DELAY_SECONDS:
        time.sleep(REQUEST_DELAY_SECONDS)
    try:
        response = _session().get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DocumentError(f"{url}: {exc}") from exc

    content = response.content
    content_type = response.headers.get("Content-Type", "")
    if not content.startswith(b"%PDF") and "pdf" not in content_type:
        # Dead links answer 200 with an HTML "page not found" body.
        raise DocumentError(
            f"{url}: not a PDF (Content-Type {content_type!r}, {len(content)} bytes)"
        )

    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(content)
    except OSError as exc:
        # A read-only container filesystem is a reason to skip the cache, not
        # to fail a document that was fetched successfully.
        logger.warning("Could not cache %s at %s: %s", url, cached, exc)

    return ZoningDocument(
        url=url, doc_id=document_id(url), content=content, from_cache=False
    )


def render_pages(
    content: bytes, *, max_pages: int = MAX_RENDER_PAGES, scale: float = RENDER_SCALE
) -> list[bytes]:
    """Rasterise the first pages of a PDF to PNG bytes.

    pypdfium2 rather than a system poppler: it ships as a self-contained wheel,
    so the Docker image needs no apt layer for it.
    """
    try:
        import pypdfium2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment problem
        raise DocumentError(
            "pypdfium2 is not installed, so the grid cannot be shown inline — "
            "`pip install pypdfium2`. The download button still works."
        ) from exc

    try:
        document = pypdfium2.PdfDocument(content)
    except Exception as exc:
        raise DocumentError(f"unreadable PDF ({exc})") from exc

    images: list[bytes] = []
    try:
        for index in range(min(len(document), max_pages)):
            bitmap = document[index].render(scale=scale)
            buffer = io.BytesIO()
            bitmap.to_pil().save(buffer, format="PNG")
            images.append(buffer.getvalue())
    finally:
        document.close()
    return images


def extract_text(content: bytes) -> str:
    """The PDF's text layer, for when the agent needs to read it rather than show it.

    No OCR, matching the dataplatform: a scan fails here the same way it fails
    the corpus, and for the same reason.
    """
    try:
        from pypdf import PdfReader  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment problem
        raise DocumentError("pypdf is not installed — `pip install pypdf`") from exc

    try:
        reader = PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise DocumentError(f"unreadable PDF ({exc})") from exc

    text = "\n\n".join(page for page in pages if page.strip())
    if not text.strip():
        raise DocumentError(
            f"no text layer over {len(pages)} page(s); a scanned document "
            "would need OCR"
        )
    return text
