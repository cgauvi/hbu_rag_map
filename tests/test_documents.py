"""Zoning PDFs: the cache key, the dead-link check, and the text layer.

The cache key matters most. It is deliberately identical to the dataplatform's
``document_id`` so the two share a directory — a test that lets it drift would
silently cost ~600 re-downloads from the city's web server.
"""

from __future__ import annotations

import hashlib

import pytest

from src.utils import documents

URL = "http://www1.ville.montreal.qc.ca/CartesInteractives/villeray/doc/zone/C01-001.pdf"


def test_document_id_matches_the_dataplatform_scheme():
    """sha256(url)[:16] — the same 16 hex characters urban_rag writes."""
    assert documents.document_id(URL) == hashlib.sha256(URL.encode()).hexdigest()[:16]
    assert len(documents.document_id(URL)) == 16


def test_cache_path_is_the_id_plus_pdf(tmp_path):
    path = documents.cache_path(URL, tmp_path)
    assert path.name == f"{documents.document_id(URL)}.pdf"
    assert path.parent == tmp_path


def test_a_cached_file_is_read_without_a_request(tmp_path, monkeypatch):
    cached = documents.cache_path(URL, tmp_path)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"%PDF-1.4 cached")

    def _explode():
        raise AssertionError("should not have opened a session")

    monkeypatch.setattr(documents, "_session", _explode)

    document = documents.fetch(URL, cache_dir=tmp_path)

    assert document.from_cache
    assert document.content == b"%PDF-1.4 cached"


def test_a_fetched_file_is_written_to_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "REQUEST_DELAY_SECONDS", 0)
    monkeypatch.setattr(documents, "_session", lambda: _FakeSession(b"%PDF-1.4 fresh"))

    document = documents.fetch(URL, cache_dir=tmp_path)

    assert not document.from_cache
    assert documents.cache_path(URL, tmp_path).read_bytes() == b"%PDF-1.4 fresh"


def test_an_html_error_page_is_not_accepted_as_a_pdf(tmp_path, monkeypatch):
    """Dead municipal links answer 200 with a 'page not found' body."""
    monkeypatch.setattr(documents, "REQUEST_DELAY_SECONDS", 0)
    monkeypatch.setattr(
        documents, "_session",
        lambda: _FakeSession(b"<html>Page introuvable</html>", "text/html"),
    )

    with pytest.raises(documents.DocumentError, match="not a PDF"):
        documents.fetch(URL, cache_dir=tmp_path)

    assert not documents.cache_path(URL, tmp_path).exists()


def test_a_non_http_link_is_refused(tmp_path):
    with pytest.raises(documents.DocumentError, match="not a fetchable link"):
        documents.fetch("", cache_dir=tmp_path)
    with pytest.raises(documents.DocumentError, match="not a fetchable link"):
        documents.fetch("ftp://example.com/x.pdf", cache_dir=tmp_path)


def test_an_unwritable_cache_does_not_fail_the_fetch(tmp_path, monkeypatch):
    """A read-only container filesystem should cost the cache, not the document."""
    monkeypatch.setattr(documents, "REQUEST_DELAY_SECONDS", 0)
    monkeypatch.setattr(documents, "_session", lambda: _FakeSession(b"%PDF-1.4 x"))

    target = tmp_path / "readonly"
    target.mkdir()
    target.chmod(0o500)
    try:
        document = documents.fetch(URL, cache_dir=target)
        assert document.content == b"%PDF-1.4 x"
    finally:
        target.chmod(0o700)


def test_filename_falls_back_to_the_doc_id():
    with_name = documents.ZoningDocument(URL, "abc", b"", False)
    assert with_name.filename == "C01-001.pdf"

    without = documents.ZoningDocument("http://x/y", "abc123", b"", False)
    assert without.filename == "abc123.pdf"


def test_a_grid_served_from_a_handler_is_named_for_its_zone():
    """Quebec City's grids all share one path and differ only in the query.

    ``.../GrillesZonage/HandlerZonage.ashx?13001Hb`` — so naming the download
    after the path would send four of a lot's zones to disk as four copies of
    ``HandlerZonage``, and the `doc_id` fallback would make them four hashes.
    """
    served = documents.ZoningDocument(
        "https://carte.ville.quebec.qc.ca/GrillesZonage/HandlerZonage.ashx?13001Hb",
        "abc123", b"", False,
    )
    assert served.filename == "13001Hb.pdf"


def test_a_query_that_is_not_a_bare_token_is_not_a_filename():
    """``?a=1&b=2`` names nothing; the doc_id at least cannot collide."""
    document = documents.ZoningDocument(
        "https://example.test/handler.ashx?a=1&b=2", "abc123", b"", False
    )
    assert document.filename == "abc123.pdf"


def test_a_query_string_does_not_hide_a_pdf_name():
    """A cache-buster on a Montreal link must not cost the sheet its name."""
    document = documents.ZoningDocument(
        "http://x/doc/zone/C01-001.pdf?v=2", "abc123", b"", False
    )
    assert document.filename == "C01-001.pdf"


def test_an_unreadable_pdf_reports_rather_than_crashes():
    with pytest.raises(documents.DocumentError):
        documents.extract_text(b"not a pdf at all")


class _FakeResponse:
    def __init__(self, content: bytes, content_type: str):
        self.content = content
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        return None


class _FakeSession:
    def __init__(self, content: bytes, content_type: str = "application/pdf"):
        self._response = _FakeResponse(content, content_type)

    def get(self, _url, timeout=None):
        return self._response


# ---------------------------------------------------------------------------
# Publishing, so `tiles` can hand the same bytes back
# ---------------------------------------------------------------------------


def test_a_document_id_is_the_only_shape_that_can_address_a_file():
    assert documents.is_document_id("784a0b4f710d1785")
    assert not documents.is_document_id("784A0B4F710D1785")   # upper case
    assert not documents.is_document_id("784a0b4f710d178")    # too short
    assert not documents.is_document_id("../../etc/passwd")
    assert not documents.is_document_id("")


def test_publishing_makes_a_document_readable_by_id():
    documents.forget_published()
    doc_id = documents.document_id("http://example.test/a.pdf")
    assert documents.publish(doc_id, b"%PDF-1.4") == doc_id
    assert documents.published(doc_id) == b"%PDF-1.4"
    documents.forget_published()


def test_nothing_is_published_under_an_id_that_is_not_one():
    documents.forget_published()
    assert documents.publish("../../etc/passwd", b"%PDF-1.4") is None
    assert documents.published("../../etc/passwd") is None


def test_the_registry_is_bounded_and_evicts_the_least_recently_used(monkeypatch):
    documents.forget_published()
    monkeypatch.setattr(documents, "PUBLISHED_LIMIT", 2)
    first, second, third = (documents.document_id(f"http://x/{n}") for n in "abc")
    documents.publish(first, b"%PDF-1")
    documents.publish(second, b"%PDF-2")
    documents.publish(first, b"%PDF-1")   # touched, so `second` is now oldest
    documents.publish(third, b"%PDF-3")

    assert documents.published(second) is None
    assert documents.published(first) == b"%PDF-1"
    assert documents.published(third) == b"%PDF-3"
    documents.forget_published()


def test_the_disk_cache_answers_when_the_registry_does_not(tmp_path, monkeypatch):
    documents.forget_published()
    monkeypatch.setattr(documents, "DEFAULT_CACHE_DIR", tmp_path)
    doc_id = documents.document_id("http://example.test/b.pdf")
    (tmp_path / f"{doc_id}.pdf").write_bytes(b"%PDF-cached")

    assert documents.published(doc_id) == b"%PDF-cached"


def test_an_empty_cached_file_is_not_a_document(tmp_path, monkeypatch):
    documents.forget_published()
    monkeypatch.setattr(documents, "DEFAULT_CACHE_DIR", tmp_path)
    doc_id = documents.document_id("http://example.test/c.pdf")
    (tmp_path / f"{doc_id}.pdf").write_bytes(b"")

    assert documents.published(doc_id) is None


def test_the_ca_bundle_falls_back_to_ssl_cert_file(tmp_path, monkeypatch):
    """The variable a managed machine actually sets.

    Only Quebec City's grids are served over https, so until the second city
    this fallback had nothing to fail on.
    """
    bundle = tmp_path / "zscaler-plus-certifi.pem"
    bundle.write_text("-----BEGIN CERTIFICATE-----\n")
    for variable in documents.CA_BUNDLE_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))

    assert documents.ca_bundle() == str(bundle)


def test_a_bundle_that_is_not_on_disk_is_not_used(tmp_path, monkeypatch):
    """A host path read inside a container names no file there.

    OpenSSL handed a filename that does not exist verifies against nothing, so
    passing it through would turn a proxy problem into a silent one.
    """
    for variable in documents.CA_BUNDLE_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", "C:/Users/nobody/.certs/missing.pem")

    assert documents.ca_bundle() is None


def test_the_requests_variables_win_over_ssl_cert_file(tmp_path, monkeypatch):
    specific = tmp_path / "urban.pem"
    specific.write_text("-----BEGIN CERTIFICATE-----\n")
    general = tmp_path / "system.pem"
    general.write_text("-----BEGIN CERTIFICATE-----\n")
    for variable in documents.CA_BUNDLE_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", str(general))
    monkeypatch.setenv("URBAN_RAG_CA_BUNDLE", str(specific))

    assert documents.ca_bundle() == str(specific)
