# ── Build stage ───────────────────────────────────────────────────────────────
# Nothing in this image needs torch: the chat model and the query encoder are
# both served by the HuggingFace Inference API, which is why the runtime layer
# is a few hundred megabytes rather than several gigabytes.
FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PROGRESS_BAR=off

WORKDIR /build
COPY requirements-docker.txt .
RUN python -m pip install --no-cache-dir --prefix=/install -r requirements-docker.txt

# ── Test stage ────────────────────────────────────────────────────────────────
FROM builder AS test

RUN cp -r /install/. /usr/local/

COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt

WORKDIR /app
COPY app.py .
COPY src/ src/
COPY tests/ tests/
COPY pyproject.toml .

# The integration marker is deselected by pyproject's addopts, so this runs the
# unit suite and needs no database.
CMD ["python", "-m", "pytest", "tests/", "-v"]

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

ARG BUILD_VERSION=local-dev
ENV BUILD_VERSION=${BUILD_VERSION}

RUN adduser --disabled-password --gecos "" appuser

# Amazon's RDS root bundle, so a deployed task can connect with
# sslmode=verify-full instead of dropping to `require`. `require` encrypts but
# authenticates nothing — it accepts any certificate presented — which is most
# of what TLS was for. The global bundle covers every region and rotates
# rarely; hbu_infra/ecs.tf points URBAN_RAG_PG_SSLROOTCERT at this exact path.
ADD --chmod=644 https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem \
    /etc/ssl/certs/rds-global-bundle.pem

ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    PYTHONUNBUFFERED=1 \
    # Written to on first use; kept under the app dir so a read-only mount is
    # the operator's choice rather than a crash. documents.py degrades to
    # fetching every time when this is not writable.
    HBU_PDF_CACHE_DIR=/app/data/cache/pdf

WORKDIR /app

COPY --from=builder /install /usr/local

COPY app.py .
COPY src/ src/
COPY .streamlit/ .streamlit/

RUN mkdir -p /app/data/cache/pdf && chown -R appuser:appuser /app/data

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

EXPOSE 8501

USER appuser

CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
