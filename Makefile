# hbu_rag_map — an interactive zoning map over the hbu Postgres.
#
#   make install          create the venv and install the app
#   make db-up            a local postgis+pgvector container, schema applied
#   make check            why is nothing showing up
#   make run              streamlit, at http://localhost:8501
#
# Every target here runs from WSL and from Git Bash alike, but they do not
# share a virtualenv: WSL builds .venv-linux, Windows builds .venv (see VENV
# below). `make install` once per shell you intend to use is the whole of it —
# a missing one announces itself as `.venv*/bin/python: No such file or
# directory`. The database, the tunnel and the docker targets are shared.
#
# Which database, in one switch - DB_TARGET, under the same name and with the
# same two values in all three urban repos (see "which database" below):
#
#   make run                     the local container (DB_TARGET=local, default)
#   make run DB_TARGET=rds       hbu-dev, through an open SSM tunnel
#   make db-target               which of the two the next command will use
#
# It overrides .env and the shell alike, so `eval "$(make -s db-app-env
# ENV=dev)"` no longer redirects `make run` on its own: name DB_TARGET=rds and
# it reads that endpoint out of the environment, or out of .env when the shell
# carries none.
#
# The dev RDS has no public endpoint, so from outside the VPC it is reached
# through an SSM tunnel. Leave `make db-tunnel ENV=dev LOCAL_PORT=5433` running
# in hbu_infra, then here:
#   make run-tunnel         natively       (= make run DB_TARGET=rds)
#   make docker-run-tunnel  in the image   (= make docker-run DB_TARGET=rds)

PYTHON      ?= python3

# One venv per kernel, because one directory cannot serve both. A venv bakes in
# its layout (bin/python vs Scripts/python.exe) and the absolute path of the
# interpreter that built it, so whichever shell ran `make install` last would
# otherwise own it — and the other one fails with a bare
#
#     make: .venv/bin/python: No such file or directory
#
# which reads as a missing install rather than as the wrong flavour of venv.
# The names are the reverse of hbu_infra's (.venv there is the Linux one)
# because this repo's .venv was built on Windows first and still works; the
# WSL/Linux side gets its own rather than clobbering it.
#
# The entry points differ with the layout, which is what BIN is for.
ifneq (,$(findstring NT,$(shell uname -s)))
VENV        ?= .venv
BIN          = $(VENV)/Scripts
else
VENV        ?= .venv-linux
BIN          = $(VENV)/bin
endif

# The uv installer drops its binary in ~/.local/bin and does not touch this
# shell's PATH — under WSL that is exactly where it lands, so without this
# `make install` reports uv as missing and falls back to a python3 -m venv.
# Prepended rather than appended so a Linux uv outranks a Windows uv.exe that
# WSL interop also puts within reach: a Windows uv builds a *Windows* venv no
# matter which kernel launched it, which is the failure this whole block is
# about.
export PATH := $(HOME)/.local/bin:$(PATH)
UV          := $(shell command -v uv 2>/dev/null)

# Sibling repos. Same convention as hbu_infra's DATAPLATFORM variable: the
# schema lives over there, and this repo runs it from here when it is present.
HBU_INFRA   ?= ../hbu_infra
ENV         ?= dev

# -- which database --------------------------------------------------------
#
# One switch, carried by all three urban repos under the same name and with
# the same two values:
#
#   DB_TARGET=local  (default) the postgis+pgvector container `make db-up`
#                    below runs, published on 127.0.0.1:$(LOCAL_PG_PORT). No
#                    AWS, no tunnel; this is where `make db-restore-local` in
#                    hbu_infra puts the dump.
#   DB_TARGET=rds    hbu-$(ENV), through an open port-forward:
#                    `cd ../hbu_infra && make db-tunnel ENV=dev LOCAL_PORT=5433`
#
# Per command - `make run DB_TARGET=rds` - or exported once for a shell.
#
# It overrides .env rather than leaning on it, and that is the whole design:
# python-dotenv does not override a variable already in the environment, so a
# value set in a recipe wins over the same name in .env. Each branch therefore
# blanks what the other one sets. .env here names both at once - DATABASE_URL
# for the container and URBAN_RAG_PG_* for the endpoint - and the failure that
# matters is not an error: it is half of one branch left standing beside the
# other, connecting to a database nobody asked for and saying nothing about it.
DB_TARGET ?= local

LOCAL_PG_PORT ?= $(or $(HBU_LOCAL_PG_PORT),5432)
# 127.0.0.1 rather than localhost, for the same reason TUNNEL_HOST below is:
# a machine that answers localhost with ::1 first makes libpq spend the whole
# connect_timeout on a dead address before falling back.
LOCAL_PG_URL  ?= postgresql://urban_rag:urban_rag@127.0.0.1:$(LOCAL_PG_PORT)/urban_rag?sslmode=disable
# The same container as seen from inside another container, where 127.0.0.1 is
# the container itself. The docker targets below already map
# host.docker.internal onto the host gateway.
DOCKER_LOCAL_PG_URL ?= postgresql://urban_rag:urban_rag@host.docker.internal:$(LOCAL_PG_PORT)/urban_rag?sslmode=disable
TUNNEL_PORT   ?= 5433
# From the shell first - `eval "$(make -s db-app-env ENV=dev)"` in hbu_infra
# exports it - and from .env otherwise.
TUNNEL_DB_HOST ?= $(if $(URBAN_RAG_PG_HOST),$(URBAN_RAG_PG_HOST),$(shell sed -n 's/^URBAN_RAG_PG_HOST=//p' .env 2>/dev/null | tail -n 1 | tr -d '\r'))

# The address a *native* run dials the tunnel on, and it is deliberately not
# `localhost`. The Session Manager plugin binds IPv4 only; Windows resolves
# localhost to ::1 first, and libpq spends the entire connect_timeout on that
# dead address before falling back to IPv4 and succeeding — so every connect
# takes exactly connect_timeout seconds and then works, which reads as a slow
# tunnel rather than as a misresolution. Worse here than elsewhere, because the
# pool's own timeout is 15s: through `localhost` it raises PoolTimeout while
# libpq is still waiting.
TUNNEL_HOST   ?= 127.0.0.1

ifeq (local,$(DB_TARGET))
DB_TARGET_DESC = the local container, 127.0.0.1:$(LOCAL_PG_PORT)
DB_TARGET_FIX  = make db-up
PROBE_HOST     = 127.0.0.1
PROBE_PORT     = $(LOCAL_PG_PORT)
PG_ENV = DATABASE_URL="$(LOCAL_PG_URL)" URBAN_RAG_PG_DSN= \
	URBAN_RAG_PG_HOST= URBAN_RAG_PG_HOSTADDR= URBAN_RAG_PG_PORT= \
	URBAN_RAG_PG_SECRET_ID= URBAN_RAG_PG_IAM_AUTH= \
	URBAN_RAG_PG_SSLMODE=disable URBAN_RAG_PG_SSLROOTCERT=
DOCKER_PG_ENV = -e DATABASE_URL="$(DOCKER_LOCAL_PG_URL)" -e URBAN_RAG_PG_DSN= \
	  -e URBAN_RAG_PG_HOST= -e URBAN_RAG_PG_HOSTADDR= -e URBAN_RAG_PG_PORT= \
	  -e URBAN_RAG_PG_SECRET_ID= -e URBAN_RAG_PG_IAM_AUTH= \
	  -e URBAN_RAG_PG_SSLMODE=disable -e URBAN_RAG_PG_SSLROOTCERT=
DOCKER_PG_HOSTS =
else ifeq (rds,$(DB_TARGET))
DB_TARGET_DESC = $(if $(TUNNEL_DB_HOST),$(TUNNEL_DB_HOST),<no URBAN_RAG_PG_HOST in .env>) through $(TUNNEL_HOST):$(TUNNEL_PORT)
DB_TARGET_FIX  = cd $(HBU_INFRA) && make db-tunnel ENV=$(ENV) LOCAL_PORT=$(TUNNEL_PORT)  (leave it running, then re-run this in another terminal)
PROBE_HOST     = $(TUNNEL_HOST)
PROBE_PORT     = $(TUNNEL_PORT)
# A native run cannot rewrite its own resolver, so it splits the two halves
# instead: URBAN_RAG_PG_HOST goes on naming the endpoint, which is what
# verify-full matches the certificate against, while URBAN_RAG_PG_HOSTADDR
# carries the loopback address the socket actually goes to. Until db.py
# understood hostaddr this had to drop to `require` - encrypted but
# authenticating nothing, leaning on the SSM session as the only authenticated
# hop. DATABASE_URL and URBAN_RAG_PG_DSN are cleared because they outrank
# URBAN_RAG_PG_* in db.py's resolution order, and either one in .env would
# silently win over everything set here.
PG_ENV = DATABASE_URL= URBAN_RAG_PG_DSN= \
	URBAN_RAG_PG_HOST="$(TUNNEL_DB_HOST)" URBAN_RAG_PG_HOSTADDR="$(TUNNEL_HOST)" \
	URBAN_RAG_PG_PORT="$(TUNNEL_PORT)" URBAN_RAG_PG_SSLMODE=verify-full
# The container keeps verify-full by mapping the RDS hostname onto the host
# gateway (DOCKER_PG_HOSTS), so the name the certificate carries still
# resolves and no HOSTADDR is needed. SSLROOTCERT names the bundle inside the
# image rather than a host path.
DOCKER_PG_ENV = -e DATABASE_URL= -e URBAN_RAG_PG_DSN= \
	  -e URBAN_RAG_PG_HOST="$(TUNNEL_DB_HOST)" \
	  -e URBAN_RAG_PG_PORT="$(TUNNEL_PORT)" \
	  -e URBAN_RAG_PG_SSLMODE=verify-full \
	  -e URBAN_RAG_PG_SSLROOTCERT=/etc/ssl/certs/rds-global-bundle.pem \
	  -e URBAN_RAG_PG_IAM_AUTH=
DOCKER_PG_HOSTS = --add-host="$(TUNNEL_DB_HOST):host-gateway"
else
$(error DB_TARGET must be `local` or `rds`, not `$(DB_TARGET)`)
endif

IMAGE       ?= hbu-rag-map
IMAGE_TAG   ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo local-dev)

# The map's second port, in the same process: the vector renderer's own
# JavaScript and the zoning grid PDFs, because Streamlit serves no routes of
# its own and the page has to fetch both. The tiles themselves are PMTiles
# archives the browser reads off S3 - HBU_TILES_URL in .env says where, see
# .env.example - or, when that names a directory, off this port too. The
# browser has to be able to reach it, so a container run has to publish it
# alongside 8501 — unpublished, the map draws a basemap and nothing else: the
# page loads, the renderer's library never arrives, and the only sign of it is
# in the browser console. Deployed, hbu_infra's ALB routes /tiles/* here
# instead, and the URLs come out relative rather than naming a port at all.
TILE_PORT   ?= 8502

# The plugin form when it resolves, the standalone binary when it does not.
# msys2 make hands recipes an environment without ProgramFiles/ProgramData,
# and the docker CLI finds its plugins — `docker compose` among them — through
# exactly those, so under make every compose call dies with `unknown shorthand
# flag: 'd'` / `unknown command: docker compose` while the same line works in
# the terminal. Docker Desktop also ships the standalone `docker-compose` on
# PATH, which needs no plugin discovery at all.
COMPOSE ?= $(if $(shell docker compose version 2>/dev/null),docker compose,docker-compose)
AWS_PROFILE ?= charles_gauvin_east_1
AWS_REGION  ?= us-east-1

# Every host path below is mounted into a container, so it has to be one the
# Docker daemon can open — and on Windows neither $(HOME) nor $(USERPROFILE)
# reliably names it. make here is an msys2 binary from a different installation
# than the shell that invokes it, so $(HOME) can be /home/<user>, and
# USERPROFILE may not survive into make at all. Either way `docker run -v`
# accepts the path and mounts an empty directory, which is the worst shape of
# this failure: the container starts, finds no credentials, and the UI says
# "Not connected to the database" with nothing about a missing mount.
# `cygpath -m -D` is answered by the msys runtime itself and returns the
# C:/Users/<user> form the daemon wants. Empty off Windows, where $(HOME) is
# correct.
WIN_HOME := $(if $(findstring NT,$(shell uname -s)),$(patsubst %/Desktop,%,$(shell cygpath -m -D 2>/dev/null)))
AWS_DIR     ?= $(firstword $(wildcard $(WIN_HOME)/.aws $(USERPROFILE)/.aws $(HOME)/.aws) $(HOME)/.aws)

# The same msys2-make quirk one more layer down: USERPROFILE may not survive
# into make at all, and any Windows binary that resolves `~` through it — the
# aws CLI, docker credential helpers — dies or looks in the wrong place.
# Same fix hbu_infra's Makefile carries. (It is NOT what breaks `docker
# compose` under make — see COMPOSE below for that one.)
ifneq (,$(WIN_HOME))
export USERPROFILE := $(WIN_HOME)
endif
DOCKER_AWS_CA_BUNDLE_PATH ?= /etc/ssl/certs/aws-ca-bundle.pem

# The same mismatch again, one layer in: msys2 make hands a recipe
# HOME=/home/<user> and drops USERPROFILE entirely, so the *Windows* Python in
# .venv cannot resolve `~`. Path.home() raises "Could not determine home
# directory" and Streamlit dies on it before reading a line of config — and
# db.py's DEFAULT_CA_BUNDLE is built from the same call. WIN_HOME is already
# the C:/Users/<user> form; hand it over under the name Windows expanduser()
# checks first. Empty off Windows, where HOME is already right.
NATIVE_HOME_ENV = $(if $(WIN_HOME),USERPROFILE="$(WIN_HOME)" HOME="$(WIN_HOME)")

# The docker targets pass the profile explicitly and the native ones did not,
# which left `make run` reading whatever default credentials the shell happens
# to hold. Those are an account that cannot see the app-role secret, so the run
# dies on a cross-account AccessDenied naming secretsmanager — a much less
# obvious message than "wrong profile". `?=` above means an AWS_PROFILE already
# exported by the caller still wins.
NATIVE_AWS_ENV = AWS_PROFILE="$(AWS_PROFILE)" AWS_REGION="$(AWS_REGION)" \
	AWS_DEFAULT_REGION="$(AWS_REGION)"

# Behind a TLS-inspecting proxy the image has no way to verify AWS: botocore
# ships its own CA list and reads neither the host trust store nor anything the
# container was not given. The very first thing the app does is read the
# app-role password out of Secrets Manager, so without this the run dies on
# CERTIFICATE_VERIFY_FAILED and the UI reports it as "Not connected to the
# database" — the proxy nowhere in sight.
#
# Requiring the caller to export AWS_CA_BUNDLE made that the normal outcome,
# since `make docker-run-tunnel` in a fresh shell has none. .env already names
# the combined corporate+certifi bundle as SSL_CERT_FILE, for the HuggingFace
# client that only honours that variable, so fall back to it — and then to the
# conventional location, for a .env that predates that line.
#
# Off Windows that .env line is a trap rather than a fallback: it holds a
# C:/Users/... path, and from inside WSL that file does not exist. OpenSSL
# handed a filename it cannot open does not quietly fall back to the system
# roots — it verifies against nothing — so the Linux branch asks the distro's
# own store instead, which already carries the corporate root (which is why
# `aws` and `terraform` work from WSL with no bundle set at all).
ifeq (,$(strip $(AWS_CA_BUNDLE)))
ifneq (,$(findstring NT,$(shell uname -s)))
AWS_CA_BUNDLE := $(firstword \
  $(shell sed -n 's/^SSL_CERT_FILE=//p' .env 2>/dev/null | tail -n 1 | tr -d '\r') \
  $(wildcard $(WIN_HOME)/.certs/zscaler-plus-certifi.pem $(HOME)/.certs/zscaler-plus-certifi.pem))
else
AWS_CA_BUNDLE := $(firstword $(wildcard \
  /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt \
  $(HOME)/.certs/zscaler-plus-certifi.pem))
endif
endif

# SSL_CERT_FILE is overridden either way, never merely passed through: the value
# in .env is a *host* path, and OpenSSL handed a filename that does not exist
# does not fall back to the system roots — it verifies against nothing.
ifneq (,$(strip $(AWS_CA_BUNDLE)))
DOCKER_AWS_CA_ARGS = -v "$(AWS_CA_BUNDLE):$(DOCKER_AWS_CA_BUNDLE_PATH):ro" \
	-e AWS_CA_BUNDLE="$(DOCKER_AWS_CA_BUNDLE_PATH)" \
	-e SSL_CERT_FILE="$(DOCKER_AWS_CA_BUNDLE_PATH)"
else
DOCKER_AWS_CA_ARGS = -e SSL_CERT_FILE=
endif

# The same override for a *native* run, and it matters most under WSL. .env
# sets SSL_CERT_FILE to a C:/Users/... path for the HuggingFace client, which
# is right for a Windows interpreter and a file that does not exist for a Linux
# one — and python-dotenv does not override a variable already in the
# environment, so naming it here is what keeps .env from winning. Empty on
# Windows, where .env is already correct, and empty anywhere AWS_CA_BUNDLE
# could not be resolved rather than pointing OpenSSL at nothing.
NATIVE_TLS_ENV = $(if $(findstring NT,$(shell uname -s)),,$(if $(AWS_CA_BUNDLE),\
	SSL_CERT_FILE="$(AWS_CA_BUNDLE)" REQUESTS_CA_BUNDLE="$(AWS_CA_BUNDLE)" \
	AWS_CA_BUNDLE="$(AWS_CA_BUNDLE)"))

.PHONY: help install run run-tunnel check test lint fmt \
        db-target db-target-check db-reachable \
        db-up db-down db-init db-shell db-url db-logs db-image-src \
        docker-build docker-run docker-run-tunnel docker-test clean

help: ## This list
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Local development
# ---------------------------------------------------------------------------

install: ## Create the venv and install the app (uv when available, pip otherwise)
ifdef UV
	uv venv $(VENV) --python 3.12
	uv pip install --python $(BIN)/python -e ".[dev]"
else
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"
endif
	@test -f .env || (cp .env.example .env && echo "Created .env — fill in HUGGINGFACE_API_TOKEN")

# `serve`, not `streamlit run app.py`: it brings the map's tile server up
# before Streamlit's, rather than on the first page load. See serve.py — the
# difference only matters behind a load balancer, but running the two the same
# way locally is what keeps that path exercised.
run: db-target-check db-reachable ## Start the app at http://localhost:8501 (renderer assets on $(TILE_PORT))
	$(NATIVE_HOME_ENV) $(NATIVE_AWS_ENV) $(NATIVE_TLS_ENV) $(PG_ENV) HBU_TILE_PORT=$(TILE_PORT) \
	$(BIN)/python -m serve

# Kept as the name the README and the runbooks use. It is now one spelling of
# the switch rather than a second configuration: everything it used to set
# inline is PG_ENV's rds branch, and the two guards below it are
# db-target-check and db-reachable, which every run goes through now.
run-tunnel: ## Start Streamlit against an open hbu_infra db-tunnel (= DB_TARGET=rds)
	@$(MAKE) run DB_TARGET=rds

# What the switch resolves to, and the two guards it resolves for.
db-target: ## Print which database DB_TARGET resolves to
	@echo "DB_TARGET=$(DB_TARGET) -> $(DB_TARGET_DESC)"

# In rds mode the endpoint *name* is required, not the address the tunnel
# listens on: sslmode=verify-full matches the certificate against it, and RDS
# issues that certificate to the endpoint. Empty (no URBAN_RAG_PG_HOST in .env)
# or a loopback address are the two ways to get this wrong, and both would
# otherwise surface as a certificate error naming neither this switch nor .env.
# Nothing to check in local mode: the URL is self-contained.
db-target-check:
ifeq (rds,$(DB_TARGET))
	@test -n "$(TUNNEL_DB_HOST)" || { \
	  echo "DB_TARGET=rds needs the RDS endpoint; URBAN_RAG_PG_HOST is unset in .env."; \
	  echo "  cd $(HBU_INFRA) && make -s db-app-env ENV=$(ENV)   # prints it"; \
	  exit 1; \
	}
	@case "$(TUNNEL_DB_HOST)" in \
	  localhost|127.*|::1) \
	    echo "TUNNEL_DB_HOST must be the RDS endpoint, not $(TUNNEL_DB_HOST), when sslmode=verify-full"; \
	    exit 1 ;; \
	esac
endif

# Nothing listening where the switch points is the one failure worth catching
# before Streamlit starts: the pool's timeout is 15s and the UI reports it as
# "Not connected to the database", which says nothing about a container that
# was never started or a tunnel that died.
db-reachable: ## Fail early when nothing is listening where DB_TARGET points
	@$(BIN)/python -c "import socket; socket.create_connection(('$(PROBE_HOST)', $(PROBE_PORT)), 2)" 2>/dev/null \
	  || { \
	    echo "Nothing is listening on $(PROBE_HOST):$(PROBE_PORT) - DB_TARGET=$(DB_TARGET) wants $(DB_TARGET_DESC)."; \
	    echo "  $(DB_TARGET_FIX)"; \
	    exit 1; \
	  }

check: db-target-check ## Report what is and is not loaded, and how to fix each gap
	$(NATIVE_HOME_ENV) $(NATIVE_AWS_ENV) $(NATIVE_TLS_ENV) $(PG_ENV) \
	$(BIN)/python scripts/doctor.py

test: ## Unit tests (the integration marker is deselected by default)
	$(BIN)/python -m pytest tests/ -v

test-integration: db-target-check ## Also run the tests that need a reachable database
	$(PG_ENV) $(BIN)/python -m pytest tests/ -v -m integration

lint: ## ruff
	$(BIN)/ruff check src/ app.py serve.py scripts/ tests/

fmt: ## ruff --fix
	$(BIN)/ruff check --fix src/ app.py serve.py scripts/ tests/

# ---------------------------------------------------------------------------
# The local database
#
# A stand-in for RDS: same extensions, same schema, no AWS account. The schema
# itself comes from hbu_infra, because that is the repo that owns it — this
# only runs it, the way hbu_infra runs the dataplatform's bootstrap file.
# ---------------------------------------------------------------------------

# What docker/postgres.Dockerfile builds pgvector from. Fetched here, on the
# host, because in-build TLS is answered by the TLS-inspecting proxy's
# certificate and the build then fails certificate verify on github.com —
# while the host's curl can be pointed at the combined bundle. Pinned to the
# version the dev RDS reports in pg_extension, so a dump restored locally
# never asks for a vector version this server does not have.
PGVECTOR_VERSION ?= 0.8.1
PGVECTOR_SRC      = docker/pgvector-src.tar.gz

db-image-src: ## Fetch the pgvector source the local database image builds from
	@test -s $(PGVECTOR_SRC) || { \
	  echo "→ fetching pgvector v$(PGVECTOR_VERSION)"; \
	  curl -fsSL $(if $(AWS_CA_BUNDLE),--cacert "$(AWS_CA_BUNDLE)") \
	    -o $(PGVECTOR_SRC) \
	    https://github.com/pgvector/pgvector/archive/refs/tags/v$(PGVECTOR_VERSION).tar.gz; \
	}

db-up: db-image-src ## Start the local postgis+pgvector container and apply the schema
	$(COMPOSE) up -d postgres
	@echo "Waiting for the server…"
	@until $(COMPOSE) exec -T postgres pg_isready -U urban_rag -d urban_rag >/dev/null 2>&1; \
	  do sleep 1; done
	@$(MAKE) db-init
	@echo
	@echo "Ready. make run points here on its own - DB_TARGET=local."
	@echo "For anything that reads .env directly, or for psql:"
	@echo "  DATABASE_URL=$(LOCAL_PG_URL)"

# Every table this app reads is created by hbu_infra — this only runs its SQL
# against the local container, the way hbu_infra runs the dataplatform's
# bootstrap file. 003_spatial_search.sql legitimately fails until rag.chunks
# exists, which is why errors here are reported rather than fatal.
db-init: ## Apply hbu_infra's sql/*.sql to the local container
	@test -d "$(HBU_INFRA)/sql" || { \
	  echo "hbu_infra not found at $(HBU_INFRA) — set HBU_INFRA=/path/to/hbu_infra"; \
	  exit 1; }
	@for f in $(HBU_INFRA)/sql/*.sql; do \
	  printf '→ %s' "$$f"; \
	  if $(COMPOSE) exec -T postgres psql -q -v ON_ERROR_STOP=1 -U urban_rag \
	       -d urban_rag < "$$f" >/dev/null 2>&1; then echo "  ok"; \
	  else echo "  skipped (needs a table that is not loaded yet)"; fi; \
	done

db-shell: ## psql into the local container
	$(COMPOSE) exec postgres psql -U urban_rag -d urban_rag

db-url: ## Print the local connection URL
	@echo "$(LOCAL_PG_URL)"

db-logs: ## Tail the container's logs
	$(COMPOSE) logs -f postgres

db-down: ## Stop the container (keeps the volume)
	$(COMPOSE) down

db-reset: ## Stop the container AND delete its data
	$(COMPOSE) down -v

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build: ## Build the runtime image
	docker build --target runtime -t $(IMAGE):$(IMAGE_TAG) \
	  --build-arg BUILD_VERSION=$(IMAGE_TAG) .
	docker tag $(IMAGE):$(IMAGE_TAG) $(IMAGE):latest

docker-run: docker-build db-target-check ## Run the image against DB_TARGET
	docker run --rm -p 8501:8501 -p $(TILE_PORT):$(TILE_PORT) --env-file .env \
	  -v "$(AWS_DIR):/home/appuser/.aws:ro" \
	  -e AWS_PROFILE="$(AWS_PROFILE)" \
	  -e AWS_REGION="$(AWS_REGION)" \
	  -e AWS_DEFAULT_REGION="$(AWS_REGION)" \
	  -e HBU_TILE_PORT="$(TILE_PORT)" \
	  $(DOCKER_PG_ENV) \
	  $(DOCKER_AWS_CA_ARGS) \
	  $(DOCKER_PG_HOSTS) \
	  --add-host=host.docker.internal:host-gateway \
	  $(IMAGE):$(IMAGE_TAG)

# The same target, pinned to the endpoint. `--env-file .env` is read before
# any -e, so the switch's values override whatever .env names - which is also
# what stops a local DATABASE_URL of 127.0.0.1 from reaching the container,
# where it would mean the container itself.
docker-run-tunnel: ## Run the image through an open hbu_infra db-tunnel (= DB_TARGET=rds)
	@$(MAKE) docker-run DB_TARGET=rds

docker-test: ## Run the unit suite inside the image
	docker build --target test -t $(IMAGE):test .
	docker run --rm $(IMAGE):test

compose-up: ## Build and run app + database together
	$(COMPOSE) --profile app up --build

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
