# A local stand-in for the RDS instance: PostGIS *and* pgvector in one server.
#
# Neither published image has both — postgis/postgis has no vector type, and
# pgvector/pgvector has no geometry type — and this database's whole point is
# asking one question of both.
#
# The alpine tag, and pgvector compiled from source rather than installed from
# apt, because the obvious route is unpullable from behind this proxy: the
# Debian postgis/postgis:16-3.4 image has a layer large enough that the Hub's
# CDN answers its GET with 403 partway through, every time, while the alpine
# tag's smaller layers come through fine. postgres:16-alpine has no pgvector
# package, so the extension is a two-minute PGXS build instead.
#
# Two more proxy accommodations, both no-ops on an open network:
#
#   - The source is COPYed in, not cloned: in-build TLS is answered by the
#     TLS-inspecting proxy's certificate, which the image does not trust, so
#     `git clone https://github.com/...` dies on certificate verify. The
#     tarball is fetched by `make db-image-src` on the host, whose trust store
#     handles the proxy, and is gitignored.
#   - apk falls back to plain http mirrors when https fails the same way.
#     Alpine packages are signature-verified against keys already in the
#     image, so http costs confidentiality of *which* packages, not integrity.
#
# v0.8.1 pins what the dev RDS reports in pg_extension (2026-10-01), so a dump
# restored into this container never asks for a vector version the local
# server does not have. with_llvm=no skips emitting JIT bitcode, which would
# otherwise want a clang matching whatever LLVM the base image was built with.
FROM postgis/postgis:16-3.4-alpine

COPY docker/pgvector-src.tar.gz /tmp/pgvector-src.tar.gz

RUN { apk add --no-cache --virtual .build-deps build-base \
      || { sed -i 's|https://|http://|' /etc/apk/repositories \
           && apk add --no-cache --virtual .build-deps build-base; }; } \
 && mkdir /tmp/pgvector \
 && tar xzf /tmp/pgvector-src.tar.gz -C /tmp/pgvector --strip-components=1 \
 && make -C /tmp/pgvector with_llvm=no \
 && make -C /tmp/pgvector install with_llvm=no \
 && rm -rf /tmp/pgvector /tmp/pgvector-src.tar.gz \
 && apk del .build-deps
