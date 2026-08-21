# A local stand-in for the RDS instance: PostGIS *and* pgvector in one server.
#
# Neither published image has both — postgis/postgis has no vector type, and
# pgvector/pgvector has no geometry type — and this database's whole point is
# asking one question of both. The PostGIS image is built on the official
# postgres image, which already has the PGDG apt repository configured, so
# pgvector is one package away rather than a compile.
FROM postgis/postgis:16-3.4

RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql-16-pgvector \
 && rm -rf /var/lib/apt/lists/*
