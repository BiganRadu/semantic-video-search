# The API, built for a host with no GPU and little memory.
#
# Three stages because the build needs Node and Go and the result needs
# neither: what ships is a static Go binary, the Python sources, and two small
# packages. Python is in the final image because the search worker is a Python
# subprocess, so a Go-only runtime is not enough.

# --- frontend ---------------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /src/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
# Baked in at build time: a built bundle has no server to ask at runtime. Empty
# means same-origin, which is right when this container serves the frontend too.
ARG VITE_API_BASE=""
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

# --- server -----------------------------------------------------------------
FROM golang:1.26 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY server/ ./server/
COPY migrations/ ./migrations/
# CGO off so the binary runs on a slim base with no matching libc.
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/server ./server

# --- runtime ----------------------------------------------------------------
FROM python:3.12-slim
WORKDIR /app

# Only what a remote searcher needs; see requirements-remote.txt.
COPY requirements-remote.txt ./
RUN pip install --no-cache-dir -r requirements-remote.txt

COPY --from=build /out/server /app/server
COPY --from=web /src/web/dist /app/web/dist
COPY python/ /app/python/

ENV PYTHON=python3 \
    PYTHON_DIR=/app/python \
    STATIC_DIR=/app/web/dist \
    REMOTE_MODELS=true

# Informational: the platform assigns the real port through PORT, which the
# server prefers over this.
EXPOSE 8080

CMD ["/app/server"]
