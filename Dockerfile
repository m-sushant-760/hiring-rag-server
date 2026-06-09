# ── Build stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install uv — the single tool for all Python env management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# System deps needed by PyMuPDF (MuPDF native libs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency manifest and lockfile FIRST — Docker caches this layer
# until pyproject.toml or uv.lock changes (not when src/ changes).
COPY pyproject.toml uv.lock ./

# Install production deps only (no dev tools in the image).
# --frozen: refuse to update lockfile (fail-fast if lockfile is stale).
# --no-dev: skip [dependency-groups] dev.
# --no-install-project: don't install the project itself yet.
RUN uv sync --frozen --no-dev --no-install-project

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy the fully-installed virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source — this layer rebuilds on every code change,
# but the .venv layer above is cached as long as deps don't change.
COPY src/ src/

# Add .venv binaries to PATH so `uvicorn` is found directly
ENV PATH="/app/.venv/bin:$PATH"

# Default port — override at runtime with: docker run -e PORT=9000 -p 9000:9000 ...
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]
