"""
RAG Hiring Server v0.2.0 — FastAPI entry point.

Run locally:
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

Changelog v0.2.0
  Phase 8  — Skills Ontology Graph (NetworkX)
  Phase 9  — Bi-directional Matching (candidate interest score)
  Phase 10 — HR Feedback Loop + training data export
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routers import resumes, jobs, feedback
from src.services import feedback_service, registry_service
from src.services.db import close_pool

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database schemas in PostgreSQL
    try:
        feedback_service.init_db()
        registry_service.init_db()
    except Exception as e:
        import sys
        print(f"CRITICAL: Failed to initialize database: {e}", file=sys.stderr)
    yield
    # Gracefully close connection pool on shutdown
    close_pool()

app = FastAPI(
    title="RAG Hiring Server",
    description=(
        "Agentic RAG system for resume screening. Phases 8-10: "
        "Skills Ontology expansion, Bi-directional candidate matching, "
        "and HR Feedback Loop for continuous reranker fine-tuning."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# ── CORS — allow all origins during development ────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(resumes.router)
app.include_router(jobs.router)
app.include_router(feedback.router)


@app.get("/", tags=["health"])
async def health_check():
    """Simple liveness probe for Fly.io health checks."""
    return {"status": "ok", "service": "rag-hiring-server"}
