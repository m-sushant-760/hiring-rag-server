# RAG Hiring Server

A **Retrieval-Augmented Generation** API for screening resumes against job descriptions.
Built with **FastAPI + Python**, **Pinecone Integrated Inference** (Starter Plan — free),
and **Google Gemini 2.5 Flash** (free via AI Studio), deployed on **Fly.io**.

> **Zero paid API keys required.** All services have a free tier that covers
> prototyping and moderate production use.

---

## Quick Start

### 1. Install `uv` (one-time)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # or restart your terminal
```

### 2. Clone & install dependencies

```bash
# uv creates .venv and installs everything from uv.lock automatically
uv sync --dev
```

### 3. Configure

```bash
cp .env.example .env
# Fill in:
#   GOOGLE_API_KEY=AIza...        ← from aistudio.google.com (free)
#   PINECONE_API_KEY=pcsk_...     ← from pinecone.io (free Starter plan)
```

### 4. Run locally

```bash
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8080
```

Open **http://localhost:8080/docs** for interactive Swagger UI.

### 5. Run tests

```bash
uv run pytest -v
```

---

## API Endpoints

| Method   | Path                          | Description                        |
|----------|-------------------------------|------------------------------------|
| `GET`    | `/`                           | Health check                       |
| `POST`   | `/api/resumes/upload`         | Upload a PDF / DOCX resume         |
| `DELETE` | `/api/resumes/{candidate_id}` | Remove a candidate from the index  |
| `POST`   | `/api/jobs/screen`            | Screen resumes against a job desc. |
| `POST`   | `/api/feedback`               | Submit recruiter feedback          |

### Upload a resume

```bash
curl -X POST http://localhost:8080/api/resumes/upload \
  -F "file=@resume.pdf"
# → {"candidate_id": "uuid...", "filename": "resume.pdf", "chunks_indexed": 14}
```

### Screen candidates

```bash
curl -X POST http://localhost:8080/api/jobs/screen \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Senior Python engineer with FastAPI and ML experience...",
    "top_k": 5
  }'
# → Ranked candidate list with match_score, strengths, gaps, recommendation
```

---

## Deploy to Fly.io

```bash
# Install flyctl
brew install flyctl
flyctl auth login

# Set secrets (never stored in code)
flyctl secrets set \
  GOOGLE_API_KEY=AIza... \
  PINECONE_API_KEY=pcsk_...

# First deploy — detects Dockerfile automatically
flyctl launch --name rag-hiring-server --region iad --no-deploy
flyctl deploy

# View logs
flyctl logs
```

Server goes live at `https://rag-hiring-server.fly.dev`.

---

## Project Structure

```
Sample RAG Server/
├── src/
│   ├── main.py                      # FastAPI entry point, CORS, health check
│   ├── config.py                    # Centralised settings from .env
│   ├── routers/
│   │   ├── resumes.py               # Upload / delete resumes
│   │   ├── jobs.py                  # Screen candidates
│   │   └── feedback.py              # Recruiter feedback endpoint
│   ├── services/
│   │   ├── embedding_service.py     # Stub (Pinecone handles embeddings)
│   │   ├── pinecone_service.py      # Integrated Inference upsert / search
│   │   ├── llm_service.py           # Gemini 2.5 Flash, 6-dim JSON scoring
│   │   ├── ontology_service.py      # Skills graph (networkx) for query expansion
│   │   └── feedback_service.py      # Recruiter feedback persistence
│   └── utils/
│       ├── chunker.py               # Overlapping text splitter
│       └── parser.py                # PDF / DOCX text extraction
├── tests/
│   ├── test_api.py                  # Mocked endpoint tests
│   ├── test_chunker.py              # Chunking edge cases
│   └── test_parser.py               # In-memory PDF/DOCX tests
├── pyproject.toml                   # Single source of truth for deps (uv)
├── uv.lock                          # Exact pinned versions (committed)
├── Dockerfile                       # Multi-stage uv build, slim runtime
├── fly.toml                         # Fly.io config (auto-stop, iad region)
├── .env.example                     # Template for environment variables
└── .gitignore
```

---

## Common `uv` Commands

| Task | Command |
|---|---|
| Install all deps | `uv sync --dev` |
| Add a new package | `uv add <package>` |
| Add a dev-only package | `uv add --dev <package>` |
| Remove a package | `uv remove <package>` |
| Run a command in venv | `uv run <command>` |
| Run the server | `uv run uvicorn src.main:app --reload --port 8080` |
| Run tests | `uv run pytest -v` |
| Upgrade all deps | `uv lock --upgrade && uv sync` |
| Check outdated | `uv tree --outdated` |
