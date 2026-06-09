# Resume Screening RAG Server — Architecture Document
## Strategy B: Layout-Aware Structural Parsing + Per-Experience-Block Chunking

> **Document type**: Greenfield architecture and design.
> **Scope**: Design only. No code has been written.

---

## 1. System Purpose

A FastAPI-based RAG (Retrieval-Augmented Generation) server that ingests candidate resumes, indexes them in a semantically structured form, and ranks candidates against a job description using a combination of vector search, structured metadata filtering, skills ontology matching, LLM multi-dimensional evaluation, and bi-directional career-fit scoring.

The system is designed specifically for the structural characteristics of resumes: short, dense, multi-column, semi-structured documents where the semantic connection between *what a candidate did* and *where and when they did it* must never be broken during chunking.

---

## 2. Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Web framework** | FastAPI + Uvicorn | Async, auto-schema, production-grade |
| **Layout parsing** | Unstructured.io (`hi_res` strategy) | Vision-model-backed multi-column PDF reading |
| **DOCX parsing** | `python-docx` | Reliable for single-column structured documents |
| **Vector database** | Pinecone Serverless (Integrated Inference) | Server-side `multilingual-e5-large` embedding; no embedding API key needed |
| **LLM** | Google Gemini 2.5 Flash via `google-genai` SDK | 1,500 req/day free; JSON response mode; low latency |
| **Skills ontology** | NetworkX in-memory directed graph | Zero infra; covers tools → frameworks → categories |
| **Date parsing** | `python-dateutil` | Handles full range of resume date formats |
| **Fuzzy matching** | `rapidfuzz` | Zone header classification before embedding fallback |
| **Config** | `python-dotenv` | `.env`-based secrets, no scattered `os.getenv` |
| **Validation** | Pydantic v2 | All data contracts typed end-to-end |
| **Packaging** | `uv` + `pyproject.toml` | Fast dependency resolution |
| **Containerisation** | Docker + Fly.io (`fly.toml`) | Production deployment target |
| **Feedback store** | SQLite (`feedback.db`) | Lightweight; no extra infra for recruiter signal collection |

---

## 3. Project Structure

```
resume-screening-rag/
├── src/
│   ├── __init__.py
│   ├── main.py                         # FastAPI app factory, router registration
│   ├── config.py                       # Settings class; all env vars in one place
│   ├── models/
│   │   ├── __init__.py
│   │   └── resume_block.py             # ResumeBlock, ZoneType — central data contract
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── layout_parser.py            # Unstructured.io wrapper (PDF + DOCX fallback)
│   │   ├── structural_chunker.py       # Elements → ResumeBlocks (zones + per-role split)
│   │   └── date_parser.py              # Date range normalisation → recency_year, duration_months
│   ├── services/
│   │   ├── __init__.py
│   │   ├── pinecone_service.py         # Index management, upsert, hybrid query, summary fetch
│   │   ├── llm_service.py              # Gemini evaluation + weighted scoring
│   │   ├── ontology_service.py         # Skills graph: expansion, extraction, match scoring
│   │   └── feedback_service.py         # SQLite recruiter feedback storage
│   └── routers/
│       ├── __init__.py
│       ├── resumes.py                  # POST /upload, DELETE /{candidate_id}
│       ├── jobs.py                     # POST /screen
│       └── feedback.py                 # POST /feedback, GET /feedback/{candidate_id}
├── tests/
│   ├── test_layout_parser.py
│   ├── test_structural_chunker.py
│   ├── test_date_parser.py
│   ├── test_pinecone_service.py
│   ├── test_llm_service.py
│   ├── test_ontology_service.py
│   └── fixtures/                       # Sample PDFs and DOCX files for integration tests
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml                  # App + optional self-hosted Unstructured.io
├── fly.toml
├── pyproject.toml
└── requirements.txt
```

---

## 4. Full System Architecture

### 4.1 Ingestion Pipeline

```
POST /api/resumes/upload  (PDF or DOCX, max 10 MB)
        │
        ▼
[resumes.py]
  ① Size guard — reject > MAX_UPLOAD_SIZE_MB
        │
        ▼
[layout_parser.py]  parse_to_elements(filename, data)
        │
        ├── PDF ──────► Unstructured.io  strategy="hi_res"
        │                │  Vision model detects columns, headings, lists, tables
        │                │  Falls back to strategy="fast" on timeout
        │                ▼
        │           list[Element]
        │           (Title | NarrativeText | ListItem | Table | Header | ...)
        │
        └── DOCX ─────► python-docx
                         │  Paragraphs → NarrativeText elements
                         │  Headings   → Title elements
                         ▼
                    list[Element]
        │
        ▼
[structural_chunker.py]  build_structured_blocks(elements)
        │
        ├── ② Zone Detection
        │      Title/Header elements → ZoneType classification
        │      Pass 1: string + fuzzy match against KNOWN_ZONE_LABELS
        │      Pass 2: embedding cosine similarity (if Pass 1 score < threshold)
        │      Accumulate body elements under each zone
        │
        ├── ③ Experience Sub-Splitting (per job role)
        │      Within "experience" zone:
        │        Detect JOB_ROLE_BOUNDARY:
        │          · Title element + date range within 2 lines
        │          · OR: line ≤ 80 chars, title-case, not ending in "."
        │            followed within 2 lines by a date range
        │        One ResumeBlock per detected role:
        │          job_title, company, date_range_text → date_parser
        │          body = remaining NarrativeText + ListItem elements
        │
        ├── ④ Metadata Enrichment (per block)
        │      skills_extracted = ontology_service.extract_skills_from_text(chunk_text)
        │      candidate_name   = first-line name heuristic (no "@", no digits, 5–60 chars)
        │      is_summary       = (zone_type == "summary")
        │      recency_year, duration_months = date_parser.parse_date_range(date_text)
        │
        └── ⑤ Fallback
               If < 2 zones detected:
               → Single ResumeBlock, zone_type="other", no job-level metadata
               → Ingestion continues; vector search works without filters
        │
        │  list[ResumeBlock]
        ▼
[pinecone_service.py]  upsert_structured_blocks(candidate_id, filename, blocks)
        │
        │  Records batched in groups of ≤ 100
        │  Pinecone Integrated Inference embeds chunk_text server-side
        │  (multilingual-e5-large)
        ▼
Pinecone Serverless Index
        │
        ▼
UploadResponse {
    candidate_id, filename, chunks_indexed,
    zones_detected, experience_blocks, message
}
```

---

### 4.2 Screening Pipeline

```
POST /api/jobs/screen
        │
        ▼
[jobs.py]
  ① JD Skills Detection + Ontology Expansion
        ontology_service.extract_skills_from_text(job_description)
        → jd_skills: list[str]

        ontology_service.expand_query_terms(jd_skills, max_hops=1)
        → expanded: set[str]   (aliases, parent categories, child tools)

        augmented_query = job_description + "\n\nExpanded skills:\n" + expanded[:50]
        │
        ▼
  ② Stage 1 — Hybrid Retrieval (vector + metadata pre-filter)
        pinecone_service.query_similar_chunks(
            query_text          = augmented_query,
            top_k               = body.top_k × TOP_K_RESULTS,
            zone_filter         = body.zone_focus,           # optional
            min_recency_year    = body.min_recency_year,     # optional
            min_duration_months = body.min_role_duration_months, # optional
            required_skills     = list(expanded)[:30],       # optional
        )
        → list[hit]   — semantically relevant blocks, pre-filtered by metadata
        │
        ▼
  ③ Stage 2 — Summary Fetch (per unique candidate, parallel)
        asyncio.gather(*[
            pinecone_service.fetch_summary_chunk(cid)
            for cid in unique_candidate_ids
        ])
        → dict[candidate_id → summary_hit | None]
        │
        ▼
  ④ Group + Limit
        _group_by_candidate(hits)
        → dict[candidate_id → {filename, experience_hits, score, metadata}]
        Keep top_k candidates by max hit score
        │
        ▼
  ⑤ Per-Candidate Evaluation  (all four steps run per candidate)
        │
        ├── [A] Small-to-Large Context Assembly
        │       context_chunks = _assemble_context(
        │           summary_hit, experience_hits, max_experience=MAX_EXP_BLOCKS_CONTEXT
        │       )
        │       → [summary_text, exp_block_1, exp_block_2, ...]
        │
        ├── [B] Recency-Weighted Ontology Score
        │       For each hit: skill_match × recency_weight(recency_year)
        │       → onto_score: float (0.0–1.0)
        │
        ├── [C] LLM Multi-Dimensional Evaluation (employer perspective)
        │       llm_service.evaluate_candidate(
        │           job_description  = body.job_description,
        │           resume_chunks    = context_chunks,
        │           context_metadata = [hit metadata dicts],
        │       )
        │       Gemini receives:
        │         · Structured career preamble (role facts from metadata)
        │         · Summary chunk (career narrative)
        │         · Top-N experience chunks (specific technical evidence)
        │       → dict with 6 dimension scores, strengths, gaps, summary, recommendation
        │
        ├── [D] Candidate Interest Score (bi-directional, Phase 9)
        │       _candidate_interest_score(job_description, context_chunks)
        │       → float (0–100): is this role an appropriate career step?
        │
        └── [E] Ensemble Final Score
                _ensemble(llm_score, onto_score, interest, bidirectional)
                Bidirectional:  LLM 50% | Ontology 20% | Bi-dir 30%
                Unidirectional: LLM 60% | Ontology 40%
        │
        ▼
  ⑥ Rank descending → assign final_rank
        │
        ▼
ScreenResponse {
    job_description_snippet, jd_skills_detected,
    total_candidates_evaluated,
    results: list[CandidateResult]
}
```

---

## 5. Data Models

### 5.1 `src/models/resume_block.py`

The central data contract flowing between the parser, chunker, and Pinecone service.

```python
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

ZoneType = Literal[
    "summary",        # Professional summary / profile / objective
    "experience",     # Work history — one block per job role
    "education",      # Degrees, academic background
    "skills",         # Technical skills, competencies, tech stack
    "certifications", # Certs, licenses, accreditations
    "other",          # Fallback: awards, publications, languages, unrecognised sections
]


class ResumeBlock(BaseModel):
    """
    One semantically coherent chunk of a resume.
    For the "experience" zone, one ResumeBlock = one job role.
    For all other zones, one ResumeBlock = the entire zone content
    (zones are typically short enough not to need further splitting).
    """
    chunk_text:       str        = Field(..., description="The raw text content of this block.")
    zone_type:        ZoneType   = Field(..., description="Semantic zone this block belongs to.")
    candidate_name:   str        = Field("",   description="Extracted from the resume header.")
    is_summary:       bool       = Field(False, description="True for summary/profile zone blocks.")

    # Experience-specific fields (null for non-experience zones)
    job_title:        str | None = Field(None, description="Detected job title for this role.")
    company:          str | None = Field(None, description="Detected company name for this role.")
    recency_year:     int | None = Field(None, description="Year the role ended (current year if 'Present').")
    duration_months:  int | None = Field(None, description="Computed length of tenure in months.")

    # Enriched at indexing time
    skills_extracted: list[str]  = Field(default_factory=list,
                                          description="Ontology-matched skills found in chunk_text.")
```

---

### 5.2 Pinecone Record Schema

Every `ResumeBlock` maps to exactly one Pinecone record.

```json
{
  "_id":              "3f8a2b1c-…#block4",
  "chunk_text":       "Designed and implemented a real-time event streaming platform using Apache Kafka and Python, reducing data latency from 8 minutes to 12 seconds across 40+ microservices.",
  "candidate_id":     "3f8a2b1c-4e9d-…",
  "filename":         "jane_smith_cv.pdf",
  "block_index":      4,
  "candidate_name":   "Jane Smith",
  "zone_type":        "experience",
  "is_summary":       false,
  "job_title":        "Senior Data Engineer",
  "company":          "StreamCo",
  "recency_year":     2025,
  "duration_months":  22,
  "skills_extracted": ["Apache Kafka", "Python", "Microservices", "Data Engineering", "ETL"]
}
```

**For a summary block:**
```json
{
  "_id":            "3f8a2b1c-…#block0",
  "chunk_text":     "Experienced data engineer with 8+ years building scalable pipelines…",
  "candidate_id":   "3f8a2b1c-…",
  "filename":       "jane_smith_cv.pdf",
  "block_index":    0,
  "candidate_name": "Jane Smith",
  "zone_type":      "summary",
  "is_summary":     true,
  "job_title":      null,
  "company":        null,
  "recency_year":   null,
  "duration_months":null,
  "skills_extracted": ["Python", "Data Pipelines", "Apache Kafka"]
}
```

**Pinecone filterable metadata fields:**

| Field | Type | Operators Supported |
|-------|------|---------------------|
| `candidate_id` | string | `$eq` |
| `zone_type` | string | `$eq`, `$in` |
| `is_summary` | boolean | `$eq` |
| `recency_year` | integer | `$gte`, `$lte`, `$eq` |
| `duration_months` | integer | `$gte`, `$lte` |
| `skills_extracted` | list[string] | `$in` |
| `company` | string | `$eq` |

> [!NOTE]
> On Pinecone Serverless plans, all metadata fields are indexed by default and support filtering without any additional configuration.

---

## 6. Component Designs

### 6.1 `src/config.py`

Single `Settings` class; all configuration is read from environment variables via `.env`. No `os.getenv` calls elsewhere in the codebase.

```python
class Settings:
    # Pinecone
    pinecone_api_key:        str   = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name:     str   = os.getenv("PINECONE_INDEX_NAME", "hiring-rag")
    pinecone_cloud:          str   = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region:         str   = os.getenv("PINECONE_REGION", "us-east-1")
    pinecone_embedding_model:str   = os.getenv("PINECONE_EMBEDDING_MODEL", "multilingual-e5-large")

    # Google Gemini
    google_api_key:          str   = os.getenv("GOOGLE_API_KEY", "")
    gemini_model:            str   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Unstructured.io
    unstructured_api_key:    str   = os.getenv("UNSTRUCTURED_API_KEY", "")
    unstructured_api_url:    str   = os.getenv("UNSTRUCTURED_API_URL",
                                        "https://api.unstructured.io/general/v0/general")
    unstructured_strategy:   str   = os.getenv("UNSTRUCTURED_STRATEGY", "hi_res")
    unstructured_timeout_s:  int   = int(os.getenv("UNSTRUCTURED_TIMEOUT_S", "30"))

    # Application
    app_env:                 str   = os.getenv("APP_ENV", "development")
    max_upload_size_bytes:   int   = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10")) * 1024 * 1024
    top_k_results:           int   = int(os.getenv("TOP_K_RESULTS", "10"))

    # Retrieval tuning
    zone_embed_threshold:    float = float(os.getenv("ZONE_EMBED_THRESHOLD", "0.6"))
    max_exp_blocks_context:  int   = int(os.getenv("MAX_EXP_BLOCKS_CONTEXT", "3"))

    # Recency weighting tiers (in years before current)
    recency_tier1_years:     int   = int(os.getenv("RECENCY_TIER1_YEARS", "2"))   # weight 1.0
    recency_tier2_years:     int   = int(os.getenv("RECENCY_TIER2_YEARS", "4"))   # weight 0.7
                                                                                    # else  0.4
```

**.env.example:**
```dotenv
# Pinecone
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=hiring-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_EMBEDDING_MODEL=multilingual-e5-large

# Gemini
GOOGLE_API_KEY=your_google_key
GEMINI_MODEL=gemini-2.5-flash

# Unstructured.io  (leave blank to use self-hosted Docker)
UNSTRUCTURED_API_KEY=your_unstructured_key
UNSTRUCTURED_API_URL=https://api.unstructured.io/general/v0/general
UNSTRUCTURED_STRATEGY=hi_res
UNSTRUCTURED_TIMEOUT_S=30

# App
MAX_UPLOAD_SIZE_MB=10
TOP_K_RESULTS=10
MAX_EXP_BLOCKS_CONTEXT=3
RECENCY_TIER1_YEARS=2
RECENCY_TIER2_YEARS=4
```

---

### 6.2 `src/utils/layout_parser.py`

**Responsibility**: Accept raw file bytes, return an ordered `list[Element]` with correct multi-column reading order and typed element classification.

**Design decisions**:
- Uses `unstructured.partition.auto.partition()` with `strategy="hi_res"` for PDFs. This internally runs a vision model (layout detection → OCR) that correctly reads two-column resumes column-by-column, not row-by-row.
- `strategy="fast"` is the automatic fallback when `hi_res` times out or fails — degrades to text-order extraction, still better than PyPDF2.
- DOCX files are processed by `python-docx` directly. Unstructured's DOCX support parses paragraph styles into `Title`/`NarrativeText` equivalents, but `python-docx` gives finer control over style attributes needed for heading detection.
- The function always returns `list[Element]`. It never raises on recoverable failures — it raises `ValueError` only on truly unreadable inputs, which becomes HTTP 422 in the router.

**Interface**:
```python
def parse_to_elements(filename: str, data: bytes) -> list[Element]:
    """
    Parse PDF or DOCX bytes into an ordered list of typed Unstructured Elements.

    PDF path: Unstructured.io hi_res → fast fallback.
    DOCX path: python-docx paragraph/heading extraction.

    Returns: list[Element] — never empty for a readable file.
    Raises:  ValueError for unsupported file types or completely unreadable inputs.
    """
```

**Element types used downstream:**

| Unstructured Type | Meaning | Used for |
|-------------------|---------|---------|
| `Title` | Bold/large heading | Zone header detection, role boundary detection |
| `Header` | Page header | Zone header detection |
| `NarrativeText` | Body paragraph | Zone body text, company name extraction |
| `ListItem` | Bullet point | Experience body, skills lists |
| `Table` | Tabular data | Skills tables, education tables |

**Failure handling:**

| Failure | Response |
|---------|---------|
| `strategy="hi_res"` timeout | Retry with `strategy="fast"`; log `WARNING` |
| Unstructured API unavailable | `strategy="fast"` local fallback; log `WARNING` |
| Scanned/image-only PDF | `hi_res` uses OCR internally; transparent |
| Encrypted PDF | `ValueError` → HTTP 422 |
| Unsupported extension | `ValueError` → HTTP 415 |

---

### 6.3 `src/utils/date_parser.py`

**Responsibility**: Extract and normalise resume date ranges into `(recency_year, duration_months)` tuples.

**Supported patterns** (handled by `python-dateutil` + regex pre-processing):

```
"Jan 2022 – Mar 2024"         → (2024, 26)
"January 2022 — March 2024"   → (2024, 26)
"2021/03 – 2023/07"           → (2023, 28)
"March 2020 to Present"       → (current_year, computed)
"2019 – 2021"                 → (2021, 24)  # mid-year approximation
"Jun '18 – Dec '19"           → (2019, 18)
"2023 – Current"              → (current_year, computed)
"2023 – Now"                  → (current_year, computed)
"2023 – Till Date"            → (current_year, computed)
"03/2021 - 07/2023"           → (2023, 28)
```

**Relative-marker normalisation**: "Present", "Current", "Now", "Till Date", "Ongoing", "—" (em-dash alone) → `datetime.date.today()`.

**Interface**:
```python
def parse_date_range(text: str) -> tuple[int | None, int | None]:
    """
    Scan text for a date range pattern and return (recency_year, duration_months).
    recency_year  = year of the end date (current year if still ongoing).
    duration_months = integer months between start and end (rounded).
    Returns (None, None) if no parseable date range is found in text.
    """

def recency_weight(recency_year: int | None, settings: Settings) -> float:
    """
    Map a recency_year to a weight for the ontology score ensemble.
    Tier 1 (≤ tier1_years ago): 1.0
    Tier 2 (≤ tier2_years ago): 0.7
    Older or unknown:           0.4
    """
```

---

### 6.4 `src/utils/structural_chunker.py`

**Responsibility**: The core transformation layer. Converts `list[Element]` into `list[ResumeBlock]` with full metadata enrichment.

**Algorithm in detail**:

```
─── Step 1: Candidate Name Extraction ───────────────────────────────────────
  Scan first 5 non-empty elements.
  First element matching: len 5–60, no "@", no digits, not all-caps → candidate_name.
  Stored on every block produced for this resume.

─── Step 2: Zone Detection ──────────────────────────────────────────────────
  Iterate elements in document order.
  For each Title or Header element:
    a. Normalise: strip punctuation, lowercase, strip leading icons/bullets.
    b. Pass 1 — String matching:
         For each (zone_type, alias_list) in KNOWN_ZONE_LABELS:
           If element text matches any alias (exact or rapidfuzz ratio ≥ 85):
             → assign zone_type, break.
    c. Pass 2 — Embedding similarity (only if Pass 1 found no match):
         Retrieve pre-cached zone label embedding vectors (loaded at startup).
         Embed element text via Pinecone Integrated Inference (single call).
         Cosine similarity vs. each zone label vector.
         If max_similarity ≥ ZONE_EMBED_THRESHOLD → assign that zone_type.
         Else → "other".
  Accumulate subsequent NarrativeText, ListItem, Table elements under current zone
  until the next Title/Header is encountered.

─── Step 3: Experience Zone Sub-Splitting (per job role) ────────────────────
  Process the accumulated "experience" elements as a sub-document.
  Role boundary signals (in order of confidence):
    HIGH:   A Title element whose text is ≤ 80 chars AND within the next 3
            elements there is a NarrativeText containing a parseable date range.
    MEDIUM: A NarrativeText line ≤ 60 chars, title-case, not ending in period,
            not starting with a verb, followed within 2 lines by a date range.
    LOW:    A line containing only a company name pattern (e.g. capitalised words
            with optional "Inc.", "Ltd.", "GmbH") adjacent to a date range.

  For each detected role boundary, create a new sub-block:
    job_title       : the boundary header text
    company         : next short NarrativeText line below job_title (heuristic;
                      may be null if pattern not confident)
    date_range_text : first date range string found in sub-block
    recency_year, duration_months : date_parser.parse_date_range(date_range_text)
    chunk_text      : all NarrativeText + ListItem text in this sub-block joined

  If no role boundaries are detected:
    → One block for the entire experience zone (no job_title/company)

─── Step 4: Non-Experience Zone Handling ────────────────────────────────────
  Each non-experience zone becomes one ResumeBlock:
    zone_type = detected zone
    chunk_text = all element text joined with "\n"
    is_summary = (zone_type == "summary")
    job_title, company, recency_year, duration_months = None

─── Step 5: Metadata Enrichment (all blocks) ────────────────────────────────
  For each ResumeBlock:
    skills_extracted = ontology_service.extract_skills_from_text(block.chunk_text)
    Cap at 50 skills to respect Pinecone metadata size limits.

─── Step 6: Fallback ────────────────────────────────────────────────────────
  If total zones detected < 2 (no headers found at all):
    Return [ResumeBlock(chunk_text=all_text, zone_type="other", ...)]
  Ingestion always succeeds; vector search degrades gracefully without filters.
```

**KNOWN_ZONE_LABELS:**

```python
KNOWN_ZONE_LABELS: dict[ZoneType, list[str]] = {
    "summary": [
        "summary", "professional summary", "profile", "about me", "about",
        "objective", "career objective", "overview", "executive summary",
        "summary of qualifications", "professional profile",
    ],
    "experience": [
        "experience", "work experience", "professional experience",
        "employment", "employment history", "work history", "career history",
        "career", "industry experience", "relevant experience",
    ],
    "education": [
        "education", "academic background", "academic qualifications",
        "qualifications", "degrees", "academic credentials", "schooling",
        "educational background",
    ],
    "skills": [
        "skills", "technical skills", "core competencies", "competencies",
        "technologies", "tech stack", "tools & technologies",
        "areas of expertise", "key skills", "proficiencies", "tools",
    ],
    "certifications": [
        "certifications", "certificates", "professional certifications",
        "licenses", "accreditations", "credentials",
    ],
    "other": [],   # Catch-all: awards, publications, languages, interests, references
}
```

**Interface**:
```python
def build_structured_blocks(
    elements: list[Element],
    candidate_name_hint: str = "",
) -> list[ResumeBlock]:
    """
    Convert an ordered list of Unstructured Elements into structured, enriched ResumeBlocks.

    - One block per detected zone (summary, education, skills, certifications, other).
    - One block per job role within the experience zone.
    - Every block carries full metadata: zone_type, candidate_name, job_title,
      company, recency_year, duration_months, skills_extracted, is_summary.
    - Always returns at least one block (fallback to zone_type="other").
    """

def _load_zone_label_embeddings() -> dict[ZoneType, list[float]]:
    """
    Pre-compute and cache zone label embedding vectors at startup.
    Called once during application initialisation.
    Each canonical zone label is embedded using Pinecone Integrated Inference.
    """
```

---

### 6.5 `src/services/pinecone_service.py`

**Responsibility**: Pinecone index lifecycle management, structured block upsert, hybrid retrieval, summary fetch, and candidate deletion.

**Index creation** (lazy, on first request):
```python
pc.create_index(
    name   = settings.pinecone_index_name,
    metric = "cosine",
    spec   = ServerlessSpec(cloud=settings.pinecone_cloud,
                             region=settings.pinecone_region),
    embed  = {
        "model":     settings.pinecone_embedding_model,   # multilingual-e5-large
        "field_map": {"text": "chunk_text"},              # Pinecone embeds this field
    },
)
```

**Functions**:

```python
def upsert_structured_blocks(
    candidate_id: str,
    filename:     str,
    blocks:       list[ResumeBlock],
) -> int:
    """
    Upsert all ResumeBlocks for one resume as Pinecone records.
    One record per block. Records batched in groups of ≤ 100.
    Pinecone embeds chunk_text server-side.
    Returns total number of records upserted.
    """


def query_similar_chunks(
    query_text:            str,
    top_k:                 int | None = None,
    zone_filter:           str | None = None,
    min_recency_year:      int | None = None,
    min_duration_months:   int | None = None,
    required_skills:       list[str] | None = None,
) -> list[dict]:
    """
    Hybrid retrieval: semantic vector search + optional metadata pre-filters.

    Filters are composed as an $and clause:
      zone_filter        → zone_type $eq
      min_recency_year   → recency_year $gte
      min_duration_months→ duration_months $gte
      required_skills    → skills_extracted $in

    Returns list of hit dicts, each containing:
      _id, _score, chunk_text, candidate_id, filename, block_index,
      candidate_name, zone_type, is_summary, job_title, company,
      recency_year, duration_months, skills_extracted.
    """


def fetch_summary_chunk(candidate_id: str) -> dict | None:
    """
    Retrieve the is_summary=true block for a specific candidate.
    Used in Stage 2 of the screening pipeline (small-to-large context).
    Returns the hit dict or None if the candidate has no summary block.
    """


def delete_resume(candidate_id: str) -> None:
    """
    Remove all Pinecone records belonging to candidate_id.
    Uses metadata filter: candidate_id $eq candidate_id.
    """
```

**Dynamic filter construction inside `query_similar_chunks`**:
```python
def _build_filter(
    zone_filter:         str | None,
    min_recency_year:    int | None,
    min_duration_months: int | None,
    required_skills:     list[str] | None,
) -> dict | None:
    clauses = []
    if zone_filter:
        clauses.append({"zone_type": {"$eq": zone_filter}})
    if min_recency_year:
        clauses.append({"recency_year": {"$gte": min_recency_year}})
    if min_duration_months:
        clauses.append({"duration_months": {"$gte": min_duration_months}})
    if required_skills:
        clauses.append({"skills_extracted": {"$in": required_skills}})
    if not clauses:
        return None
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]
```

---

### 6.6 `src/routers/resumes.py`

**Endpoints**: `POST /api/resumes/upload`, `DELETE /api/resumes/{candidate_id}`

**Request/Response models**:

```python
class UploadResponse(BaseModel):
    candidate_id:      str
    filename:          str
    chunks_indexed:    int           # total Pinecone records created
    zones_detected:    list[str]     # e.g. ["summary", "experience", "skills", "education"]
    experience_blocks: int           # number of individual job-role blocks created
    message:           str
    parse_quality:     str           # "structured" | "partial" | "fallback"
    # "structured"  = ≥3 zones detected, ≥1 experience role block
    # "partial"     = zones detected but no per-role splitting
    # "fallback"    = fewer than 2 zones; indexed as a single block


class DeleteResponse(BaseModel):
    candidate_id: str
    message:      str
```

**Upload handler logic**:
```
① Size guard
② parse_to_elements(filename, data) → list[Element]
③ build_structured_blocks(elements) → list[ResumeBlock]
④ Guard: if not blocks → HTTP 422
⑤ candidate_id = uuid4()
⑥ upsert_structured_blocks(candidate_id, filename, blocks) → count
⑦ Derive parse_quality from zones_detected and experience_blocks
⑧ Return UploadResponse
```

---

### 6.7 `src/routers/jobs.py`

**Endpoint**: `POST /api/jobs/screen`

**Request model**:

```python
class ScreenRequest(BaseModel):
    job_description:           str       = Field(..., min_length=20)
    top_k:                     int       = Field(default=5, ge=1, le=20)

    # Pre-filter options (metadata-level, applied before vector scoring)
    min_recency_year:          int | None = Field(
        default=None,
        description="Exclude experience blocks older than this year."
    )
    min_role_duration_months:  int | None = Field(
        default=None,
        description="Exclude roles shorter than N months."
    )
    zone_focus:                str | None = Field(
        default=None,
        description="Restrict Stage 1 retrieval to a single zone: "
                    "'experience', 'skills', 'education', etc."
    )

    # Scoring options
    use_ontology:              bool = Field(
        default=True,
        description="Expand JD terms via skills ontology graph."
    )
    bidirectional:             bool = Field(
        default=True,
        description="Include candidate-interest bi-directional score."
    )
    required_certifications:   list[str] = Field(
        default_factory=list,
        description="Pre-filter: only candidates with these certifications in skills_extracted."
    )
```

**Response models**:

```python
class DimensionScores(BaseModel):
    technical:      float | None = None   # technical_skills_score
    relevance:      float | None = None   # experience_relevance_score
    depth:          float | None = None   # experience_depth_score
    education:      float | None = None   # education_score
    certifications: float | None = None   # certifications_score
    communication:  float | None = None   # communication_score


class CandidateResult(BaseModel):
    candidate_id:             str
    candidate_name:           str            # from block metadata
    filename:                 str
    final_rank:               int
    match_score:              int            # ensemble 0–100
    employer_score:           float          # LLM weighted (employer perspective)
    candidate_interest_score: float          # bi-directional career-fit
    ontology_skill_score:     float          # recency-weighted ontology overlap %
    recommendation:           str            # "Strong Yes | Yes | Maybe | No"
    strengths:                list[str]
    gaps:                     list[str]
    summary:                  str            # 2–3 sentence Gemini narrative
    dimension_scores:         DimensionScores
    matched_jd_skills:        list[str]      # JD skills found in candidate blocks
    expanded_skills:          list[str]      # ontology-expanded terms used in query
    top_roles:                list[str]      # "job_title @ company" from top-scored blocks
    most_recent_role_year:    int | None     # max(recency_year) across retrieved blocks
    zones_matched:            list[str]      # which zones contributed to the score
    parse_quality:            str            # "structured" | "partial" | "fallback"


class ScreenResponse(BaseModel):
    job_description_snippet:    str
    jd_skills_detected:         list[str]
    total_candidates_evaluated: int
    results:                    list[CandidateResult]
```

**Key internal helpers**:

```python
def _group_by_candidate(hits: list[dict]) -> dict[str, dict]:
    """
    Group Stage 1 Pinecone hits by candidate_id.
    Per candidate accumulates: filename, candidate_name, experience_hits,
    max _score, all recency_year values, all zone_types, top_roles list.
    """


def _assemble_context(
    summary_hit:       dict | None,
    experience_hits:   list[dict],
    max_experience:    int,
) -> tuple[list[str], list[dict]]:
    """
    Build (context_chunks, context_metadata) for the LLM.
    context_chunks   : [summary_text, exp_block_1_text, ...]
    context_metadata : [summary_hit_dict, exp_hit_1_dict, ...]
    Always leads with summary (if present) for career-level grounding.
    """


def _recency_weighted_ontology_score(
    hits:       list[dict],
    jd_skills:  list[str],
    settings:   Settings,
) -> float:
    """
    Compute ontology match score weighted by recency of the block where the skill appeared.
    Score = Σ(match_i × weight_i) / Σ(weight_i)
    where weight_i = recency_weight(hit["recency_year"], settings).
    Returns float 0.0–1.0.
    """


def _ensemble(
    llm_score:  float,
    onto_score: float,   # 0–1
    interest:   float,   # 0–100
    bidir:      bool,
) -> float:
    """
    Final score (0–100).
    Bidirectional:   LLM 50% | Ontology 20% | Bi-dir 30%
                     where Bi-dir = 70% × employer_llm + 30% × candidate_interest
    Unidirectional:  LLM 60% | Ontology 40%
    """
```

---

### 6.8 `src/services/llm_service.py`

**Responsibility**: Gemini-based multi-dimensional evaluation + weighted scoring.

**System prompt** (6 dimensions, JSON output):
```
You are a senior talent acquisition specialist.
Given a JOB DESCRIPTION and CANDIDATE CAREER CONTEXT + RESUME EXCERPTS, evaluate:

{
  "technical_skills_score":     <0-100>,
  "experience_relevance_score": <0-100>,
  "experience_depth_score":     <0-100>,
  "education_score":            <0-100>,
  "certifications_score":       <0-100>,
  "communication_score":        <0-100>,
  "strengths": ["...", "...", "..."],
  "gaps":      ["...", "...", "..."],
  "summary":   "2-3 sentence narrative for the recruiter",
  "recommendation": "Strong Yes | Yes | Maybe | No"
}

Return ONLY the JSON object.
```

**Structured career preamble** (injected before resume excerpts when metadata is available):
```
CANDIDATE CAREER CONTEXT:
  Name            : Jane Smith
  Most recent role: Senior Data Engineer @ StreamCo (2025, 22 months)
  Previous role   : Data Engineer @ DataCorp (2022, 18 months)
  Skills detected : Apache Kafka, Python, Microservices, BigQuery, dbt

RESUME EXCERPTS:
[Professional Summary]
Experienced data engineer with 8+ years…

---

[Experience: Senior Data Engineer @ StreamCo, 2023–2025]
Designed and implemented a real-time event streaming platform…

---

[Experience: Data Engineer @ DataCorp, 2021–2023]
Built and maintained ELT pipelines…
```

**Interface**:

```python
def evaluate_candidate(
    job_description:  str,
    resume_chunks:    list[str],
    context_metadata: list[dict] | None = None,
) -> dict:
    """
    Evaluate a candidate against a JD using Gemini.

    context_metadata: list of Pinecone hit dicts for each chunk in resume_chunks.
    When provided, a structured career context preamble is prepended to the prompt,
    giving Gemini explicit role facts (title, company, recency, skills) before
    it reads the raw text excerpts.

    Returns parsed dict with 6 scores, strengths, gaps, summary, recommendation.
    Falls back to all-zero safe dict on JSON parse failure.
    """


def compute_weighted_score(evaluation: dict) -> float:
    """
    Weighted composite score (0–100).
    Weights:
      technical_skills_score     0.30
      experience_relevance_score 0.25
      experience_depth_score     0.15
      education_score            0.10
      certifications_score       0.10
      communication_score        0.10
    """
```

---

### 6.9 `src/services/ontology_service.py`

**Responsibility**: In-memory NetworkX directed skills graph. Provides skills expansion, extraction, and structured match scoring. Unchanged in design from the baseline architecture, but benefits significantly from cleaner per-zone and per-role chunk inputs.

**Key public functions**:
```python
def extract_skills_from_text(text: str) -> list[str]:
    """Return all ontology skill nodes found (case-insensitive) in text."""

def expand_query_terms(terms: list[str], max_hops: int = 1) -> set[str]:
    """Expand JD terms with ontology neighbours (aliases, parents, children)."""

def skills_match_score(
    candidate_skills: list[str],
    jd_skills:        list[str],
    max_hops:         int = 2,
) -> float:
    """Structured overlap score 0.0–1.0 using ontology expansion."""
```

---

### 6.10 `src/services/feedback_service.py`

**Responsibility**: Persist recruiter feedback (hire/reject decisions, rating overrides) in SQLite for future fine-tuning signal collection.

Unchanged in design. No interaction with the new chunking or retrieval components.

---

## 7. API Surface

### Resume Management

| Method | Path | Description | Request | Response |
|--------|------|-------------|---------|---------|
| `POST` | `/api/resumes/upload` | Upload and index a resume | `multipart/form-data` file (PDF or DOCX) | `UploadResponse` (201) |
| `DELETE` | `/api/resumes/{candidate_id}` | Remove all vectors for a candidate | path param | `DeleteResponse` (200) |

### Screening

| Method | Path | Description | Request | Response |
|--------|------|-------------|---------|---------|
| `POST` | `/api/jobs/screen` | Screen indexed resumes against a JD | `ScreenRequest` JSON | `ScreenResponse` (200) |

### Feedback

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/feedback` | Submit recruiter decision signal |
| `GET` | `/api/feedback/{candidate_id}` | Retrieve stored feedback for a candidate |

### Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — returns `{"status": "ok"}` |

---

## 8. Data Flow — Complete End-to-End

### Upload Flow (detailed)

```
Client
  │  POST /api/resumes/upload  multipart PDF
  ▼
FastAPI [resumes.py]
  │  ① await file.read()
  │  ② size guard (> 10 MB → HTTP 413)
  │
  │  ③ layout_parser.parse_to_elements(filename, data)
  │         ├── PDF → Unstructured.io hi_res
  │         │         ↓ timeout → fast fallback
  │         └── DOCX → python-docx
  │         → list[Element]  (Title, NarrativeText, ListItem, …)
  │
  │  ④ structural_chunker.build_structured_blocks(elements)
  │     ├── candidate_name extracted
  │     ├── zones detected (string → fuzzy → embedding)
  │     ├── experience sub-split per role (boundary detection)
  │     ├── dates parsed (date_parser.parse_date_range)
  │     ├── skills extracted (ontology_service.extract_skills_from_text)
  │     └── is_summary flagged
  │     → list[ResumeBlock]
  │
  │  ⑤ pinecone_service.upsert_structured_blocks(candidate_id, filename, blocks)
  │     ├── Build record dicts from ResumeBlock fields
  │     ├── Batch upsert in groups of ≤ 100
  │     └── Pinecone embeds chunk_text via multilingual-e5-large
  │     → count: int
  │
  └── HTTP 201  UploadResponse {
        candidate_id, filename, chunks_indexed,
        zones_detected, experience_blocks,
        parse_quality, message
      }
```

### Screen Flow (detailed)

```
Client
  │  POST /api/jobs/screen  {job_description, top_k, min_recency_year, …}
  ▼
FastAPI [jobs.py]
  │
  │  ① JD Processing
  │     jd_skills  = ontology_service.extract_skills_from_text(job_description)
  │     expanded   = ontology_service.expand_query_terms(jd_skills, max_hops=1)
  │     aug_query  = job_description + "\nExpanded skills:\n" + expanded[:50]
  │
  │  ② Stage 1 — Hybrid Retrieval
  │     hits = pinecone_service.query_similar_chunks(
  │                aug_query,
  │                top_k              = top_k × TOP_K_RESULTS,
  │                zone_filter        = zone_focus,
  │                min_recency_year   = min_recency_year,
  │                min_duration_months= min_role_duration_months,
  │                required_skills    = expanded[:30] if use_ontology,
  │            )
  │     → list[hit]  (blocks passing both vector similarity AND metadata filters)
  │
  │  ③ Stage 2 — Summary Fetch (parallel)
  │     summaries = await asyncio.gather(*[
  │         fetch_summary_chunk(cid)
  │         for cid in unique_candidate_ids(hits)
  │     ])
  │
  │  ④ Group + Trim
  │     grouped = _group_by_candidate(hits)
  │     candidates = list(grouped.items())[:top_k]
  │
  │  ⑤ Per-Candidate Scoring
  │     for cid, info in candidates:
  │
  │       context_chunks, context_meta = _assemble_context(
  │           summaries[cid], info["experience_hits"],
  │           max_experience = MAX_EXP_BLOCKS_CONTEXT
  │       )
  │
  │       onto_score = _recency_weighted_ontology_score(
  │           info["experience_hits"], jd_skills, settings
  │       )
  │
  │       evaluation = llm_service.evaluate_candidate(
  │           job_description, context_chunks, context_meta
  │       )
  │       employer_score = llm_service.compute_weighted_score(evaluation)
  │
  │       interest = _candidate_interest_score(job_description, context_chunks)
  │                  if bidirectional else 50.0
  │
  │       final_score = _ensemble(employer_score, onto_score, interest, bidirectional)
  │
  │  ⑥ Sort descending, assign final_rank
  │
  └── HTTP 200  ScreenResponse {
        jd_skills_detected, total_candidates_evaluated,
        results: list[CandidateResult]
      }
```

---

## 9. Dependencies

### Python packages (`requirements.txt`)

```
# Web framework
fastapi>=0.111
uvicorn[standard]>=0.30
python-multipart>=0.0.9

# Layout-aware parsing
unstructured[pdf]>=0.13          # Includes hi_res vision model + OCR
python-docx>=1.1                 # DOCX paragraph/heading extraction

# Date parsing
python-dateutil>=2.9

# Fuzzy matching (zone label classification)
rapidfuzz>=3.0

# Vector database
pinecone>=5.0                    # Serverless + Integrated Inference

# LLM
google-genai>=1.0                # google.genai (not deprecated generativeai)

# Skills ontology graph
networkx>=3.0

# Data validation
pydantic>=2.7
pydantic-settings>=2.0

# Configuration
python-dotenv>=1.0

# Feedback store
# SQLite is part of Python stdlib — no extra package

# Testing
pytest>=8.0
pytest-asyncio>=0.23
httpx>=0.27                      # ASGI test client
```

> [!NOTE]
> `unstructured[pdf]` includes `detectron2` and `pytesseract` as transitive dependencies for `hi_res` processing. First install may take several minutes. Docker build caches this layer.

---

## 10. Infrastructure Requirements

| Component | Requirement | Options |
|-----------|------------|---------|
| **Unstructured.io** | API key OR self-hosted Docker | SaaS free tier: 1,000 pages/month. Production: paid plan or self-hosted. |
| **Pinecone** | Starter Serverless account | Free: 2 GB storage, 5M tokens/month. Sufficient for hundreds of resumes. |
| **Google Gemini** | AI Studio API key | Free: 1,500 req/day. Each screen request uses 1 evaluation + 1 bi-directional call per candidate. |
| **Application host** | Any Python WSGI host or Docker | Fly.io config included. 512 MB RAM minimum for `hi_res` fallback; 4 GB for self-hosted Unstructured. |
| **Application memory** | +~50 MB above baseline | Zone label embedding vectors cached at startup. |

### Self-Hosted Unstructured.io (no SaaS cost)

```yaml
# docker-compose.yml
services:
  app:
    build: .
    ports: ["8080:8080"]
    environment:
      - UNSTRUCTURED_API_URL=http://unstructured:8000/general/v0/general
      - UNSTRUCTURED_API_KEY=local
    depends_on: [unstructured]

  unstructured:
    image: downloads.unstructured.io/unstructured-io/unstructured:latest
    ports: ["8000:8000"]
    environment:
      - UNSTRUCTURED_API_KEY=local
    deploy:
      resources:
        limits:
          memory: 4G
```

Set `UNSTRUCTURED_API_KEY=local` and `UNSTRUCTURED_API_URL=http://unstructured:8000/general/v0/general` in the app's `.env`.

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| `hi_res` timeout on large multi-page PDFs | Medium | Medium | Auto-fallback to `strategy="fast"`; configurable `UNSTRUCTURED_TIMEOUT_S`; 10 MB upload cap limits pages |
| Per-role boundary detection fails on functional or skills-based resume formats | Medium | Medium | Falls back to one block per experience section; vector search still works, just without per-role metadata |
| Date parsing returns `None` for exotic formats | Medium | Low | `recency_year=None` → block not filterable but still retrieved by vector score; no data loss; filter is opt-in |
| `skills_extracted` list exceeds Pinecone metadata size (40 KB/record) | Low | Low | Cap at 50 skills per block before upsert |
| `required_skills` filter too narrow → zero Pinecone hits | Low | Medium | `required_skills=None` by default; caller opts in; system returns empty results gracefully with clear message |
| Zone embedding fallback adds latency per unknown header | Low | Low | Zone label vectors pre-computed and cached at startup; one embedding call per unknown header only |
| Unstructured.io SaaS pricing at production resume volume | High | Medium | Self-hosted Docker option; async upload queue (e.g. Celery + Redis) to absorb burst traffic |
| Gemini API rate limit (1,500 req/day free) | Medium | Medium | Each `/screen` with top_k=5 uses 10 Gemini calls; 150 screenings/day on free tier. Upgrade to paid for production. |
| `multilingual-e5-large` token limit (512 tokens) silently truncates long blocks | Low | Low | Per-role blocks are typically 200–400 tokens; summary blocks rarely exceed 300. Monitor via Pinecone dashboard. |
| `candidate_name` extraction heuristic fails for non-Latin names | Medium | Low | Stored as empty string; scoring and retrieval unaffected; display shows filename instead |

---

## 12. Build Order

Dependency-ordered sequence. Each step is independently testable before moving to the next.

```
Step 1 ── src/models/resume_block.py
          Pydantic data classes (ResumeBlock, ZoneType).
          No project dependencies. Write + test first.

Step 2 ── src/config.py
          Settings class with all env vars.
          No project dependencies.

Step 3 ── src/utils/date_parser.py
          parse_date_range(), recency_weight().
          Depends only on: python-dateutil, config.py.
          Unit-testable with fixture strings — no I/O.

Step 4 ── src/services/ontology_service.py
          NetworkX skills graph, extract_skills_from_text(),
          expand_query_terms(), skills_match_score().
          Depends only on: networkx.
          Fully in-memory, no I/O.

Step 5 ── src/utils/layout_parser.py
          parse_to_elements() — Unstructured.io + python-docx.
          Depends on: config.py, unstructured, python-docx.
          Integration-test with sample PDF/DOCX fixtures.

Step 6 ── src/utils/structural_chunker.py
          build_structured_blocks() — the core transformation.
          Depends on: steps 1, 3, 4, 5.
          Unit-test with mocked Element lists + integration-test with fixture files.

Step 7 ── src/services/pinecone_service.py
          Index creation, upsert_structured_blocks(),
          query_similar_chunks(), fetch_summary_chunk(), delete_resume().
          Depends on: steps 1, 2. Test against a real Pinecone dev index.

Step 8 ── src/routers/resumes.py
          POST /upload, DELETE /{candidate_id}.
          Wires steps 5 → 6 → 7. Integration-test the full upload path.

Step 9 ── src/services/llm_service.py
          evaluate_candidate() with structured preamble,
          compute_weighted_score().
          Depends on: config.py, google-genai. Mock Gemini in unit tests.

Step 10 ─ src/routers/jobs.py
           POST /screen — two-stage retrieval, ensemble scoring.
           Depends on: steps 4, 7, 9.
           Integration-test end-to-end with seeded Pinecone index.

Step 11 ─ src/services/feedback_service.py + src/routers/feedback.py
           SQLite feedback persistence. Independent of all above.

Step 12 ─ src/main.py
           FastAPI app factory. Register all routers.
           Add startup event: _load_zone_label_embeddings().

Step 13 ─ Dockerfile + docker-compose.yml
           Multi-stage build. App + optional Unstructured.io container.

Step 14 ─ pyproject.toml + requirements.txt
           Pin all versions. Verify unstructured[pdf] installs cleanly.

Step 15 ─ tests/ fixtures/
           Compile test suite. Ensure all unit and integration tests pass
           before any deployment.
```
