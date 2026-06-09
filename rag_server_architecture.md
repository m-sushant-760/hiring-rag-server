# RAG Hiring Server — Architecture Pipeline

> **Version**: v0.2.0 · FastAPI + Pinecone Integrated Inference + Gemini 2.5 Flash  
> **Deployment**: Fly.io (serverless) · **Storage**: Pinecone (vectors) + SQLite (feedback)

---

## 1. High-Level System Overview

```mermaid
graph TB
    subgraph Client["👤 Client (Recruiter / HR)"]
        A[Resume PDF/DOCX]
        B[Job Description]
        C[HR Decision]
    end

    subgraph API["🌐 FastAPI Server"]
        R1["/api/resumes/upload<br>/api/resumes/bulk-upload"]
        R2["/api/jobs/screen"]
        R3["/api/feedback/*"]
    end

    subgraph Ingestion["📥 Ingestion Pipeline"]
        P[Parser<br>PyMuPDF / python-docx]
        S[Sectioner<br>3-pass header detection]
        CK[Chunker<br>per-section size tuning]
    end

    subgraph Retrieval["🔍 Retrieval & Scoring Pipeline"]
        OE[Ontology Service<br>Skills Graph - NetworkX]
        PC[Pinecone<br>Integrated Inference<br>multilingual-e5-large]
        LLM["LLM Service<br>Gemini 2.5 Flash<br>6-dim scoring"]
        ENS[Ensemble Scorer]
    end

    subgraph Storage["🗄️ Storage"]
        PINE[(Pinecone Index<br>hiring-rag<br>cosine / serverless)]
        SQLITE[(SQLite<br>feedback.db)]
    end

    A --> R1 --> Ingestion --> PINE
    B --> R2 --> Retrieval --> PINE
    Retrieval --> LLM
    C --> R3 --> SQLITE
    SQLITE -- "export training pairs" --> FT[Fine-tuning Dataset]
```

---

## 2. Ingestion Pipeline (Resume Upload)

### Flow: `POST /api/resumes/upload` or `/bulk-upload`

```mermaid
flowchart TD
    A([PDF or DOCX file]) --> B{Size ≤ 10 MB?}
    B -- No --> ERR1([HTTP 413])
    B -- Yes --> C[Parser\nextract_text]

    C --> D{Extension?}
    D -- .pdf --> E[PyMuPDF fitz\nlayout-aware extraction]
    D -- .docx --> F[python-docx\nparagraph join]
    D -- other --> ERR2([HTTP 415])

    E & F --> G{Text non-empty?}
    G -- No --> ERR3([HTTP 422])
    G -- Yes --> H[Sectioner\nsplit_into_sections]

    H --> I{3-Pass Header Detection}
    I -- Pass 1 --> J[Exact alias lookup\n_ALIAS_MAP]
    I -- Pass 2 --> K[Fuzzy match\nSequenceMatcher ≥ 0.82]
    I -- Pass 3 --> L[Strip icon prefix ▶●★\nretry passes 1 & 2]

    J & K & L --> M{≥ 2 canonical\nsections found?}
    M -- No --> N[Fallback:\nentire text → 'other']
    M -- Yes --> O[Dict: section_label → text]

    N & O --> P[Chunker: chunk_section\nper-section size tuning]

    P --> Q{Section type?}
    Q -- skills --> R[400 chars, 0 overlap]
    Q -- experience --> S[1200 chars, 150 overlap]
    Q -- education/summary/certs --> T[600 chars, 50 overlap]
    Q -- projects/publications --> U[800 chars, 100 overlap]
    Q -- other --> V[1000 chars, 100 overlap]

    R & S & T & U & V --> W[list of tuple\nchunk_text, section_label]

    W --> X[Name extraction\nregex heuristic\n_NAME_RE]
    X --> Y[uuid4 → candidate_id]
    Y --> Z[Pinecone upsert_records\nbatches of 100]

    Z --> AA([✅ chunks_indexed: N\ncandidate_id: uuid])
```

### Key: Pinecone Record Schema

| Field | Description |
|---|---|
| `_id` | `{candidate_id}#chunk{i}` |
| `chunk_text` | Raw text sent to Pinecone for server-side embedding |
| `candidate_id` | UUID for the resume |
| `filename` | Original filename |
| `chunk_index` | Positional index within the resume |
| `section` | Canonical label: `experience`, `skills`, `education`, etc. |
| `candidate_name` | Heuristically extracted from the first lines |

---

## 3. Retrieval & Screening Pipeline (Job Screening)

### Flow: `POST /api/jobs/screen`

```mermaid
flowchart TD
    JD([Job Description text]) --> A

    subgraph StepA["A · JD Skill Detection (Phase 8)"]
        A[extract_skills_from_text\nontology_service] --> B[List of JD skills\nmatched against ontology nodes]
    end

    B --> C
    subgraph StepB["B · Query Expansion via Ontology (Phase 8)"]
        C{use_ontology?}
        C -- Yes --> D[expand_query_terms\nmax_hops=1\nNetworkX BFS]
        C -- No --> E[Use raw JD text]
        D --> F[Augmented Query =\nJD + expanded skill list]
        E --> F
    end

    F --> G
    subgraph StepC["C · Vector Retrieval (Pinecone)"]
        G[query_similar_chunks\nPinecone server-side embed\nmultilingual-e5-large] --> H{section_filter set?}
        H -- Yes --> I[Metadata pre-filter:\nsection = value]
        H -- No --> J[All sections retrieved]
        I & J --> K[Top K × top_k_results hits\nranked by cosine similarity]
    end

    K --> L
    subgraph StepD["D · Group & Pre-Rank"]
        L[_group_by_candidate\nGroup hits by candidate_id] --> M[Track: best score,\nchunks, sections_retrieved,\nchunks_by_section, candidate_name]
        M --> N[Sort by best Pinecone score\nSlice to top_k candidates]
    end

    N --> O
    subgraph StepE["E · Per-Candidate Evaluation"]
        O[For each candidate] --> O1 & O2 & O3

        O1[Phase 8 — Ontology Score\nextract_skills from chunks\nskills_match_score vs JD skills\nmax_hops=2]
        O2[LLM Evaluation — Employer\nevaluate_candidate\nGemini 2.5 Flash\n6-dimension JSON scoring]
        O3[Phase 9 — Bi-dir Interest\n_candidate_interest_score\nGemini: candidate perspective\nprefer experience+summary chunks]

        O1 & O2 & O3 --> P[compute_weighted_score\nLLM dims × fixed weights]
        P --> Q[_ensemble\nwith bidirectional:\nLLM 50% + Ontology 20% + Bidir 30%\nwithout:\nLLM 60% + Ontology 40%]
    end

    Q --> R
    subgraph StepF["F · Final Ranking"]
        R[Sort all candidates\nby match_score desc] --> S[Assign final_rank 1…N]
    end

    S --> T([ScreenResponse\nranked CandidateResult list])
```

### LLM Dimension Weights (Employer Score)

| Dimension | Weight |
|---|---|
| `technical_skills_score` | 30% |
| `experience_relevance_score` | 25% |
| `experience_depth_score` | 15% |
| `education_score` | 10% |
| `certifications_score` | 10% |
| `communication_score` | 10% |

### Ensemble Score Formula

```
With bidirectional ON:
  bidir_score  = 0.70 × employer_score + 0.30 × candidate_interest_score
  final_score  = 0.50 × employer_score + 0.20 × (ontology_score × 100) + 0.30 × bidir_score

Without bidirectional:
  final_score  = 0.60 × employer_score + 0.40 × (ontology_score × 100)
```

---

## 4. Skills Ontology Graph (Phase 8)

The ontology is an in-memory directed graph (`networkx.DiGraph`) with **three edge types**:

| Edge Type | Example |
|---|---|
| `IS_A` | `FastAPI → REST Framework → REST API → API Development` |
| `ALIAS` | `K8s → Kubernetes` |
| `RELATED_TO` | `Deep Learning → MLOps` |

### Ontology Coverage

| Domain | Examples |
|---|---|
| API / Backend | REST API, GraphQL, gRPC, FastAPI, Flask, Spring Boot |
| Python Ecosystem | Django, SQLAlchemy, Pydantic, Celery, NumPy, Pandas |
| Cloud & Infra | AWS, GCP, Azure, Fly.io, Lambda, S3, EC2 |
| Container / DevOps | Docker, Kubernetes, Terraform, Helm, GitHub Actions |
| Data Engineering | Kafka, Spark, Airflow, dbt, Snowflake, BigQuery |
| ML / AI | PyTorch, TensorFlow, Scikit-learn, XGBoost, Hugging Face |
| LLM / Gen AI | LangChain, RAG, Pinecone, Weaviate, Prompt Engineering |
| Frontend | React, Vue.js, Next.js, TypeScript |
| Leadership | Scrum, Agile, Mentoring, Engineering Manager |

### How Query Expansion Works

```
JD skill: "REST API development"
  ↳ exact match: "REST API"
  ↳ 1-hop BFS: REST Framework, API Development
  ↳ children of REST Framework: FastAPI, Flask, Django REST Framework,
                                  Express.js, Spring Boot, ASP.NET Core, Rails API
Result: augmented query now surfaces candidates listing FastAPI even if
        the JD only says "REST API"
```

---

## 5. HR Feedback Loop (Phase 10)

```mermaid
sequenceDiagram
    participant HR as HR Recruiter
    participant API as FastAPI
    participant FB as Feedback Service
    participant DB as SQLite (feedback.db)
    participant FT as Fine-tuning

    HR->>API: POST /api/jobs/screen
    API-->>HR: Ranked candidates

    HR->>API: POST /api/feedback/decision\n{job_id, candidate_id, decision: accepted|shortlisted|rejected}
    API->>FB: record_decision()
    FB->>DB: INSERT INTO candidate_feedback

    HR->>API: GET /api/feedback/stats
    API->>DB: COUNT, GROUP BY decision
    API-->>HR: {total, by_decision, avg_match_score}

    HR->>API: GET /api/feedback/export?min_pairs=50
    API->>DB: SELECT job_description, resume_text, decision
    API-->>HR: {pairs: [{query, document, label}], ready_to_finetune: true}

    HR->>FT: Fine-tune CrossEncoder\n(cross-encoder/ms-marco-MiniLM-L-6-v2)
```

### Feedback DB Schema

```sql
CREATE TABLE candidate_feedback (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           TEXT NOT NULL,
    job_description  TEXT NOT NULL,
    candidate_id     TEXT NOT NULL,
    filename         TEXT DEFAULT '',
    resume_text      TEXT DEFAULT '',
    decision         TEXT CHECK(decision IN ('accepted','shortlisted','rejected')),
    match_score      REAL,
    dimension_scores TEXT,   -- JSON blob
    notes            TEXT,
    recruiter_id     TEXT,
    created_at       TEXT
);
```

### Export Label Mapping

| HR Decision | Fine-tune Label |
|---|---|
| `accepted` | `1` |
| `shortlisted` | `1` |
| `rejected` | `0` |

---

## 6. API Surface

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check / liveness probe |
| `POST` | `/api/resumes/upload` | Single resume upload (PDF/DOCX) |
| `POST` | `/api/resumes/bulk-upload` | Batch upload (up to 20 files) |
| `DELETE` | `/api/resumes/{candidate_id}` | Remove all vectors for a candidate |
| `POST` | `/api/jobs/screen` | Screen all indexed resumes against a JD |
| `POST` | `/api/feedback/decision` | Record one HR decision |
| `POST` | `/api/feedback/bulk` | Record multiple HR decisions |
| `GET` | `/api/feedback/stats` | Summary statistics |
| `GET` | `/api/feedback/export` | Export labelled pairs for fine-tuning |
| `GET` | `/api/feedback/decisions` | Query stored decisions (filterable) |

---

## 7. Component Map

```
Sample RAG Server/
├── src/
│   ├── main.py              FastAPI app, CORS middleware, router registration
│   ├── config.py            Centralised Settings from .env (chunk sizes, API keys, model names)
│   │
│   ├── routers/
│   │   ├── resumes.py       Upload pipeline orchestrator; single + bulk upload; delete
│   │   ├── jobs.py          Screening pipeline orchestrator; ontology + LLM + ensemble
│   │   └── feedback.py      Feedback CRUD; export endpoint
│   │
│   ├── services/
│   │   ├── pinecone_service.py   Lazy index creation; upsert_records; search; delete by prefix
│   │   ├── llm_service.py        Gemini client; 6-dimension prompt; weighted score
│   │   ├── ontology_service.py   NetworkX graph; expand_query_terms; skills_match_score
│   │   ├── feedback_service.py   SQLite persistence; export training pairs; stats
│   │   └── embedding_service.py  Stub — Pinecone handles embedding internally
│   │
│   └── utils/
│       ├── parser.py        PyMuPDF (PDF) + python-docx (DOCX) text extraction
│       ├── sectioner.py     3-pass section header classifier; fuzzy matching; fallback
│       └── chunker.py       RecursiveCharacterTextSplitter; per-section chunk params
│
└── tests/
    ├── test_api.py              Mocked endpoint integration tests
    ├── test_bulk_upload.py      Bulk upload edge cases
    ├── test_chunker.py          Chunking edge cases
    ├── test_parser.py           In-memory PDF/DOCX parsing
    ├── test_pinecone_service.py Pinecone service unit tests
    └── test_feedback.py         Feedback service + endpoint tests
```

---

## 8. Configuration Reference

All values configurable via `.env` (no code changes needed):

| Env Variable | Default | Description |
|---|---|---|
| `PINECONE_API_KEY` | — | Pinecone API key |
| `PINECONE_INDEX_NAME` | `hiring-rag` | Vector index name |
| `PINECONE_CLOUD` | `aws` | Cloud provider |
| `PINECONE_REGION` | `us-east-1` | Index region |
| `PINECONE_EMBEDDING_MODEL` | `multilingual-e5-large` | Server-side embedding model |
| `GOOGLE_API_KEY` | — | Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Generation model |
| `MAX_UPLOAD_SIZE_MB` | `10` | Per-file size limit |
| `TOP_K_RESULTS` | `10` | Default Pinecone retrieval k |
| `CHUNK_SIZE_SKILLS` | `400` | Skills section chunk size |
| `CHUNK_SIZE_EXPERIENCE` | `1200` | Experience section chunk size |
| `CHUNK_SIZE_DEFAULT` | `1000` | Default section chunk size |
| `CHUNK_OVERLAP_DEFAULT` | `100` | Default chunk overlap |
| `FEEDBACK_DB_PATH` | `feedback.db` | SQLite database path |

---

## 9. Deployment Topology

```mermaid
graph LR
    subgraph Fly["☁️ Fly.io (iad region)"]
        APP[FastAPI App\nuvicorn + Docker\nauto-stop enabled]
        DB2[("SQLite\nfeedback.db\n(ephemeral)")]
        APP --- DB2
    end

    subgraph External["External Services (Free Tier)"]
        PC2[("Pinecone Serverless\nStarter Plan\naws / us-east-1\n2GB / 5M tokens/month")]
        GEM[Google AI Studio\nGemini 2.5 Flash\n1,500 req/day]
    end

    APP -- "HTTPS REST" --> PC2
    APP -- "HTTPS REST" --> GEM
    Client([Client]) -- "HTTPS" --> APP
```

> [!WARNING]
> SQLite (`feedback.db`) on Fly.io is **ephemeral** — it resets on redeploy. For production, swap `DB_PATH` to a mounted volume or migrate to PostgreSQL.

> [!TIP]
> Pinecone delete on the Starter (free) plan does **not** support filter-based deletes. The service works around this by using `index.list(prefix=candidate_id#)` to discover chunk IDs, then deletes them by explicit ID list.

---

## 10. Data Flow Summary

```
                      INGESTION
File ──► Parser ──► Sectioner ──► Chunker ──► Pinecone (embed + store)
         PDF/DOCX   9 sections   per-section  multilingual-e5-large
                    fuzzy match  size tuning  cosine, serverless

                      SCREENING
Job Description
  ──► Ontology (detect + expand JD skills)
  ──► Pinecone query (augmented query, optional section_filter)
  ──► Group hits by candidate
  ──► Per candidate:
        Ontology score  (skills_match_score, max_hops=2)
        LLM score       (Gemini, 6 dims, employer perspective)
        Interest score  (Gemini, candidate perspective, Phase 9)
        Ensemble        (50/20/30 or 60/40 weights)
  ──► Sort & rank ──► ScreenResponse

                      FEEDBACK LOOP
HR Decision ──► SQLite ──► export_training_pairs
                            ──► CrossEncoder fine-tuning dataset
```
