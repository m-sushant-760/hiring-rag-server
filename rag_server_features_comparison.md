# RAG Hiring Server — Features, Benefits & Version Comparison

> **Baseline**: v0.1.x — Basic RAG (parse → chunk → embed → retrieve → LLM score)  
> **Current**: v0.2.0 — Agentic RAG with Ontology, Bi-directional Matching, Feedback Loop & Section-Aware Chunking

---

## 1. Version-at-a-Glance

| Capability | v0.1.x (Baseline) | v0.2.0 (Current) |
|---|:---:|:---:|
| Resume upload (single) | ✅ | ✅ |
| Resume upload (bulk, up to 20) | ❌ | ✅ |
| Resume delete | ✅ | ✅ |
| Fixed-size text chunking | ✅ | ✅ (kept for compat) |
| **Section-aware chunking** | ❌ | ✅ Phase A |
| **Section header detection** | ❌ | ✅ Phase A |
| Pinecone Integrated Inference | ✅ | ✅ |
| Basic LLM scoring (Gemini) | ✅ | ✅ |
| 6-dimension LLM scoring | ✅ | ✅ |
| **Skills Ontology Graph** | ❌ | ✅ Phase 8 |
| **JD Query Expansion** | ❌ | ✅ Phase 8 |
| **Ontology skill match score** | ❌ | ✅ Phase 8 |
| **Bi-directional candidate matching** | ❌ | ✅ Phase 9 |
| **Ensemble scoring (3-source)** | ❌ | ✅ Phase 9 |
| **HR Feedback recording** | ❌ | ✅ Phase 10 |
| **Fine-tuning data export** | ❌ | ✅ Phase 10 |
| Section-filtered retrieval | ❌ | ✅ Phase A |
| Candidate name attribution | ❌ | ✅ Phase A |
| Section metadata in results | ❌ | ✅ Phase A |
| Pre-filter by experience / certs | ❌ | ✅ |
| Swagger UI (`/docs`) | ✅ | ✅ |
| Fly.io deployment | ✅ | ✅ |
| Zero paid API keys required | ✅ | ✅ |

---

## 2. Feature Deep-Dive & Benefits

### 2.1 — Phase 8: Skills Ontology Graph

#### What it is
An in-memory directed graph (`networkx.DiGraph`) of ~100+ technology nodes connected by `IS_A`, `ALIAS`, and `RELATED_TO` edges. Built once at startup, zero extra infrastructure.

#### Feature breakdown

| Feature | v0.1.x | v0.2.0 |
|---|---|---|
| Skill detection in JD | ❌ Raw text only | ✅ `extract_skills_from_text` scans ontology nodes |
| Query expansion | ❌ JD sent verbatim | ✅ `expand_query_terms` BFS expands up to 1 hop |
| Structured skill overlap score | ❌ | ✅ `skills_match_score` (0.0–1.0), max_hops=2 |
| Alias resolution | ❌ | ✅ `K8s` → `Kubernetes` automatically |
| Runtime ontology extension | ❌ | ✅ `add_custom_skill(skill, parent)` |
| `jd_skills_detected` in response | ❌ | ✅ Returned in `ScreenResponse` |
| `matched_jd_skills` per candidate | ❌ | ✅ Returned in `CandidateResult` |
| `expanded_skills` per candidate | ❌ | ✅ Top 20 expansions returned |

#### Key benefit
Pure embedding similarity often misses **specific-tool ↔ broad-category** matches. The ontology ensures:
```
JD says: "REST API development"
Candidate lists: "FastAPI"

v0.1.x: Low similarity → candidate may be buried
v0.2.0: FastAPI → REST Framework → REST API (2-hop) → full credit
```

#### Coverage domains
Cloud, Python, Data Engineering, ML/AI, LLM/GenAI, DevOps, Frontend, Leadership, Certifications (AWS SA, CKA, CKAD, PMP, GCP Pro)

---

### 2.2 — Phase 9: Bi-directional Candidate Matching

#### What it is
Alongside the traditional employer-fit score, a second Gemini call assesses whether the **candidate would genuinely want** this role — penalising over-qualification and career trajectory mismatches.

#### Feature breakdown

| Feature | v0.1.x | v0.2.0 |
|---|---|---|
| Employer-perspective scoring | ✅ | ✅ |
| Candidate-perspective scoring | ❌ | ✅ `_candidate_interest_score` |
| Career trajectory analysis | ❌ | ✅ Gemini assesses step-up vs. step-down |
| Over-qualification detection | ❌ | ✅ Penalises trajectory mismatch |
| Ensemble final score | ❌ (LLM only) | ✅ 3-source weighted blend |
| `bidirectional` toggle | ❌ | ✅ Optional — defaults to `true` |
| `candidate_interest_score` in output | ❌ | ✅ 0–100 in `CandidateResult` |
| Preferred chunk selection | ❌ | ✅ Uses `experience` + `summary` chunks (Phase A) |

#### Scoring formula comparison

**v0.1.x** (single-source):
```
final_score = LLM weighted score (6 dims)
```

**v0.2.0** (ensemble):
```
# With bidirectional ON (default):
bidir_score  = 0.70 × employer_score + 0.30 × candidate_interest
final_score  = 0.50 × employer_score
             + 0.20 × (ontology_score × 100)
             + 0.30 × bidir_score

# With bidirectional OFF:
final_score  = 0.60 × employer_score
             + 0.40 × (ontology_score × 100)
```

#### Key benefit
Prevents placing a Principal Engineer into a junior IC role, and avoids surfacing candidates whose trajectory clearly points away from the role — reducing unnecessary interview rounds.

---

### 2.3 — Phase 10: HR Feedback Loop

#### What it is
A SQLite-backed persistence layer that captures recruiter accept/reject/shortlist decisions and exports labelled `(job_description, resume_text, label)` pairs for cross-encoder fine-tuning.

#### Feature breakdown

| Feature | v0.1.x | v0.2.0 |
|---|---|---|
| HR decision recording (single) | ❌ | ✅ `POST /api/feedback/decision` |
| HR decision recording (bulk) | ❌ | ✅ `POST /api/feedback/bulk` |
| Decision query / audit trail | ❌ | ✅ `GET /api/feedback/decisions` (filterable) |
| Summary statistics | ❌ | ✅ `GET /api/feedback/stats` |
| Fine-tuning dataset export | ❌ | ✅ `GET /api/feedback/export` |
| `ready_to_finetune` readiness flag | ❌ | ✅ Auto-set when ≥ N pairs collected |
| Dimension scores storage | ❌ | ✅ Stored as JSON blob |
| Recruiter attribution | ❌ | ✅ `recruiter_id` field |
| Notes / free-text annotations | ❌ | ✅ `notes` field |
| SQLite schema (zero infra) | ❌ | ✅ Auto-created on import |

#### Feedback label mapping

| HR Decision | Fine-tune Label | Meaning |
|---|:---:|---|
| `accepted` | `1` | Strong positive signal |
| `shortlisted` | `1` | Positive (not yet hired but qualified) |
| `rejected` | `0` | Negative signal |

#### Key benefit
Turns every hiring cycle into a **self-improving training dataset**. Once ≥ 50 labelled pairs accumulate, a `CrossEncoder` can be fine-tuned on domain-specific hiring signals — reducing the model's reliance on generic pre-trained weights over time.

---

### 2.4 — Phase A: Section-Aware Chunking

#### What it is
Resumes are split into canonical sections before chunking, with each section using independently tuned chunk sizes. Every Pinecone record carries a `section` metadata tag enabling filtered retrieval.

#### Feature breakdown

| Feature | v0.1.x | v0.2.0 |
|---|---|---|
| Uniform 1000-char chunks | ✅ (only option) | ✅ (fallback) |
| Section header detection | ❌ | ✅ 3-pass: exact → fuzzy → icon-stripped |
| Fuzzy header matching | ❌ | ✅ SequenceMatcher ≥ 0.82 (catches typos) |
| Icon/bullet prefix stripping | ❌ | ✅ (`▶`, `●`, `★`, `•`, `–`, `—`) |
| Per-section chunk sizes | ❌ | ✅ Tuned per section type |
| `section` metadata on records | ❌ | ✅ Stored in Pinecone |
| `section_filter` on `/screen` | ❌ | ✅ Restricts retrieval to one section |
| `sections_retrieved` in output | ❌ | ✅ List of sections that contributed |
| `candidate_name` extraction | ❌ | ✅ Regex heuristic from resume header |
| `candidate_name` in output | ❌ | ✅ `CandidateResult.candidate_name` |
| Bulk upload (up to 20 files) | ❌ | ✅ `POST /api/resumes/bulk-upload` |
| Soft per-file error handling | ❌ | ✅ Failed files don't abort batch |

#### Per-section chunk tuning

| Section | Chunk Size | Overlap | Rationale |
|---|:---:|:---:|---|
| `skills` | 400 chars | 0 | Dense bullet lists — small, atomic chunks |
| `experience` | 1200 chars | 150 | Role narratives need context preservation |
| `education` | 600 chars | 50 | Short, factual blocks |
| `summary` | 600 chars | 50 | Usually single paragraph |
| `certifications` | 600 chars | 50 | Compact credential lists |
| `projects` | 800 chars | 100 | Medium narratives |
| `publications` | 800 chars | 100 | Medium narratives |
| `languages` | 400 chars | 0 | Very compact |
| `awards` | 400 chars | 0 | Very compact |
| `other` (fallback) | 1000 chars | 100 | Previous default |

#### Key benefit
In v0.1.x, a 1000-char chunk might blend a candidate's job title with unrelated bullet points from the next role. Section-aware chunking ensures:
- Skills bullets stay in dense, independently-searchable chunks
- Experience narratives keep enough context (150-char overlap) to preserve which role each achievement belongs to
- Retrieval can be **surgically scoped** — e.g., `section_filter: "skills"` for a skills-only search

---

## 3. API Surface Comparison

### New endpoints in v0.2.0

| Method | Path | Phase | Description |
|---|---|---|---|
| `POST` | `/api/resumes/bulk-upload` | A | Upload up to 20 resumes in one multipart call |
| `POST` | `/api/feedback/decision` | 10 | Record a single HR decision |
| `POST` | `/api/feedback/bulk` | 10 | Record multiple HR decisions |
| `GET` | `/api/feedback/stats` | 10 | Decision summary statistics |
| `GET` | `/api/feedback/export` | 10 | Export fine-tuning pairs |
| `GET` | `/api/feedback/decisions` | 10 | Query decisions with filters |

### Enhanced endpoints in v0.2.0

| Method | Path | What changed |
|---|---|---|
| `POST` | `/api/resumes/upload` | Ingestion refactored through `_process_single_resume`; section-aware chunking; `candidate_name` extraction |
| `DELETE` | `/api/resumes/{candidate_id}` | Delete now uses `index.list(prefix=…)` + batch delete (Starter plan compatible) |
| `POST` | `/api/jobs/screen` | 5 new request fields; 6 new response fields per candidate; ontology + bi-dir scoring; pre-filter support |

### New `ScreenRequest` fields (v0.2.0)

| Field | Type | Default | Description |
|---|---|---|---|
| `min_experience` | `int` | `0` | Metadata pre-filter: min years experience |
| `required_certifications` | `list[str]` | `[]` | Metadata pre-filter: must-have certs |
| `use_ontology` | `bool` | `true` | Toggle Phase 8 query expansion |
| `bidirectional` | `bool` | `true` | Toggle Phase 9 candidate interest score |
| `section_filter` | `str \| None` | `None` | Restrict retrieval to one section |

### New `CandidateResult` fields (v0.2.0)

| Field | Phase | Description |
|---|---|---|
| `candidate_interest_score` | 9 | Gemini's 0–100 candidate-perspective score |
| `ontology_skill_score` | 8 | Structured skill overlap % |
| `matched_jd_skills` | 8 | JD skills found verbatim in candidate text |
| `expanded_skills` | 8 | Top 20 ontology expansions applied |
| `candidate_name` | A | Name extracted from resume header |
| `sections_retrieved` | A | Sections that contributed chunks to ranking |

---

## 4. Retrieval Quality Improvements

| Problem in v0.1.x | Solution in v0.2.0 | Phase |
|---|---|---|
| "FastAPI" résumé not matched to "REST API" JD | Ontology expansion: FastAPI → REST Framework → REST API | 8 |
| "K8s" not matched to "Kubernetes" requirement | ALIAS edge in ontology graph | 8 |
| Semantic drift for specific tools vs. broad categories | Structured `skills_match_score` (0–1) as a separate signal | 8 |
| Overqualified candidates ranked highly | Bi-directional interest score penalises mismatch | 9 |
| Single-source score (LLM only) too noisy | 3-source ensemble (LLM + Ontology + Bi-dir) | 9 |
| Skills and experience text mixed in same chunks | Section-aware chunking separates semantic contexts | A |
| Experience chunks lose cross-role context | 150-char overlap on experience chunks | A |
| No way to search "show me only skills sections" | `section_filter` on `/api/jobs/screen` | A |
| Candidate insertion-order bias in grouping | Sort by best Pinecone score before `top_k` slice | A |

---

## 5. Infrastructure & Compatibility

| Concern | v0.1.x | v0.2.0 |
|---|---|---|
| Paid API keys required | None | None |
| New runtime dependencies | — | `networkx>=3.0` (ontology) |
| Embedding API calls | None (Pinecone Integrated) | None (unchanged) |
| SQLite auto-creation | ❌ | ✅ On module import |
| Pinecone Starter-plan delete | ❌ (filter delete unsupported) | ✅ Prefix-list + ID batch delete |
| Old records without `section` tag | ❌ Would break | ✅ Backward compatible — fields simply absent |
| `chunk_text()` function | ✅ | ✅ Preserved for backward compat |
| Env-variable tuning (no code change) | Partial | ✅ All chunk sizes, DB path, model names |

---

## 6. Test Coverage Added in v0.2.0

| Test File | New Test Classes / Coverage |
|---|---|
| `test_bulk_upload.py` | `TestSingleUploadAfterRefactor`, `TestBulkUploadSuccess`, `TestBulkUploadPartialFailure`, `TestBulkUploadValidation`, `TestBulkUploadResponseShape` |
| `test_feedback.py` | Feedback decision recording, bulk decisions, stats, export, audit query |
| `test_pinecone_service.py` | Prefix-list delete, section metadata in upsert/query |
| `test_api.py` | Pinecone score-sort regression, delete no-op, unsupported format |
| `test_chunker.py` | Per-section parameter verification |

### Key regression tests

| Test | What it guards against |
|---|---|
| `test_screen_candidates_sorted_by_pinecone_score` | Insertion-order bias — high-score candidate arriving 2nd in Pinecone hits must still win |
| `test_unsupported_file_fails_softly` | Single bad file in bulk batch must not abort the rest |
| `test_all_files_fail_returns_200_with_zero_succeeded` | Full-failure batch still returns HTTP 200 (soft error contract) |
| `test_delete_unknown_candidate_is_noop` | Unknown `candidate_id` delete must not raise an error |
| `test_exactly_max_files_is_accepted` | Boundary test: exactly 20 files must succeed, 21 must fail |

---

## 7. Summary: Why upgrade to v0.2.0?

> [!IMPORTANT]
> All four phases work together as a cohesive system — not as isolated add-ons.

| Dimension | Improvement |
|---|---|
| **Recall** | Ontology query expansion surfaces candidates who use specific tools when the JD uses category names — reducing false negatives |
| **Precision** | Ensemble scoring (3 sources) is harder to game than a single LLM call — reducing false positives |
| **Fairness** | Bi-directional matching penalises over-qualification, producing more realistic shortlists |
| **Relevance** | Section-aware chunking ensures the right semantic context reaches the LLM for evaluation |
| **Scalability** | Bulk upload (20 files/request) enables ATS integration without sequential round-trips |
| **Operability** | All chunk sizes, model names, and DB paths are env-configurable — no code changes needed for tuning |
| **Continuous Improvement** | Feedback loop turns every hire/reject decision into a structured training dataset for future re-ranking |
| **Auditability** | `sections_retrieved`, `candidate_name`, `matched_jd_skills`, and `dimension_scores` make every ranking decision explainable |
