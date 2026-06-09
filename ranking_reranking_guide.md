# Candidate Resume Ranking & Re-Ranking — Full Pipeline Guide

> Tailored to your existing **FastAPI + Pinecone + OpenAI** stack in `Sample RAG Server`.

---

## The Problem with Your Current Approach

Your current `jobs.py` pipeline ranks candidates by a **single LLM-generated `match_score`**. This has key weaknesses:

```python
# Current — single-signal ranking (jobs.py line 112)
results.sort(key=lambda r: r.match_score, reverse=True)
```

| Weakness | Impact |
|---|---|
| Cosine similarity alone misses exact skill keywords | "AWS SAA-C03" and "cloud certifications" score differently even if same |
| LLM scores are not calibrated across candidates (all evaluated independently) | Candidate A gets 85, Candidate B gets 84 — difference is noise, not signal |
| No pre-filtering — LLM is called for every candidate | Expensive and slow; a candidate missing 3 hard requirements wastes a call |
| Chunk-level retrieval groups by first appearance | Better chunks for a candidate may be missed if they appear after `top_k` cutoff |

---

## Full 6-Stage Pipeline

```
Resumes in Pinecone
        │
        ▼
┌─────────────────────┐
│ Stage 1: Pre-Filter │  ← Hard requirements (years exp, certs, location)
│ (Metadata Filter)   │     eliminates unqualified candidates cheaply
└────────┬────────────┘
         │
         ▼
┌─────────────────────────┐
│ Stage 2: Initial Recall │  ← Dual retrieval: Vector (Pinecone) + BM25
│ (Hybrid Retrieval)      │     wide net to maximize recall
└────────┬────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Stage 3: RRF Fusion      │  ← Merge vector + keyword results
│ (Reciprocal Rank Fusion) │     normalize two rank lists into one
└────────┬─────────────────┘
         │
         ▼
┌───────────────────────────┐
│ Stage 4: Cross-Encoder    │  ← Precise pairwise relevance scoring
│ Re-Ranking                │     "Is this resume relevant to THIS JD?"
└────────┬──────────────────┘
         │
         ▼
┌────────────────────────────────┐
│ Stage 5: Multi-Dimensional     │  ← LLM scores 6 dimensions per candidate
│ LLM Scoring                    │     only on top-N after re-ranking
└────────┬───────────────────────┘
         │
         ▼
┌───────────────────────────┐
│ Stage 6: Ensemble Ranking │  ← Weighted combination of all signals
│ (Final Score)             │     produces explainable final rank
└───────────────────────────┘
```

---

## Stage 1: Pre-Filtering (Metadata Filter)

**Goal:** Eliminate unqualified candidates *before* any embedding or LLM work.

**How it works in Pinecone:** Store structured fields as metadata at upsert time, then filter at query time.

### Tools
| Tool | Purpose |
|---|---|
| `pyresparser` | Lightweight resume parser — extracts name, skills, experience years |
| `spaCy` + custom NER | More accurate entity extraction (certifications, universities, job titles) |
| `pypdf2` / `pdfminer` | Already in your stack via `parser.py` |
| Pinecone metadata filter | Native support for `$gte`, `$in`, `$eq` operators |

### Code — Enrich metadata at upload time

Add to your **`src/utils/parser.py`** or a new `src/utils/resume_extractor.py`:

```python
# src/utils/resume_extractor.py
import re
from pyresparser import ResumeParser  # pip install pyresparser

def extract_structured_fields(resume_text: str, filepath: str) -> dict:
    """
    Extract structured fields from resume text for metadata pre-filtering.
    Returns a dict safe to store as Pinecone vector metadata.
    """
    try:
        data = ResumeParser(filepath).get_extracted_data()
    except Exception:
        data = {}

    # Calculate total years of experience from regex as fallback
    year_mentions = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)', resume_text, re.I)
    max_exp = max((int(y) for y in year_mentions), default=0)

    return {
        "skills":          data.get("skills", []),               # list[str]
        "experience_years": data.get("total_experience", max_exp), # int
        "education":       data.get("degree", ["Unknown"])[0],   # str
        "certifications":  _extract_certs(resume_text),          # list[str]
        "job_titles":      data.get("designation", []),          # list[str]
    }


_CERT_PATTERNS = [
    r'\bAWS\s+[\w\-]+\b', r'\bPMP\b', r'\bCKA\b', r'\bCKAD\b',
    r'\bGCP\s+[\w\-]+\b', r'\bAzure\s+[\w\-]+\b', r'\bCISSP\b',
    r'\bCPA\b', r'\bCFA\b', r'\bSeries\s+\d+\b',
]

def _extract_certs(text: str) -> list[str]:
    found = []
    for pat in _CERT_PATTERNS:
        found.extend(re.findall(pat, text, re.I))
    return list(set(found))
```

### Code — Update `pinecone_service.py` upsert to include structured metadata

```python
# In upsert_resume_chunks() — add structured fields to every vector's metadata
"metadata": {
    "candidate_id": candidate_id,
    "filename":     filename,
    "chunk_index":  i,
    "text":         chunk,
    # ── NEW: structured fields for pre-filtering ──
    "experience_years": structured["experience_years"],
    "education":        structured["education"],
    "skills":           structured["skills"],          # stored as list
    "certifications":   structured["certifications"],
}
```

### Code — Apply metadata filter in `jobs.py`

```python
# Build a Pinecone metadata filter from JD requirements
def build_prefilter(min_experience: int = 0, required_certs: list[str] = []) -> dict:
    filter_dict = {}
    if min_experience > 0:
        filter_dict["experience_years"] = {"$gte": min_experience}
    if required_certs:
        # Candidate must have at least one of the required certs
        filter_dict["certifications"] = {"$in": required_certs}
    return filter_dict or None

# In screen_candidates()
prefilter = build_prefilter(min_experience=5, required_certs=["PMP"])
raw_matches = pinecone_service.query_similar_chunks(
    query_embedding=query_embedding,
    top_k=body.top_k * settings.top_k_results,
    filter=prefilter,   # ← already supported in your pinecone_service.py!
)
```

> [!TIP]
> Your `query_similar_chunks()` already accepts a `filter` parameter — you just need to start passing it!

---

## Stage 2: Hybrid Retrieval (Vector + BM25)

**Goal:** Combine semantic search with exact keyword matching to improve recall.

**Why needed:** Embeddings compress meaning. "Kubernetes container orchestration" and "K8s CKA certified" may have low cosine similarity but a keyword match would catch the acronym.

### Tools
| Tool | When to use |
|---|---|
| `rank_bm25` (pure Python) | Simple drop-in, no extra infrastructure |
| `elasticsearch-py` | Production-grade, handles large corpora (10K+ resumes) |
| `opensearch-py` | AWS-native alternative to Elasticsearch |
| Pinecone sparse-dense | Pinecone's built-in hybrid — uses SPLADE sparse vectors alongside dense |

### Recommended: `rank_bm25` (zero new infrastructure)

```bash
pip install rank-bm25
```

```python
# src/services/bm25_service.py
from rank_bm25 import BM25Okapi
import re

_corpus: list[str] = []           # raw resume texts
_candidate_ids: list[str] = []    # parallel list of candidate IDs
_bm25: BM25Okapi | None = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


def index_resume(candidate_id: str, full_text: str):
    """Call this after embedding — adds resume to BM25 corpus."""
    global _bm25
    _corpus.append(full_text)
    _candidate_ids.append(candidate_id)
    _bm25 = BM25Okapi([_tokenize(doc) for doc in _corpus])


def bm25_search(query: str, top_k: int = 20) -> list[dict]:
    """
    Returns [{candidate_id, bm25_score}, ...] sorted by score descending.
    """
    if _bm25 is None or not _corpus:
        return []
    tokens = _tokenize(query)
    scores = _bm25.get_scores(tokens)
    ranked = sorted(
        zip(_candidate_ids, scores),
        key=lambda x: x[1], reverse=True
    )[:top_k]
    return [{"candidate_id": cid, "bm25_score": float(score)}
            for cid, score in ranked]
```

> [!IMPORTANT]
> `BM25Okapi` rebuilds the index on every new addition. For production with thousands of resumes, switch to Elasticsearch or Pinecone sparse-dense hybrid to avoid rebuilding.

---

## Stage 3: RRF Fusion (Merging Two Rank Lists)

**Goal:** Combine the Pinecone vector ranking and BM25 keyword ranking into a single unified rank list.

**Reciprocal Rank Fusion formula:**
```
RRF_score(candidate) = Σ  1 / (k + rank_i)
                       over all result lists
```
where `k = 60` is a smoothing constant (standard default).

### Code — `src/services/fusion_service.py`

```python
# src/services/fusion_service.py

def reciprocal_rank_fusion(
    vector_results: list[dict],   # [{candidate_id, score}, ...]
    bm25_results:   list[dict],   # [{candidate_id, bm25_score}, ...]
    k: int = 60,
) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    Returns a unified list sorted by fused score descending.
    """
    rrf_scores: dict[str, float] = {}

    for rank, item in enumerate(vector_results, start=1):
        cid = item["candidate_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank)

    for rank, item in enumerate(bm25_results, start=1):
        cid = item["candidate_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank)

    merged = [
        {"candidate_id": cid, "rrf_score": score}
        for cid, score in rrf_scores.items()
    ]
    return sorted(merged, key=lambda x: x["rrf_score"], reverse=True)
```

---

## Stage 4: Cross-Encoder Re-Ranking ⭐ (Highest Impact Stage)

**Goal:** Re-score each candidate using a model that reads **both** the JD and the resume together (not just their embeddings separately).

**Why this is the highest-impact upgrade:** A bi-encoder (like your current setup) encodes JD and resume independently and compares vectors. A **cross-encoder** reads both texts jointly — it understands context, negation, and nuance far better.

### Tools
| Tool | Model | Size | Speed | Accuracy |
|---|---|---|---|---|
| `sentence-transformers` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 23MB | Fast | ⭐⭐⭐⭐ |
| `sentence-transformers` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | 34MB | Medium | ⭐⭐⭐⭐⭐ |
| Cohere Rerank API | `rerank-english-v3.0` | Cloud | Fast | ⭐⭐⭐⭐⭐ |
| Jina AI Reranker | `jina-reranker-v2-base-en` | Cloud | Fast | ⭐⭐⭐⭐⭐ |
| Voyage AI Reranker | `rerank-2` | Cloud | Fast | ⭐⭐⭐⭐ |

### Recommended Option A — Local (no API cost): `sentence-transformers`

```bash
pip install sentence-transformers
```

```python
# src/services/reranker_service.py
from sentence_transformers import CrossEncoder

# Load once at module level — model is cached after first download (~23MB)
_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)


def rerank_candidates(
    job_description: str,
    candidates: list[dict],        # [{candidate_id, filename, resume_text}, ...]
    top_n: int = 10,
) -> list[dict]:
    """
    Score each (JD, resume) pair with a cross-encoder and re-rank.
    Returns the top_n candidates sorted by cross-encoder score descending.
    """
    if not candidates:
        return []

    # Build (query, passage) pairs for batch scoring
    pairs = [(job_description, c["resume_text"][:1024]) for c in candidates]

    scores = _model.predict(pairs)  # returns numpy array of floats

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_n]
```

### Recommended Option B — Cloud (best accuracy): Cohere Rerank

```bash
pip install cohere
```

```python
# src/services/reranker_service.py (Cohere variant)
import cohere
from src.config import settings

_client = cohere.Client(settings.cohere_api_key)

def rerank_candidates_cohere(
    job_description: str,
    candidates: list[dict],
    top_n: int = 10,
) -> list[dict]:
    documents = [c["resume_text"][:2000] for c in candidates]

    response = _client.rerank(
        model="rerank-english-v3.0",
        query=job_description,
        documents=documents,
        top_n=top_n,
    )

    reranked = []
    for result in response.results:
        c = candidates[result.index].copy()
        c["rerank_score"] = result.relevance_score
        reranked.append(c)

    return reranked
```

> [!NOTE]
> Cohere Rerank has a generous free tier (1000 calls/month) — ideal for prototyping. Switch to local cross-encoder for production to eliminate per-call cost.

---

## Stage 5: Multi-Dimensional LLM Scoring

**Goal:** Replace the single `match_score` in your `llm_service.py` with a weighted multi-dimensional scorecard.

**Current prompt asks for:** `match_score`, `strengths`, `gaps`, `summary`, `recommendation`

**Upgraded prompt asks for 6 weighted dimensions:**

```python
# Replace _SYSTEM_PROMPT in src/services/llm_service.py

_SYSTEM_PROMPT = """You are a senior talent acquisition specialist.
You will be given:
  1. A JOB DESCRIPTION written by a hiring manager.
  2. Resume excerpts from a candidate.

Evaluate the candidate across EXACTLY these 6 dimensions and return a JSON object:

{
  "technical_skills_score": <0-100>,       // Match of candidate skills to JD requirements
  "experience_relevance_score": <0-100>,   // How relevant past roles are to this position
  "experience_depth_score": <0-100>,       // Seniority, leadership, ownership demonstrated
  "education_score": <0-100>,              // Degree level and field relevance
  "certifications_score": <0-100>,         // Required/preferred certs present
  "communication_score": <0-100>,          // Clarity, impact statements, measurable results
  "strengths": ["...", "...", "..."],       // Top 3 specific strengths for THIS role
  "gaps": ["...", "...", "..."],            // Top 3 specific gaps for THIS role
  "summary": "...",                         // 2-3 sentence narrative for the recruiter
  "recommendation": "Strong Yes|Yes|Maybe|No"
}

Return ONLY the JSON object. Be precise and consistent across candidates.
"""


def compute_weighted_score(evaluation: dict) -> float:
    """
    Compute a single weighted composite score from the 6 LLM dimensions.
    Weights reflect typical hiring priorities — tune per role type.
    """
    weights = {
        "technical_skills_score":    0.30,
        "experience_relevance_score": 0.25,
        "experience_depth_score":    0.15,
        "education_score":           0.10,
        "certifications_score":      0.10,
        "communication_score":       0.10,
    }
    return sum(
        evaluation.get(dim, 0) * weight
        for dim, weight in weights.items()
    )
```

---

## Stage 6: Ensemble Final Ranking

**Goal:** Combine all signals into a single, transparent, tuneable final score.

```python
# src/services/ensemble_ranker.py

def compute_ensemble_score(
    rrf_score:      float,   # from Stage 3 (0.01 – 0.05 typical range)
    rerank_score:   float,   # from Stage 4 (0 – 10 range for cross-encoder)
    llm_score:      float,   # from Stage 5 (0 – 100 range)
    weights: dict | None = None,
) -> float:
    """
    Normalize each signal to [0, 1] then apply weights.
    Default weights lean on LLM score as the most reliable signal.
    """
    w = weights or {
        "rrf":    0.20,   # broad recall signal
        "rerank": 0.30,   # semantic precision signal
        "llm":    0.50,   # structured reasoning signal
    }

    # Normalize to [0, 1] with rough empirical bounds
    rrf_norm    = min(rrf_score / 0.05, 1.0)        # max realistic RRF ~0.05
    rerank_norm = max(min((rerank_score + 10) / 20, 1.0), 0.0)  # cross-encoder [-10,10]
    llm_norm    = llm_score / 100.0                  # already 0-100

    return (
        w["rrf"]    * rrf_norm +
        w["rerank"] * rerank_norm +
        w["llm"]    * llm_norm
    ) * 100  # scale back to 0-100 for readability


def rank_candidates(candidates: list[dict]) -> list[dict]:
    """Sort by ensemble_score descending and attach rank."""
    ranked = sorted(candidates, key=lambda c: c["ensemble_score"], reverse=True)
    for i, c in enumerate(ranked, start=1):
        c["final_rank"] = i
    return ranked
```

---

## Updated `jobs.py` — Putting It All Together

```python
@router.post("/screen", response_model=ScreenResponse)
async def screen_candidates(body: ScreenRequest):
    """
    6-Stage ranking pipeline:
      1. Pre-filter by metadata (hard requirements)
      2. Hybrid retrieval (Pinecone vector + BM25 keyword)
      3. RRF fusion
      4. Cross-encoder re-ranking (top 20)
      5. Multi-dimensional LLM scoring (top 10)
      6. Ensemble ranking → final sorted list
    """
    query_embedding = embed_text(body.job_description)

    # Stage 1+2: Hybrid retrieval with pre-filter
    prefilter = build_prefilter(
        min_experience=body.min_experience,
        required_certs=body.required_certifications,
    )
    vector_matches = pinecone_service.query_similar_chunks(
        query_embedding=query_embedding,
        top_k=50,           # wide net
        filter=prefilter,
    )
    bm25_matches = bm25_service.bm25_search(body.job_description, top_k=50)

    # Stage 3: RRF Fusion
    vector_list = [{"candidate_id": m.metadata["candidate_id"],
                    "score": m.score} for m in vector_matches]
    fused = fusion_service.reciprocal_rank_fusion(vector_list, bm25_matches)[:20]

    # Gather full resume text per candidate for re-ranking
    candidates_for_rerank = _build_candidate_contexts(fused, vector_matches)

    # Stage 4: Cross-encoder re-ranking
    reranked = reranker_service.rerank_candidates(
        job_description=body.job_description,
        candidates=candidates_for_rerank,
        top_n=10,
    )

    # Stage 5+6: LLM scoring + ensemble final rank
    results = []
    for c in reranked:
        evaluation = evaluate_candidate(
            job_description=body.job_description,
            resume_chunks=c["chunks"],
        )
        llm_score = compute_weighted_score(evaluation)
        ensemble = compute_ensemble_score(
            rrf_score=c.get("rrf_score", 0),
            rerank_score=c.get("rerank_score", 0),
            llm_score=llm_score,
        )
        results.append(CandidateResult(
            candidate_id=c["candidate_id"],
            filename=c["filename"],
            match_score=round(ensemble),
            recommendation=evaluation.get("recommendation", "No"),
            strengths=evaluation.get("strengths", []),
            gaps=evaluation.get("gaps", []),
            summary=evaluation.get("summary", ""),
            dimension_scores={            # ← new field for transparency
                "technical":     evaluation.get("technical_skills_score"),
                "relevance":     evaluation.get("experience_relevance_score"),
                "depth":         evaluation.get("experience_depth_score"),
                "education":     evaluation.get("education_score"),
                "certifications":evaluation.get("certifications_score"),
                "communication": evaluation.get("communication_score"),
            }
        ))

    results = rank_candidates(results)
    return ScreenResponse(
        job_description_snippet=body.job_description[:200],
        total_candidates_evaluated=len(results),
        results=results,
    )
```

---

## Tools Summary Table

| Stage | Tool | Install | Free? | Best For |
|---|---|---|---|---|
| Structured parsing | `pyresparser` | `pip install pyresparser` | ✅ | Small teams, quick start |
| Structured parsing | `spaCy` + custom NER | `pip install spacy` | ✅ | Custom entity types |
| BM25 keyword search | `rank-bm25` | `pip install rank-bm25` | ✅ | In-memory, <5K resumes |
| BM25 at scale | Elasticsearch | Docker / cloud | ✅ (self-host) | Production, >5K resumes |
| RRF Fusion | custom code | — | ✅ | Any scale |
| Cross-encoder (local) | `sentence-transformers` | `pip install sentence-transformers` | ✅ | No API cost, offline |
| Cross-encoder (cloud) | Cohere Rerank | `pip install cohere` | ✅ (1K/month free) | Best accuracy, low infra |
| Cross-encoder (cloud) | Jina Reranker | `pip install jina` | ✅ (free tier) | Good alternative |
| LLM scoring | OpenAI GPT-4o | Already in your stack | ❌ (pay-per-use) | Structured JSON scoring |
| LLM scoring | Gemini 1.5 Flash | `pip install google-generativeai` | ✅ (free tier) | Cost-effective at scale |
| Ensemble scoring | custom code | — | ✅ | Any scale |

---

## Implementation Priority

| Priority | Stage | Effort | Expected Accuracy Gain |
|---|---|---|---|
| ✅ Do first | Stage 1: Metadata pre-filter | 2 hrs | Eliminates ~30-40% unqualified candidates before LLM calls |
| ✅ Do first | Stage 5: Multi-dimensional LLM scoring | 1 hr | Much more explainable and accurate than single score |
| ✅ Do next | Stage 4: Cross-encoder re-ranking | 2 hrs | +20-30% ranking accuracy (highest single improvement) |
| ⏳ Then | Stage 2+3: BM25 + RRF Fusion | 3 hrs | Improves recall of keyword-specific requirements |
| ⏳ Later | Stage 6: Ensemble scoring | 2 hrs | Final polish, makes the whole system tuneable |
