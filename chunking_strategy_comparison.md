# Chunking Strategy Comparison
## Section-Aware (Plan A) vs. Layout-Aware Structural Parsing (Suggested)

> **Reference**: [Section-Aware Chunking Change Analysis](section_aware_chunking_analysis.md)
> **Scope**: Analysis only. No code has been changed.

---

## Quick Reference — Strategy Definitions

| | **Strategy A — Section-Aware Chunking** | **Strategy B — Layout-Aware + Per-Experience Chunking** |
|---|---|---|
| **Parser** | PyMuPDF (current) + regex section splitter | Azure AI Doc Intelligence / Unstructured.io / Marker |
| **Split unit** | Resume section (`experience`, `skills`, etc.) | Individual job-role blocks + discrete zones |
| **Chunk granularity** | Section → overlapping character windows | One chunk per job role / project |
| **Metadata enrichment** | `section`, `candidate_name` | `job_title`, `company`, `duration_months`, `recency_year`, `skills_extracted` |
| **Retrieval** | Vector search + optional `section` filter | Hybrid: vector search + metadata keyword filters |
| **LLM context** | Top-K chunks passed to evaluator | Small-to-large: project chunk + full summary passed together |
| **Implementation cost** | Low–Medium (6 files, no new infra) | High (new parsing service / API dependency) |

---

## 1. Strategy A — Section-Aware Chunking

*As described in the [change analysis document](section_aware_chunking_analysis.md).*

### ✅ Benefits

**1. No new infrastructure dependency**
Uses the existing PyMuPDF extraction pipeline. The new `sectioner.py` is pure Python regex — no external API key, no network call, zero added latency at index time.

**2. Eliminates cross-section bleed**
The primary failure mode of the current fixed-size strategy — an "experience" bullet leaking into an "education" chunk — is fully solved. The `section` metadata filter then allows targeted retrieval (e.g., query only the `skills` section for a skills-heavy JD).

**3. Per-section chunk size tuning is a real win**
A `skills` section is a dense bullet list; 400-char chunks with no overlap produce sharper vectors. An `experience` section is narrative; 1,200-char chunks with 150-char overlap preserve role context. This is not possible in the current strategy without section awareness.

**4. Backward compatible and low-risk rollout**
Old vectors without `section` metadata still return from Pinecone queries — they're just not filterable. The system degrades gracefully. No index recreation required.

**5. Directly improves the existing ontology and LLM layers**
`ontology_service.extract_skills_from_text` called on a clean `skills`-section chunk will produce far fewer false positives than when called on a mixed blob of text.

**6. Fast, deterministic, testable**
Regex section detection is unit-testable with 100% coverage. There are no flaky API calls at index time.

### ⚠️ Limitations

**1. Still character/token splits within sections**
The `experience` section (which may contain 3–5 distinct jobs) is still chunked by character count. A 1,200-char window may split the boundary between Company A's role and Company B's role — the core fragmentation problem Strategy B directly solves.

**2. Regex section detection is fragile against non-standard headers**
See Section 3 below for the full treatment of this problem.

**3. Metadata enrichment is shallow**
The only new metadata fields are `section` and `candidate_name`. There is no `company`, `job_title`, `duration_months`, or `recency_year`. This means:
- You cannot filter by recency (e.g., "show candidates whose most recent role was post-2023").
- The LLM receives chunks without explicit temporal context — it must infer recency from the text content alone.

**4. "Small-to-large" retrieval not implemented**
When a specific project chunk is the top hit, the evaluator still only receives the top-K chunks by vector score. It does not automatically pull in the candidate's professional summary for context. The LLM may score a strong technical match without the full career narrative.

**5. Multi-column layout corruption survives into chunking**
PyMuPDF with `page.get_text("text")` is much better than PyPDF2 but still scrambles some two-column resume layouts. A corrupted text stream makes both regex section detection and chunking less reliable.

---

## 2. Strategy B — Layout-Aware + Per-Experience-Block Chunking

### ✅ Benefits

**1. Solves the root cause: layout corruption**
Tools like **Unstructured.io** or **Azure AI Document Intelligence** read a two-column resume as two logical columns, not as scrambled horizontal rows. This is the foundational fix — downstream chunking quality improves automatically because the input text is structurally clean.

**2. Per-job-role chunks preserve semantic unity**
"Designed a serverless pipeline using Python Durable Functions…" stays bound to "Senior Data Engineer at OrbitLabs, 2024–2026". The job title, company, and bullet points are never separated. This is the single biggest accuracy improvement for the vector search — the embedding of a coherent work block is far more discriminative than an embedding of a random character window.

**3. Rich metadata enables powerful hybrid retrieval**
With `recency_year`, `duration_months`, `company`, and `job_title` in metadata, Pinecone queries can combine semantic similarity with hard business-rule filters:
```
query: "distributed systems experience"
filter: recency_year >= 2023 AND duration_months >= 12
```
This is impossible with Strategy A's metadata schema.

**4. Small-to-large context window is the right LLM input**
Passing `[professional_summary_chunk] + [matched_experience_chunk]` to `evaluate_candidate` and `_candidate_interest_score` is strictly better than passing the top-4 arbitrary chunks. The LLM gets career trajectory context plus the precise technical block that matched — matching how a human recruiter reads a resume.

**5. `skills_extracted` per-chunk enables attribution**
Knowing that "Python, Azure Durable Functions, Bicep" came from a 14-month senior role in 2026 is much more useful to the ontology scorer than knowing they appear somewhere in the resume. The `skills_match_score` in `ontology_service.py` could be weighted by recency.

### ⚠️ Limitations

**1. Major infrastructure dependency**
Unstructured.io (self-hosted or API), Azure AI Document Intelligence, and Marker all introduce:
- A new external API key / cost ($$$) or a self-hosted container.
- Network latency on every resume upload.
- A new failure mode: if the layout parser API is down, resume uploads fail entirely.
- A SaaS vendor lock-in concern (especially Azure AI Doc Intelligence).

**2. Per-role chunking requires structured entity extraction**
To create one chunk *per job role*, you must first identify where each job role begins and ends. This requires either:
- A second LLM call at index time (expensive, slow, non-deterministic), or
- A layout parser that returns structured JSON with `{job_title, company, dates, bullets}` fields (only Azure AI Doc Intelligence and Unstructured.io with fine-tuned models do this reliably).

A regex heuristic for "role boundary detection" inside an experience section is even harder than section header detection — roles don't have consistent delimiters.

**3. Skills extraction at chunk level needs its own pipeline**
`"skills_extracted": ["Python", "Azure Durable Functions", "Bicep"]` must be populated at index time. Options:
- Run `ontology_service.extract_skills_from_text(chunk_text)` per chunk — already in the codebase, feasible.
- Run a separate NER model — adds latency and complexity.

**4. Recency and duration parsing is brittle**
Dates on resumes appear in dozens of formats: "Jan 2024 – Present", "2022/03 – 2023/07", "March '21 to now". A robust date parser is a non-trivial component. Failure to parse a date means `recency_year` and `duration_months` are null — and a filter on null fields returns zero results.

**5. High implementation cost relative to current codebase**
This is not a 6-file change. It requires:
- Replacing or wrapping `parser.py` with a layout-aware library.
- A new entity extraction pipeline (role boundaries, dates, companies).
- A new metadata enrichment layer.
- Significant changes to `pinecone_service.py`, `resumes.py`, and `jobs.py`.
- New test fixtures for structured resume JSON.

This is a Phase 10+ effort, not an incremental improvement.

**6. Risk of over-segmentation on short or poorly formatted resumes**
A resume with only one job role produces one experience chunk. A fresh graduate's resume might produce zero valid experience chunks. The pipeline needs fallback logic for every degenerate case.

---

## 3. The Variable Section Title Problem

> *"In case the section titles are different for each resume, how will section-aware chunking work?"*

This is the most important practical weakness of Strategy A and deserves its own section.

### The Real-World Distribution of Section Titles

Resumes in the wild use an enormous variety of header labels for the same semantic zone:

| Semantic Zone | Seen in Practice |
|---|---|
| Work experience | "Experience", "Work History", "Career History", "Professional Experience", "Employment", "Relevant Experience", "Industry Experience", "Work Experience", "Career Summary" |
| Skills | "Technical Skills", "Core Competencies", "Skills & Technologies", "Areas of Expertise", "Tech Stack", "Proficiencies", "Tools & Technologies", "Key Skills" |
| Summary | "Profile", "About Me", "Professional Summary", "Executive Summary", "Objective", "Career Objective", "Overview", "Summary of Qualifications" |
| Education | "Education", "Academic Background", "Qualifications", "Degrees", "Academic Credentials", "Schooling" |

Additionally:
- **Non-English resumes**: Headers in French ("Expérience professionnelle"), German ("Berufserfahrung"), Spanish ("Experiencia laboral") — the `multilingual-e5-large` embedding model handles these, but the regex in `sectioner.py` would not.
- **Graphic/icon-based headers**: Many modern resume templates use icons (▶, ●, ★) before section names, or render headers in a separate text block that PyMuPDF may extract as a separate paragraph.
- **Abbreviated headers**: "Exp.", "Edu.", "Certs."
- **No explicit headers at all**: Some resume formats use visual separation (horizontal rules, shading) with no text header — PyMuPDF extracts these as blank lines with no label.
- **Merged sections**: "Education & Certifications", "Projects & Publications" — these don't map cleanly to any single canonical label.

### How Strategy A (Regex) Handles This

The planned `KNOWN_SECTIONS` alias list in `sectioner.py` covers the most common English variants. Performance degrades in this order:

| Scenario | Outcome |
|---|---|
| Standard English header ("Work Experience") | ✅ Detected correctly |
| Common alias ("Employment History") | ✅ Detected if in alias list |
| Uncommon alias ("Career Highlights") | ⚠️ Falls into `"other"` bucket |
| Non-English header | ❌ Not detected; entire section falls into `"other"` |
| Icon-prefixed header ("▶ Experience") | ⚠️ May fail substring match |
| No text headers | ❌ Entire resume becomes `"other"` — degrades to current behaviour |
| Merged sections | ⚠️ Matches on first keyword; second zone is misclassified |

The `fallback=True` path means the system **never breaks** — it just loses the section-isolation benefit for these cases. The resume still gets indexed; it just gets flat chunked like today.

### How Strategy B (Layout-Aware) Handles This

Layout parsers like Unstructured.io use visual heuristics — font size, bold, position, whitespace — not text matching. A heading is a heading because it is visually styled like one, regardless of its exact wording.

- Non-English headers → detected as headings by font/position, content classified by semantic embedding.
- Icon-prefixed headers → the icon is stripped or treated as decoration.
- No text headers → the parser infers structural blocks from whitespace/visual zones.

This is a **structural advantage of Strategy B**. It does not depend on knowing what the header says.

### Mitigation Paths for Strategy A

If Strategy A is implemented, the variable-title problem can be partially mitigated by:

1. **Expand the alias list aggressively** — cover the top-50 English variants, add common non-English equivalents for target markets.

2. **Fuzzy matching** — use `difflib.SequenceMatcher` or `rapidfuzz` to match candidate headers within a Levenshtein distance of 2 from any known alias. Catches typos ("Experiance") and partial matches.

3. **Embedding-based header classification (LLM-lite)** — embed the candidate header line and compare cosine similarity to canonical section label embeddings. Since Pinecone Integrated Inference is already in the stack, this could be done with a single embedding call per detected header line. This is much cheaper than a full LLM call and handles multilingual headers correctly.

4. **`use_llm_sectioner` flag** (already noted in the change analysis) — for high-value resumes or when fewer than 2 sections are detected, fall back to a Gemini call: *"Classify each block of this resume text into one of: summary, experience, education, skills, certifications, other."*

---

## 4. Synthesis: What Should This Project Actually Do?

Neither strategy is perfect in isolation. The right path for this codebase is a **phased hybrid**:

```
Phase A (implement now — 6-file change, low risk)
  └── Strategy A: Section-Aware Chunking with regex + fuzzy matching
      + Embedding-based fallback for unknown headers
      + candidate_name metadata

Phase B (next — medium effort, high value)
  └── Within the experience section: sub-split by job role boundary
      (date-line heuristic: lines matching "YYYY–YYYY" or "MMM YYYY – Present")
      + Inject job_title, company, recency_year into chunk metadata
      (No layout parser needed — role boundaries are detectable from text alone)

Phase C (future — high effort, infrastructure decision required)
  └── Replace PyMuPDF with Unstructured.io for multi-column layout correctness
      + Full per-role structured extraction
      + Small-to-large retrieval in jobs.py
```

### Why Phase B before Phase C?

The most impactful part of Strategy B is **per-role chunking within the experience section** — and that can be achieved without a layout parser for the majority of resumes. Date ranges are a reliable role boundary signal. Implementing Phase B requires modifying only `sectioner.py` (add an experience sub-splitter) and `pinecone_service.py` (two new metadata fields). The layout parser (Phase C) solves multi-column corruption, which affects a subset of resumes and requires significant infrastructure investment.

---

## 5. Decision Matrix

| Criterion | Strategy A | Strategy B | Phased Hybrid |
|---|---|---|---|
| Implementation time | Low | High | Medium |
| New infra required | None | Yes (layout parser) | None for A+B |
| Handles variable section titles | Partial (alias + fuzzy) | Yes (visual structure) | Partial → Full in Phase C |
| Eliminates cross-section bleed | ✅ | ✅ | ✅ |
| Per-role semantic unity | ❌ | ✅ | ✅ (Phase B) |
| Rich temporal metadata | ❌ | ✅ | ✅ (Phase B) |
| Hybrid retrieval (vector + filter) | Partial | Full | Full (Phase B) |
| Small-to-large LLM context | ❌ | ✅ | ✅ (Phase B) |
| Handles non-English resumes | ❌ | ✅ | Partial (embedding fallback) |
| Multi-column layout correctness | Partial (PyMuPDF) | ✅ | ❌ → ✅ (Phase C) |
| Risk of breaking existing flow | Low | High | Low |
