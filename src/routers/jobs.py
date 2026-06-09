"""
Jobs / screening router — enhanced with Phase 8 + 9 + Phase A + Phase B + Phase C
==================================================================================
Phase 8  Skills Ontology: JD skills are detected and expanded via the
         ontology graph before embedding, improving recall for candidates
         who list specific tools when the JD uses a broader category name.

Phase 9  Bi-directional Matching: alongside the employer-side fit score,
         the LLM also estimates candidate interest (is this role an
         appropriate career step?). Final score = 70% employer fit +
         30% candidate interest, blended inside the ensemble.

Phase A  Section-Aware Chunking: the /screen endpoint gains an optional
         `section_filter` field that restricts Pinecone retrieval to a
         specific resume section (e.g. "skills", "experience").  The bi-
         directional scorer now preferentially uses experience + summary
         chunks rather than an arbitrary first-4 slice.  Two new fields
         are returned per candidate: candidate_name and sections_retrieved.

Phase B  OR-Group Skill Matching: JD skills that are alternatives in the
         job description (e.g. "AWS, GCP, or Azure") are grouped into
         OR-requirements via shared ontology parent (Strategy B) and
         confirmed/corrected by scanning the JD text for OR/AND signals
         (Strategy A).  The ontology_skill_score now uses
         satisfied_groups / total_groups instead of a flat skill count,
         eliminating false negatives from OR-alternatives being counted
         as individual gaps.  Two new response fields expose the group
         structure: jd_skill_groups (ScreenResponse) and
         satisfied_groups / unsatisfied_groups (CandidateResult).

Phase C  Structured Role & Experience Extraction:
         - designation, role, location accepted as structured payload fields
         - JDMetadata (seniority, domain, min experience) inferred inline
         - experience context injected into LLM prompt per candidate
         - _prefilter() wired to Pinecone for experience-based pre-filtering
         - New response fields: jd_metadata, total_experience_years,
           relevant_experience_years, seniority_level, experience_level_match
"""

import json
import re
from dataclasses import dataclass, asdict
from google import genai
from google.genai import types
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.services import pinecone_service
from src.services.llm_service import evaluate_candidate, compute_weighted_score
from src.services.ontology_service import (
    extract_skills_from_text,
    expand_query_terms,
    skills_match_score,
    group_jd_skills_by_parent,
    refine_groups_with_text,
    score_skill_groups,
    get_candidate_coverage,
    get_jd_skill_variants,
    JDSkillGroup,
)

_job_genai_client: genai.Client | None = None


def _get_genai_client() -> genai.Client:
    """Lazy Gemini client — avoids import-time ValueError when no API key is set."""
    global _job_genai_client
    if _job_genai_client is None:
        _job_genai_client = genai.Client(api_key=settings.google_api_key)
    return _job_genai_client


router = APIRouter(prefix="/api/jobs", tags=["jobs"])


# ── Request / Response models ──────────────────────────────────────────────

class ScreenRequest(BaseModel):
    job_description: str = Field(..., min_length=20)
    top_k: int           = Field(default=5, ge=1, le=20,
                                 description="Max candidates in shortlist.")
    min_experience: int  = Field(default=0, ge=0,
                                 description="Min years experience (0 = no filter).")
    required_certifications: list[str] = Field(
        default_factory=list,
        description="Certifications required (pre-filter)."
    )
    use_ontology:   bool = Field(default=True,
                                 description="Expand JD terms via skills ontology graph.")
    bidirectional:  bool = Field(default=True,
                                 description="Include candidate-interest bi-directional score.")
    # Phase A — section-aware retrieval filter.
    section_filter: str | None = Field(
        default=None,
        description=(
            "Restrict Pinecone retrieval to a specific resume section. "
            "Accepted values: 'summary', 'experience', 'education', "
            "'skills', 'certifications', 'projects', 'publications', "
            "'languages', 'awards', 'other'. "
            "Defaults to None (all sections retrieved)."
        ),
    )
    # Phase C — structured role metadata from payload.
    designation: str | None = Field(
        default=None,
        description="Formal job title (e.g. 'Engineering Manager - Python').",
    )
    role: str | None = Field(
        default=None,
        description="Functional role label (may equal designation or differ).",
    )
    location: str | None = Field(
        default=None,
        description="Job location (e.g. 'Bengaluru').",
    )


class DimensionScores(BaseModel):
    technical:      float | None = None
    relevance:      float | None = None
    depth:          float | None = None
    education:      float | None = None
    certifications: float | None = None
    communication:  float | None = None


# ── Phase B: OR-group response models ─────────────────────────────────────

class JDSkillGroupResult(BaseModel):
    """
    One skill requirement group from the JD, with satisfaction status.

    group_type   : "OR"  — any one skill in this group satisfies the requirement
                   "AND" — all skills must be present (or singleton)
    label        : human-readable name (shared ontology parent for OR-groups,
                   skill name for singletons)
    skills       : individual JD skill nodes in this group
    satisfied    : whether this candidate satisfied the requirement
    satisfied_by : candidate skills that provided the coverage
    """
    group_type:   str
    label:        str
    skills:       list[str]
    satisfied:    bool
    satisfied_by: list[str]


class JDSkillGroupsSummary(BaseModel):
    """
    JD-level OR/AND group structure, returned once in ScreenResponse.

    groups    : the full list of detected skill requirement groups
    or_count  : number of OR-groups (JD alternatives, e.g. "AWS or GCP or Azure")
    and_count : number of AND-groups (standalone requirements)
    """
    groups:    list[JDSkillGroupResult]
    or_count:  int
    and_count: int


class CandidateResult(BaseModel):
    candidate_id:             str
    filename:                 str
    final_rank:               int
    match_score:              int              # ensemble 0-100
    employer_score:           float            # LLM weighted (employer perspective)
    candidate_interest_score: float            # Phase 9 — bi-directional
    ontology_skill_score:     float            # Phase B — group-aware overlap %
    recommendation:           str
    strengths:                list[str]
    gaps:                     list[str]
    summary:                  str
    dimension_scores:         DimensionScores
    matched_jd_skills:        list[str]        # individual skills from satisfied groups
    expanded_skills:          list[str]        # JD-level ontology expansion terms
    # Phase B — group satisfaction detail
    satisfied_groups:         list[JDSkillGroupResult]   # groups this candidate satisfied
    unsatisfied_groups:       list[JDSkillGroupResult]   # gaps (real missing requirements)
    # Phase A — section-aware metadata
    candidate_name:           str              # extracted from resume header
    sections_retrieved:       list[str]        # which sections contributed to this ranking
    # Phase C — experience metadata
    total_experience_years:     float | None = None
    relevant_experience_years:  float | None = None
    seniority_level:            str | None   = None
    experience_level_match:     str | None   = None   # "match" / "under" / "over" / "unknown"


# ── Phase C: JD Metadata ──────────────────────────────────────────────────

@dataclass
class JDMetadata:
    """Structured metadata inferred from the screening payload."""
    designation:          str
    role:                 str
    location:             str | None
    seniority_level:      str
    min_experience_years: int | None
    max_experience_years: int | None
    role_domain:          str


class JDMetadataModel(BaseModel):
    """Pydantic model for JDMetadata in the API response."""
    designation:          str
    role:                 str
    location:             str | None = None
    seniority_level:      str
    min_experience_years: int | None = None
    max_experience_years: int | None = None
    role_domain:          str


class ScreenResponse(BaseModel):
    job_description_snippet:    str
    jd_skills_detected:         list[str]         # Phase 8 — flat list (for compat)
    jd_skill_groups:            JDSkillGroupsSummary  # Phase B — grouped structure
    jd_metadata:                JDMetadataModel | None = None  # Phase C
    total_candidates_evaluated: int
    results:                    list[CandidateResult]


# ── Phase C: Inline seniority / domain / experience helpers ───────────────

# Ordered by priority (first match wins).
_SENIORITY_KEYWORDS: list[tuple[list[str], str]] = [
    (["intern", "trainee"],                                    "intern"),
    (["junior", "entry", "associate", "grad"],                  "junior"),
    (["mid", "intermediate", "ii"],                             "mid"),
    (["senior", "sr.", "iii"],                                  "senior"),
    (["lead", "tech lead", "team lead"],                        "lead"),
    (["staff", "principal", "iv"],                              "staff"),
    (["engineering manager", "manager"],                        "manager"),
    (["director", "head of"],                                   "director"),
    (["vp", "vice president"],                                  "executive"),
]

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "backend":  ["python", "backend", "api", "flask", "django", "fastapi"],
    "devops":   ["devops", "sre", "platform", "infrastructure", "kubernetes"],
    "data":     ["data", "analytics", "pipeline", "etl", "spark"],
    "ml":       ["machine learning", "ml", "ai", "nlp", "llm"],
    "frontend": ["fullstack", "full stack", "frontend", "react"],
}

_DOMAIN_TITLE_KEYWORDS: dict[str, list[str]] = {
    "backend":  ["engineer", "developer", "backend", "api", "server"],
    "devops":   ["devops", "sre", "platform", "infrastructure", "cloud"],
    "data":     ["data", "analytics", "pipeline", "etl", "snowflake"],
    "ml":       ["ml", "machine learning", "ai", "research", "nlp"],
}

# Seniority → minimum expected experience years (for mismatch detection).
_SENIORITY_MIN_YEARS: dict[str, int] = {
    "intern": 0, "junior": 0, "mid": 2, "senior": 4,
    "lead": 5, "staff": 7, "manager": 6, "director": 10, "executive": 12,
}

# Seniority → numeric rank (for comparison).
_SENIORITY_RANK: dict[str, int] = {
    "intern": 0, "junior": 1, "mid": 2, "senior": 3,
    "lead": 4, "staff": 5, "manager": 5, "director": 6, "executive": 7,
    "unknown": 2,
}

# Regex patterns for extracting min/max experience from JD text.
_EXP_PATTERNS = [
    re.compile(r"(\d+)\+\s*years?\s+of\s+(?:professional\s+)?experience", re.IGNORECASE),
    re.compile(r"minimum\s+(\d+)\s+years?", re.IGNORECASE),
    re.compile(r"at\s+least\s+(\d+)\s+years?", re.IGNORECASE),
    re.compile(r"(\d+)[–\-–]+(\d+)\s+years?", re.IGNORECASE),  # range
    re.compile(r"(\d+)\+\s*years?", re.IGNORECASE),
]


def _infer_seniority(title: str | None) -> str:
    """Infer seniority level from a job designation or role string."""
    if not title:
        return "mid"
    t = title.lower()
    for keywords, level in _SENIORITY_KEYWORDS:
        for kw in keywords:
            if kw in t:
                return level
    return "mid"


def _infer_domain(role: str | None, jd_text: str) -> str:
    """
    Infer role domain from the role/designation + JD text.

    For manager-level roles the domain is inferred from secondary keywords.
    """
    sources = []
    if role:
        sources.append(role.lower())
    sources.append(jd_text.lower())
    combined = " ".join(sources)

    # For manager roles, infer from secondary keyword.
    if role and "manager" in role.lower():
        # Look after the dash/hyphen for the domain keyword.
        parts = re.split(r"[-–—]", role)
        if len(parts) > 1:
            secondary = parts[-1].strip().lower()
            for domain, keywords in _DOMAIN_KEYWORDS.items():
                for kw in keywords:
                    if kw in secondary:
                        return domain

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                return domain
    return "other"


def _extract_min_experience(jd_text: str) -> tuple[int | None, int | None]:
    """Extract min/max experience years from JD text via regex."""
    for pattern in _EXP_PATTERNS:
        m = pattern.search(jd_text)
        if m:
            groups = m.groups()
            if len(groups) == 2 and groups[1] is not None:
                return int(groups[0]), int(groups[1])
            return int(groups[0]), None
    return None, None


def _build_jd_metadata(body: ScreenRequest) -> JDMetadata:
    """Assemble JDMetadata from the request payload (no LLM call)."""
    designation = body.designation or ""
    role = body.role or ""
    seniority = _infer_seniority(designation or role)
    domain = _infer_domain(role or designation, body.job_description)
    min_exp, max_exp = _extract_min_experience(body.job_description)

    return JDMetadata(
        designation=designation,
        role=role,
        location=body.location,
        seniority_level=seniority,
        min_experience_years=min_exp,
        max_experience_years=max_exp,
        role_domain=domain,
    )


def _compute_relevant_experience(
    timeline: list[dict],
    role_domain: str,
) -> float:
    """
    Compute relevant experience years from timeline entries.

    Matches job titles against role-domain keywords.
    Only counts 'professional' and 'transitional' segments.
    """
    keywords = _DOMAIN_TITLE_KEYWORDS.get(role_domain, [])
    if not keywords:
        return 0.0

    relevant_months = 0.0
    for entry in timeline:
        if entry.get("segment") == "pre_graduation":
            continue
        title = (entry.get("title") or "").lower()
        if any(kw in title for kw in keywords):
            relevant_months += entry.get("months", 0)
    return round(relevant_months / 12, 1)


def _experience_level_match(
    candidate_years: float | None,
    candidate_seniority: str,
    jd_meta: JDMetadata,
) -> str:
    """Determine if candidate experience level matches JD requirements."""
    if candidate_years is None:
        return "unknown"

    jd_min = jd_meta.min_experience_years
    jd_max = jd_meta.max_experience_years
    jd_seniority_rank = _SENIORITY_RANK.get(jd_meta.seniority_level, 2)
    cand_seniority_rank = _SENIORITY_RANK.get(candidate_seniority, 2)

    # Under-qualified checks.
    if jd_min is not None and candidate_years < jd_min * 0.5:
        return "under"
    if cand_seniority_rank < jd_seniority_rank - 1:
        return "under"

    # Over-qualified checks.
    if jd_max is not None and candidate_years > jd_max + 3:
        return "over"

    return "match"


def _build_experience_context(
    jd_meta: JDMetadata,
    candidate_years: float | None,
    candidate_seniority: str,
    relevant_years: float,
    level_match: str,
) -> str:
    """
    Build the structured experience context prefix for the LLM prompt.

    This block is prepended to resume excerpts so the LLM has unambiguous
    structured context about experience level.
    """
    lines = ["JOB REQUIREMENTS (structured):"]
    if jd_meta.designation:
        lines.append(f"  Designation   : {jd_meta.designation}")
    if jd_meta.role:
        lines.append(f"  Role          : {jd_meta.role}")
    if jd_meta.location:
        lines.append(f"  Location      : {jd_meta.location}")

    seniority_info = jd_meta.seniority_level
    expected_min = _SENIORITY_MIN_YEARS.get(jd_meta.seniority_level, 0)
    if expected_min > 0:
        seniority_info += f"  (≥ {expected_min} years expected)"
    lines.append(f"  Seniority     : {seniority_info}")

    if jd_meta.min_experience_years is not None:
        lines.append(f"  Min Experience: {jd_meta.min_experience_years} years required (from JD text)")
    lines.append(f"  Domain        : {jd_meta.role_domain}")

    lines.append("")
    lines.append("CANDIDATE PROFILE:")
    if candidate_years is not None:
        lines.append(f"  Total experience    : {candidate_years} years")
    else:
        lines.append("  Total experience    : unknown")
    lines.append(f"  Seniority level     : {candidate_seniority}")
    if relevant_years > 0:
        lines.append(f"  Relevant experience : {relevant_years} years ({jd_meta.role_domain} domain)")

    # Mismatch warnings.
    if level_match == "under":
        jd_min = jd_meta.min_experience_years
        lines.append(f"  ⚠ LEVEL MISMATCH    : candidate seniority ({candidate_seniority}) is significantly below required ({jd_meta.seniority_level}).")
        if jd_min is not None and candidate_years is not None:
            lines.append(f"                         JD requires ≥ {jd_min} years; candidate has {candidate_years} years.")
    elif level_match == "over":
        lines.append(f"  ℹ OVER-QUALIFIED    : candidate may be over-qualified for this role.")

    return "\n".join(lines)


# ── Helpers ────────────────────────────────────────────────────────────────

def _prefilter(min_exp: int, certs: list[str]) -> dict | None:
    f: dict = {}
    if min_exp > 0:
        f["experience_years"] = {"$gte": min_exp}
    if certs:
        f["certifications"] = {"$in": certs}
    return f or None


def _group_by_candidate(hits: list) -> dict[str, dict]:
    """
    Group Pinecone Integrated Inference hits by candidate_id.
    Hit fields are top-level (not nested under .metadata).

    Phase A additions:
      - candidate_name    : stored from the first hit that carries it.
      - sections_retrieved: set of section labels seen across all hits.
      - chunks_by_section : dict mapping section label → list of chunk texts,
                            used by _candidate_interest_score to prefer
                            experience + summary chunks.
    """
    grouped: dict[str, dict] = {}
    for h in hits:
        cid = h.get("candidate_id", "unknown")
        if cid not in grouped:
            grouped[cid] = {
                "filename":          h.get("filename", "unknown"),
                "chunks":            [],
                "score":             float(h.get("_score", 0)),
                # Phase A
                "candidate_name":    h.get("candidate_name", ""),
                "sections_retrieved": set(),
                "chunks_by_section": {},
            }
        chunk_text = h.get("chunk_text", "")
        section    = h.get("section", "unknown") or "unknown"
        grouped[cid]["chunks"].append(chunk_text)
        grouped[cid]["score"] = max(grouped[cid]["score"], float(h.get("_score", 0)))
        # Phase A — accumulate section metadata.
        grouped[cid]["sections_retrieved"].add(section)
        grouped[cid]["chunks_by_section"].setdefault(section, []).append(chunk_text)
        # Keep the first non-empty candidate_name seen.
        if not grouped[cid]["candidate_name"] and h.get("candidate_name"):
            grouped[cid]["candidate_name"] = h["candidate_name"]
    return grouped


def _candidate_interest_score(
    job_description: str,
    chunks: list[str],
    chunks_by_section: dict[str, list[str]] | None = None,
) -> float:
    """
    Phase 9 — Bi-directional Matching.
    Ask Gemini whether the candidate would genuinely be interested in
    and well-positioned for this role (candidate's perspective).
    Returns 0-100. Penalises over-qualification or trajectory mismatch.

    Phase A: when chunks_by_section is provided, the prompt is built from
    experience + summary chunks specifically (more representative of career
    trajectory) rather than an arbitrary first-4 slice.
    """
    if chunks_by_section:
        preferred = (
            chunks_by_section.get("experience", [])
            + chunks_by_section.get("summary", [])
        )
        context_chunks = preferred[:4] if preferred else chunks[:4]
    else:
        context_chunks = chunks[:4]
    context = "\n\n---\n\n".join(context_chunks)
    prompt  = (
        "Assess this from the CANDIDATE'S perspective.\n\n"
        f"JOB DESCRIPTION:\n{job_description}\n\n"
        f"CANDIDATE RESUME:\n{context}\n\n"
        "Questions to consider:\n"
        "1. Is this role an appropriate next career step (not a step down)?\n"
        "2. Does the candidate's trajectory suggest genuine interest in this type of role?\n"
        "3. Any over-qualification or career-direction mismatch signals?\n\n"
        'Return ONLY JSON: {"candidate_interest_score": <0-100>, "reasoning": "<1 sentence>"}'
    )
    try:
        resp = _get_genai_client().models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=150,
                response_mime_type="application/json",
            ),
        )
        data = json.loads(resp.text or "{}")
        return float(data.get("candidate_interest_score", 50))
    except Exception:
        return 50.0   # neutral fallback


def _ensemble(
    llm_score:  float,
    onto_score: float,       # 0-1
    interest:   float,
    bidir:      bool,
) -> float:
    """
    Ensemble final score (0-100).

    With bi-directional  → LLM 50% | Ontology 20% | Bi-dir 30%
                            (Bi-dir itself = 70% employer + 30% candidate interest)
    Without bi-directional → LLM 60% | Ontology 40%
    """
    if bidir:
        bidir_score = 0.70 * llm_score + 0.30 * interest
        return 0.50 * llm_score + 0.20 * (onto_score * 100) + 0.30 * bidir_score
    return 0.60 * llm_score + 0.40 * (onto_score * 100)


def _group_to_result(g: JDSkillGroup) -> JDSkillGroupResult:
    """Convert a JDSkillGroup dataclass to its Pydantic response model."""
    return JDSkillGroupResult(
        group_type   = g.group_type,
        label        = g.label,
        skills       = g.skills,
        satisfied    = g.satisfied,
        satisfied_by = g.satisfied_by,
    )


# ── Endpoint ───────────────────────────────────────────────────────────────

@router.post("/screen", response_model=ScreenResponse)
async def screen_candidates(body: ScreenRequest):
    """
    Screen all indexed resumes against a job description.

    Pipeline (Phase 8 + 9 + A + B enhanced):
      A.  Detect skills in JD via ontology                [Phase 8]
      A2. Group JD skills into OR/AND requirement groups  [Phase B Strategy B+A]
      B.  Expand JD skills → augmented query              [Phase 8]
      C.  Embed augmented query → Pinecone retrieval with metadata pre-filter
      D.  Group chunks by candidate; keep top_k
      E.  Per candidate:
            i.   Group-aware ontology skill score         [Phase B]
            ii.  6-dimension LLM evaluation               (employer perspective)
            iii. Candidate interest score                  [Phase 9 bi-directional]
            iv.  Ensemble final score
      F.  Sort descending → ranked shortlist
    """
    # ── A. Detect JD skills ────────────────────────────────────────────────
    jd_skills: list[str] = (
        # deduplicate_aliases=True: drop ALIAS shorthand nodes (e.g. 'Kafka',
        # 'Mongo') when their canonical form ('Apache Kafka', 'MongoDB') is
        # also present in the JD text.  JD authors use canonical names; aliases
        # are only needed for resume shorthand detection (candidate scan below).
        extract_skills_from_text(body.job_description, deduplicate_aliases=True)
        if body.use_ontology else []
    )

    # ── A2. Phase B — Build OR/AND skill groups ────────────────────────────
    # Strategy B: group siblings under shared ontology parent into OR-groups.
    # Strategy A: refine group types using OR/AND signals in the JD text.
    if body.use_ontology and jd_skills:
        jd_groups = group_jd_skills_by_parent(jd_skills)
        jd_groups = refine_groups_with_text(body.job_description, jd_groups)
    else:
        # No ontology: treat every detected skill as a singleton AND-requirement
        jd_groups = [
            JDSkillGroup(group_type="AND", label=s, skills=[s])
            for s in jd_skills
        ]

    # ── B. Expand via ontology ─────────────────────────────────────────────
    if body.use_ontology and jd_skills:
        expanded: set[str] = expand_query_terms(jd_skills, max_hops=1)
        augmented_query = (
            body.job_description
            + "\n\nExpanded skills:\n"
            + ", ".join(sorted(expanded)[:50])
        )
    else:
        expanded = set()
        augmented_query = body.job_description

    # ── Phase C: Build JD metadata ─────────────────────────────────────────
    jd_meta = _build_jd_metadata(body)
    jd_metadata_model = JDMetadataModel(**asdict(jd_meta))

    # ── C. Retrieve via Pinecone Integrated Inference ─────────────────────
    # Pinecone embeds the query text server-side — no embed_text() call needed.
    # Phase A: pass section_filter so Pinecone applies it as a metadata $eq
    # pre-filter before scoring.  None means all sections are returned.
    raw = pinecone_service.query_similar_chunks(
        query_text=augmented_query,
        top_k=body.top_k * settings.top_k_results,
        section_filter=body.section_filter,
    )

    # Build the JD-level group summary (same for all candidates — computed once)
    jd_groups_summary = JDSkillGroupsSummary(
        groups    = [_group_to_result(g) for g in jd_groups],
        or_count  = sum(1 for g in jd_groups if g.group_type == "OR"),
        and_count = sum(1 for g in jd_groups if g.group_type == "AND"),
    )

    if not raw:
        return ScreenResponse(
            job_description_snippet    = body.job_description[:200],
            jd_skills_detected         = jd_skills,
            jd_skill_groups            = jd_groups_summary,
            jd_metadata                = jd_metadata_model,
            total_candidates_evaluated = 0,
            results                    = [],
        )

    # ── D. Group + limit ───────────────────────────────────────────────────
    grouped = _group_by_candidate(raw)
    # Sort by each candidate's best Pinecone similarity score before slicing.
    # Without this, top_k is applied in dict insertion order (first-seen hit),
    # which may favour candidates with many mid-scoring chunks over one with a
    # single high-scoring chunk — the opposite of what we want.
    top_candidates = sorted(
        grouped.items(),
        key=lambda kv: kv[1]["score"],
        reverse=True,
    )[: body.top_k]

    # ── E. Evaluate ────────────────────────────────────────────────────────
    results: list[CandidateResult] = []
    for cid, info in top_candidates:
        chunks    = info["chunks"]
        full_text = " ".join(chunks)

        # Phase B — group-aware ontology skill score
        # extract candidate skills, then score against OR/AND groups.
        cand_skills = extract_skills_from_text(full_text)
        onto_score, scored_groups = (
            score_skill_groups(cand_skills, jd_groups, max_hops=2)
            if jd_groups else (0.5, [])
        )

        # T4 — matched_jd_skills: individual skills from satisfied groups,
        # using ontology-aware matching (consistent with ontology_skill_score).
        matched_jd: list[str] = sorted({
            skill
            for g in scored_groups
            if g.satisfied
            for skill in g.skills
        })

        # ── Phase C: Extract candidate experience from experience_summary chunk
        cand_exp_years: float | None = None
        cand_seniority = "unknown"
        cand_timeline: list[dict] = []
        for chunk in chunks:
            if chunk.startswith("EXPERIENCE SUMMARY"):
                # Parse total years from the synthetic chunk.
                exp_match = re.search(r"Total work experience\s*:\s*([\d.]+)", chunk)
                if exp_match:
                    cand_exp_years = float(exp_match.group(1))
                sen_match = re.search(r"Seniority level\s*:\s*(\S+)", chunk)
                if sen_match:
                    cand_seniority = sen_match.group(1).lower()
                break

        # Compute relevant experience.
        relevant_years = _compute_relevant_experience(cand_timeline, jd_meta.role_domain)
        level_match = _experience_level_match(cand_exp_years, cand_seniority, jd_meta)

        # Phase C: Build experience context prefix for LLM prompt.
        exp_context = ""
        if jd_meta.designation or jd_meta.role:
            exp_context = _build_experience_context(
                jd_meta, cand_exp_years, cand_seniority, relevant_years, level_match,
            )

        # LLM multi-dimensional (employer)
        # Phase C: pass experience context prefix to the LLM.
        evaluation   = evaluate_candidate(
            body.job_description, chunks, experience_context=exp_context,
        )
        employer_scr = compute_weighted_score(evaluation)

        # Phase 9 — candidate interest (bi-directional)
        # Phase A: pass chunks_by_section so the scorer can prefer
        # experience + summary chunks over an arbitrary first-4 slice.
        interest = (
            _candidate_interest_score(
                body.job_description,
                chunks,
                chunks_by_section=info.get("chunks_by_section"),
            )
            if body.bidirectional else 50.0
        )

        # Ensemble
        final = _ensemble(employer_scr, onto_score, interest, body.bidirectional)

        results.append(CandidateResult(
            candidate_id             = cid,
            filename                 = info["filename"],
            final_rank               = 0,
            match_score              = round(final),
            employer_score           = round(employer_scr, 1),
            candidate_interest_score = round(interest, 1),
            ontology_skill_score     = round(onto_score * 100, 1),
            recommendation           = evaluation.get("recommendation", "No"),
            strengths                = evaluation.get("strengths", []),
            gaps                     = evaluation.get("gaps", []),
            summary                  = evaluation.get("summary", ""),
            dimension_scores         = DimensionScores(
                technical      = evaluation.get("technical_skills_score"),
                relevance      = evaluation.get("experience_relevance_score"),
                depth          = evaluation.get("experience_depth_score"),
                education      = evaluation.get("education_score"),
                certifications = evaluation.get("certifications_score"),
                communication  = evaluation.get("communication_score"),
            ),
            matched_jd_skills        = matched_jd,
            expanded_skills          = sorted(expanded)[:20],
            # Phase B — group satisfaction detail
            satisfied_groups   = [_group_to_result(g) for g in scored_groups if g.satisfied],
            unsatisfied_groups = [_group_to_result(g) for g in scored_groups if not g.satisfied],
            # Phase A — section-aware metadata
            candidate_name     = info.get("candidate_name", ""),
            sections_retrieved = sorted(info.get("sections_retrieved", set())),
            # Phase C — experience metadata
            total_experience_years    = cand_exp_years,
            relevant_experience_years = relevant_years if relevant_years > 0 else None,
            seniority_level           = cand_seniority if cand_seniority != "unknown" else None,
            experience_level_match    = level_match,
        ))

    # ── F. Rank ────────────────────────────────────────────────────────────
    results.sort(key=lambda r: r.match_score, reverse=True)
    for i, r in enumerate(results, start=1):
        r.final_rank = i

    return ScreenResponse(
        job_description_snippet    = body.job_description[:200],
        jd_skills_detected         = jd_skills,
        jd_skill_groups            = jd_groups_summary,
        jd_metadata                = jd_metadata_model,
        total_candidates_evaluated = len(results),
        results                    = results,
    )
