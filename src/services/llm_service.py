"""
LLM service — constructs a structured prompt and calls Google Gemini.

Uses the model configured in GEMINI_MODEL (default: gemini-2.5-flash-lite).
Uses the current google-genai SDK (google.genai), not the deprecated
google-generativeai package.

Recommended free-tier models (as of June 2026)
───────────────────────────────────────────────
  gemini-2.5-flash-lite  — RECOMMENDED default
      Stable, fastest & most budget-friendly model in the 2.5 family.
      Free tier RPM/RPD: check aistudio.google.com/rate-limit for your
      project's exact limits (Google no longer publishes fixed numbers).

  gemini-2.5-flash       — Higher quality, lower free quota
      Best price-performance for reasoning tasks.
      Free tier is more restricted than flash-lite.

  gemini-2.5-pro         — Most capable, very limited free quota
      For complex tasks requiring deep reasoning.

Deprecated / Shut-down models (DO NOT USE)
───────────────────────────────────────────
  gemini-2.0-flash      — SHUT DOWN (June 2026)
  gemini-2.0-flash-lite — SHUT DOWN (June 2026)
  gemini-1.5-flash      — SHUT DOWN

Thinking tokens (gemini-2.5-series)
─────────────────────────────────────
Gemini 2.5 Flash and Flash-Lite spend internal "thinking" tokens before
writing output. These count against max_output_tokens and would truncate
the JSON response. We disable thinking for this task (thinking_budget=0)
since the output schema is rigid and fully specified — no chain-of-thought
is needed. thinking_config is only sent for 2.5-series models.

Error handling
──────────────
• 429 RESOURCE_EXHAUSTED → HTTP 503 with retry guidance (not a 500).
• Other Gemini API errors  → HTTP 502 with the error detail.
• Truncated / unparseable JSON → safe fallback dict (zeros + error note).
"""

import json
import logging

from fastapi import HTTPException
from google import genai
from google.genai import types
from google.genai.errors import ClientError

from src.config import settings

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazy Gemini client — created on first call so tests can import without a key."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client


_SYSTEM_PROMPT = """You are a senior talent acquisition specialist.
You will be given:
  1. A JOB DESCRIPTION written by a hiring manager.
  2. One or more RESUME EXCERPTS from a candidate.
  3. Optionally, structured JOB REQUIREMENTS and CANDIDATE PROFILE context.

Evaluate the candidate on EXACTLY these 6 dimensions and return a JSON object:

{
  "technical_skills_score":     <integer 0-100>,
  "experience_relevance_score": <integer 0-100>,
  "experience_depth_score":     <integer 0-100>,
  "education_score":            <integer 0-100>,
  "certifications_score":       <integer 0-100>,
  "communication_score":        <integer 0-100>,
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "gaps":      ["gap 1",      "gap 2",      "gap 3"],
  "summary":   "2-3 sentence narrative for the recruiter",
  "recommendation": "Strong Yes | Yes | Maybe | No"
}

Dimension guidance:
  technical_skills_score     — match of candidate skills to JD requirements
  experience_relevance_score — similarity of past roles to this position
  experience_depth_score     — seniority, ownership, leadership demonstrated
  education_score            — degree level and field relevance
  certifications_score       — required or preferred certifications present
  communication_score        — clarity, measurable achievements, impact statements

Additional scoring guidance when CANDIDATE PROFILE context is provided:
  experience_depth_score — if the candidate's total experience is less than
    50% of the JD minimum, this score MUST NOT exceed 50.
    If less than 25% of JD minimum, MUST NOT exceed 30.
    If the candidate exceeds JD maximum by 3+ years, apply a mild
    over-qualification penalty (−5 to −10 points).
  If a LEVEL MISMATCH warning is present, factor it strongly into
  experience_depth_score and experience_relevance_score.

Return ONLY the JSON object. Be precise and consistent across candidates."""


def _build_config() -> types.GenerateContentConfig:
    """
    Build GenerateContentConfig for the configured model.

    Thinking tokens are disabled for 2.5-series models (flash, flash-lite, pro):
      - Gemini 2.5 models use extended thinking by default.
      - Thinking tokens count against max_output_tokens, truncating JSON.
      - This structured scoring task gains nothing from chain-of-thought.
      - Non-2.5 models do not support thinking_config and will error if included.
    """
    config = types.GenerateContentConfig(
        system_instruction=_SYSTEM_PROMPT,
        temperature=0.2,
        max_output_tokens=800,
        response_mime_type="application/json",
    )
    if "2.5" in settings.gemini_model:
        config.thinking_config = types.ThinkingConfig(thinking_budget=0)
    return config


def evaluate_candidate(
    job_description: str,
    resume_chunks: list[str],
    experience_context: str = "",
) -> dict:
    """
    Ask Gemini to evaluate a candidate against a job description.

    *resume_chunks* should be the top-K semantically relevant excerpts from
    the candidate's resume retrieved from Pinecone.

    *experience_context* — optional structured context prefix (Phase C).
    When provided, it is prepended before the resume excerpts so the LLM
    has unambiguous structured context about experience level and role fit.

    Returns a parsed dict with 6 dimension scores, strengths, gaps, summary,
    and recommendation — or a safe fallback dict on parse failure.

    Raises:
        HTTPException 503: Gemini API rate limit hit (429). Caller should retry.
        HTTPException 502: Other Gemini API error.
    """
    context = "\n\n---\n\n".join(resume_chunks)

    # Phase C: prepend structured experience context if available.
    if experience_context:
        user_message = (
            f"{experience_context}\n\n"
            f"JOB DESCRIPTION:\n{job_description}\n\n"
            f"RESUME EXCERPTS:\n{context}"
        )
    else:
        user_message = (
            f"JOB DESCRIPTION:\n{job_description}\n\n"
            f"RESUME EXCERPTS:\n{context}"
        )

    try:
        response = _get_client().models.generate_content(
            model=settings.gemini_model,
            contents=user_message,
            config=_build_config(),
        )
    except ClientError as exc:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status == 429:
            # Extract retry delay from the error detail when available.
            detail_msg = str(exc)
            retry_hint = ""
            if "retry" in detail_msg.lower():
                import re
                m = re.search(r"retry in ([\d.]+)s", detail_msg, re.IGNORECASE)
                if m:
                    retry_hint = f" Retry after {float(m.group(1)):.0f}s."
            logger.warning("Gemini quota exhausted (429). %s", detail_msg[:200])
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Gemini API rate limit reached for model '{settings.gemini_model}'.{retry_hint} "
                    "Check your project's exact limits at aistudio.google.com/rate-limit. "
                    "Switch GEMINI_MODEL in .env to gemini-2.5-flash-lite for the highest "
                    "free-tier quota, or gemini-2.5-flash for better reasoning quality."
                ),
            )
        logger.error("Gemini API error (%s): %s", status, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error ({status}): {str(exc)[:300]}",
        )

    raw = response.text or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "Failed to parse Gemini JSON response (model=%s, finish=%s, len=%d)",
            settings.gemini_model,
            getattr(response.candidates[0], "finish_reason", "?") if response.candidates else "?",
            len(raw),
        )
        return {
            "technical_skills_score": 0, "experience_relevance_score": 0,
            "experience_depth_score": 0, "education_score": 0,
            "certifications_score": 0,  "communication_score": 0,
            "strengths": [], "gaps": [],
            "summary": "Failed to parse LLM response.",
            "recommendation": "No", "raw": raw,
        }


def compute_weighted_score(evaluation: dict) -> float:
    """
    Weighted composite score (0-100) from the 6 LLM dimension scores.
    Weights reflect typical hiring priorities — tune per role type.
    """
    weights = {
        "technical_skills_score":     0.30,
        "experience_relevance_score": 0.25,
        "experience_depth_score":     0.15,
        "education_score":            0.10,
        "certifications_score":       0.10,
        "communication_score":        0.10,
    }
    return sum(evaluation.get(dim, 0) * w for dim, w in weights.items())
