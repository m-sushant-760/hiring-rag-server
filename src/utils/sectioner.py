"""
Resume section detector — Strategy A / Phase A.

Splits raw resume text into a dict {section_label: text} so that the
chunker can apply per-section chunk-size tuning and every Pinecone record
carries a `section` metadata field for filtered retrieval.

Detection strategy (three passes, in order):
  1. Exact / case-insensitive alias match against KNOWN_SECTIONS.
  2. Fuzzy match using difflib.SequenceMatcher (ratio ≥ 0.82) — catches
     typos ("Experiance") and minor wording differences.
  3. Prefix stripping: many templates prefix headers with icon characters
     (▶ ● ★ • – —).  The header text is re-checked after stripping leading
     non-alphanumeric characters before applying passes 1 and 2.

Fallback: if fewer than 2 sections are detected the entire text is returned
as {"other": full_text} so the pipeline degrades gracefully to the existing
fixed-size chunking behaviour — no data is lost.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Section alias registry
# ---------------------------------------------------------------------------

KNOWN_SECTIONS: dict[str, list[str]] = {
    "summary": [
        "summary", "professional summary", "profile", "professional profile",
        "about me", "about", "objective", "career objective", "overview",
        "executive summary", "summary of qualifications", "personal statement",
        "career summary",
    ],
    "experience": [
        "experience", "work experience", "professional experience",
        "employment", "employment history", "work history", "career history",
        "career", "industry experience", "relevant experience",
        "professional background", "positions held",
    ],
    "education": [
        "education", "academic background", "academic qualifications",
        "qualifications", "degrees", "academic credentials", "schooling",
        "educational background", "academic history",
    ],
    "skills": [
        "skills", "technical skills", "core competencies", "competencies",
        "technologies", "tech stack", "tools & technologies",
        "tools and technologies", "areas of expertise", "key skills",
        "proficiencies", "tools", "technical proficiencies",
        "skills & expertise", "skills and expertise",
    ],
    "certifications": [
        "certifications", "certificates", "professional certifications",
        "licenses", "accreditations", "credentials", "professional licenses",
        "certifications & licenses", "certifications and licenses",
    ],
    "projects": [
        "projects", "personal projects", "open source", "open-source",
        "side projects", "portfolio", "key projects", "notable projects",
    ],
    "publications": [
        "publications", "papers", "research", "research & publications",
        "articles", "conference papers",
    ],
    "languages": [
        "languages", "spoken languages", "language skills",
    ],
    "awards": [
        "awards", "honors", "honours", "achievements", "accomplishments",
        "recognition", "awards & honors", "awards and honors",
    ],
}

# Pre-flatten for O(1) lookup: alias → canonical label.
_ALIAS_MAP: dict[str, str] = {
    alias: label
    for label, aliases in KNOWN_SECTIONS.items()
    for alias in aliases
}

# All canonical aliases as a flat list for fuzzy matching.
_ALL_ALIASES: list[str] = list(_ALIAS_MAP.keys())

# A line is a candidate section header only if it is short enough.
_MAX_HEADER_LEN = 60

# Fuzzy match threshold: ratio ≥ this value is accepted.
_FUZZY_THRESHOLD = 0.82

# Regex to strip leading icon / bullet characters before matching.
_ICON_PREFIX_RE = re.compile(r"^[^\w\s]+\s*")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_icons(text: str) -> str:
    """Remove leading non-alphanumeric decoration (▶, ●, –, —, •, etc.)."""
    return _ICON_PREFIX_RE.sub("", text).strip()


def _classify_header(line: str) -> str | None:
    """
    Try to classify *line* as a section header.

    Returns the canonical section label (e.g. "experience") or None if the
    line does not match any known section.

    Two passes are attempted: exact alias lookup, then fuzzy matching.
    Both passes are also retried after stripping leading icon characters.
    """
    candidates = [line.strip(), _strip_icons(line.strip())]

    for candidate in candidates:
        normalised = candidate.lower()

        # Pass 1 — exact alias lookup.
        if normalised in _ALIAS_MAP:
            return _ALIAS_MAP[normalised]

        # Pass 2 — fuzzy match against all known aliases.
        best_ratio = 0.0
        best_label: str | None = None
        for alias in _ALL_ALIASES:
            ratio = SequenceMatcher(None, normalised, alias).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_label = _ALIAS_MAP[alias]
        if best_ratio >= _FUZZY_THRESHOLD and best_label is not None:
            return best_label

    return None


def _is_header_candidate(line: str) -> bool:
    """
    Heuristic guard: only short lines that do not end with a period are
    considered as potential section headers to avoid false positives on
    sentence-initial words like "Experience shows that…".
    """
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > _MAX_HEADER_LEN:
        return False
    if stripped.endswith(".") or stripped.endswith(","):
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_into_sections(text: str) -> dict[str, str]:
    """
    Split *text* into a dict mapping canonical section labels to their text.

    The dict is ordered by appearance in the resume.  Unrecognised sections
    accumulate under the "other" key.  Text before the first detected header
    (typically contact info) is stored under "other" as well.

    Fallback: if fewer than 2 distinct canonical sections are found, the
    entire text is returned as {"other": text} so that the downstream
    chunker behaves exactly as before this change.

    Returns an empty dict only for blank input.
    """
    if not text or not text.strip():
        return {}

    lines = text.splitlines()

    # Accumulate sections as a list of (label, lines) to preserve order
    # while allowing multiple "other" spans to merge correctly.
    sections_raw: list[tuple[str, list[str]]] = []
    current_label = "other"
    current_lines: list[str] = []

    for line in lines:
        if _is_header_candidate(line):
            label = _classify_header(line)
            if label is not None:
                # Save whatever we have collected so far.
                if current_lines:
                    sections_raw.append((current_label, current_lines))
                current_label = label
                current_lines = []
                # Do NOT include the header line itself in the body text —
                # the label already captures the semantic meaning.
                continue

        current_lines.append(line)

    # Flush the final section.
    if current_lines:
        sections_raw.append((current_label, current_lines))

    # Merge spans with the same label that are adjacent (e.g. two "other"
    # blocks before the first real header).
    merged: dict[str, list[str]] = {}
    for label, body_lines in sections_raw:
        merged.setdefault(label, []).extend(body_lines)

    # Build the final output — skip sections whose body is blank after strip.
    result: dict[str, str] = {
        label: "\n".join(body_lines).strip()
        for label, body_lines in merged.items()
        if "\n".join(body_lines).strip()
    }

    # Fallback: fewer than 2 recognised canonical sections means the detector
    # found no meaningful structure.  Return the full text as "other" so the
    # existing fixed-size chunking path handles it transparently.
    canonical_detected = {k for k in result if k != "other"}
    if len(canonical_detected) < 2:
        return {"other": text.strip()}

    return result
