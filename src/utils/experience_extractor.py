"""
Experience extractor — three-layer cascade for resume experience extraction.

Phase 2 of the Structured Role & Experience Extraction plan.

Layers (executed in order — first match wins):
  1. Explicit phrase scan (regex, highest confidence)
     Finds self-stated totals like "9+ years of experience".
  2. Timeline calculation (date math, medium confidence)
     Parses job date ranges, computes durations, handles overlaps.
     2b. Post-graduation classification — classifies each job entry as
         pre_graduation / professional / transitional using the highest
         degree's graduation year as cutoff.
  3. Graduation year fallback (lowest confidence)
     Uses current_year - graduation_year when no job dates are found.

Output: ExperienceProfile dataclass with total_years, timeline, etc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperienceProfile:
    """Structured experience data extracted from a resume."""
    total_years:            float | None = None   # post-graduation professional only
    total_years_all:        float | None = None   # all jobs including pre-graduation
    graduation_year:        int | None   = None   # from highest completed degree
    degree_level:           str | None   = None   # "phd" / "masters" / "bachelors" / "diploma" / None
    timeline:               list[dict]   = field(default_factory=list)
    inferred:               bool         = True   # True = calculated from dates
    experience_floor_only:  bool         = False  # True = only graduation year was available
    extraction_method:      str          = "unknown"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CURRENT_YEAR = date.today().year

# Month name → number mapping for date parsing.
_MONTH_MAP: dict[str, int] = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6,
    "jul": 7, "july": 7, "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# Degree hierarchy for graduation year extraction.
# Higher number = higher degree level.
_DEGREE_HIERARCHY: dict[str, tuple[int, str]] = {
    # keyword → (rank, canonical_level)
    "phd":        (4, "phd"),
    "ph.d":       (4, "phd"),
    "doctorate":  (4, "phd"),
    "doctoral":   (4, "phd"),
    "m.tech":     (3, "masters"),
    "mtech":      (3, "masters"),
    "m.s.":       (3, "masters"),
    "m.sc":       (3, "masters"),
    "ms ":        (3, "masters"),
    "mba":        (3, "masters"),
    "masters":    (3, "masters"),
    "master":     (3, "masters"),
    "m.e.":       (3, "masters"),
    "m.a.":       (3, "masters"),
    "mca":        (3, "masters"),
    "b.tech":     (2, "bachelors"),
    "btech":      (2, "bachelors"),
    "b.e.":       (2, "bachelors"),
    "b.sc":       (2, "bachelors"),
    "b.s.":       (2, "bachelors"),
    "bs ":        (2, "bachelors"),
    "bachelors":  (2, "bachelors"),
    "bachelor":   (2, "bachelors"),
    "b.a.":       (2, "bachelors"),
    "bca":        (2, "bachelors"),
    "b.com":      (2, "bachelors"),
    "diploma":    (1, "diploma"),
}

# Patterns that indicate an uncertain graduation year — skip filter.
_UNCERTAIN_PATTERNS = re.compile(
    r"\b(expected|pursuing|enrolled|ongoing|current)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Layer 1 — Explicit phrase scan
# ---------------------------------------------------------------------------

# Patterns to detect self-stated experience totals.
_EXPLICIT_PATTERNS = [
    re.compile(r"(\d+)\+?\s*years?\s+of\s+(?:total\s+)?experience", re.IGNORECASE),
    re.compile(r"over\s+(\d+)\s+years?", re.IGNORECASE),
    re.compile(r"(\d+)\s+years?\s+of\s+professional\s+experience", re.IGNORECASE),
    re.compile(r"(\d+)\s+years?\s+in\s+the\s+(?:industry|field|domain)", re.IGNORECASE),
]


def _layer1_explicit(text: str) -> float | None:
    """Scan for explicit experience phrases. Returns years or None."""
    for pattern in _EXPLICIT_PATTERNS:
        m = pattern.search(text)
        if m:
            return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Layer 2 — Timeline calculation
# ---------------------------------------------------------------------------

# Date range patterns.
# Group 1 = month_or_year_start, Group 2 = year_start (if month given),
# then separator, then end date or Present/Current.

# Pattern: "Jan 2020 – Mar 2023" or "January 2020 - March 2023"
_DATE_MONTH_YEAR = re.compile(
    r"([A-Za-z]+)\s+(\d{4})\s*[–\-—to]+\s*"
    r"(?:(present|current|now|till\s+date)|([A-Za-z]+)\s+(\d{4}))",
    re.IGNORECASE,
)

# Pattern: "06/2019 – 12/2023" or "06/2019 – Present"
_DATE_MMYYYY = re.compile(
    r"(\d{1,2})/(\d{4})\s*[–\-—to]+\s*"
    r"(?:(present|current|now|till\s+date)|(\d{1,2})/(\d{4}))",
    re.IGNORECASE,
)

# Pattern: "2018 – 2022" or "2018 - Present" (year only)
_DATE_YEAR_ONLY = re.compile(
    r"(?<!\d)(\d{4})\s*[–\-—to]+\s*"
    r"(?:(present|current|now|till\s+date)|(\d{4}))(?!\d)",
    re.IGNORECASE,
)


def _parse_month(name: str) -> int:
    """Convert month name/abbreviation to number (1-12). Returns 1 on failure."""
    return _MONTH_MAP.get(name.lower().strip("."), 1)


def _to_date(year: int, month: int = 1) -> date:
    """Create a date from year+month, clamped to valid range."""
    month = max(1, min(12, month))
    return date(year, month, 1)


def _extract_date_ranges(text: str) -> list[tuple[date, date, int, int, str]]:
    """
    Extract all (start_date, end_date, match_start, match_end, match_text) tuples from text.
    "Present"/"Current"/"Now" → today's date.
    """
    today = date.today()
    ranges: list[tuple[date, date, int, int, str]] = []

    # Pattern 1: Month Year – Month Year
    for m in _DATE_MONTH_YEAR.finditer(text):
        start_month = _parse_month(m.group(1))
        start_year = int(m.group(2))
        if m.group(3):  # present/current
            end = today
        else:
            end_month = _parse_month(m.group(4))
            end_year = int(m.group(5))
            end = _to_date(end_year, end_month)
        start = _to_date(start_year, start_month)
        if start <= end:
            ranges.append((start, end, m.start(), m.end(), m.group(0)))

    # Pattern 2: MM/YYYY – MM/YYYY
    for m in _DATE_MMYYYY.finditer(text):
        start_month = int(m.group(1))
        start_year = int(m.group(2))
        if m.group(3):  # present/current
            end = today
        else:
            end_month = int(m.group(4))
            end_year = int(m.group(5))
            end = _to_date(end_year, end_month)
        start = _to_date(start_year, start_month)
        if start <= end:
            ranges.append((start, end, m.start(), m.end(), m.group(0)))

    # Only use year-only pattern if no month-based ranges were found
    # to avoid double-counting the same date ranges.
    if not ranges:
        for m in _DATE_YEAR_ONLY.finditer(text):
            start_year = int(m.group(1))
            if start_year < 1980 or start_year > _CURRENT_YEAR + 1:
                continue
            if m.group(2):  # present/current
                end = today
            else:
                end_year = int(m.group(3))
                if end_year < 1980 or end_year > _CURRENT_YEAR + 1:
                    continue
                end = _to_date(end_year, 12)  # year-only → treat as Dec
            start = _to_date(start_year, 1)   # year-only → treat as Jan
            if start <= end:
                ranges.append((start, end, m.start(), m.end(), m.group(0)))

    return sorted(ranges, key=lambda r: r[2])


def _compute_months_no_overlap(ranges: list[tuple[date, date]]) -> float:
    """
    Sum durations across all date ranges, handling overlaps.

    Sort by start date; if a new range starts before the previous one ends,
    credit only the non-overlapping portion. Gaps are not counted.
    """
    if not ranges:
        return 0.0

    sorted_ranges = sorted(ranges, key=lambda r: r[0])
    total_months = 0.0
    current_start, current_end = sorted_ranges[0]

    for start, end in sorted_ranges[1:]:
        if start <= current_end:
            # Overlapping — extend the current span if needed.
            current_end = max(current_end, end)
        else:
            # Non-overlapping — finalize the previous span.
            months = (current_end.year - current_start.year) * 12 + (current_end.month - current_start.month)
            total_months += max(0, months)
            current_start, current_end = start, end

    # Finalize the last span.
    months = (current_end.year - current_start.year) * 12 + (current_end.month - current_start.month)
    total_months += max(0, months)

    return total_months


def _extract_title_company(block: str) -> tuple[str, str]:
    """
    Best-effort extraction of job title and company from a text block.

    Looks for the first non-empty line that is not a date pattern
    as the title, and the next non-empty line as company.
    """
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    title = ""
    company = ""
    for line in lines:
        # Skip lines that are primarily date patterns.
        if re.match(r"^[\d/\-–—\s]+(present|current|now)?$", line, re.IGNORECASE):
            continue
        if re.search(r"\d{4}\s*[–\-—]", line):
            # Line contains a date range — try to extract title before the date.
            pre_date = re.split(r"\d{4}", line)[0].strip().rstrip(",|-–—")
            if pre_date and not title:
                title = pre_date
            continue
        if not title:
            title = line
        elif not company:
            company = line
            break
    return title, company


def _layer2_timeline(
    experience_text: str,
    full_text: str,
) -> tuple[list[dict], float | None]:
    """
    Parse job date ranges from the experience section.

    Returns (timeline_entries, total_months) or ([], None) if no dates found.
    Each timeline entry: {title, company, start, end, months, segment}
    Segment is initially set to "professional" — Layer 2b reclassifies.
    """
    text = experience_text or full_text
    if not text:
        return [], None

    ranges = _extract_date_ranges(text)
    if not ranges:
        return [], None

    all_entries: list[dict] = []
    all_ranges: list[tuple[date, date]] = []

    prev_end = 0
    for start, end, m_start, m_end, m_text in ranges:
        pre_text = text[prev_end:m_start]

        # Extract job header lines immediately preceding the date range
        raw_lines = [l.strip() for l in pre_text.splitlines()]
        raw_lines = [l for l in raw_lines if l]

        header_lines = []
        for line in reversed(raw_lines):
            # Stop or skip if it looks like a description/bullet line
            if line.startswith(('•', '-', '*', '+')):
                break
            if line.endswith(('.', '?', '!')):
                break
            if len(line) > 100:
                break
            header_lines.insert(0, line)
            if len(header_lines) >= 2:
                break

        if not header_lines and raw_lines:
            header_lines = [raw_lines[-1]]

        block = "\n".join(header_lines) + "\n" + m_text
        title, company = _extract_title_company(block)
        months = max(0, (end.year - start.year) * 12 + (end.month - start.month))

        all_entries.append({
            "title":   title,
            "company": company,
            "start":   start.strftime("%Y-%m"),
            "end":     "present" if end >= date.today().replace(day=1) else end.strftime("%Y-%m"),
            "months":  months,
            "segment": "professional",  # default; reclassified by Layer 2b
            # Internal: store raw dates for Layer 2b classification.
            "_start_date": start,
            "_end_date":   end,
        })
        all_ranges.append((start, end))
        prev_end = m_end

    if not all_entries:
        return [], None

    total_months = _compute_months_no_overlap(all_ranges)
    return all_entries, total_months


# ---------------------------------------------------------------------------
# Layer 2b — Post-graduation classification
# ---------------------------------------------------------------------------

def _extract_graduation_year(education_text: str) -> tuple[int | None, str | None]:
    """
    Extract graduation year from education section using degree-level awareness.

    Returns (graduation_year, degree_level) or (None, None) if not found
    or if confidence is low (expected/pursuing/future year).
    """
    if not education_text:
        return None, None

    text_lower = education_text.lower()

    # Check for uncertainty markers in the education section.
    if _UNCERTAIN_PATTERNS.search(education_text):
        return None, None

    best_rank = 0
    best_year: int | None = None
    best_level: str | None = None

    # Scan for each degree keyword and find the associated year.
    for keyword, (rank, level) in _DEGREE_HIERARCHY.items():
        # Find keyword in text.
        idx = text_lower.find(keyword)
        if idx == -1:
            continue

        # Search for a 4-digit year near this keyword (within ~100 chars).
        context = education_text[max(0, idx - 30): idx + len(keyword) + 100]
        year_matches = re.findall(r"((?:19|20)\d{2})", context)

        for ym in year_matches:
            year = int(ym)
            if year > _CURRENT_YEAR:
                # Future year — uncertain.
                continue
            if year < 1970:
                continue

            # Pick the latest year from the highest degree level.
            if rank > best_rank or (rank == best_rank and year > (best_year or 0)):
                best_rank = rank
                best_year = year
                best_level = level

    # If no degree keyword matched, try generic year extraction as fallback.
    if best_year is None:
        year_matches = re.findall(r"((?:19|20)\d{2})", education_text)
        for ym in year_matches:
            year = int(ym)
            if 1970 <= year <= _CURRENT_YEAR:
                if best_year is None or year > best_year:
                    best_year = year

    return best_year, best_level


def _classify_timeline(
    entries: list[dict],
    graduation_year: int | None,
) -> tuple[list[dict], float]:
    """
    Classify each timeline entry as pre_graduation / professional / transitional.

    Returns (classified_entries, professional_months).
    If graduation_year is None, all entries are classified as "professional".
    """
    if graduation_year is None:
        # No graduation year — all jobs count as professional.
        total = sum(e["months"] for e in entries)
        return entries, total

    grad_date = _to_date(graduation_year, 6)  # Assume mid-year graduation (June).

    professional_ranges: list[tuple[date, date]] = []

    for entry in entries:
        start = entry["_start_date"]
        end = entry["_end_date"]

        if end < grad_date:
            entry["segment"] = "pre_graduation"
        elif start >= grad_date:
            entry["segment"] = "professional"
            professional_ranges.append((start, end))
        else:
            # Transitional: spans graduation date.
            entry["segment"] = "transitional"
            # Credit only the post-graduation portion.
            professional_ranges.append((grad_date, end))
            # Update months to reflect only the credited portion.
            credited_months = max(0, (end.year - grad_date.year) * 12 + (end.month - grad_date.month))
            entry["months"] = credited_months

    professional_months = _compute_months_no_overlap(professional_ranges)
    return entries, professional_months


# ---------------------------------------------------------------------------
# Layer 3 — Graduation year fallback
# ---------------------------------------------------------------------------

def _layer3_graduation_fallback(
    education_text: str,
) -> tuple[float | None, int | None, str | None]:
    """
    When no job dates are found, estimate experience from graduation year.

    Returns (total_years, graduation_year, degree_level) or (None, None, None).
    """
    grad_year, degree_level = _extract_graduation_year(education_text)
    if grad_year is None:
        return None, None, None

    floor_years = max(0.0, _CURRENT_YEAR - grad_year)
    return floor_years, grad_year, degree_level


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_experience(
    full_text: str,
    experience_text: str = "",
    education_text: str = "",
) -> ExperienceProfile:
    """
    Extract structured experience data from resume text.

    Uses a three-layer cascade:
      Layer 1: Explicit phrase scan (e.g. "9+ years of experience")
      Layer 2: Timeline date math + post-graduation classification
      Layer 3: Graduation year fallback

    Parameters
    ----------
    full_text : str
        The complete resume text (used for Layer 1 phrase scan).
    experience_text : str
        The experience/work-history section text (used for Layer 2).
    education_text : str
        The education section text (used for graduation year extraction).

    Returns
    -------
    ExperienceProfile
        Structured experience data with total_years, timeline, etc.
    """
    profile = ExperienceProfile()

    # ── Layer 1: Explicit phrase scan ──────────────────────────────────
    explicit = _layer1_explicit(full_text)
    if explicit is not None:
        profile.total_years = explicit
        profile.total_years_all = explicit
        profile.inferred = False
        profile.extraction_method = "explicit"

        # Still try to extract graduation year for metadata.
        grad_year, degree_level = _extract_graduation_year(education_text)
        profile.graduation_year = grad_year
        profile.degree_level = degree_level

        # Still try to build timeline for the synthetic chunk.
        entries, _ = _layer2_timeline(experience_text, full_text)
        if entries:
            entries, _ = _classify_timeline(entries, grad_year)
            # Clean internal fields before storing.
            for e in entries:
                e.pop("_start_date", None)
                e.pop("_end_date", None)
            profile.timeline = entries

        return profile

    # ── Layer 2: Timeline calculation ──────────────────────────────────
    entries, total_months = _layer2_timeline(experience_text, full_text)
    if entries and total_months is not None:
        # Extract graduation year for classification.
        grad_year, degree_level = _extract_graduation_year(education_text)
        profile.graduation_year = grad_year
        profile.degree_level = degree_level

        # Layer 2b: Post-graduation classification.
        entries, professional_months = _classify_timeline(entries, grad_year)

        # Clean internal date fields.
        for e in entries:
            e.pop("_start_date", None)
            e.pop("_end_date", None)

        profile.timeline = entries
        profile.total_years_all = round(total_months / 12, 1)
        profile.total_years = round(professional_months / 12, 1)
        profile.inferred = True
        profile.extraction_method = "timeline"
        return profile

    # ── Layer 3: Graduation year fallback ──────────────────────────────
    floor_years, grad_year, degree_level = _layer3_graduation_fallback(education_text)
    if floor_years is not None:
        profile.total_years = floor_years
        profile.total_years_all = floor_years
        profile.graduation_year = grad_year
        profile.degree_level = degree_level
        profile.inferred = True
        profile.experience_floor_only = True
        profile.extraction_method = "graduation"
        return profile

    # ── All layers failed ──────────────────────────────────────────────
    profile.extraction_method = "unknown"
    return profile


def infer_seniority_from_years(years: float | None) -> str:
    """
    Infer seniority level from total experience years.

    Applied at index time (no LLM needed).
    """
    if years is None:
        return "unknown"
    if years <= 2:
        return "junior"
    if years <= 5:
        return "mid"
    if years <= 9:
        return "senior"
    return "staff"


def build_experience_summary(profile: ExperienceProfile) -> str:
    """
    Generate the synthetic experience_summary chunk text.

    Used by Phase 4 (chunker.py) to prepend a structured experience
    summary to the candidate's chunk list.
    """
    if profile.extraction_method == "unknown":
        return (
            "EXPERIENCE SUMMARY\n"
            "Total experience: unknown\n"
            "Graduation year : unknown"
        )

    seniority = infer_seniority_from_years(profile.total_years)

    lines = [
        "EXPERIENCE SUMMARY (auto-generated)",
        "=====================================",
    ]

    # Total years.
    if profile.total_years is not None:
        lines.append(f"Total work experience : {profile.total_years} years (professional, post-graduation)")
    if profile.total_years_all is not None and profile.total_years_all != profile.total_years:
        lines.append(f"All work experience   : {profile.total_years_all} years (including pre-graduation)")

    # Graduation info.
    grad_info = str(profile.graduation_year) if profile.graduation_year else "unknown"
    if profile.degree_level:
        grad_info += f" ({profile.degree_level.title()})"
    lines.append(f"College graduation    : {grad_info}")

    lines.append(f"Seniority level       : {seniority.title()}")
    lines.append(f"Extraction method     : {profile.extraction_method}")

    if profile.experience_floor_only:
        lines.append("Note                  : experience estimated from graduation year only (no job dates found)")

    # Timeline.
    if profile.timeline:
        lines.append("")
        lines.append("Career timeline:")
        for entry in profile.timeline:
            start = entry.get("start", "?")
            end = entry.get("end", "?")
            title = entry.get("title", "")
            company = entry.get("company", "")
            segment = entry.get("segment", "professional")

            end_display = "Present" if end == "present" else end
            title_display = title[:40].ljust(40) if title else "".ljust(40)
            company_display = f" — {company}" if company else ""
            lines.append(f"  [{start} → {end_display:>8}]  {title_display}{company_display} [{segment}]")

    return "\n".join(lines)
