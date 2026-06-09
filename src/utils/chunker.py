"""
Text chunker — splits a raw string into overlapping token-aware windows.

Strategy A / Phase A adds `chunk_section`, which accepts a section label
and returns (chunk_text, section_label) tuples with per-section chunk-size
tuning.  The original `chunk_text` function is preserved unchanged for
backward compatibility with any existing callers and tests.

Per-section defaults (overridable via config.py env vars):
  skills        → 400 chars, 0 overlap   (dense bullet lists)
  experience    → 1200 chars, 150 overlap (narratives; preserve role context)
  education     → 600 chars, 50 overlap  (short, factual blocks)
  summary       → 600 chars, 50 overlap  (usually a single chunk)
  other / rest  → 1000 chars, 100 overlap (previous default, unchanged)
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import settings

# ---------------------------------------------------------------------------
# Legacy constants — unchanged; used by the original chunk_text() function.
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1_000   # characters
CHUNK_OVERLAP = 100  # characters — ~10% ensures cross-boundary context

# ---------------------------------------------------------------------------
# Per-section chunk parameters.
# Values are read from config so they can be tuned via .env without code edits.
# ---------------------------------------------------------------------------

_SECTION_PARAMS: dict[str, tuple[int, int]] = {
    # section_label → (chunk_size, chunk_overlap)
    "skills":              (settings.section_chunk_size_skills,      0),
    "experience":          (settings.section_chunk_size_experience,   150),
    "education":           (600,                                       50),
    "summary":             (600,                                       50),
    "certifications":      (600,                                       50),
    "projects":            (800,                                      100),
    "publications":        (800,                                      100),
    "languages":           (400,                                        0),
    "awards":              (400,                                        0),
    "experience_summary":  (600,                                        0),  # Phase C: synthetic chunk
}

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_section(text: str, section: str) -> list[tuple[str, str]]:
    """
    Split *text* (the body of one resume section) into overlapping chunks
    and tag every chunk with *section*.

    Returns a list of (chunk_text, section_label) tuples.
    Returns an empty list for blank input.

    The chunk size and overlap are chosen based on *section*; sections not
    in _SECTION_PARAMS fall back to the configurable defaults.
    """
    if not text or not text.strip():
        return []

    chunk_size, chunk_overlap = _SECTION_PARAMS.get(
        section,
        (settings.section_chunk_size_default, settings.section_chunk_overlap_default),
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=_SEPARATORS,
    )
    return [(chunk, section) for chunk in splitter.split_text(text)]


def chunk_text(text: str) -> list[str]:
    """
    Split *text* into overlapping chunks (original implementation).

    Preserved unchanged for backward compatibility — not called by the new
    section-aware ingestion path, but may be used by existing tests or
    other callers.

    Returns an empty list for blank input so callers don't need to guard.
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        # Try natural sentence / paragraph breaks first before hard splits.
        separators=_SEPARATORS,
    )
    return splitter.split_text(text)
