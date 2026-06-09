"""
Shared helpers — PDF / DOCX text extraction.

PyMuPDF (fitz) is substantially better than pdf-parse or PyPDF2 at handling:
  - multi-column résumé layouts
  - embedded tables
  - non-ASCII characters and special bullets
"""

import io
import fitz                          # PyMuPDF
import docx                          # python-docx


def extract_text_from_pdf(data: bytes) -> str:
    """Return plain text extracted from a PDF byte-string."""
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for page in doc:
        # Use layout-aware extraction to preserve column order.
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages).strip()


def extract_text_from_docx(data: bytes) -> str:
    """Return plain text extracted from a DOCX byte-string."""
    doc = docx.Document(io.BytesIO(data))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs).strip()


def extract_text(filename: str, data: bytes) -> str:
    """
    Dispatch to the right extractor based on file extension.
    Raises ValueError for unsupported file types.
    """
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(data)
    if lower.endswith(".docx"):
        return extract_text_from_docx(data)
    raise ValueError(f"Unsupported file type: {filename!r}. Use PDF or DOCX.")
