"""
Tests for src/utils/parser.py

Uses small in-memory PDFs and DOCX files to validate extraction logic
without relying on external fixture files.
"""

import pytest
import fitz  # PyMuPDF
import io
from docx import Document

from src.utils.parser import extract_text, extract_text_from_pdf, extract_text_from_docx


def _make_pdf(text: str) -> bytes:
    """Create a minimal single-page PDF in memory."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def _make_docx(text: str) -> bytes:
    """Create a minimal DOCX in memory."""
    doc = Document()
    for para in text.split("\n"):
        doc.add_paragraph(para)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


class TestPDFExtraction:
    def test_basic_text(self):
        pdf = _make_pdf("John Doe — Software Engineer")
        result = extract_text_from_pdf(pdf)
        assert "John Doe" in result

    def test_multiline(self):
        pdf = _make_pdf("Line 1\nLine 2\nLine 3")
        result = extract_text_from_pdf(pdf)
        assert "Line 1" in result
        assert "Line 3" in result


class TestDOCXExtraction:
    def test_basic_text(self):
        docx_bytes = _make_docx("Jane Smith\nData Scientist")
        result = extract_text_from_docx(docx_bytes)
        assert "Jane Smith" in result
        assert "Data Scientist" in result


class TestExtractText:
    def test_pdf_dispatch(self):
        pdf = _make_pdf("Test PDF content")
        result = extract_text("resume.pdf", pdf)
        assert "Test PDF content" in result

    def test_docx_dispatch(self):
        docx_bytes = _make_docx("Test DOCX content")
        result = extract_text("resume.docx", docx_bytes)
        assert "Test DOCX content" in result

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text("resume.txt", b"plain text")
