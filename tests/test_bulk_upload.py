"""
Tests for POST /api/resumes/bulk-upload and GET /api/resumes.

External services (Pinecone, registry) are mocked so tests run offline.
The single-upload /upload endpoint is also re-tested here to confirm
it still works correctly after the _process_single_resume refactor.

Response contract (client-aligned)
───────────────────────────────────
POST /api/resumes/bulk-upload →
  {
    "total_files":   int,
    "indexed_count": int,
    "soft_errors":   [{"filename": str, "reason": str}],
    "candidates":    [{"candidate_id": str, "filename": str, "chunks_indexed": int}]
  }

GET /api/resumes →
  {
    "candidates": [
      {"candidate_id": str, "name": str, "filename": str, "chunks": int, "indexed_at": str}
    ]
  }
"""

import io
import pytest
import fitz
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pdf(text: str) -> bytes:
    """Create a minimal valid PDF containing *text*."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def _pdf_file(text: str, name: str = "resume.pdf"):
    """Return a (name, BytesIO, content-type) tuple for multipart upload."""
    return (name, io.BytesIO(_make_pdf(text)), "application/pdf")


def _txt_file(name: str = "notes.txt"):
    """Return an unsupported-format file tuple."""
    return (name, io.BytesIO(b"plain text content"), "text/plain")


# Patch both external services for every upload test.
_PATCH_PC  = "src.routers.resumes.pinecone_service"
_PATCH_REG = "src.routers.resumes.registry_service"


# ---------------------------------------------------------------------------
# Regression: single /upload still works after the refactor
# ---------------------------------------------------------------------------

class TestSingleUploadAfterRefactor:
    """Confirm /upload behaviour is unchanged after extracting the helper."""

    @patch(_PATCH_REG)
    @patch(_PATCH_PC)
    def test_upload_pdf_still_returns_201(self, mock_pc, mock_reg):
        mock_pc.upsert_resume_chunks.return_value = 3

        res = client.post(
            "/api/resumes/upload",
            files={"file": _pdf_file("Alice Smith — 5 years Python FastAPI")},
        )

        assert res.status_code == 201
        body = res.json()
        assert "candidate_id" in body
        assert body["filename"]       == "resume.pdf"
        assert body["chunks_indexed"] >= 1
        assert body["message"]        == "Resume successfully indexed."

    def test_upload_unsupported_format_still_returns_415(self):
        res = client.post(
            "/api/resumes/upload",
            files={"file": _txt_file()},
        )
        assert res.status_code == 415

    @patch(_PATCH_REG)
    @patch(_PATCH_PC)
    def test_upload_413_on_oversized_file(self, mock_pc, mock_reg):
        with patch("src.routers.resumes.settings") as mock_cfg:
            mock_cfg.max_upload_size_bytes = 1
            res = client.post(
                "/api/resumes/upload",
                files={"file": _pdf_file("Alice Smith")},
            )
        assert res.status_code == 413


# ---------------------------------------------------------------------------
# Bulk upload — happy paths
# ---------------------------------------------------------------------------

class TestBulkUploadSuccess:
    """POST /api/resumes/bulk-upload — fully successful batches."""

    @patch(_PATCH_REG)
    @patch(_PATCH_PC)
    def test_single_file_in_bulk_returns_200(self, mock_pc, mock_reg):
        mock_pc.upsert_resume_chunks.return_value = 5

        res = client.post(
            "/api/resumes/bulk-upload",
            files=[("files", _pdf_file("Alice Smith — Senior Engineer"))],
        )

        assert res.status_code == 200
        body = res.json()
        assert body["total_files"]   == 1
        assert body["indexed_count"] == 1
        assert body["soft_errors"]   == []
        assert len(body["candidates"]) == 1

    @patch(_PATCH_REG)
    @patch(_PATCH_PC)
    def test_bulk_upload_with_singular_file_parameter_succeeds(self, mock_pc, mock_reg):
        mock_pc.upsert_resume_chunks.return_value = 3

        res = client.post(
            "/api/resumes/bulk-upload",
            files=[
                ("file", _pdf_file("Alice Smith", "alice.pdf")),
                ("file", _pdf_file("Bob Jones", "bob.pdf")),
            ],
        )

        assert res.status_code == 200
        body = res.json()
        assert body["total_files"]   == 2
        assert body["indexed_count"] == 2
        assert body["soft_errors"]   == []
        assert len(body["candidates"]) == 2

    @patch(_PATCH_REG)
    @patch(_PATCH_PC)
    def test_three_files_all_succeed(self, mock_pc, mock_reg):
        mock_pc.upsert_resume_chunks.return_value = 8

        res = client.post(
            "/api/resumes/bulk-upload",
            files=[
                ("files", _pdf_file("Alice Smith — Python Engineer",  "alice.pdf")),
                ("files", _pdf_file("Bob Jones — ML Researcher",       "bob.pdf")),
                ("files", _pdf_file("Carol Lee — Data Scientist",      "carol.pdf")),
            ],
        )

        assert res.status_code == 200
        body = res.json()
        assert body["total_files"]   == 3
        assert body["indexed_count"] == 3
        assert body["soft_errors"]   == []
        assert len(body["candidates"]) == 3

    @patch(_PATCH_REG)
    @patch(_PATCH_PC)
    def test_each_candidate_has_required_fields(self, mock_pc, mock_reg):
        mock_pc.upsert_resume_chunks.return_value = 4

        res = client.post(
            "/api/resumes/bulk-upload",
            files=[
                ("files", _pdf_file("Alice Smith", "alice.pdf")),
                ("files", _pdf_file("Bob Jones",   "bob.pdf")),
            ],
        )

        for c in res.json()["candidates"]:
            assert c["candidate_id"]  is not None
            assert c["chunks_indexed"] > 0
            assert "filename" in c

    @patch(_PATCH_REG)
    @patch(_PATCH_PC)
    def test_each_candidate_preserves_original_filename(self, mock_pc, mock_reg):
        mock_pc.upsert_resume_chunks.return_value = 3

        res = client.post(
            "/api/resumes/bulk-upload",
            files=[
                ("files", _pdf_file("Alice Smith", "alice_cv.pdf")),
                ("files", _pdf_file("Bob Jones",   "bob_resume.pdf")),
            ],
        )

        names = [c["filename"] for c in res.json()["candidates"]]
        assert "alice_cv.pdf"   in names
        assert "bob_resume.pdf" in names

    @patch(_PATCH_REG)
    @patch(_PATCH_PC)
    def test_candidate_ids_are_unique_across_batch(self, mock_pc, mock_reg):
        mock_pc.upsert_resume_chunks.return_value = 5

        res = client.post(
            "/api/resumes/bulk-upload",
            files=[
                ("files", _pdf_file("Alice Smith", "alice.pdf")),
                ("files", _pdf_file("Bob Jones",   "bob.pdf")),
                ("files", _pdf_file("Carol Lee",   "carol.pdf")),
            ],
        )

        ids = [c["candidate_id"] for c in res.json()["candidates"]]
        assert len(ids) == len(set(ids)), "Every candidate_id must be unique."


# ---------------------------------------------------------------------------
# Bulk upload — partial failure (mixed batch)
# ---------------------------------------------------------------------------

class TestBulkUploadPartialFailure:
    """One bad file must not abort the rest of the batch."""

    @patch(_PATCH_REG)
    @patch(_PATCH_PC)
    def test_unsupported_file_fails_softly(self, mock_pc, mock_reg):
        mock_pc.upsert_resume_chunks.return_value = 4

        res = client.post(
            "/api/resumes/bulk-upload",
            files=[
                ("files", _pdf_file("Alice Smith", "alice.pdf")),
                ("files", _txt_file("bad.txt")),           # unsupported format
                ("files", _pdf_file("Carol Lee", "carol.pdf")),
            ],
        )

        assert res.status_code == 200
        body = res.json()
        assert body["total_files"]   == 3
        assert body["indexed_count"] == 2
        assert len(body["soft_errors"]) == 1

        err = body["soft_errors"][0]
        assert err["filename"] == "bad.txt"
        assert "Unsupported" in err["reason"]

    @patch(_PATCH_REG)
    @patch(_PATCH_PC)
    def test_oversized_file_fails_softly(self, mock_pc, mock_reg):
        mock_pc.upsert_resume_chunks.return_value = 3

        with patch("src.routers.resumes.settings") as mock_cfg:
            mock_cfg.max_upload_size_bytes = 1   # anything bigger → error
            res = client.post(
                "/api/resumes/bulk-upload",
                files=[
                    ("files", _pdf_file("Alice Smith", "alice.pdf")),  # too big
                    ("files", _pdf_file("Bob Jones",   "bob.pdf")),    # too big
                ],
            )

        assert res.status_code == 200
        body = res.json()
        assert len(body["soft_errors"]) == 2
        assert body["indexed_count"]    == 0

    @patch(_PATCH_REG)
    @patch(_PATCH_PC)
    def test_all_files_fail_returns_200_with_zero_indexed(self, mock_pc, mock_reg):
        """Even an all-failure batch must return 200, not 4xx/5xx."""
        res = client.post(
            "/api/resumes/bulk-upload",
            files=[
                ("files", _txt_file("a.txt")),
                ("files", _txt_file("b.txt")),
            ],
        )

        assert res.status_code == 200
        body = res.json()
        assert body["indexed_count"]      == 0
        assert len(body["soft_errors"])   == 2
        assert body["candidates"]         == []

    @patch(_PATCH_REG)
    @patch(_PATCH_PC)
    def test_successful_files_still_indexed_after_partial_failure(self, mock_pc, mock_reg):
        """Pinecone upsert must be called for every successful file."""
        mock_pc.upsert_resume_chunks.return_value = 5

        client.post(
            "/api/resumes/bulk-upload",
            files=[
                ("files", _pdf_file("Alice Smith", "alice.pdf")),
                ("files", _txt_file("bad.txt")),
                ("files", _pdf_file("Carol Lee",   "carol.pdf")),
            ],
        )

        # upsert_resume_chunks should have been called exactly twice
        # (once per valid PDF, not for the txt file).
        assert mock_pc.upsert_resume_chunks.call_count == 2


# ---------------------------------------------------------------------------
# Bulk upload — validation / limit guards
# ---------------------------------------------------------------------------

class TestBulkUploadValidation:
    """Request-level validation: empty batch, too many files."""

    def test_empty_files_list_returns_400(self):
        """Sending no files at all should be rejected with 400."""
        res = client.post("/api/resumes/bulk-upload", files=[])
        assert res.status_code in (400, 422)

    @patch(_PATCH_REG)
    @patch(_PATCH_PC)
    def test_exceeding_max_files_returns_400(self, mock_pc, mock_reg):
        """Batches larger than _MAX_BULK_FILES must be rejected with 400."""
        from src.routers.resumes import _MAX_BULK_FILES

        mock_pc.upsert_resume_chunks.return_value = 2

        files = [
            ("files", _pdf_file(f"Candidate {i}", f"resume_{i}.pdf"))
            for i in range(_MAX_BULK_FILES + 1)
        ]
        res = client.post("/api/resumes/bulk-upload", files=files)

        assert res.status_code == 400
        assert "Too many files" in res.json()["detail"]

    @patch(_PATCH_REG)
    @patch(_PATCH_PC)
    def test_exactly_max_files_is_accepted(self, mock_pc, mock_reg):
        """A batch of exactly _MAX_BULK_FILES must succeed."""
        from src.routers.resumes import _MAX_BULK_FILES

        mock_pc.upsert_resume_chunks.return_value = 2

        files = [
            ("files", _pdf_file(f"Candidate {i}", f"resume_{i}.pdf"))
            for i in range(_MAX_BULK_FILES)
        ]
        res = client.post("/api/resumes/bulk-upload", files=files)

        assert res.status_code == 200
        assert res.json()["total_files"] == _MAX_BULK_FILES


# ---------------------------------------------------------------------------
# Bulk upload — response shape contract
# ---------------------------------------------------------------------------

class TestBulkUploadResponseShape:
    """The response must always include all required top-level fields."""

    @patch(_PATCH_REG)
    @patch(_PATCH_PC)
    def test_response_has_all_top_level_fields(self, mock_pc, mock_reg):
        mock_pc.upsert_resume_chunks.return_value = 3

        body = client.post(
            "/api/resumes/bulk-upload",
            files=[("files", _pdf_file("Alice Smith", "alice.pdf"))],
        ).json()

        assert "total_files"   in body
        assert "indexed_count" in body
        assert "soft_errors"   in body
        assert "candidates"    in body

    @patch(_PATCH_REG)
    @patch(_PATCH_PC)
    def test_each_candidate_has_all_fields(self, mock_pc, mock_reg):
        mock_pc.upsert_resume_chunks.return_value = 2

        candidates = client.post(
            "/api/resumes/bulk-upload",
            files=[("files", _pdf_file("Alice Smith", "alice.pdf"))],
        ).json()["candidates"]

        for c in candidates:
            assert "filename"       in c
            assert "candidate_id"   in c
            assert "chunks_indexed" in c

    @patch(_PATCH_REG)
    @patch(_PATCH_PC)
    def test_total_files_equals_indexed_plus_errors(self, mock_pc, mock_reg):
        mock_pc.upsert_resume_chunks.return_value = 4

        body = client.post(
            "/api/resumes/bulk-upload",
            files=[
                ("files", _pdf_file("Alice", "a.pdf")),
                ("files", _txt_file("b.txt")),
                ("files", _pdf_file("Carol", "c.pdf")),
            ],
        ).json()

        assert body["total_files"] == body["indexed_count"] + len(body["soft_errors"])


# ---------------------------------------------------------------------------
# GET /api/resumes — resume directory
# ---------------------------------------------------------------------------

class TestResumeDirectory:
    """GET /api/resumes returns the candidate directory from the registry."""

    @patch(_PATCH_REG)
    def test_returns_200_with_candidates_key(self, mock_reg):
        mock_reg.list_all.return_value = []

        res = client.get("/api/resumes")

        assert res.status_code == 200
        assert "candidates" in res.json()

    @patch(_PATCH_REG)
    def test_maps_registry_rows_to_response(self, mock_reg):
        mock_reg.list_all.return_value = [
            {
                "candidate_id": "abc-123",
                "name":         "Alice Smith",
                "filename":     "alice.pdf",
                "chunks":       12,
                "indexed_at":   "2026-06-02T10:00:00+00:00",
            }
        ]

        body = client.get("/api/resumes").json()
        candidates = body["candidates"]

        assert len(candidates) == 1
        c = candidates[0]
        assert c["candidate_id"] == "abc-123"
        assert c["name"]         == "Alice Smith"
        assert c["filename"]     == "alice.pdf"
        assert c["chunks"]       == 12
        assert c["indexed_at"]   == "2026-06-02T10:00:00+00:00"

    @patch(_PATCH_REG)
    def test_empty_registry_returns_empty_list(self, mock_reg):
        mock_reg.list_all.return_value = []

        body = client.get("/api/resumes").json()
        assert body["candidates"] == []
