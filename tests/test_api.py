"""
Integration tests for the FastAPI endpoints.

External services (Gemini, Pinecone) are mocked so tests run offline and
don't consume API credits.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import io
import fitz

from src.main import app

client = TestClient(app)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


# ── Health check ───────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_root_returns_ok(self):
        res = client.get("/")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"


# ── Resume upload ──────────────────────────────────────────────────────────

class TestResumeUpload:
    @patch("src.routers.resumes.registry_service")
    @patch("src.routers.resumes.pinecone_service")
    def test_upload_pdf_success(self, mock_pinecone, mock_registry):
        # Pinecone Integrated Inference: no separate embed call needed.
        # upsert_resume_chunks now accepts only text chunks (no embeddings arg).
        mock_pinecone.upsert_resume_chunks.return_value = 1
        mock_registry.register.return_value = None

        pdf = _make_pdf("John Doe — 5 years Python experience")

        res = client.post(
            "/api/resumes/upload",
            files={"file": ("john_doe.pdf", io.BytesIO(pdf), "application/pdf")},
        )

        assert res.status_code == 201
        body = res.json()
        assert body["chunks_indexed"] >= 1
        assert body["filename"] == "john_doe.pdf"
        assert "candidate_id" in body

    def test_upload_unsupported_format(self):
        res = client.post(
            "/api/resumes/upload",
            files={"file": ("notes.txt", io.BytesIO(b"plain text"), "text/plain")},
        )
        assert res.status_code == 415


# ── Resume delete ──────────────────────────────────────────────────────────

class TestResumeDelete:
    """
    Tests for DELETE /api/resumes/{candidate_id}.

    The underlying pinecone_service.delete_resume now uses prefix-based
    listing (index.list) followed by explicit ID deletion (index.delete)
    instead of the unsupported metadata-filter delete on Starter plan.
    We mock at the service level so the router logic is fully exercised.
    """

    @patch("src.routers.resumes.registry_service")
    @patch("src.routers.resumes.pinecone_service")
    def test_delete_known_candidate_returns_200(self, mock_pinecone, mock_registry):
        """Router returns 200 and the correct candidate_id on a successful delete."""
        mock_pinecone.delete_resume.return_value = None  # void function
        mock_registry.remove.return_value = None

        res = client.delete("/api/resumes/abc-123")

        assert res.status_code == 200
        body = res.json()
        assert body["candidate_id"] == "abc-123"
        assert "removed" in body["message"].lower()
        mock_pinecone.delete_resume.assert_called_once_with("abc-123")
        mock_registry.remove.assert_called_once_with("abc-123")

    @patch("src.routers.resumes.registry_service")
    @patch("src.routers.resumes.pinecone_service")
    def test_delete_unknown_candidate_is_noop(self, mock_pinecone, mock_registry):
        """
        Deleting a candidate_id that has no indexed chunks should still return
        200 — delete_resume is a no-op (not an error) for unknown IDs.
        """
        mock_pinecone.delete_resume.return_value = None
        mock_registry.remove.return_value = None

        res = client.delete("/api/resumes/does-not-exist")

        assert res.status_code == 200
        mock_pinecone.delete_resume.assert_called_once_with("does-not-exist")
        mock_registry.remove.assert_called_once_with("does-not-exist")


# ── Screening ─────────────────────────────────────────────────────────────

class TestJobScreening:
    @patch("src.routers.jobs.evaluate_candidate")
    @patch("src.routers.jobs.pinecone_service")
    def test_screen_returns_results(self, mock_pinecone, mock_llm):
        # Pinecone Integrated Inference returns a list of hit dicts
        # with top-level fields (_id, _score, candidate_id, chunk_text, filename).
        mock_hit = {
            "_id": "abc-123#chunk0",
            "_score": 0.92,
            "candidate_id": "abc-123",
            "filename": "john_doe.pdf",
            "chunk_text": "5 years Python, FastAPI, ML experience",
        }
        mock_pinecone.query_similar_chunks.return_value = [mock_hit]

        mock_llm.return_value = {
            "technical_skills_score": 88,
            "experience_relevance_score": 85,
            "experience_depth_score": 80,
            "education_score": 75,
            "certifications_score": 70,
            "communication_score": 82,
            "strengths": ["Python", "ML"],
            "gaps": ["Kubernetes"],
            "summary": "Strong backend engineer.",
            "recommendation": "Yes",
        }

        res = client.post(
            "/api/jobs/screen",
            json={
                "job_description": "We need a senior Python engineer with ML experience and cloud deployment skills.",
                "top_k": 5,
            },
        )

        assert res.status_code == 200
        body = res.json()
        assert body["total_candidates_evaluated"] == 1
        assert body["results"][0]["recommendation"] == "Yes"
        assert body["results"][0]["match_score"] > 0

    def test_screen_short_description_rejected(self):
        res = client.post(
            "/api/jobs/screen",
            json={"job_description": "Short", "top_k": 5},
        )
        assert res.status_code == 422  # Pydantic validation

    @patch("src.routers.jobs.evaluate_candidate")
    @patch("src.routers.jobs.pinecone_service")
    def test_screen_candidates_sorted_by_pinecone_score(self, mock_pinecone, mock_llm):
        """
        Step 4 regression test: when Pinecone returns hits for two candidates,
        the one with the HIGHER similarity score must be evaluated first
        (i.e. appear in the results list), regardless of insertion order.

        We set top_k=1 so only one candidate enters the LLM evaluation step;
        if the sort is missing the wrong candidate would win.
        """
        # low-score candidate arrives FIRST in Pinecone's response
        low_score_hit = {
            "_id": "low-cand#chunk0",
            "_score": 0.55,
            "candidate_id": "low-cand",
            "filename": "low_score.pdf",
            "chunk_text": "Junior developer with 1 year of experience.",
        }
        # high-score candidate arrives SECOND
        high_score_hit = {
            "_id": "high-cand#chunk0",
            "_score": 0.97,
            "candidate_id": "high-cand",
            "filename": "high_score.pdf",
            "chunk_text": "Senior Python ML engineer, 8 years experience.",
        }
        mock_pinecone.query_similar_chunks.return_value = [low_score_hit, high_score_hit]

        mock_llm.return_value = {
            "technical_skills_score": 90, "experience_relevance_score": 88,
            "experience_depth_score": 85, "education_score": 80,
            "certifications_score": 75,  "communication_score": 85,
            "strengths": ["Python"], "gaps": [],
            "summary": "Excellent candidate.", "recommendation": "Strong Yes",
        }

        res = client.post(
            "/api/jobs/screen",
            json={
                "job_description": "Senior Python ML engineer with 5+ years of experience and cloud skills.",
                "top_k": 1,  # only the top candidate should be evaluated
            },
        )

        assert res.status_code == 200
        body = res.json()
        # Only one result because top_k=1
        assert body["total_candidates_evaluated"] == 1
        # The high-score candidate (second in Pinecone response) must win
        assert body["results"][0]["candidate_id"] == "high-cand"
