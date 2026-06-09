"""
Tests for all five /api/feedback/* endpoints (Phase 10).

The feedback_service (SQLite) is mocked throughout so tests:
  - run offline with no database writes
  - remain deterministic regardless of db state
  - are fast (no I/O)

Endpoints covered
-----------------
  POST  /api/feedback/decision   — record one HR decision
  POST  /api/feedback/bulk       — record multiple decisions at once
  GET   /api/feedback/stats      — summary statistics
  GET   /api/feedback/export     — export labelled pairs for fine-tuning
  GET   /api/feedback/decisions  — query stored decisions (with filters)
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_VALID_DECISION = {
    "job_id":          "job-001",
    "job_description": "Senior Python engineer with FastAPI and ML experience.",
    "candidate_id":    "cand-abc",
    "filename":        "john_doe.pdf",
    "resume_text":     "John Doe — 5 years Python FastAPI ML.",
    "decision":        "accepted",
    "match_score":     87.0,
    "dimension_scores": {"technical": 90, "relevance": 85},
    "notes":           "Strong fit",
    "recruiter_id":    "recruiter-alice",
}

_STATS_RESPONSE = {
    "total_decisions": 3,
    "unique_jobs":     1,
    "by_decision":     {"accepted": 1, "shortlisted": 1, "rejected": 1},
    "avg_match_score": 72.5,
}

_EXPORT_RESPONSE = {
    "total_pairs":       2,
    "positives":         1,
    "negatives":         1,
    "ready_to_finetune": False,
    "pairs": [
        {"query": "Senior Python…", "document": "John Doe…", "label": 1},
        {"query": "Senior Python…", "document": "Jane Smith…", "label": 0},
    ],
}


# ===========================================================================
# POST /api/feedback/decision
# ===========================================================================

class TestRecordDecision:
    """Single-decision recording endpoint."""

    @patch("src.routers.feedback.feedback_service")
    def test_valid_accepted_decision_returns_201(self, mock_svc):
        mock_svc.record_decision.return_value = 1  # new row id

        res = client.post("/api/feedback/decision", json=_VALID_DECISION)

        assert res.status_code == 201
        body = res.json()
        assert body["id"] == 1
        assert "accepted" in body["message"]

    @patch("src.routers.feedback.feedback_service")
    def test_valid_shortlisted_decision_returns_201(self, mock_svc):
        mock_svc.record_decision.return_value = 2

        payload = {**_VALID_DECISION, "decision": "shortlisted"}
        res = client.post("/api/feedback/decision", json=payload)

        assert res.status_code == 201
        assert "shortlisted" in res.json()["message"]

    @patch("src.routers.feedback.feedback_service")
    def test_valid_rejected_decision_returns_201(self, mock_svc):
        mock_svc.record_decision.return_value = 3

        payload = {**_VALID_DECISION, "decision": "rejected"}
        res = client.post("/api/feedback/decision", json=payload)

        assert res.status_code == 201
        assert "rejected" in res.json()["message"]

    def test_invalid_decision_value_rejected_by_pydantic(self):
        """Pattern validator on 'decision' field must block unknown values."""
        payload = {**_VALID_DECISION, "decision": "maybe"}
        res = client.post("/api/feedback/decision", json=payload)
        assert res.status_code == 422

    def test_missing_required_fields_rejected(self):
        """job_id, job_description, candidate_id and decision are all required."""
        res = client.post("/api/feedback/decision", json={"decision": "accepted"})
        assert res.status_code == 422

    def test_job_description_too_short_rejected(self):
        """job_description has min_length=10."""
        payload = {**_VALID_DECISION, "job_description": "Too short"}
        res = client.post("/api/feedback/decision", json=payload)
        assert res.status_code == 422

    @patch("src.routers.feedback.feedback_service")
    def test_service_value_error_becomes_422(self, mock_svc):
        """If feedback_service raises ValueError the router must return 422."""
        mock_svc.record_decision.side_effect = ValueError("Invalid decision 'bad'.")

        payload = {**_VALID_DECISION, "decision": "accepted"}  # passes Pydantic
        res = client.post("/api/feedback/decision", json=payload)

        assert res.status_code == 422

    @patch("src.routers.feedback.feedback_service")
    def test_service_called_with_correct_kwargs(self, mock_svc):
        """All DecisionRequest fields must be forwarded to feedback_service."""
        mock_svc.record_decision.return_value = 5

        client.post("/api/feedback/decision", json=_VALID_DECISION)

        mock_svc.record_decision.assert_called_once_with(
            job_id          ="job-001",
            job_description ="Senior Python engineer with FastAPI and ML experience.",
            candidate_id    ="cand-abc",
            filename        ="john_doe.pdf",
            resume_text     ="John Doe — 5 years Python FastAPI ML.",
            decision        ="accepted",
            match_score     =87.0,
            dimension_scores={"technical": 90, "relevance": 85},
            notes           ="Strong fit",
            recruiter_id    ="recruiter-alice",
        )

    @patch("src.routers.feedback.feedback_service")
    def test_optional_fields_default_correctly(self, mock_svc):
        """filename, resume_text, notes, recruiter_id all have defaults."""
        mock_svc.record_decision.return_value = 6

        minimal = {
            "job_id":          "job-001",
            "job_description": "Senior Python engineer with 5+ years of experience.",
            "candidate_id":    "cand-xyz",
            "decision":        "rejected",
        }
        res = client.post("/api/feedback/decision", json=minimal)

        assert res.status_code == 201
        call_kwargs = mock_svc.record_decision.call_args.kwargs
        assert call_kwargs["filename"]     == ""
        assert call_kwargs["resume_text"]  == ""
        assert call_kwargs["notes"]        == ""
        assert call_kwargs["recruiter_id"] == ""
        assert call_kwargs["match_score"]  is None


# ===========================================================================
# POST /api/feedback/bulk
# ===========================================================================

class TestRecordBulk:
    """Bulk decision recording endpoint."""

    @patch("src.routers.feedback.feedback_service")
    def test_bulk_two_decisions_returns_201(self, mock_svc):
        mock_svc.record_decision.side_effect = [10, 11]  # two row ids

        payload = {
            "decisions": [
                {**_VALID_DECISION, "candidate_id": "cand-1", "decision": "accepted"},
                {**_VALID_DECISION, "candidate_id": "cand-2", "decision": "rejected"},
            ]
        }
        res = client.post("/api/feedback/bulk", json=payload)

        assert res.status_code == 201
        body = res.json()
        assert body["recorded"] == 2
        assert body["total"]    == 2

    @patch("src.routers.feedback.feedback_service")
    def test_bulk_partial_failure_counts_successes(self, mock_svc):
        """
        If one decision raises ValueError the rest should still be recorded.
        The response should reflect the actual recorded count, not total.
        """
        mock_svc.record_decision.side_effect = [ValueError("bad"), 12]

        payload = {
            "decisions": [
                {**_VALID_DECISION, "candidate_id": "cand-fail",  "decision": "accepted"},
                {**_VALID_DECISION, "candidate_id": "cand-ok",    "decision": "rejected"},
            ]
        }
        res = client.post("/api/feedback/bulk", json=payload)

        assert res.status_code == 201
        body = res.json()
        assert body["recorded"] == 1
        assert body["total"]    == 2

    def test_empty_decisions_list_rejected(self):
        """min_length=1 on the decisions list must block empty lists."""
        res = client.post("/api/feedback/bulk", json={"decisions": []})
        assert res.status_code == 422

    @patch("src.routers.feedback.feedback_service")
    def test_bulk_calls_service_once_per_decision(self, mock_svc):
        """feedback_service.record_decision must be called for each item."""
        mock_svc.record_decision.side_effect = [1, 2, 3]

        payload = {
            "decisions": [
                {**_VALID_DECISION, "candidate_id": f"cand-{i}", "decision": "rejected"}
                for i in range(3)
            ]
        }
        client.post("/api/feedback/bulk", json=payload)

        assert mock_svc.record_decision.call_count == 3


# ===========================================================================
# GET /api/feedback/stats
# ===========================================================================

class TestFeedbackStats:
    """Summary statistics endpoint."""

    @patch("src.routers.feedback.feedback_service")
    def test_stats_returns_200_with_expected_shape(self, mock_svc):
        mock_svc.get_stats.return_value = _STATS_RESPONSE

        res = client.get("/api/feedback/stats")

        assert res.status_code == 200
        body = res.json()
        assert "total_decisions"  in body
        assert "unique_jobs"      in body
        assert "by_decision"      in body
        assert "avg_match_score"  in body

    @patch("src.routers.feedback.feedback_service")
    def test_stats_delegates_to_service(self, mock_svc):
        """The endpoint must return exactly what the service returns."""
        mock_svc.get_stats.return_value = _STATS_RESPONSE

        res = client.get("/api/feedback/stats")

        mock_svc.get_stats.assert_called_once()
        assert res.json() == _STATS_RESPONSE

    @patch("src.routers.feedback.feedback_service")
    def test_stats_empty_db_returns_zeros(self, mock_svc):
        mock_svc.get_stats.return_value = {
            "total_decisions": 0,
            "unique_jobs":     0,
            "by_decision":     {},
            "avg_match_score": None,
        }

        res = client.get("/api/feedback/stats")

        assert res.status_code == 200
        assert res.json()["total_decisions"] == 0
        assert res.json()["avg_match_score"] is None


# ===========================================================================
# GET /api/feedback/export
# ===========================================================================

class TestFeedbackExport:
    """Training-pair export endpoint."""

    @patch("src.routers.feedback.feedback_service")
    def test_export_default_min_pairs(self, mock_svc):
        """Without ?min_pairs the default of 10 is used."""
        mock_svc.export_training_pairs.return_value = _EXPORT_RESPONSE

        res = client.get("/api/feedback/export")

        assert res.status_code == 200
        mock_svc.export_training_pairs.assert_called_once_with(min_pairs=10)

    @patch("src.routers.feedback.feedback_service")
    def test_export_custom_min_pairs(self, mock_svc):
        """?min_pairs=25 must be forwarded to the service."""
        mock_svc.export_training_pairs.return_value = _EXPORT_RESPONSE

        res = client.get("/api/feedback/export?min_pairs=25")

        assert res.status_code == 200
        mock_svc.export_training_pairs.assert_called_once_with(min_pairs=25)

    @patch("src.routers.feedback.feedback_service")
    def test_export_response_shape(self, mock_svc):
        """Response must include total_pairs, positives, negatives, ready_to_finetune, pairs."""
        mock_svc.export_training_pairs.return_value = _EXPORT_RESPONSE

        body = client.get("/api/feedback/export").json()

        assert "total_pairs"       in body
        assert "positives"         in body
        assert "negatives"         in body
        assert "ready_to_finetune" in body
        assert "pairs"             in body
        assert isinstance(body["pairs"], list)

    def test_export_min_pairs_zero_rejected(self):
        """min_pairs has ge=1 so 0 must be rejected by FastAPI query validation."""
        res = client.get("/api/feedback/export?min_pairs=0")
        assert res.status_code == 422

    @patch("src.routers.feedback.feedback_service")
    def test_export_not_ready_when_insufficient_pairs(self, mock_svc):
        mock_svc.export_training_pairs.return_value = {
            "total_pairs": 3, "positives": 2, "negatives": 1,
            "ready_to_finetune": False, "pairs": [],
        }

        body = client.get("/api/feedback/export?min_pairs=50").json()

        assert body["ready_to_finetune"] is False

    @patch("src.routers.feedback.feedback_service")
    def test_export_ready_when_sufficient_pairs(self, mock_svc):
        mock_svc.export_training_pairs.return_value = {
            "total_pairs": 60, "positives": 35, "negatives": 25,
            "ready_to_finetune": True, "pairs": [],
        }

        body = client.get("/api/feedback/export?min_pairs=50").json()

        assert body["ready_to_finetune"] is True


# ===========================================================================
# GET /api/feedback/decisions
# ===========================================================================

class TestListDecisions:
    """Decision query/audit endpoint."""

    @patch("src.routers.feedback.feedback_service")
    def test_list_all_no_filters(self, mock_svc):
        """Without any query params all decisions are returned."""
        mock_svc.get_decisions.return_value = [
            {"id": 1, "decision": "accepted", "candidate_id": "cand-a"},
            {"id": 2, "decision": "rejected", "candidate_id": "cand-b"},
        ]

        res = client.get("/api/feedback/decisions")

        assert res.status_code == 200
        body = res.json()
        assert body["count"] == 2
        assert len(body["decisions"]) == 2
        mock_svc.get_decisions.assert_called_once_with(
            job_id=None, decision_filter=None, limit=100
        )

    @patch("src.routers.feedback.feedback_service")
    def test_list_filter_by_job_id(self, mock_svc):
        """?job_id must be forwarded as the job_id filter."""
        mock_svc.get_decisions.return_value = []

        client.get("/api/feedback/decisions?job_id=job-001")

        mock_svc.get_decisions.assert_called_once_with(
            job_id="job-001", decision_filter=None, limit=100
        )

    @patch("src.routers.feedback.feedback_service")
    def test_list_filter_by_decision_type(self, mock_svc):
        """?decision=accepted must be forwarded as the decision_filter."""
        mock_svc.get_decisions.return_value = []

        client.get("/api/feedback/decisions?decision=accepted")

        mock_svc.get_decisions.assert_called_once_with(
            job_id=None, decision_filter="accepted", limit=100
        )

    @patch("src.routers.feedback.feedback_service")
    def test_list_combined_filters_and_limit(self, mock_svc):
        """All three query params must be forwarded together."""
        mock_svc.get_decisions.return_value = []

        client.get("/api/feedback/decisions?job_id=job-42&decision=rejected&limit=5")

        mock_svc.get_decisions.assert_called_once_with(
            job_id="job-42", decision_filter="rejected", limit=5
        )

    @patch("src.routers.feedback.feedback_service")
    def test_list_response_count_matches_decisions_length(self, mock_svc):
        """'count' in response must equal len(decisions)."""
        mock_svc.get_decisions.return_value = [
            {"id": i, "decision": "accepted"} for i in range(7)
        ]

        body = client.get("/api/feedback/decisions").json()

        assert body["count"] == 7
        assert len(body["decisions"]) == 7

    def test_list_limit_above_max_rejected(self):
        """limit has le=1000; sending 1001 must be rejected."""
        res = client.get("/api/feedback/decisions?limit=1001")
        assert res.status_code == 422

    def test_list_limit_below_min_rejected(self):
        """limit has ge=1; sending 0 must be rejected."""
        res = client.get("/api/feedback/decisions?limit=0")
        assert res.status_code == 422

    @patch("src.routers.feedback.feedback_service")
    def test_list_empty_result_returns_zero_count(self, mock_svc):
        mock_svc.get_decisions.return_value = []

        body = client.get("/api/feedback/decisions").json()

        assert body["count"] == 0
        assert body["decisions"] == []
