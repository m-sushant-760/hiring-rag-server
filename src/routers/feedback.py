"""
Feedback router — Phase 10: HR Decision API
============================================
REST endpoints for HR to submit decisions and export training data.

Endpoints
---------
  POST  /api/feedback/decision   — record one HR decision
  POST  /api/feedback/bulk       — record multiple decisions at once
  GET   /api/feedback/stats      — summary statistics
  GET   /api/feedback/export     — export labelled pairs for fine-tuning
  GET   /api/feedback/decisions  — query stored decisions (with filters)
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.services import feedback_service

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


# ── Models ─────────────────────────────────────────────────────────────────

class DecisionRequest(BaseModel):
    job_id: str            = Field(..., description="Unique ID for this screening session / job posting.")
    job_description: str   = Field(..., min_length=10)
    candidate_id: str      = Field(...)
    filename: str          = Field(default="")
    resume_text: str       = Field(default="", description="Full resume text — stored for fine-tuning.")
    decision: str          = Field(..., pattern="^(accepted|shortlisted|rejected)$",
                                   description="accepted | shortlisted | rejected")
    match_score: float | None    = Field(None)
    dimension_scores: dict | None = Field(None)
    notes: str             = Field(default="")
    recruiter_id: str      = Field(default="")


class BulkRequest(BaseModel):
    decisions: list[DecisionRequest] = Field(..., min_length=1)


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/decision", status_code=201)
async def record_decision(body: DecisionRequest):
    """
    Record one HR decision for a candidate.

    Call this after the recruiter reviews a candidate from /api/jobs/screen.
    Decisions accumulate as a training dataset; once you have ≥ 50
    positive + negative pairs, export and fine-tune the cross-encoder.
    """
    try:
        row_id = feedback_service.record_decision(
            job_id=body.job_id,
            job_description=body.job_description,
            candidate_id=body.candidate_id,
            filename=body.filename,
            resume_text=body.resume_text,
            decision=body.decision,
            match_score=body.match_score,
            dimension_scores=body.dimension_scores,
            notes=body.notes,
            recruiter_id=body.recruiter_id,
        )
        return {"id": row_id, "message": f"Decision '{body.decision}' recorded."}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/bulk", status_code=201)
async def record_bulk(body: BulkRequest):
    """Record multiple HR decisions in one call (useful after reviewing a full shortlist)."""
    count = 0
    for d in body.decisions:
        try:
            feedback_service.record_decision(
                job_id=d.job_id,
                job_description=d.job_description,
                candidate_id=d.candidate_id,
                filename=d.filename,
                resume_text=d.resume_text,
                decision=d.decision,
                match_score=d.match_score,
                dimension_scores=d.dimension_scores,
                notes=d.notes,
                recruiter_id=d.recruiter_id,
            )
            count += 1
        except ValueError:
            pass
    return {"recorded": count, "total": len(body.decisions)}


@router.get("/stats")
async def stats():
    """
    Summary statistics about stored HR feedback.

    Shows total decisions, accepted/rejected breakdown, and average
    system match_score — lets you audit model accuracy vs human judgement.
    """
    return feedback_service.get_stats()


@router.get("/export")
async def export(
    min_pairs: int = Query(
        default=10, ge=1,
        description="Minimum pairs before ready_to_finetune=true."
    )
):
    """
    Export labelled (job_description, resume_text, label) pairs
    for cross-encoder fine-tuning.

      label=1  →  accepted / shortlisted
      label=0  →  rejected

    Download this JSON periodically and retrain when ready_to_finetune=true.
    """
    return feedback_service.export_training_pairs(min_pairs=min_pairs)


@router.get("/decisions")
async def list_decisions(
    job_id:   str | None = Query(None),
    decision: str | None = Query(None),
    limit:    int        = Query(default=100, ge=1, le=1000),
):
    """Query stored decisions for audit or reporting."""
    rows = feedback_service.get_decisions(
        job_id=job_id, decision_filter=decision, limit=limit
    )
    return {"count": len(rows), "decisions": rows}
