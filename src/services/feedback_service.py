"""
Feedback Service — Phase 10: HR Decision Feedback Loop
=======================================================
Persists HR accept / reject / shortlist decisions to PostgreSQL and exports
labelled (JD, resume, label) pairs for cross-encoder fine-tuning.

Storage: Supabase PostgreSQL (replaces SQLite feedback.db for scalability)
"""

import json
from datetime import datetime, timezone
from psycopg.rows import dict_row
from src.services.db import get_pool

_DDL = """
CREATE TABLE IF NOT EXISTS candidate_feedback (
    id               SERIAL PRIMARY KEY,
    job_id           VARCHAR(255) NOT NULL,
    job_description  TEXT NOT NULL,
    candidate_id     VARCHAR(255) NOT NULL,
    filename         VARCHAR(255) NOT NULL DEFAULT '',
    resume_text      TEXT NOT NULL DEFAULT '',
    decision         VARCHAR(50) NOT NULL
                     CHECK(decision IN ('accepted','shortlisted','rejected')),
    match_score      DOUBLE PRECISION,
    dimension_scores JSONB,
    notes            TEXT DEFAULT '',
    recruiter_id     VARCHAR(255) DEFAULT '',
    created_at       TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_job  ON candidate_feedback (job_id);
CREATE INDEX IF NOT EXISTS idx_dec  ON candidate_feedback (decision);
"""

def init_db() -> None:
    """Create schema — safe to call repeatedly."""
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL)
            conn.commit()

# Remove the top-level call to init_db() on import.
# It will be called from the lifespan startup event instead.

def record_decision(
    *,
    job_id: str,
    job_description: str,
    candidate_id: str,
    filename: str = "",
    resume_text: str = "",
    decision: str,                        # accepted | shortlisted | rejected
    match_score: float | None = None,
    dimension_scores: dict | None = None,
    notes: str = "",
    recruiter_id: str = "",
) -> int:
    """Persist one HR decision. Returns new row id."""
    if decision not in ("accepted", "shortlisted", "rejected"):
        raise ValueError(f"Invalid decision '{decision}'.")
    
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO candidate_feedback
                   (job_id, job_description, candidate_id, filename, resume_text,
                    decision, match_score, dimension_scores, notes, recruiter_id, created_at)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   RETURNING id""",
                (job_id, job_description, candidate_id, filename, resume_text,
                 decision, match_score,
                 json.dumps(dimension_scores) if dimension_scores else None,
                 notes, recruiter_id,
                 datetime.now(timezone.utc)),
            )
            inserted_id = cur.fetchone()[0]
            conn.commit()
            return inserted_id

def get_decisions(
    job_id: str | None = None,
    decision_filter: str | None = None,
    limit: int = 500,
) -> list[dict]:
    """Return stored decisions with optional filters."""
    sql = "SELECT * FROM candidate_feedback WHERE 1=1"
    params: list = []
    if job_id:
        sql += " AND job_id = %s";   params.append(job_id)
    if decision_filter:
        sql += " AND decision = %s"; params.append(decision_filter)
    sql += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)
    
    with get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchall()

def export_training_pairs(min_pairs: int = 10) -> dict:
    """
    Export (query, document, label) triples for cross-encoder fine-tuning.

    label=1 → accepted / shortlisted
    label=0 → rejected
    """
    with get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT job_description, resume_text, decision "
                "FROM candidate_feedback "
                "WHERE decision IN ('accepted','shortlisted','rejected')"
            )
            rows = cur.fetchall()

    pairs = [
        {
            "query":    r["job_description"],
            "document": r["resume_text"],
            "label":    1 if r["decision"] in ("accepted", "shortlisted") else 0,
        }
        for r in rows
    ]
    positives = sum(1 for p in pairs if p["label"] == 1)
    negatives  = len(pairs) - positives
    return {
        "total_pairs":       len(pairs),
        "positives":         positives,
        "negatives":         negatives,
        "ready_to_finetune": len(pairs) >= min_pairs
                             and positives > 0 and negatives > 0,
        "pairs":             pairs,
    }

def get_stats() -> dict:
    """Summary statistics about stored HR feedback."""
    with get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT COUNT(*) as n FROM candidate_feedback")
            total = cur.fetchone()["n"]
            
            cur.execute(
                "SELECT decision, COUNT(*) as n FROM candidate_feedback GROUP BY decision"
            )
            by_dec = cur.fetchall()
            
            cur.execute(
                "SELECT COUNT(DISTINCT job_id) as n FROM candidate_feedback"
            )
            uniq_jobs = cur.fetchone()["n"]
            
            cur.execute(
                "SELECT AVG(match_score) as a FROM candidate_feedback WHERE match_score IS NOT NULL"
            )
            avg = cur.fetchone()["a"]
            
    return {
        "total_decisions":  total,
        "unique_jobs":      uniq_jobs,
        "by_decision":      {r["decision"]: r["n"] for r in by_dec},
        "avg_match_score":  round(avg, 2) if avg else None,
    }
