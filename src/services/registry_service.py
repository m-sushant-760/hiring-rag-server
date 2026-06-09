"""
Candidate registry — lightweight PostgreSQL store for indexed resume metadata.

We use Supabase (PostgreSQL) instead of SQLite to keep a record of every candidate
indexed so that GET /api/resumes can return a fast, accurate list, even when running
multiple backend instances on Render.
"""

import threading
from datetime import datetime, timezone
from psycopg.rows import dict_row
from src.services.db import get_pool

_lock = threading.Lock()

def init_db() -> None:
    """Create the candidates schema in PostgreSQL — safe to call repeatedly."""
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id     VARCHAR(255) PRIMARY KEY,
                    name             VARCHAR(255) NOT NULL DEFAULT '',
                    filename         VARCHAR(255) NOT NULL DEFAULT '',
                    chunks           INTEGER NOT NULL DEFAULT 0,
                    indexed_at       TIMESTAMPTZ NOT NULL,
                    experience_years DOUBLE PRECISION,
                    graduation_year  INTEGER,
                    seniority_level  VARCHAR(50)
                )
            """)
            conn.commit()

def register(
    candidate_id: str,
    filename: str,
    chunks: int,
    name: str = "",
    experience_years: float | None = None,
    graduation_year: int | None = None,
    seniority_level: str | None = None,
) -> None:
    """Insert or update a candidate record after successful indexing."""
    indexed_at = datetime.now(timezone.utc)
    with _lock:
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO candidates
                        (candidate_id, name, filename, chunks, indexed_at,
                         experience_years, graduation_year, seniority_level)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (candidate_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        filename = EXCLUDED.filename,
                        chunks = EXCLUDED.chunks,
                        indexed_at = EXCLUDED.indexed_at,
                        experience_years = EXCLUDED.experience_years,
                        graduation_year = EXCLUDED.graduation_year,
                        seniority_level = EXCLUDED.seniority_level
                    """,
                    (candidate_id, name, filename, chunks, indexed_at,
                     experience_years, graduation_year, seniority_level),
                )
                conn.commit()

def remove(candidate_id: str) -> None:
    """Delete a candidate record when its vectors are removed from Pinecone."""
    with _lock:
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM candidates WHERE candidate_id = %s",
                    (candidate_id,)
                )
                conn.commit()

def list_all() -> list[dict]:
    """
    Return all indexed candidates sorted by indexed_at descending (newest first).
    """
    with _lock:
        with get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT candidate_id, name, filename, chunks, indexed_at,
                           experience_years, graduation_year, seniority_level
                    FROM   candidates
                    ORDER  BY indexed_at DESC
                    """
                )
                return cur.fetchall()
