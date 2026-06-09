"""
Resume router — handles file upload, parsing, chunking, and indexing.

Strategy A / Phase A changes
─────────────────────────────
The ingestion pipeline now runs section-aware chunking:

  1. Read bytes and enforce size limit.             (unchanged)
  2. Extract text with PyMuPDF / python-docx.       (unchanged)
  3a. Detect sections via sectioner.split_into_sections().
  3b. Chunk each section via chunker.chunk_section() with per-section tuning.
  4. Upsert section-tagged chunks to Pinecone with candidate_name metadata.
  5. Record candidate in the local SQLite registry (new — powers GET /api/resumes).

Bulk upload
───────────
POST /api/resumes/bulk-upload accepts up to _MAX_BULK_FILES files in one
multipart request.  Files are processed sequentially through the same
pipeline; per-file failures are captured as soft errors so the rest of
the batch still succeeds.

Response shape is aligned with the Retro Recruit Console client contract:
  {
    "total_files":   <int>,
    "indexed_count": <int>,
    "soft_errors":   [{"filename": ..., "reason": ...}],
    "candidates":    [{"candidate_id": ..., "filename": ..., "chunks_indexed": ...}]
  }

GET /api/resumes
────────────────
Returns the full candidate directory from the local SQLite registry.
Response:
  {
    "candidates": [
      {"candidate_id": ..., "name": ..., "chunks": ..., "indexed_at": ...}
    ]
  }
"""

import re
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.config import settings
from src.utils.parser import extract_text
from src.utils.sectioner import split_into_sections
from src.utils.chunker import chunk_section
from src.utils.experience_extractor import (
    extract_experience,
    build_experience_summary,
    infer_seniority_from_years,
)
from src.services import pinecone_service
from src.services import registry_service

router = APIRouter(prefix="/api/resumes", tags=["resumes"])

# Maximum number of files accepted in a single bulk-upload call.
_MAX_BULK_FILES: int = 20


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    candidate_id:   str
    filename:       str
    chunks_indexed: int
    message:        str


class DeleteResponse(BaseModel):
    candidate_id: str
    message:      str


# ── Vector inspection endpoints ──────────────────────────────────────────────

class VectorListResponse(BaseModel):
    """
    Response from GET /api/resumes/vectors.

    Fields
    ------
    namespace   : the Pinecone namespace that was queried.
    total       : total number of IDs returned.
    ids         : sorted list of all matching vector IDs.
    """
    namespace: str
    total:     int
    ids:       list[str]


class VectorFetchRequest(BaseModel):
    """
    Request body for POST /api/resumes/vectors/fetch.

    Fields
    ------
    ids       : list of vector IDs to retrieve (1–1000 per call).
    namespace : Pinecone namespace to query (defaults to "default").
    """
    ids:       list[str]
    namespace: str = "default"


class VectorRecord(BaseModel):
    """A single fetched vector record."""
    id:       str
    values:   list[float]
    metadata: dict


class VectorFetchResponse(BaseModel):
    """
    Response from POST /api/resumes/vectors/fetch.

    Fields
    ------
    namespace : the Pinecone namespace that was queried.
    fetched   : number of IDs that were found and returned.
    vectors   : one VectorRecord per found ID.
    """
    namespace: str
    fetched:   int
    vectors:   list[VectorRecord]


# ── Bulk upload (client-contract shape) ─────────────────────────────────────

class IngestionRecord(BaseModel):
    """One successfully indexed resume inside BulkUploadResponse."""
    candidate_id:   str
    filename:       str
    chunks_indexed: int


class SoftError(BaseModel):
    """One file that could not be indexed inside BulkUploadResponse."""
    filename: str
    reason:   str


class BulkUploadResponse(BaseModel):
    """
    Response from POST /api/resumes/bulk-upload.

    Field names match the Retro Recruit Console client contract (types.ts):
      total_files   — total number of files received
      indexed_count — number successfully indexed
      soft_errors   — per-file failures (non-fatal; rest of batch continues)
      candidates    — indexed candidate records
    """
    total_files:   int
    indexed_count: int
    soft_errors:   list[SoftError]
    candidates:    list[IngestionRecord]


# ── Directory (GET /api/resumes) ─────────────────────────────────────────────

class DirectoryCandidate(BaseModel):
    candidate_id: str
    name:         str
    filename:     str
    chunks:       int
    indexed_at:   str


class ResumeDirectoryResponse(BaseModel):
    candidates: list[DirectoryCandidate]


# ---------------------------------------------------------------------------
# Internal pipeline result (used between helpers)
# ---------------------------------------------------------------------------

class _PipelineResult:
    """Internal (not exposed in API) result of the ingestion pipeline."""
    __slots__ = ("filename", "candidate_id", "chunks_indexed", "name", "error")

    def __init__(
        self,
        filename:       str,
        candidate_id:   str | None = None,
        chunks_indexed: int = 0,
        name:           str = "",
        error:          str = "",
    ):
        self.filename       = filename
        self.candidate_id   = candidate_id
        self.chunks_indexed = chunks_indexed
        self.name           = name
        self.error          = error

    @property
    def ok(self) -> bool:
        return not self.error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches lines that look like a name: 5–60 chars, no "@" or digits,
# allowing letters, spaces, hyphens, apostrophes, and common diacritics.
_NAME_RE = re.compile(r"^[^\d@]{5,60}$")


def _extract_candidate_name(text: str) -> str:
    """
    Best-effort extraction of the candidate's name from the top of the resume.

    Scans the first five non-empty lines and returns the first that:
      - is 5–60 characters long,
      - contains no "@" (so it is not an e-mail),
      - contains no digits (so it is not a phone number or address line).

    Returns an empty string if no such line is found — callers must handle
    the empty-string case gracefully (it simply means no attribution stored).
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and _NAME_RE.match(stripped):
            return stripped
    return ""


async def _process_single_resume(file: UploadFile) -> _PipelineResult:
    """
    Run the full ingestion pipeline for one UploadFile.

    Pipeline:
      1. Read bytes — enforce per-file size limit.
      2. Extract text with PyMuPDF / python-docx.
      3. Detect sections; chunk each section with per-section size tuning.
      4. Upsert section-tagged chunks to Pinecone (server-side embedding).
      5. Register candidate in the local SQLite directory.

    Returns a _PipelineResult.  Never raises — callers do not need try/except.
    """
    filename = file.filename or "resume.pdf"

    # ── 1. Size guard ──────────────────────────────────────────────────────
    data = await file.read()
    if len(data) > settings.max_upload_size_bytes:
        return _PipelineResult(
            filename=filename,
            error=f"File exceeds {settings.max_upload_size_bytes // (1024 * 1024)} MB limit.",
        )

    # ── 2. Extract text ────────────────────────────────────────────────────
    try:
        text = extract_text(filename, data)
    except ValueError as exc:
        return _PipelineResult(filename=filename, error=str(exc))

    if not text.strip():
        return _PipelineResult(filename=filename, error="No readable text found in the file.")

    # ── 3. Section-aware chunking ──────────────────────────────────────────
    sections = split_into_sections(text)
    section_chunks: list[tuple[str, str]] = []
    for section_label, section_text in sections.items():
        section_chunks.extend(chunk_section(section_text, section_label))

    if not section_chunks:
        return _PipelineResult(filename=filename, error="Text could not be chunked.")

    # ── 3b. Phase C: Experience extraction ─────────────────────────────────
    experience_text = sections.get("experience", "")
    education_text = sections.get("education", "")
    exp_profile = extract_experience(
        full_text=text,
        experience_text=experience_text,
        education_text=education_text,
    )

    # Phase 4: Generate and prepend the synthetic experience_summary chunk.
    exp_summary_text = build_experience_summary(exp_profile)
    section_chunks.insert(0, (exp_summary_text, "experience_summary"))

    # Derive seniority for metadata storage.
    seniority = infer_seniority_from_years(exp_profile.total_years)

    # ── 4. Upsert to Pinecone ─────────────────────────────────────────────
    candidate_id   = str(uuid.uuid4())
    candidate_name = _extract_candidate_name(text)
    count = pinecone_service.upsert_resume_chunks(
        candidate_id        = candidate_id,
        filename            = filename,
        section_chunks      = section_chunks,
        candidate_name      = candidate_name,
        experience_years    = exp_profile.total_years,
        graduation_year     = exp_profile.graduation_year,
        experience_inferred = exp_profile.inferred,
        seniority_level     = seniority,
    )

    # ── 5. Register in local directory ────────────────────────────────────
    registry_service.register(
        candidate_id    = candidate_id,
        filename        = filename,
        chunks          = count,
        name            = candidate_name,
        experience_years = exp_profile.total_years,
        graduation_year  = exp_profile.graduation_year,
        seniority_level  = seniority,
    )

    return _PipelineResult(
        filename       = filename,
        candidate_id   = candidate_id,
        chunks_indexed = count,
        name           = candidate_name,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=ResumeDirectoryResponse, tags=["resumes"])
async def list_resumes():
    """
    List all indexed candidates from the local registry.

    Returns candidates sorted newest-first.  Each entry includes the
    candidate_id, detected name, filename, chunk count, and index timestamp.
    """
    rows = registry_service.list_all()
    return ResumeDirectoryResponse(
        candidates=[
            DirectoryCandidate(
                candidate_id = r["candidate_id"],
                name         = r["name"],
                filename     = r["filename"],
                chunks       = r["chunks"],
                indexed_at   = r["indexed_at"],
            )
            for r in rows
        ]
    )


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload a single resume (PDF or DOCX).

    Raises HTTP 413 / 415 / 422 on error so callers get a clear status code.
    For uploading multiple resumes in one call, use POST /bulk-upload instead.
    """
    result = await _process_single_resume(file)

    if not result.ok:
        detail = result.error
        if "exceeds" in detail and "MB" in detail:
            raise HTTPException(status_code=413, detail=detail)
        if "Unsupported file type" in detail:
            raise HTTPException(status_code=415, detail=detail)
        raise HTTPException(status_code=422, detail=detail)

    return UploadResponse(
        candidate_id   = result.candidate_id,   # type: ignore[arg-type]
        filename       = result.filename,
        chunks_indexed = result.chunks_indexed,
        message        = "Resume successfully indexed.",
    )


@router.post("/bulk-upload", response_model=BulkUploadResponse)
async def bulk_upload_resumes(
    files: list[UploadFile] = File(default=[]),
    file: list[UploadFile] = File(default=[]),
):
    """
    Upload multiple resumes (PDF or DOCX) in a single multipart request.

    Each file is run through the same ingestion pipeline as /upload.
    Per-file failures are captured as soft errors — the rest of the batch
    continues and is still indexed.

    Limits:
      - Maximum _MAX_BULK_FILES (20) files per call.
      - Per-file size limit is the same as /upload (MAX_UPLOAD_SIZE_MB).

    Returns HTTP 200 with:
      total_files   — total files received
      indexed_count — files successfully indexed
      soft_errors   — list of per-file failures
      candidates    — list of successfully indexed candidate records

    Example curl:
        curl -X POST http://localhost:8000/api/resumes/bulk-upload \\
          -F "files=@alice.pdf" \\
          -F "files=@bob.docx" \\
          -F "files=@carol.pdf"
    """
    all_files = files + file
    if not all_files:
        raise HTTPException(status_code=400, detail="No files provided.")

    if len(all_files) > _MAX_BULK_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {_MAX_BULK_FILES} files per request.",
        )

    candidates: list[IngestionRecord] = []
    soft_errors: list[SoftError]      = []

    for f in all_files:
        result = await _process_single_resume(f)
        if result.ok:
            candidates.append(IngestionRecord(
                candidate_id   = result.candidate_id,  # type: ignore[arg-type]
                filename       = result.filename,
                chunks_indexed = result.chunks_indexed,
            ))
        else:
            soft_errors.append(SoftError(
                filename = result.filename,
                reason   = result.error,
            ))

    return BulkUploadResponse(
        total_files   = len(all_files),
        indexed_count = len(candidates),
        soft_errors   = soft_errors,
        candidates    = candidates,
    )


@router.get("/vectors", response_model=VectorListResponse, tags=["vectors"])
async def list_vector_ids(
    prefix:    str | None = None,
    namespace: str = "default",
):
    """
    List all vector IDs stored in a Pinecone namespace.

    Query parameters
    ----------------
    prefix    (optional) — restrict results to IDs that start with this string.
                           Pass a ``candidate_id`` to scope to one candidate,
                           or omit to list every ID in the namespace.
    namespace (optional) — Pinecone namespace to query (default: "default").

    Returns
    -------
    A JSON object with:
      ``namespace`` — the namespace queried
      ``total``     — total number of IDs returned
      ``ids``       — sorted list of matching vector IDs

    Example
    -------
    # All IDs in the namespace:
    GET /api/resumes/vectors

    # Only IDs for one candidate:
    GET /api/resumes/vectors?prefix=<candidate_id>
    """
    ids = pinecone_service.list_namespace_ids(prefix=prefix, namespace=namespace)
    return VectorListResponse(namespace=namespace, total=len(ids), ids=ids)


_MAX_FETCH_IDS: int = 1_000  # Pinecone practical per-request limit


@router.post("/vectors/fetch", response_model=VectorFetchResponse, tags=["vectors"])
async def fetch_vectors(
    body: VectorFetchRequest,
):
    """
    Fetch raw vector records (dense embeddings + metadata) by explicit ID.

    Uses ``index.fetch()`` to retrieve dense embedding values and all stored
    metadata fields for the requested IDs.  IDs not present in the index are
    silently omitted from the response (Pinecone behaviour).

    Request body
    ------------
    ids       — list of vector IDs (1–1 000 per call; required)
    namespace — Pinecone namespace to query (default: "default")

    Returns
    -------
    A JSON object with:
      ``namespace`` — the namespace queried
      ``fetched``   — number of IDs found and returned
      ``vectors``   — list of records, each containing:
                        ``id``       — vector ID
                        ``values``   — dense embedding (list of floats)
                        ``metadata`` — all stored metadata fields

    Example
    -------
    POST /api/resumes/vectors/fetch
    {
      "ids": ["<candidate_id>#chunk0", "<candidate_id>#chunk1"],
      "namespace": "default"
    }
    """
    if not body.ids:
        raise HTTPException(status_code=400, detail="ids must not be empty.")

    if len(body.ids) > _MAX_FETCH_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many IDs. Maximum {_MAX_FETCH_IDS} IDs per request.",
        )

    raw_vectors = pinecone_service.fetch_vectors_by_ids(
        ids=body.ids,
        namespace=body.namespace,
    )

    vectors = [
        VectorRecord(
            id       = v["_id"],
            values   = v["values"],
            metadata = v["metadata"],
        )
        for v in raw_vectors
    ]

    return VectorFetchResponse(
        namespace = body.namespace,
        fetched   = len(vectors),
        vectors   = vectors,
    )


@router.delete("/{candidate_id}", response_model=DeleteResponse)
async def delete_resume(candidate_id: str):
    """Remove all indexed chunks for the given candidate from Pinecone and the registry."""
    pinecone_service.delete_resume(candidate_id)
    registry_service.remove(candidate_id)
    return DeleteResponse(
        candidate_id = candidate_id,
        message      = "Candidate vectors removed from index.",
    )
