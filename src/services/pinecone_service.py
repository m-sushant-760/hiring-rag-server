"""
Pinecone service — index creation, text-native upsert, and semantic search.

Uses Pinecone Integrated Inference so raw text is sent directly to Pinecone,
which embeds it internally using `multilingual-e5-large`.
No external embedding API key is required.

Pinecone Starter plan: serverless, aws/us-east-1, 2GB storage, 5M tokens/month.

Strategy A / Phase A changes
─────────────────────────────
• upsert_resume_chunks now accepts section_chunks: list[tuple[str, str]] and
  an optional candidate_name.  Each record gains two new metadata fields:
    - section        : canonical section label ("experience", "skills", …)
    - candidate_name : first-line name heuristic (empty string if not detected)
• query_similar_chunks accepts an optional section_filter that is translated
  to a Pinecone metadata $eq filter.  Passing None (the default) preserves
  the previous unfiltered behaviour.
• Both new fields are added to the `fields` list returned by search() so
  callers receive them in every hit dict.

Backward compatibility
──────────────────────
Existing Pinecone records (indexed before this change) simply lack the
`section` and `candidate_name` fields.  They are still retrieved by vector
search and are returned with those fields absent/null — they are never
excluded unless an explicit section_filter is set.

delete_resume — Starter plan note
──────────────────────────────────
Pinecone serverless indexes (Starter / free tier) do NOT support
`delete(filter=...)`.  delete_resume therefore uses `index.list(prefix=…)`
to discover all chunk IDs that belong to the candidate (they share the
`{candidate_id}#chunk{n}` prefix) and deletes them by explicit ID list.
This works on every Pinecone plan.

SDK version note (pinecone >= 6.x)
────────────────────────────────────
The Integrated Inference API (upsert_records, search, IntegratedSpec) was
introduced in pinecone-python v6.  This file requires >= 6.0 and is tested
against v9.x.  Key v9 changes vs v5:
  • create_index: embed config moves into IntegratedSpec, not a top-level kwarg.
  • upsert_records / search: namespace must be a non-empty string ("default").
  • search: flat kwargs (inputs=, top_k=, filter=, fields=), not a nested query
    dict.  Returns a SearchRecordsResponse object; hits live in
    response.result.hits and each Hit has .id, .score, and .fields dict.
"""

from pinecone import Pinecone, IntegratedSpec
from pinecone.models.indexes.specs import EmbedConfig

from src.config import settings

_pc: Pinecone | None = None
_index = None

# Namespace used for all records.  Must be a non-empty string in pinecone >= 6.
_NAMESPACE = "default"


def _get_client() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=settings.pinecone_api_key)
    return _pc


def _get_index():
    """
    Return (or lazily create) the Pinecone serverless index.

    The index is created with an IntegratedSpec so Pinecone Integrated
    Inference automatically embeds any text sent via upsert_records / search.
    """
    global _index
    if _index is not None:
        return _index

    pc = _get_client()
    existing = [idx.name for idx in pc.list_indexes()]

    if settings.pinecone_index_name not in existing:
        pc.create_index(
            name=settings.pinecone_index_name,
            metric="cosine",
            # IntegratedSpec bundles cloud/region with the embedding model config.
            # This replaces the `embed=` top-level kwarg used in pinecone v5.
            spec=IntegratedSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
                embed=EmbedConfig(
                    model=settings.pinecone_embedding_model,
                    field_map={"text": "chunk_text"},
                ),
            ),
        )

    _index = pc.Index(settings.pinecone_index_name)
    return _index


def upsert_resume_chunks(
    candidate_id:   str,
    filename:       str,
    section_chunks: list[tuple[str, str]],
    candidate_name: str = "",
    experience_years:    float | None = None,
    graduation_year:     int | None = None,
    experience_inferred: bool = True,
    seniority_level:     str = "unknown",
) -> int:
    """
    Upsert all section-tagged text chunks for a single resume into Pinecone.

    *section_chunks* is a list of (chunk_text, section_label) tuples produced
    by `chunk_section()`.  Each record is stored with metadata fields:
      - section           : the canonical section label for filtered retrieval.
      - candidate_name    : used for attribution in search results.
      - experience_years  : Phase C — total extracted years (enables $gte filter).
      - graduation_year   : Phase C — most recent degree year.
      - experience_inferred: Phase C — was this calculated vs self-stated.
      - seniority_level   : Phase C — "junior" / "mid" / "senior" / "staff".

    Pinecone Integrated Inference converts `chunk_text` to a vector internally —
    no embeddings are computed on the application side.

    Returns the number of records upserted.
    """
    index = _get_index()

    records = []
    for i, (chunk, section) in enumerate(section_chunks):
        record = {
            "_id":            f"{candidate_id}#chunk{i}",
            "chunk_text":     chunk,           # field Pinecone will embed
            "candidate_id":   candidate_id,
            "filename":       filename,
            "chunk_index":    i,
            "section":        section,         # Phase A — section label
            "candidate_name": candidate_name,  # Phase A — attribution
            "seniority_level": seniority_level, # Phase C
        }
        # Phase C: add numeric fields only if available (avoid null metadata).
        if experience_years is not None:
            record["experience_years"] = experience_years
        if graduation_year is not None:
            record["graduation_year"] = graduation_year
        record["experience_inferred"] = experience_inferred
        records.append(record)

    # Pinecone recommends batches of ≤100 records.
    batch_size = 100
    for start in range(0, len(records), batch_size):
        index.upsert_records(
            namespace=_NAMESPACE,
            records=records[start : start + batch_size],
        )

    return len(records)


def query_similar_chunks(
    query_text:     str,
    top_k:          int | None = None,
    section_filter: str | None = None,
) -> list[dict]:
    """
    Find the *top_k* most relevant resume chunks for *query_text*.

    Pinecone embeds the query text server-side using the same model used at
    index time, then returns the closest matching records with their metadata.

    *section_filter* — if provided, restricts results to chunks whose
    `section` metadata field exactly equals this value (e.g. "skills",
    "experience").  Defaults to None (no filter — all sections returned).

    Returns a list of hit dicts each with: _id, _score, chunk_text,
    candidate_id, filename, chunk_index, section, candidate_name.
    Fields absent on pre-Phase-A records will be missing from the hit dict.
    """
    index = _get_index()
    k = top_k or settings.top_k_results

    # Build the optional metadata filter
    filter_expr: dict | None = None
    if section_filter:
        filter_expr = {"section": {"$eq": section_filter}}

    response = index.search(
        namespace=_NAMESPACE,
        inputs={"text": query_text},
        top_k=k,
        filter=filter_expr,
        fields=[
            "chunk_text", "candidate_id", "filename", "chunk_index",
            "section", "candidate_name",          # Phase A — new fields
            "experience_years", "graduation_year", # Phase C — experience metadata
            "experience_inferred", "seniority_level",
        ],
    )

    # response.result.hits is a list of Hit objects (pinecone >= 6).
    # Each Hit has: .id (str), .score (float), .fields (dict).
    # We normalise to plain dicts so the rest of the codebase is unchanged.
    hits: list[dict] = []
    for hit in response.result.hits:
        record = {
            "_id":    hit.id,
            "_score": hit.score,
        }
        record.update(hit.fields)   # chunk_text, candidate_id, filename, …
        hits.append(record)

    return hits


def list_namespace_ids(
    prefix:    str | None = None,
    namespace: str = _NAMESPACE,
) -> list[str]:
    """
    Return every vector ID stored in *namespace*, optionally filtered by *prefix*.

    Uses ``index.list(prefix=…, namespace=…)`` which is a generator that yields
    one page of IDs at a time.  All pages are flattened into a single list.

    Parameters
    ----------
    prefix : str | None
        If provided, only IDs that start with this string are returned.
        Pass a ``candidate_id`` to scope the results to one candidate,
        or ``None`` / ``""`` to list every ID in the namespace.
    namespace : str
        Pinecone namespace to query.  Defaults to the project-wide
        ``_NAMESPACE`` ("default").

    Returns
    -------
    list[str]
        Sorted list of vector IDs (sorted for deterministic output).
    """
    index = _get_index()

    kwargs: dict = {"namespace": namespace}
    if prefix:
        kwargs["prefix"] = prefix

    # index.list() yields pages of ListItem objects — extract the .id string.
    ids: list[str] = [
        record_id.id
        for id_page in index.list(**kwargs)
        for record_id in id_page
    ]

    return sorted(ids)


def fetch_vectors_by_ids(
    ids:       list[str],
    namespace: str = _NAMESPACE,
) -> list[dict]:
    """
    Fetch raw vector records (dense embedding values + metadata) by explicit ID.

    Uses ``index.fetch(ids=[…], namespace=…)`` which returns a ``FetchResponse``
    whose ``.vectors`` attribute is a dict mapping each ID to a ``Vector`` object
    with ``.id``, ``.values`` (the embedding float list), and ``.metadata``.

    Records are fetched in batches of 100 (Pinecone recommended limit) and the
    results are merged before returning, so callers do not need to paginate.

    Parameters
    ----------
    ids : list[str]
        One or more vector IDs to retrieve.  IDs not found in the index are
        silently omitted from the result (Pinecone behaviour).
    namespace : str
        Pinecone namespace to query.  Defaults to ``_NAMESPACE``.

    Returns
    -------
    list[dict]
        One dict per found vector, each containing:
          ``_id``      — the vector ID (str)
          ``values``   — the dense embedding (list[float])
          ``metadata`` — all stored metadata fields (dict)
        The list preserves insertion order (order of *ids* as found).
    """
    index = _get_index()

    results: list[dict] = []
    _FETCH_BATCH = 100

    for start in range(0, len(ids), _FETCH_BATCH):
        batch = ids[start : start + _FETCH_BATCH]
        response = index.fetch(ids=batch, namespace=namespace)

        # response.vectors is a dict[str, Vector]; iterate in requested order
        # so the output list matches the caller's id ordering where possible.
        for vid in batch:
            vec = response.vectors.get(vid)
            if vec is None:
                continue  # ID not found — silently skip (mirrors Pinecone behaviour)
            results.append(
                {
                    "_id":      vec.id,
                    "values":   vec.values,      # list[float] — the dense embedding
                    "metadata": vec.metadata or {},
                }
            )

    return results


def delete_resume(candidate_id: str) -> None:
    """
    Remove all vectors belonging to *candidate_id* from the index.

    Implementation note — Pinecone Starter plan limitation
    ───────────────────────────────────────────────────────
    Serverless indexes on the free Starter plan do not support
    metadata-filter-based deletes (``index.delete(filter=…)``).  That call
    would silently succeed but leave the vectors in place.

    Instead we:
      1. Use ``index.list(prefix=…)`` to paginate over every chunk ID that
         starts with ``{candidate_id}#`` — matching the pattern used at
         upsert time: ``{candidate_id}#chunk{i}``.
      2. Delete the discovered IDs in batches of ≤1 000 (Pinecone limit).

    The function is a no-op (no error) when no matching IDs are found,
    which mirrors the previous behaviour for unknown candidate IDs.
    """
    index = _get_index()

    # ── 1. Collect all chunk IDs for this candidate ────────────────────────
    # index.list() is a generator that yields one page of IDs at a time.
    # Each page is a plain list[str].  We flatten everything into one list.
    # Note: list() accepts namespace="" (the default) — it does not require
    # a non-empty namespace like upsert_records / search do.
    # index.list() yields pages of ListItem objects — extract the .id string.
    chunk_ids: list[str] = [
        record_id.id
        for id_page in index.list(prefix=f"{candidate_id}#", namespace=_NAMESPACE)
        for record_id in id_page
    ]

    if not chunk_ids:
        return  # nothing indexed under this candidate_id

    # ── 2. Delete in batches ───────────────────────────────────────────────
    # Pinecone recommends ≤1 000 IDs per delete call.
    _DELETE_BATCH = 1_000
    for start in range(0, len(chunk_ids), _DELETE_BATCH):
        index.delete(
            ids=chunk_ids[start : start + _DELETE_BATCH],
            namespace=_NAMESPACE,
        )
