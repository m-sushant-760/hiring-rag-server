"""
Unit tests for pinecone_service.delete_resume.

These tests mock the Pinecone client and index objects entirely so they run
offline and consume zero API credits.  They verify the new prefix-list-then-
delete implementation that replaces the broken metadata-filter delete which
does not work on Pinecone serverless (Starter plan).
"""

import pytest
from unittest.mock import MagicMock, patch, call


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_mock_index(id_pages: list[list[str]]) -> MagicMock:
    """
    Return a MagicMock index where index.list(...) yields *id_pages*
    wrapped in objects with an 'id' attribute, simulating Pinecone's ListItem.
    """
    wrapped_pages = []
    for page in id_pages:
        wrapped_page = []
        for item in page:
            mock_item = MagicMock()
            mock_item.id = item
            wrapped_page.append(mock_item)
        wrapped_pages.append(wrapped_page)

    mock_index = MagicMock()
    mock_index.list.return_value = iter(wrapped_pages)
    return mock_index


# ── Tests ──────────────────────────────────────────────────────────────────

class TestDeleteResume:
    """Tests for pinecone_service.delete_resume (prefix-list-then-delete)."""

    def _run_delete(self, mock_index: MagicMock, candidate_id: str = "cand-abc"):
        """Patch _get_index and call delete_resume, returning the mock index."""
        import src.services.pinecone_service as svc
        with patch.object(svc, "_get_index", return_value=mock_index):
            svc.delete_resume(candidate_id)
        return mock_index

    # ------------------------------------------------------------------
    # Normal path — candidate has chunks
    # ------------------------------------------------------------------

    def test_list_called_with_correct_prefix(self):
        """index.list() must be called with the candidate's ID prefix."""
        ids = ["cand-abc#chunk0", "cand-abc#chunk1", "cand-abc#chunk2"]
        mock_index = _make_mock_index([ids])

        self._run_delete(mock_index, "cand-abc")

        mock_index.list.assert_called_once_with(prefix="cand-abc#", namespace="default")

    def test_delete_called_with_discovered_ids(self):
        """index.delete() must be called with all IDs returned by list()."""
        ids = ["cand-abc#chunk0", "cand-abc#chunk1"]
        mock_index = _make_mock_index([ids])

        self._run_delete(mock_index, "cand-abc")

        mock_index.delete.assert_called_once_with(ids=ids, namespace="default")

    def test_multi_page_list_flattened(self):
        """IDs spread across multiple pages must all be collected and deleted."""
        page1 = ["cand-abc#chunk0", "cand-abc#chunk1"]
        page2 = ["cand-abc#chunk2"]
        mock_index = _make_mock_index([page1, page2])

        self._run_delete(mock_index, "cand-abc")

        expected_ids = page1 + page2
        mock_index.delete.assert_called_once_with(ids=expected_ids, namespace="default")

    # ------------------------------------------------------------------
    # No-op path — unknown candidate
    # ------------------------------------------------------------------

    def test_noop_when_no_chunks_found(self):
        """
        If list() returns no IDs the function must return without calling
        delete() — avoids an empty-IDs API call that may raise an error.
        """
        mock_index = _make_mock_index([[]])   # one empty page

        self._run_delete(mock_index, "does-not-exist")

        mock_index.delete.assert_not_called()

    def test_noop_when_list_returns_no_pages(self):
        """Generator with zero pages (completely empty) is also a no-op."""
        mock_index = _make_mock_index([])     # zero pages

        self._run_delete(mock_index, "does-not-exist")

        mock_index.delete.assert_not_called()

    # ------------------------------------------------------------------
    # Batch path — >1 000 chunks triggers multiple delete calls
    # ------------------------------------------------------------------

    def test_large_candidate_deleted_in_batches(self):
        """
        Candidates with > 1 000 chunks must be deleted in batches of 1 000
        to stay within Pinecone's per-call limit.
        """
        # Simulate 1 500 chunks spread across two pages
        all_ids = [f"cand-big#chunk{i}" for i in range(1_500)]
        page1, page2 = all_ids[:1_000], all_ids[1_000:]
        mock_index = _make_mock_index([page1, page2])

        self._run_delete(mock_index, "cand-big")

        assert mock_index.delete.call_count == 2
        first_call, second_call = mock_index.delete.call_args_list
        assert first_call == call(ids=all_ids[:1_000],  namespace="default")
        assert second_call == call(ids=all_ids[1_000:], namespace="default")
