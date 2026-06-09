"""
Tests for src/utils/chunker.py
"""

import pytest
from src.utils.chunker import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP


class TestChunkText:
    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   \n\n  ") == []

    def test_short_text_single_chunk(self):
        text = "Hello, I am a software engineer."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_produces_multiple_chunks(self):
        # Create text longer than CHUNK_SIZE
        text = "word " * (CHUNK_SIZE // 3)
        chunks = chunk_text(text)
        assert len(chunks) > 1

    def test_chunks_overlap_contains_shared_content(self):
        """Adjacent chunks should share some characters (the overlap region)."""
        text = "sentence. " * (CHUNK_SIZE // 5)
        chunks = chunk_text(text)
        if len(chunks) >= 2:
            # The end of chunk 0 and start of chunk 1 should share content.
            tail = chunks[0][-CHUNK_OVERLAP:]
            assert tail in chunks[1], (
                "Expected overlap region not found in next chunk."
            )

    def test_none_input_returns_empty_list(self):
        assert chunk_text(None) == []
