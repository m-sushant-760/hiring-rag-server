"""
Embedding service — stub module kept for import compatibility.

With Pinecone Integrated Inference the embedding step is handled directly
inside pinecone_service.py via `upsert_records` and `search`.
This module is intentionally minimal; no external embedding API is called.
"""

# No external embedding client needed.
# Pinecone's multilingual-e5-large model is invoked server-side when
# upsert_records() or index.search() are called in pinecone_service.py.
