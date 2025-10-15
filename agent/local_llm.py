"""Local LLM and embedding utilities backed by Ollama and SentenceTransformers."""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import List

import ollama
from sentence_transformers import SentenceTransformer


_embedding_lock = threading.Lock()
_embedding_model: SentenceTransformer | None = None


def _load_embedding_model() -> SentenceTransformer:
    """Load the sentence transformer lazily to avoid heavy startup cost."""
    global _embedding_model
    with _embedding_lock:
        if _embedding_model is None:
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def generate_response(prompt: str, model: str = "llama3") -> str:
    """
    Generate natural language completions using the local Ollama model.

    Parameters
    ----------
    prompt:
        Full text prompt passed to the llama3 model.
    """
    response = ollama.generate(model=model, prompt=prompt)
    return response["response"].strip()


@lru_cache(maxsize=1024)
def _cached_embedding(text: str) -> List[float]:
    """Memoize embeddings for repeated segments."""
    model = _load_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True).tolist()
    return embedding


def get_embeddings(text: str) -> List[float]:
    """
    Compute deterministic embeddings for a text snippet.

    Parameters
    ----------
    text:
        The text to embed in vector space.
    """
    return _cached_embedding(text)
