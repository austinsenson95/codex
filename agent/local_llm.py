"""Local LLM and embedding utilities backed by Ollama."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Union

import ollama

# Default local embedding model served by Ollama
_EMBED_MODEL = "nomic-embed-text"


def _embed(text: str) -> List[float]:
    """Embed a single text using the locally served Ollama model."""
    response = ollama.embeddings(model=_EMBED_MODEL, prompt=text)
    embedding = response.get("embedding")
    if embedding is None:
        raise ValueError("Ollama embeddings response missing 'embedding' payload.")
    return embedding


def generate_response(prompt: str, model: str = "llama3") -> str:
    """
    Generate a natural language completion using the local Ollama model.

    Parameters
    ----------
    prompt : str
        Full text prompt passed to the llama3 model.
    model : str, optional
        Model name served by Ollama (default: 'llama3').
    """
    response = ollama.generate(model=model, prompt=prompt)
    return response["response"].strip()


@lru_cache(maxsize=1024)
def _cached_embedding(text: str) -> List[float]:
    """Cache embeddings for repeated string inputs."""
    return _embed(text)


def get_embeddings(input_text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    """
    Compute deterministic embeddings for one or more text snippets.

    Parameters
    ----------
    input_text : str | List[str]
        Single text or list of texts to embed in vector space.

    Returns
    -------
    List[float] | List[List[float]]
        Embedding vector(s) corresponding to the input text(s).
    """
    if isinstance(input_text, list):
        # Handle list of texts by embedding each individually
        vectors = []
        for t in input_text:
            # Convert to string before caching to ensure hashability
            vectors.append(_cached_embedding(str(t)))
        return vectors
    else:
        # Single string input
        return _cached_embedding(str(input_text))
