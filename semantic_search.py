# NOTE: Offline RAG mode — uses Ollama embeddings only (no network calls).

import os
import sys
from typing import List
import chromadb

from agent.local_llm import get_embeddings


class LocalOllamaEmbedder:
    """Embedding function wrapper using Ollama-based embeddings (Chroma ≥0.4.16)."""

    def __init__(self):
        self._name = "local_ollama_embedder"

    # Chroma still checks for a __call__ method with (self, input)
    def __call__(self, input):
        if isinstance(input, list):
            return self.embed_documents(input)
        return self.embed_query(input)

    def embed_documents(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            vec = get_embeddings(text)
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            vectors.append(vec)
        return vectors

    def embed_query(self, input):
        """Return a list-of-list embedding suitable for Chroma queries."""
        vec = get_embeddings(input)

        # Handle all possible shapes: float, list[float], list[list[float]]
        if isinstance(vec, float):
            vec = [vec]
        elif isinstance(vec, list) and all(isinstance(v, (int, float)) for v in vec):
            # single vector → wrap it
            vec = [vec]
        elif isinstance(vec, list) and len(vec) == 1 and isinstance(vec[0], list):
            # already [[floats]] → leave as-is
            pass
        else:
            raise TypeError(f"Unexpected embedding format: {type(vec)}")

        # Ensure everything is basic Python lists, not numpy arrays
        if hasattr(vec, "tolist"):
            vec = vec.tolist()

        return vec


    def name(self):
        return self._name


def ensure_persistent_collection(db_path: str, collection_name: str):
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    embedder = LocalOllamaEmbedder()
    print("Using embedder:", type(embedder))
    return client.get_or_create_collection(
        name="firmware_docs_ollama_v2",
        embedding_function=embedder
    )


def seed_collection(collection) -> None:
    """
    Upsert a handful of representative firmware snippets so the demo works out of the box.
    """
    sample_docs: List[str] = [
        "CAN ID 0x18FEEE: battery management heartbeat OK",
        "Bootloader: firmware version 1.4.7, CRC verified",
        "MCU log: watchdog reset occurred after 32 seconds idle",
        "Diagnostic trouble code P0A1F: drive inverter overtemp warning",
        "LIN bus frame 0x22: HVAC actuator calibration complete",
        "Identity of user, Austin",
    ]
    doc_ids = [f"firmware-snippet-{i}" for i in range(len(sample_docs))]
    collection.upsert(ids=doc_ids, documents=sample_docs)


def prompt_and_search(collection) -> None:
    """
    Read a natural-language query from stdin, run the semantic search, and print the top match.
    """
    try:
        query = input("Ask something about the firmware logs: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNo query provided, exiting.", file=sys.stderr)
        return

    if not query:
        print("Empty query, nothing to search.", file=sys.stderr)
        return

    results = collection.query(query_texts=[query], n_results=1)

    if not results or not results.get("documents") or not results["documents"][0]:
        print("No matches found.")
        return

    top_document = results["documents"][0][0]
    score = None
    if results.get("distances") and results["distances"][0]:
        score = results["distances"][0][0]

    if score is not None:
        print(f"Most similar snippet (distance {score:.4f}):\n{top_document}")
    else:
        print(f"Most similar snippet:\n{top_document}")


def main() -> None:
    """
    Main entrypoint for the semantic search demo.
    """
    collection = ensure_persistent_collection(
        db_path="./db",
        collection_name="firmware_docs_ollama_v2",
    )
    seed_collection(collection)
    prompt_and_search(collection)


if __name__ == "__main__":
    main()
