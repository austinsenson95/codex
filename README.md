# Codex Local Memory Agent

Codex is a local-first conversational assistant that runs entirely on top of an Ollama-hosted large language model. The project exposes two command-line entry points:

* `run_local.py` – launches a chat-oriented REPL that augments every request with memories stored in a ChromaDB persistent vector database.
* `semantic_search.py` – provides an offline semantic search demo seeded with representative firmware diagnostic snippets.

## Features

- **Offline-friendly LLM usage** – all completions and embeddings are served by a locally running Ollama instance; no cloud APIs are required.
- **Persistent conversational memory** – every exchange with the REPL agent is embedded and stored in a Chroma collection so relevant snippets can be retrieved in future sessions.
- **Sample RAG workflow** – the semantic search CLI demonstrates how to index firmware-oriented notes and query them using the same embedding stack.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com/) with the `llama3` chat model and `nomic-embed-text` embedding model pulled locally.
- Project Python dependencies listed in `requirements.txt`.

## Installation

1. Install the Python dependencies:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

2. Ensure Ollama is running and that the required models are available. You can pull them with:

   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

## Usage

### Conversational memory agent

Launch the REPL assistant, which will create (or reuse) a persistent Chroma collection under `./data/memory_store`:

```bash
python3 run_local.py
```

Type messages at the `You:` prompt. The assistant will consult similar past exchanges before responding. Use `exit` or `quit` to terminate the session.

### Semantic search demo

Seed and query a `./db` Chroma collection containing sample firmware telemetry snippets:

```bash
python3 semantic_search.py
```

Enter a natural language query when prompted. The script will print the most similar stored snippet and its distance score when available.

## Project Structure

```
agent/
  codex_agent.py      # Conversational loop that augments prompts with retrieved memories
  local_llm.py        # Thin wrappers for Ollama text generation and embeddings
memory/
  vector_store.py     # Persistent Chroma-backed storage for conversation snippets
semantic_search.py    # Standalone semantic search CLI using the same embedding pipeline
run_local.py          # Entry point for the memory-enabled chat REPL
```

## Development

- Run automated checks with your preferred tooling (e.g., `python3 -m pytest`) before submitting changes.
- For dependency updates, verify consistency with `python3 -m pip check`.
- Persistent vector stores are created on demand under `./data/` and `./db/`; they are safe to delete when you need a clean slate.
