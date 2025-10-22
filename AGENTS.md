# Repository Guidelines

## Project Structure & Module Organization
- `semantic_search.py` holds the semantic search CLI that seeds and queries the `firmware_docs` collection.
- `requirements.txt` lists runtime dependencies; install with `python3 -m pip install -r requirements.txt`.
- `db/` (created at runtime) stores the persistent Chroma database; keep it under version control ignore rules unless snapshots are needed.

## Build, Test, and Development Commands
- `python3 -m pip install -r requirements.txt` installs runtime dependencies locally.
- `python3 semantic_search.py` launches the semantic search prompt and persists embeddings to `./db`.
- `python3 -m pip check` verifies dependency consistency after upgrades.

## Coding Style & Naming Conventions
- Target Python 3.9+; use 4-space indentation and type hints for new functions.
- Favor descriptive module names (`semantic_search.py`) and snake_case for functions/variables.
- Document non-obvious logic with concise comments; avoid verbose narration.

## Testing Guidelines
- Add regression scripts under `tests/` (create if absent) using `pytest`; name files `test_<feature>.py`.
- Run `python3 -m pytest` before submitting changes; include sample data fixtures for new retrieval behaviors.
- If scripts depend on persistent data, document setup steps in the test module docstring.

## Commit & Pull Request Guidelines
- Follow Conventional Commit style when practical (e.g., `feat: add can bus retrieval flow`, `fix: guard empty query input`).
- Keep commits scoped to a single concern and include relevant docs updates.
- Pull requests should describe behavior changes, list manual or automated test results, and link to any tracked tasks.

## Security & Configuration Tips
- Avoid committing API tokens or Ollama models; prefer `.env` files referenced via `python-dotenv`.
- On macOS system Python, add `/Users/austinsenson/Library/Python/3.9/bin` to `PATH` if you need CLI tools such as `chromadb` or `torchrun`.
