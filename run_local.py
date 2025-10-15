"""Simple REPL entrypoint for the memory-enabled Codex agent."""

from __future__ import annotations

import sys
from typing import NoReturn

from agent.codex_agent import CodexAgent
from memory.vector_store import MemoryStore


def launch_repl() -> NoReturn:
    """Start an interactive command-line chat loop."""
    memory_store = MemoryStore(persist_directory="./data/memory_store")
    agent = CodexAgent(memory_store=memory_store)

    print("Codex Memory Agent (Running in offline Ollama mode)")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Session ended.")
            break

        try:
            response = agent.generate_response(user_input)
        except Exception as exc:  # pragma: no cover - safety net for CLI usage
            print(f"Assistant error: {exc}")
            continue

        print(f"Assistant: {response}\n")

    sys.exit(0)


if __name__ == "__main__":
    launch_repl()
