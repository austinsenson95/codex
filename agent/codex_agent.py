"""Conversational agent that leverages a local Ollama model with persistent memory."""

from __future__ import annotations

from typing import List

from memory.vector_store import MemoryStore
from agent.local_llm import generate_response


class CodexAgent:
    """
    Minimal conversational agent that recalls past interactions via ChromaDB.

    Each call retrieves relevant memories, augments the prompt, and persists
    the latest exchange for future runs.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        model: str = "llama3",
    ) -> None:
        self._memory_store = memory_store
        self._model = model
        self._session_history: List[dict[str, str]] = []

    def generate_response(self, user_input: str) -> str:
        """
        Run a local LLM request augmented with relevant memories.

        Parameters
        ----------
        user_input:
            The message provided by the human user.
        """
        retrieved_memories = self._memory_store.search(user_input)
        memory_context = "\n".join(retrieved_memories) if retrieved_memories else "No prior memories found."

        base_system_prompt = (
            "You are Codex, a helpful local assistant. "
            "Ground your answers in the provided memories when relevant. "
            "If the memories are not applicable, proceed normally."
        )

        transcript = "\n".join(
            f"{entry['role'].capitalize()}: {entry['content']}"
            for entry in self._session_history[-6:]
        )

        prompt = (
            f"{base_system_prompt}\n\n"
            f"Relevant stored memories:\n{memory_context}\n\n"
            f"Recent conversation:\n{transcript if transcript else 'None'}\n\n"
            f"User: {user_input}\n"
            "Assistant:"
        )

        response = generate_response(prompt, model=self._model)
        self._persist_interaction(user_input=user_input, assistant_response=response)
        return response

    def _persist_interaction(self, user_input: str, assistant_response: str) -> None:
        """Update in-memory transcript and persist the interaction to the vector store."""
        self._session_history.append({"role": "user", "content": user_input})
        self._session_history.append({"role": "assistant", "content": assistant_response})

        memory_entry = (
            "Conversation snippet:\n"
            f"User: {user_input}\n"
            f"Assistant: {assistant_response}"
        )
        self._memory_store.add_memory(memory_entry)
