"""Lightweight session memory for Agent conversations.

This module stores only the current in-process conversation history. It does
not use a database, vector search, or long-term user profile storage.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Literal, TypedDict

from app.agent.facts import SessionFacts
from app.agent.state import SessionState


MessageRole = Literal["user", "assistant"]


class MemoryMessage(TypedDict):
    """Single message stored in session memory."""

    role: MessageRole
    content: str


class TurnRecord(TypedDict):
    """Latest-turn diagnostics, never passed to the model as product evidence."""

    user_input: str
    input_check: dict[str, Any]
    candidate_reply: str | None
    output_check: dict[str, Any] | None
    displayed_reply: str
    history_user_input: str


class SessionMemory:
    """Simple list-backed memory for one user session."""

    def __init__(self) -> None:
        """Create an empty session history."""
        self._messages: list[MemoryMessage] = []
        self._last_turn: TurnRecord | None = None
        self._state = SessionState()
        self.facts = SessionFacts()
        self._clear_callbacks: list[Callable[[], None]] = []

    def register_clear_callback(self, callback: Callable[[], None]) -> None:
        """Include session-owned external diagnostics in this memory's reset."""
        self._clear_callbacks.append(callback)

    def update_state(self, user_context: str, *, restricted: bool = False) -> SessionState:
        """Apply explicit user context; never call this with assistant/tool text."""
        self._state.update(user_context, restricted=restricted)
        return self.get_state()

    def get_state(self) -> SessionState:
        """Expose a snapshot so callers cannot mutate another turn's state."""
        return deepcopy(self._state)

    def resolve_name(self, query: str, sku: str | None, clarification: str | None) -> SessionState:
        """Commit the program's verified name resolution for the pending query."""
        self._state.resolve_name(query, sku, clarification)
        return self.get_state()

    def add_message(self, role: MessageRole, content: str) -> None:
        """Add one user or assistant message to the session history."""
        self._messages.append({"role": role, "content": content})

    def get_history(self) -> list[MemoryMessage]:
        """Return a copy of the current session history."""
        return list(self._messages)

    def add_turn(
        self,
        *,
        user_input: str,
        input_check: dict[str, Any],
        candidate_reply: str | None,
        output_check: dict[str, Any] | None,
        displayed_reply: str,
        history_user_input: str,
    ) -> None:
        """Append one visible turn; keep raw/candidate text in diagnostics only."""
        blocked = input_check.get("risk") or (
            output_check is not None and not output_check.get("safe", True)
        )
        assistant_history = displayed_reply
        if blocked:
            assistant_history = f"【安全拒答记录，不是商品事实】{displayed_reply}"
        self._messages.extend([
            {"role": "user", "content": history_user_input},
            {"role": "assistant", "content": assistant_history},
        ])
        # Retain only the latest diagnostic record, in process, not a second history.
        self._last_turn = deepcopy(TurnRecord(
            user_input=user_input,
            input_check=input_check,
            candidate_reply=candidate_reply,
            output_check=output_check,
            displayed_reply=displayed_reply,
            history_user_input=history_user_input,
        ))

    def get_last_turn(self) -> TurnRecord | None:
        """Return a diagnostic copy; callers must not inject it into model history."""
        return deepcopy(self._last_turn)

    def clear(self) -> None:
        """Clear session history, state, facts and registered diagnostics."""
        self._messages.clear()
        self._last_turn = None
        self._state = SessionState()
        self.facts.clear()
        for callback in self._clear_callbacks:
            callback()


def create_memory() -> SessionMemory:
    """Create an empty in-memory conversation state."""
    return SessionMemory()

