"""ConversationSession — the turn-handling core.

Framework users subclass TurnHandler to plug in any LLM / agent logic.
The session itself is intentionally thin: it manages message history,
calls the handler, and returns spoken text. No transport, no TTS/ASR here.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Public data types ─────────────────────────────────────────────────────────

@dataclass
class Message:
    role: str   # "user" | "assistant" | "system"
    content: str


@dataclass
class TurnResult:
    """What the agent wants to say and any structured side-effects."""
    spoken_text: str
    # Arbitrary app-level events that the server should fan-out via SSE.
    # Each item is {"type": str, "data": dict}.
    events: list[dict] = field(default_factory=list)
    # Set to True by the handler to signal the session should end.
    end_session: bool = False


# ── Abstract base ─────────────────────────────────────────────────────────────

class TurnHandler(ABC):
    """Override this to implement your agent / LLM logic.

    on_session_start() is called once when the session begins.
    on_turn() is called for every user utterance.
    on_session_end() is called once when the session ends.

    All three can be async or sync; the framework awaits them either way.
    """

    async def on_session_start(self, session: "ConversationSession") -> TurnResult | None:
        """Optional opening turn. Return None to stay silent."""
        return None

    @abstractmethod
    async def on_turn(self, transcript: str, session: "ConversationSession") -> TurnResult:
        """Process one user utterance and return what to say + any events."""

    async def on_session_end(self, session: "ConversationSession") -> None:
        """Called on teardown. Override for cleanup."""


# ── Session ───────────────────────────────────────────────────────────────────

class ConversationSession:
    """Holds conversation history and drives the turn handler.

    Lifecycle:
        session = ConversationSession(handler)
        result  = await session.start()          # opening greeting
        result  = await session.handle("hello")  # each user turn
        await session.end()
    """

    def __init__(self, handler: TurnHandler, metadata: dict | None = None):
        self.handler  = handler
        self.history: list[Message] = []
        self.metadata: dict = metadata or {}   # app can store arbitrary state here
        self.active   = False
        self._lock    = asyncio.Lock()

    def add_message(self, role: str, content: str) -> None:
        self.history.append(Message(role=role, content=content))

    def messages_as_dicts(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self.history]

    async def start(self) -> TurnResult | None:
        """Initialise the session and optionally get an opening greeting."""
        self.active = True
        result = await _await_maybe(self.handler.on_session_start(self))
        if result and result.spoken_text:
            self.add_message("assistant", result.spoken_text)
        return result

    async def handle(self, transcript: str) -> TurnResult:
        """Process one user turn. Thread-safe via async lock."""
        async with self._lock:
            self.add_message("user", transcript)
            try:
                result = await _await_maybe(self.handler.on_turn(transcript, self))
            except Exception as e:
                logger.error(f"TurnHandler.on_turn() raised: {e}", exc_info=True)
                result = TurnResult(spoken_text="Sorry, something went wrong.")
            if result.spoken_text:
                self.add_message("assistant", result.spoken_text)
            return result

    async def end(self) -> None:
        self.active = False
        await _await_maybe(self.handler.on_session_end(self))


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _await_maybe(obj: Any) -> Any:
    """Await coroutines; return plain values as-is."""
    if asyncio.iscoroutine(obj):
        return await obj
    return obj
