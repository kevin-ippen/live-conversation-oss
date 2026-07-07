"""Tests for ConversationSession and TurnHandler lifecycle."""

import asyncio
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from liveconv.session import ConversationSession, TurnHandler, TurnResult, Message


# ── Fixtures ──────────────────────────────────────────────────────────────────

class SimpleHandler(TurnHandler):
    def __init__(self):
        self.started = False
        self.ended   = False
        self.turns   = []

    async def on_session_start(self, session):
        self.started = True
        return TurnResult(spoken_text="Hello!")

    async def on_turn(self, transcript, session):
        self.turns.append(transcript)
        return TurnResult(spoken_text=f"You said: {transcript}")

    async def on_session_end(self, session):
        self.ended = True


class NoneStartHandler(TurnHandler):
    """on_session_start returns None — should be valid."""
    async def on_session_start(self, session):
        return None

    async def on_turn(self, transcript, session):
        return TurnResult(spoken_text="ok")

    async def on_session_end(self, session):
        pass


class EndSessionHandler(TurnHandler):
    """Sets end_session=True on the second turn."""
    def __init__(self):
        self.call_count = 0

    async def on_session_start(self, session):
        return None

    async def on_turn(self, transcript, session):
        self.call_count += 1
        return TurnResult(
            spoken_text="goodbye",
            end_session=(self.call_count >= 2),
        )

    async def on_session_end(self, session):
        pass


# ── Tests: lifecycle ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_calls_on_session_start():
    handler = SimpleHandler()
    session = ConversationSession(handler)
    result = await session.start()
    assert handler.started
    assert result is not None
    assert result.spoken_text == "Hello!"


@pytest.mark.asyncio
async def test_start_returns_none_when_handler_returns_none():
    handler = NoneStartHandler()
    session = ConversationSession(handler)
    result = await session.start()
    assert result is None


@pytest.mark.asyncio
async def test_end_calls_on_session_end():
    handler = SimpleHandler()
    session = ConversationSession(handler)
    await session.start()
    await session.end()
    assert handler.ended


@pytest.mark.asyncio
async def test_handle_calls_on_turn():
    handler = SimpleHandler()
    session = ConversationSession(handler)
    await session.start()
    result = await session.handle("hello world")
    assert result.spoken_text == "You said: hello world"
    assert handler.turns == ["hello world"]


# ── Tests: history ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_history_records_user_and_assistant():
    handler = SimpleHandler()
    session = ConversationSession(handler)
    await session.start()   # on_session_start returns "Hello!" → logged as assistant
    await session.handle("first")
    await session.handle("second")
    # Strip out any assistant-only greeting from session start
    turn_msgs = [m for m in session.history if not (m.role == "assistant" and m.content == "Hello!")]
    roles   = [m.role    for m in turn_msgs]
    content = [m.content for m in turn_msgs]
    assert roles    == ["user", "assistant", "user", "assistant"]
    assert content  == ["first", "You said: first", "second", "You said: second"]


@pytest.mark.asyncio
async def test_history_starts_empty():
    handler = SimpleHandler()
    session = ConversationSession(handler)
    assert session.history == []


@pytest.mark.asyncio
async def test_history_accessible_during_turn():
    seen_history = []

    class HistorySnapshotHandler(TurnHandler):
        async def on_session_start(self, session): return None
        async def on_turn(self, transcript, session):
            seen_history.append(list(session.history))
            return TurnResult(spoken_text="ok")
        async def on_session_end(self, session): pass

    session = ConversationSession(HistorySnapshotHandler())
    await session.start()
    await session.handle("turn one")
    await session.handle("turn two")
    # On turn one, history should have one user message
    assert seen_history[0][-1].content == "turn one"
    # On turn two, history should have user+assistant from turn one, then new user message
    assert seen_history[1][-1].content == "turn two"


# ── Tests: TurnResult fields ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_turn_result_end_session_flag():
    handler = EndSessionHandler()
    session = ConversationSession(handler)
    await session.start()
    r1 = await session.handle("first")
    r2 = await session.handle("second")
    assert not r1.end_session
    assert r2.end_session


@pytest.mark.asyncio
async def test_turn_result_custom_events():
    class EventHandler(TurnHandler):
        async def on_session_start(self, session): return None
        async def on_turn(self, transcript, session):
            return TurnResult(
                spoken_text="ok",
                events=[{"type": "custom_event", "data": {"key": "value"}}],
            )
        async def on_session_end(self, session): pass

    session = ConversationSession(EventHandler())
    await session.start()
    result = await session.handle("test")
    assert result.events == [{"type": "custom_event", "data": {"key": "value"}}]


# ── Tests: concurrency guard ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_concurrent_turns_are_serialized():
    """Turns should not interleave — the session lock must serialize them."""
    order = []

    class SlowHandler(TurnHandler):
        async def on_session_start(self, session): return None
        async def on_turn(self, transcript, session):
            order.append(f"start:{transcript}")
            await asyncio.sleep(0.05)
            order.append(f"end:{transcript}")
            return TurnResult(spoken_text="ok")
        async def on_session_end(self, session): pass

    session = ConversationSession(SlowHandler())
    await session.start()
    await asyncio.gather(
        session.handle("A"),
        session.handle("B"),
    )
    # start:A must appear before start:B (lock serializes)
    assert order.index("start:A") < order.index("start:B")
    assert order.index("end:A")   < order.index("start:B")
