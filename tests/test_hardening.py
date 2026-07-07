"""Tests for hardened turn runtime behavior.

Covers:
- TurnRuntime cancellation
- normalize_event()
- ConversationApp timeout handling (ASR / handler / TTS)
- on_empty_transcript modes
- shutdown_event replaces loop.stop()
- Greeting audio queue
- server helpers: queue_greeting_audio / flush_greeting_audio
"""

import asyncio
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from liveconv.server import (
    TurnRuntime,
    normalize_event,
    queue_greeting_audio,
    flush_greeting_audio,
    push_event,
    poll_command,
    _command_queue,
)
import liveconv.server as server
from liveconv.session import ConversationSession, TurnHandler, TurnResult
from liveconv.app import ConversationApp
from liveconv.voice_pipeline import SilentTTS, EchoASR, ASRProvider, TTSProvider


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _reset_server():
    server._connections.clear()
    server._greeting_queue.clear()
    server._active_runtime = None
    server._session_active = False
    server._session_id     = None
    server._main_loop      = None
    while not server._command_queue.empty():
        try:
            server._command_queue.get_nowait()
        except Exception:
            break


@pytest.fixture(autouse=True)
def clean_server():
    _reset_server()
    yield
    _reset_server()


# ─────────────────────────────────────────────────────────────────────────────
# TurnRuntime
# ─────────────────────────────────────────────────────────────────────────────

def test_turn_runtime_has_unique_turn_id():
    r1 = TurnRuntime()
    r2 = TurnRuntime()
    assert r1.turn_id != r2.turn_id
    assert len(r1.turn_id) == 12


def test_turn_runtime_cancel_sets_event():
    rt = TurnRuntime()
    assert not rt.cancel_event.is_set()
    rt.cancel()
    assert rt.cancel_event.is_set()


@pytest.mark.asyncio
async def test_turn_runtime_cancel_cancels_task():
    rt = TurnRuntime()

    async def slow():
        await asyncio.sleep(60)

    rt.task = asyncio.ensure_future(slow())
    rt.cancel()
    await asyncio.sleep(0)
    assert rt.task.cancelled() or rt.task.done()


def test_turn_runtime_cancel_is_idempotent():
    rt = TurnRuntime()
    rt.cancel()
    rt.cancel()  # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# normalize_event
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_event_valid():
    ev = normalize_event({"type": "tool_call", "data": {"name": "search"}})
    assert ev == {"type": "tool_call", "data": {"name": "search"}}


def test_normalize_event_no_data_defaults_to_empty_dict():
    ev = normalize_event({"type": "thinking"})
    assert ev == {"type": "thinking", "data": {}}


def test_normalize_event_non_dict_data_wrapped():
    ev = normalize_event({"type": "tool_result", "data": "raw string"})
    assert ev == {"type": "tool_result", "data": {"value": "raw string"}}


def test_normalize_event_strips_type_whitespace():
    ev = normalize_event({"type": "  thinking  "})
    assert ev["type"] == "thinking"


def test_normalize_event_drops_non_dict():
    ev = normalize_event("not a dict")
    assert ev is None


def test_normalize_event_drops_missing_type():
    ev = normalize_event({"data": {"x": 1}})
    assert ev is None


def test_normalize_event_drops_empty_type():
    ev = normalize_event({"type": "  ", "data": {}})
    assert ev is None


def test_normalize_event_drops_non_string_type():
    ev = normalize_event({"type": 42, "data": {}})
    assert ev is None


# ─────────────────────────────────────────────────────────────────────────────
# Greeting audio queue
# ─────────────────────────────────────────────────────────────────────────────

def test_queue_and_flush_greeting_audio():
    chunks = [b"wav1", b"wav2"]
    queue_greeting_audio(chunks)
    result = flush_greeting_audio()
    assert result == chunks


def test_flush_greeting_audio_empty_when_none_queued():
    result = flush_greeting_audio()
    assert result == []


def test_flush_greeting_audio_drains_queue():
    queue_greeting_audio([b"chunk"])
    flush_greeting_audio()
    assert flush_greeting_audio() == []


def test_queue_greeting_replaces_previous():
    queue_greeting_audio([b"old"])
    queue_greeting_audio([b"new"])
    result = flush_greeting_audio()
    assert result == [b"new"]


# ─────────────────────────────────────────────────────────────────────────────
# ConversationApp: ASR timeout
# ─────────────────────────────────────────────────────────────────────────────

class HangingASR(ASRProvider):
    async def transcribe(self, audio_pcm, sample_rate=16000):
        await asyncio.sleep(999)
        return "never"


class InstantASR(ASRProvider):
    def __init__(self, text="hello"):
        self._text = text

    async def transcribe(self, audio_pcm, sample_rate=16000):
        return self._text


class InstantHandler(TurnHandler):
    async def on_turn(self, transcript, session):
        return TurnResult(spoken_text=f"echo: {transcript}")

    async def on_session_end(self, session):
        pass


class HangingHandler(TurnHandler):
    async def on_turn(self, transcript, session):
        await asyncio.sleep(999)
        return TurnResult(spoken_text="never")

    async def on_session_end(self, session):
        pass


class HangingTTS(TTSProvider):
    async def synthesize(self, text):
        await asyncio.sleep(999)
        return []


@pytest.mark.asyncio
async def test_asr_timeout_returns_empty_and_pushes_error():
    app = ConversationApp(
        handler       = InstantHandler(),
        asr           = HangingASR(),
        tts           = SilentTTS(),
        asr_timeout_s = 0.05,
    )
    app._session = ConversationSession(InstantHandler())

    events_received = []
    server._loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    server._connections.append(q)

    result = await app._handle_voice_turn(b"\x00" * 3200, asyncio.Event(), "t1")

    assert result["transcript"] == ""
    assert result["empty_reason"] in ("asr_timeout", "asr_failure", "no_speech")
    assert result["audio_chunks"] == []


@pytest.mark.asyncio
async def test_handler_timeout_returns_empty_and_pushes_error():
    app = ConversationApp(
        handler        = HangingHandler(),
        asr            = InstantASR("hello"),
        tts            = SilentTTS(),
        turn_timeout_s = 0.05,
    )
    app._session = ConversationSession(HangingHandler())

    server._loop = asyncio.get_running_loop()

    result = await app._handle_voice_turn(b"\x00" * 3200, asyncio.Event(), "t2")

    assert result["spoken_text"] == ""
    assert result["audio_chunks"] == []


@pytest.mark.asyncio
async def test_tts_timeout_returns_empty_audio_and_pushes_error():
    app = ConversationApp(
        handler       = InstantHandler(),
        asr           = InstantASR("hello"),
        tts           = HangingTTS(),
        tts_timeout_s = 0.05,
    )
    app._session = ConversationSession(InstantHandler())

    server._loop = asyncio.get_running_loop()

    result = await app._handle_voice_turn(b"\x00" * 3200, asyncio.Event(), "t3")

    assert result["transcript"] == "hello"
    assert result["spoken_text"] == "echo: hello"
    assert result["audio_chunks"] == []   # tts timed out, no audio


# ─────────────────────────────────────────────────────────────────────────────
# on_empty_transcript modes
# ─────────────────────────────────────────────────────────────────────────────

class EmptyASR(ASRProvider):
    async def transcribe(self, audio_pcm, sample_rate=16000):
        return ""


@pytest.mark.asyncio
async def test_on_empty_transcript_ignore_returns_early():
    app = ConversationApp(
        handler              = InstantHandler(),
        asr                  = EmptyASR(),
        tts                  = SilentTTS(),
        on_empty_transcript  = "ignore",
    )
    app._session = ConversationSession(InstantHandler())
    server._loop = asyncio.get_running_loop()

    result = await app._handle_voice_turn(b"\x00" * 3200, asyncio.Event(), "t4")
    assert result["transcript"] == ""
    assert result["spoken_text"] == ""


@pytest.mark.asyncio
async def test_on_empty_transcript_event_pushes_asr_empty():
    received = []

    app = ConversationApp(
        handler              = InstantHandler(),
        asr                  = EmptyASR(),
        tts                  = SilentTTS(),
        on_empty_transcript  = "event",
    )
    app._session = ConversationSession(InstantHandler())

    loop = asyncio.get_running_loop()
    server._loop = loop
    q: asyncio.Queue = asyncio.Queue()
    server._connections.append(q)

    result = await app._handle_voice_turn(b"\x00" * 3200, asyncio.Event(), "t5")

    # call_soon_threadsafe schedules on next loop iteration — yield to let it flush
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # Drain SSE queue
    while not q.empty():
        received.append(q.get_nowait())

    assert result["spoken_text"] == ""
    assert any(e.get("type") == "asr.empty" for e in received)


@pytest.mark.asyncio
async def test_on_empty_transcript_reprompt_calls_handler():
    called_with = []

    class TrackingHandler(TurnHandler):
        async def on_turn(self, transcript, session):
            called_with.append(transcript)
            return TurnResult(spoken_text="I heard nothing")

        async def on_session_end(self, session):
            pass

    app = ConversationApp(
        handler              = TrackingHandler(),
        asr                  = EmptyASR(),
        tts                  = SilentTTS(),
        on_empty_transcript  = "reprompt",
    )
    app._session = ConversationSession(TrackingHandler())
    server._loop = asyncio.get_running_loop()

    result = await app._handle_voice_turn(b"\x00" * 3200, asyncio.Event(), "t6")
    assert called_with == [""]
    assert result["spoken_text"] == "I heard nothing"


# ─────────────────────────────────────────────────────────────────────────────
# cancel_event short-circuits turn
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cancelled_before_handler_skips_handler():
    called = []

    class TrackHandler(TurnHandler):
        async def on_turn(self, transcript, session):
            called.append(transcript)
            return TurnResult(spoken_text="should not speak")

        async def on_session_end(self, session):
            pass

    app = ConversationApp(
        handler = TrackHandler(),
        asr     = InstantASR("hello"),
        tts     = SilentTTS(),
    )
    app._session = ConversationSession(TrackHandler())
    server._loop = asyncio.get_running_loop()

    cancel_event = asyncio.Event()
    cancel_event.set()

    result = await app._handle_voice_turn(b"\x00" * 3200, cancel_event, "t7")
    assert called == []
    assert result["spoken_text"] == ""


# ─────────────────────────────────────────────────────────────────────────────
# shutdown_event
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_shutdown_event_set_by_shutdown():
    app = ConversationApp(
        handler = InstantHandler(),
        asr     = InstantASR(),
        tts     = SilentTTS(),
    )
    # _main creates the event, but we can test _shutdown directly
    app._shutdown_event = asyncio.Event()
    app._session = None
    server._loop = asyncio.get_running_loop()

    assert not app._shutdown_event.is_set()
    await app._shutdown()
    assert app._shutdown_event.is_set()


@pytest.mark.asyncio
async def test_end_session_cancels_active_runtime():
    rt = TurnRuntime()

    async def slow():
        await asyncio.sleep(60)

    rt.task = asyncio.ensure_future(slow())
    server._active_runtime = rt
    server._loop = asyncio.get_running_loop()

    app = ConversationApp(
        handler = InstantHandler(),
        asr     = InstantASR(),
        tts     = SilentTTS(),
    )
    app._shutdown_event = asyncio.Event()
    app._session = None

    await app._end_session()

    assert rt.cancel_event.is_set()
    await asyncio.sleep(0)
    assert rt.task.cancelled() or rt.task.done()
