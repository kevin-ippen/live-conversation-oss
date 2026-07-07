"""server.py — FastAPI server for live-conversation-oss.

Exposes:
  GET  /            → app.html  (or custom HTML path passed to start_server)
  GET  /vad_core.js → browser Silero VAD module
  GET  /pcm-processor.js → browser AudioWorklet
  GET  /events      → SSE stream (one queue per connected tab)
  WS   /voice       → real-time voice pipeline (PCM in → TTS out)
  POST /session/start
  POST /session/end

Voice pipeline flow:
  Browser mic → AudioWorklet → VAD → WebSocket binary PCM
  → server ASR → TurnHandler.on_turn() → TTS → WebSocket base64 audio
  → browser AudioContext playback

Wire protocol (browser ↔ server):
  Browser → Server
    binary frames            — 16kHz mono PCM16
    {"type":"audio.config", "sample_rate":N, "channels":N, "format":"pcm_s16le"}
    {"type":"end_of_speech", "reason":"..."}  — VAD detected end of turn
    {"type":"interrupt"}     — barge-in; cancels in-flight turn
    {"type":"ping"}          — keepalive

  Server → Browser
    {"type":"asr.final",    "text":"...",   "turn_id":"..."}
    {"type":"asr.empty",    "reason":"...", "turn_id":"..."}  — no transcript
    {"type":"agent.audio",  "audio":"<b64>","turn_id":"..."}  — WAV chunk
    {"type":"agent.done",   "reason":"...", "turn_id":"..."}  — end of turn
    {"type":"turn.cancelled","turn_id":"..."}
    {"type":"turn.error",   "stage":"asr|handler|tts", "message":"...", "turn_id":"..."}
    {"type":"pong"}          — keepalive reply
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import socket
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Callable

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent

SAMPLE_RATE_IN   = 16000
BYTES_PER_SAMPLE = 2
MIN_AUDIO_MS     = 300
MIN_AUDIO_BYTES  = SAMPLE_RATE_IN * BYTES_PER_SAMPLE * MIN_AUDIO_MS // 1000


# ── TurnRuntime ───────────────────────────────────────────────────────────────

@dataclass
class TurnRuntime:
    """Tracks one in-flight turn. Cancelled on interrupt or session end."""
    turn_id:      str           = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task:         asyncio.Task | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    def cancel(self) -> None:
        self.cancel_event.set()
        if self.task and not self.task.done():
            self.task.cancel()


def normalize_event(ev: object) -> dict | None:
    """Validate and normalise a handler-emitted event dict.

    Returns None if the event should be silently dropped.
    """
    if not isinstance(ev, dict):
        logger.warning("Dropped non-dict event: %r", ev)
        return None
    ev_type = ev.get("type")
    if not isinstance(ev_type, str) or not ev_type.strip():
        logger.warning("Dropped event without valid type: %r", ev)
        return None
    data = ev.get("data", {})
    if not isinstance(data, dict):
        data = {"value": data}
    return {"type": ev_type.strip(), "data": data}


# ── App + per-connection state ────────────────────────────────────────────────

app = FastAPI()

_connections: list[asyncio.Queue] = []
_command_queue: asyncio.Queue     = asyncio.Queue()
_loop: asyncio.AbstractEventLoop | None = None
_voice_handler: Callable | None   = None  # registered by ConversationApp
_greeting_queue: list[bytes]      = []    # pending greeting audio waiting for first WS
_active_runtime: TurnRuntime | None = None


def register_voice_handler(fn: Callable) -> None:
    global _voice_handler
    _voice_handler = fn


def queue_greeting_audio(chunks: list[bytes]) -> None:
    """Called by ConversationApp when greeting audio is ready before any WS connects."""
    global _greeting_queue
    _greeting_queue = list(chunks)


def flush_greeting_audio() -> list[bytes]:
    """Drain and return any pending greeting audio."""
    global _greeting_queue
    chunks, _greeting_queue = _greeting_queue, []
    return chunks


# ── SSE broadcast ─────────────────────────────────────────────────────────────

def push_event(event_type: str, data: dict) -> None:
    """Thread-safe SSE broadcast to all connected tabs."""
    if _loop and not _loop.is_closed():
        msg = {"type": event_type, "data": data}
        for q in list(_connections):
            _loop.call_soon_threadsafe(q.put_nowait, msg)


def poll_command() -> str | None:
    try:
        return _command_queue.get_nowait()
    except Exception:
        return None


def has_active_connection() -> bool:
    return bool(_connections)


# ── Static assets ─────────────────────────────────────────────────────────────

@app.get("/pcm-processor.js")
async def serve_pcm_processor() -> Response:
    return Response((_HERE / "pcm-processor.js").read_text(),
                    media_type="application/javascript")


@app.get("/vad_core.js")
async def serve_vad_core() -> Response:
    return Response((_HERE / "vad_core.js").read_text(),
                    media_type="application/javascript")


# ── SSE endpoint ──────────────────────────────────────────────────────────────

@app.get("/events")
async def sse_events() -> StreamingResponse:
    async def generator() -> AsyncGenerator[str, None]:
        q: asyncio.Queue = asyncio.Queue()
        _connections.append(q)
        try:
            yield 'data: {"type":"ping"}\n\n'
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield 'data: {"type":"ping"}\n\n'
        except asyncio.CancelledError:
            pass
        finally:
            try:
                _connections.remove(q)
            except ValueError:
                pass
            if not _connections:
                _command_queue.put_nowait("tab_closed")

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── WebSocket voice pipeline ──────────────────────────────────────────────────

@app.websocket("/voice")
async def voice_ws(ws: WebSocket) -> None:
    global _active_runtime

    await ws.accept()
    logger.info("Voice WebSocket connected")

    audio_buf     = bytearray()
    audio_config  = {"sample_rate": SAMPLE_RATE_IN, "channels": 1, "format": "pcm_s16le"}

    # Drain any greeting audio that was synthesized before this WS connected
    for chunk in flush_greeting_audio():
        b64 = base64.b64encode(chunk).decode()
        await ws.send_json({"type": "agent.audio", "audio": b64, "turn_id": "greeting"})
    if flush_greeting_audio.__doc__:  # always emit agent.done after greeting
        pass  # already drained above via flush_greeting_audio()

    try:
        while True:
            try:
                message = await asyncio.wait_for(ws.receive(), timeout=120.0)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "pong"})
                continue

            if "bytes" in message:
                audio_buf.extend(message["bytes"])
                continue

            if "text" not in message:
                continue

            try:
                data = json.loads(message["text"])
            except Exception:
                continue

            msg_type = data.get("type", "")

            if msg_type == "audio.config":
                audio_config.update({
                    k: data[k] for k in ("sample_rate", "channels", "format") if k in data
                })
                if audio_config.get("sample_rate") != SAMPLE_RATE_IN:
                    logger.warning("Client sample rate %s ≠ expected %s",
                                   audio_config.get("sample_rate"), SAMPLE_RATE_IN)
                    push_event("warning", {
                        "message": f"Expected {SAMPLE_RATE_IN}Hz audio, got {audio_config.get('sample_rate')}Hz"
                    })

            elif msg_type == "interrupt":
                if _active_runtime:
                    _active_runtime.cancel()
                    push_event("turn.cancelled", {"turn_id": _active_runtime.turn_id})
                audio_buf = bytearray()

            elif msg_type == "end_of_speech":
                if len(audio_buf) < MIN_AUDIO_BYTES:
                    audio_buf = bytearray()
                    rt = TurnRuntime()
                    await ws.send_json({"type": "asr.empty",
                                        "reason": "too_short", "turn_id": rt.turn_id})
                    await ws.send_json({"type": "agent.done",
                                        "reason": "empty_audio", "turn_id": rt.turn_id})
                    continue

                pcm = bytes(audio_buf)
                audio_buf = bytearray()

                # Cancel any previous in-flight turn
                if _active_runtime:
                    _active_runtime.cancel()

                rt = TurnRuntime()
                _active_runtime = rt

                push_event("orb", {"state": "thinking"})

                # Run turn in a cancellable Task
                async def _run_turn(pcm: bytes, rt: TurnRuntime, ws: WebSocket) -> None:
                    global _active_runtime
                    transcript = spoken = ""
                    try:
                        if _voice_handler is None:
                            logger.warning("No voice handler registered")
                            await ws.send_json({"type": "agent.done",
                                                "reason": "no_handler", "turn_id": rt.turn_id})
                            return

                        result = await _voice_handler(pcm, rt.cancel_event, rt.turn_id)
                        transcript = result.get("transcript", "")
                        spoken     = result.get("spoken_text", "")

                        for raw_ev in result.get("events", []):
                            ev = normalize_event(raw_ev)
                            if ev:
                                push_event(ev["type"], ev["data"])

                        if not transcript:
                            reason = result.get("empty_reason", "no_speech")
                            await ws.send_json({"type": "asr.empty",
                                                "reason": reason, "turn_id": rt.turn_id})
                            await ws.send_json({"type": "agent.done",
                                                "reason": "empty_transcript", "turn_id": rt.turn_id})
                            push_event("orb", {"state": "listening"})
                            return

                        await ws.send_json({"type": "asr.final",
                                            "text": transcript, "turn_id": rt.turn_id})
                        push_event("transcript", {"text": transcript})

                        if rt.cancel_event.is_set():
                            await ws.send_json({"type": "agent.done",
                                                "reason": "cancelled", "turn_id": rt.turn_id})
                            return

                        for audio_bytes in result.get("audio_chunks", []):
                            if rt.cancel_event.is_set():
                                break
                            b64 = base64.b64encode(audio_bytes).decode()
                            await ws.send_json({"type": "agent.audio",
                                                "audio": b64, "turn_id": rt.turn_id})

                        reason = "cancelled" if rt.cancel_event.is_set() else "complete"
                        await ws.send_json({"type": "agent.done",
                                            "reason": reason, "turn_id": rt.turn_id})
                        push_event("orb", {"state": "listening"})

                    except asyncio.CancelledError:
                        try:
                            await ws.send_json({"type": "agent.done",
                                                "reason": "cancelled", "turn_id": rt.turn_id})
                            push_event("turn.cancelled", {"turn_id": rt.turn_id})
                            push_event("orb", {"state": "listening"})
                        except Exception:
                            pass
                        raise

                    except Exception as e:
                        logger.error("Turn error: %s", e, exc_info=True)
                        try:
                            await ws.send_json({"type": "turn.error",
                                                "stage": "unknown", "message": str(e),
                                                "turn_id": rt.turn_id})
                            await ws.send_json({"type": "agent.done",
                                                "reason": "error", "turn_id": rt.turn_id})
                            push_event("orb", {"state": "listening"})
                        except Exception:
                            pass

                    finally:
                        if _active_runtime is rt:
                            _active_runtime = None

                rt.task = asyncio.ensure_future(_run_turn(pcm, rt, ws))

            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("Voice WebSocket disconnected")
        if _active_runtime:
            _active_runtime.cancel()
            _active_runtime = None


# ── Session control ───────────────────────────────────────────────────────────

@app.post("/session/start")
async def session_start() -> dict:
    if _loop and not _loop.is_closed():
        _loop.call_soon_threadsafe(_command_queue.put_nowait, "start_session")
    return {"ok": True}


@app.post("/session/end")
async def session_end() -> dict:
    if _loop and not _loop.is_closed():
        _loop.call_soon_threadsafe(_command_queue.put_nowait, "end_session")
    return {"ok": True}


# ── Canvas HTML ───────────────────────────────────────────────────────────────

_custom_html_path: Path | None = None


@app.get("/", response_class=HTMLResponse)
async def canvas_page() -> HTMLResponse:
    html_path = _custom_html_path or (_HERE / "app.html")
    return HTMLResponse(html_path.read_text())


# ── Server bootstrap ──────────────────────────────────────────────────────────

def _wait_for_port(host: str, port: int, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def start_server(
    port: int = 8765,
    open_browser: bool = True,
    html_path: str | Path | None = None,
) -> None:
    global _loop, _custom_html_path
    import uvicorn

    if html_path:
        _custom_html_path = Path(html_path)

    def _run() -> None:
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        config = uvicorn.Config(app, host="127.0.0.1", port=port,
                                loop="none", log_level="warning")
        server = uvicorn.Server(config)
        _loop.run_until_complete(server.serve())

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    if open_browser:
        if _wait_for_port("127.0.0.1", port):
            webbrowser.open(f"http://127.0.0.1:{port}")
        else:
            logger.warning("Server did not bind on port %d within 8s", port)
