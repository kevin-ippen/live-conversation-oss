"""server.py — FastAPI server for live-conversation-oss.

Exposes:
  GET  /            → app.html  (or custom HTML path passed to start_server)
  GET  /vad_core.js → browser Silero VAD module
  GET  /pcm-processor.js → browser AudioWorklet
  GET  /events      → SSE stream (one queue per connected tab)
  WS   /voice       → real-time voice pipeline (PCM in → TTS out)
  POST /session/start
  POST /session/end

The voice pipeline flow:
  Browser mic → AudioWorklet → VAD → WebSocket binary PCM
  → server ASR → TurnHandler.on_turn() → TTS → WebSocket base64 audio
  → browser AudioContext playback
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
import webbrowser
from pathlib import Path
from typing import AsyncGenerator, Callable

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from .session import ConversationSession, TurnHandler
from .voice_pipeline import ASRProvider, TTSProvider, SilentTTS

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent

SAMPLE_RATE_IN  = 16000
MIN_AUDIO_BYTES = SAMPLE_RATE_IN * 2 * 300 // 1000  # 300 ms minimum utterance


# ── App + per-connection state ────────────────────────────────────────────────

app = FastAPI()

_connections: list[asyncio.Queue] = []
_command_queue: asyncio.Queue     = asyncio.Queue()
_loop: asyncio.AbstractEventLoop | None = None
_voice_handler: Callable | None   = None  # set by the daemon on startup


def register_voice_handler(fn: Callable) -> None:
    """The application registers its voice turn handler here."""
    global _voice_handler
    _voice_handler = fn


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
    """Real-time voice pipeline.

    Browser → Server:
      binary frames            — 16kHz mono PCM16
      {"type":"end_of_speech"} — VAD detected end of turn
      {"type":"interrupt"}     — barge-in
      {"type":"ping"}          — keepalive

    Server → Browser:
      {"type":"asr.final",    "text":"..."}    — transcript
      {"type":"agent.audio",  "audio":"<b64>"} — one TTS audio chunk (WAV)
      {"type":"agent.done"}                    — finished speaking
      {"type":"pong"}                          — keepalive reply
    """
    await ws.accept()
    audio_buf    = bytearray()
    cancel_event = asyncio.Event()
    logger.info("Voice WebSocket connected")

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

            if msg_type == "interrupt":
                cancel_event.set()
                audio_buf = bytearray()

            elif msg_type == "end_of_speech":
                if len(audio_buf) < MIN_AUDIO_BYTES:
                    audio_buf = bytearray()
                    await ws.send_json({"type": "agent.done"})
                    continue

                pcm = bytes(audio_buf)
                audio_buf = bytearray()
                cancel_event.clear()

                push_event("orb", {"state": "thinking"})
                transcript = ""
                spoken     = ""

                if _voice_handler is not None:
                    try:
                        result = await _voice_handler(pcm, cancel_event)
                        transcript = result.get("transcript", "")
                        spoken     = result.get("spoken_text", "")
                        for ev in result.get("events", []):
                            push_event(ev["type"], ev.get("data", {}))
                    except Exception as e:
                        logger.error(f"Voice handler error: {e}", exc_info=True)
                        spoken = "Sorry, something went wrong."
                else:
                    logger.warning("No voice handler registered")

                if transcript:
                    await ws.send_json({"type": "asr.final", "text": transcript})
                    push_event("transcript", {"text": transcript})

                if cancel_event.is_set():
                    await ws.send_json({"type": "agent.done"})
                    push_event("orb", {"state": "listening"})
                    continue

                if spoken:
                    for audio_bytes in result.get("audio_chunks", []):
                        if cancel_event.is_set():
                            break
                        b64 = base64.b64encode(audio_bytes).decode()
                        await ws.send_json({"type": "agent.audio", "audio": b64})

                await ws.send_json({"type": "agent.done"})
                push_event("orb", {"state": "listening"})

            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("Voice WebSocket disconnected")


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
    """Start the canvas server in a background thread.

    Args:
        port:         TCP port to bind (default 8765).
        open_browser: Open the default browser on startup.
        html_path:    Path to a custom HTML file (overrides built-in app.html).
    """
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
            logger.warning(f"Server did not bind on port {port} within 8s")
