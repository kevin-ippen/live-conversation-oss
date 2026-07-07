"""app.py — ConversationApp: the top-level integration point."""

from __future__ import annotations

import asyncio
import logging
import signal
import uuid
from pathlib import Path
from typing import Literal

from . import server as _server
from .server import normalize_event
from .session import ConversationSession, TurnHandler, TurnResult
from .voice_pipeline import ASRProvider, TTSProvider, SilentTTS

logger = logging.getLogger(__name__)


class ConversationApp:
    """Wire together a TurnHandler, ASR, TTS, and the FastAPI server.

    Parameters
    ----------
    handler:
        Your TurnHandler subclass.
    asr:
        ASRProvider for speech-to-text.
    tts:
        TTSProvider for text-to-speech (default: SilentTTS).
    port:
        HTTP port for the browser canvas (default 8765).
    open_browser:
        Auto-open the canvas in the default browser on startup.
    html_path:
        Custom HTML file to serve at / (overrides built-in app.html).
    asr_timeout_s:
        Seconds before ASR is cancelled (default 30).
    turn_timeout_s:
        Seconds before the full handler turn is cancelled (default 120).
    tts_timeout_s:
        Seconds before TTS synthesis is cancelled (default 45).
    on_empty_transcript:
        What to do when ASR returns empty text.
        "ignore"   — do nothing, stay listening (default).
        "event"    — push asr.empty SSE event, stay listening.
        "reprompt" — call handler.on_turn with transcript="" so it can respond.
    """

    def __init__(
        self,
        handler:             TurnHandler,
        asr:                 ASRProvider  | None = None,
        tts:                 TTSProvider  | None = None,
        port:                int                 = 8765,
        open_browser:        bool                = True,
        html_path:           "str | Path | None" = None,
        asr_timeout_s:       float               = 30.0,
        turn_timeout_s:      float               = 120.0,
        tts_timeout_s:       float               = 45.0,
        on_empty_transcript: Literal["ignore", "event", "reprompt"] = "ignore",
    ):
        self.handler             = handler
        self.asr                 = asr
        self.tts                 = tts or SilentTTS()
        self.port                = port
        self.open_browser        = open_browser
        self.html_path           = html_path
        self.asr_timeout_s       = asr_timeout_s
        self.turn_timeout_s      = turn_timeout_s
        self.tts_timeout_s       = tts_timeout_s
        self.on_empty_transcript = on_empty_transcript

        self._session: ConversationSession | None = None
        self._session_id: str | None              = None
        self._shutdown_event = asyncio.Event()

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the server and block until Ctrl-C."""
        asyncio.run(self._main())

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _main(self) -> None:
        self._shutdown_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        def _sigint():
            loop.create_task(self._shutdown())

        loop.add_signal_handler(signal.SIGINT,  _sigint)
        loop.add_signal_handler(signal.SIGTERM, _sigint)

        _server.start_server(
            port         = self.port,
            open_browser = self.open_browser,
            html_path    = self.html_path,
        )
        # Pass the main loop so voice handler calls are bridged here —
        # ensures _session is always accessed from a single event loop.
        _server.register_voice_handler(self._handle_voice_turn, loop)

        print(f"  Canvas → http://127.0.0.1:{self.port}")
        print("  Ctrl-C to stop\n")

        while not self._shutdown_event.is_set():
            await asyncio.sleep(0.5)
            cmd = _server.poll_command()
            if cmd == "start_session":
                if self._session is None:
                    await self._start_session()
                else:
                    logger.info("start_session ignored — session already active")
            elif cmd == "end_session" and self._session is not None:
                await self._end_session()
            elif cmd == "tab_closed" and self._session is not None:
                logger.info("Tab closed (debounced) — ending session")
                await self._end_session()

    async def _start_session(self) -> None:
        self._session_id = uuid.uuid4().hex[:16]
        _server.set_session_state(active=True, session_id=self._session_id)

        _server.push_session_event("orb", {"state": "thinking"})
        _server.push_session_event("session_start", {})

        self._session = ConversationSession(self.handler)
        result = await self._session.start()

        if result and result.spoken_text:
            for raw_ev in result.events:
                ev = normalize_event(raw_ev)
                if ev:
                    _server.push_session_event(ev["type"], ev["data"])

            # Synthesize greeting and queue for the next WS connection to drain
            try:
                chunks = await asyncio.wait_for(
                    self.tts.synthesize(result.spoken_text),
                    timeout=self.tts_timeout_s,
                )
                if chunks:
                    _server.push_session_event("agent_speaking", {})
                    _server.queue_greeting_audio(chunks)
                    _server.push_session_event("agent_done", {})
            except asyncio.TimeoutError:
                logger.error("TTS timeout during opening greeting")
                _server.push_session_event("turn.error", {
                    "stage": "tts", "message": "Greeting synthesis timed out"
                })
            except Exception as e:
                logger.error("TTS error during opening greeting: %s", e)

        _server.push_session_event("orb", {"state": "listening"})

    async def _end_session(self, *, cancel_active: bool = True) -> None:
        if cancel_active and _server._active_runtime:
            _server._active_runtime.cancel()

        if self._session:
            try:
                await self._session.end()
            except Exception as e:
                logger.error("session.end() raised: %s", e)
            self._session = None

        _server.set_session_state(active=False, session_id=None)
        self._session_id = None

        _server.push_event("session_end", {})
        _server.push_event("orb", {"state": "idle"})

    async def _shutdown(self) -> None:
        await self._end_session()
        self._shutdown_event.set()

    async def _handle_voice_turn(
        self,
        pcm:          bytes,
        cancel_event: asyncio.Event,
        turn_id:      str,
    ) -> dict:
        """Invoked from _call_voice_handler — always runs on the main loop."""
        transcript   = ""
        empty_reason = ""

        # ── ASR ───────────────────────────────────────────────────────────────
        if self.asr:
            try:
                transcript = await asyncio.wait_for(
                    self.asr.transcribe(pcm),
                    timeout=self.asr_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.error("ASR timeout (turn_id=%s)", turn_id)
                _server.push_session_event("turn.error", {
                    "stage": "asr",
                    "message": f"ASR timed out after {self.asr_timeout_s}s",
                    "turn_id": turn_id,
                })
                empty_reason = "asr_timeout"
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("ASR error (turn_id=%s): %s", turn_id, e)
                _server.push_session_event("turn.error", {
                    "stage": "asr", "message": str(e), "turn_id": turn_id,
                })
                empty_reason = "asr_failure"

        if not transcript:
            empty_reason = empty_reason or "no_speech"
            if self.on_empty_transcript == "event":
                _server.push_session_event("asr.empty",
                                           {"reason": empty_reason, "turn_id": turn_id})
            if self.on_empty_transcript != "reprompt":
                return {"transcript": "", "spoken_text": "", "audio_chunks": [],
                        "events": [], "empty_reason": empty_reason}

        if cancel_event.is_set():
            return {"transcript": transcript, "spoken_text": "", "audio_chunks": [],
                    "events": [], "empty_reason": ""}

        if self._session is None:
            _server.push_session_event("turn.error", {
                "stage": "handler", "message": "No active session", "turn_id": turn_id,
            })
            return {"transcript": transcript, "spoken_text": "", "audio_chunks": [],
                    "events": [], "empty_reason": ""}

        # ── Handler ───────────────────────────────────────────────────────────
        try:
            result: TurnResult = await asyncio.wait_for(
                self._session.handle(transcript),
                timeout=self.turn_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.error("Handler timeout (turn_id=%s)", turn_id)
            _server.push_session_event("turn.error", {
                "stage": "handler",
                "message": f"Handler timed out after {self.turn_timeout_s}s",
                "turn_id": turn_id,
            })
            return {"transcript": transcript, "spoken_text": "", "audio_chunks": [],
                    "events": [], "empty_reason": ""}
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Handler error (turn_id=%s): %s", turn_id, e)
            _server.push_session_event("turn.error", {
                "stage": "handler", "message": str(e), "turn_id": turn_id,
            })
            return {"transcript": transcript, "spoken_text": "Sorry, something went wrong.",
                    "audio_chunks": [], "events": [], "empty_reason": ""}

        if result.end_session:
            asyncio.get_running_loop().create_task(self._end_session())

        # ── TTS ───────────────────────────────────────────────────────────────
        audio_chunks: list[bytes] = []
        if result.spoken_text and not cancel_event.is_set():
            try:
                audio_chunks = await asyncio.wait_for(
                    self.tts.synthesize(result.spoken_text),
                    timeout=self.tts_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.error("TTS timeout (turn_id=%s)", turn_id)
                _server.push_session_event("turn.error", {
                    "stage": "tts",
                    "message": f"TTS timed out after {self.tts_timeout_s}s",
                    "turn_id": turn_id,
                })
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("TTS error (turn_id=%s): %s", turn_id, e)
                _server.push_session_event("turn.error", {
                    "stage": "tts", "message": str(e), "turn_id": turn_id,
                })

        return {
            "transcript":   transcript,
            "spoken_text":  result.spoken_text,
            "audio_chunks": audio_chunks,
            "events":       result.events,
            "empty_reason": "",
        }
