"""app.py — ConversationApp: the top-level integration point.

Usage:
    from liveconv import ConversationApp, TurnResult
    from liveconv.voice_pipeline import WhisperASR, OpenAITTS

    class MyHandler(liveconv.TurnHandler):
        async def on_turn(self, transcript, session):
            return TurnResult(spoken_text=f"You said: {transcript}")

    app = ConversationApp(
        handler = MyHandler(),
        asr     = WhisperASR(),
        tts     = OpenAITTS(voice="nova"),
    )
    app.run()   # blocks; Ctrl-C to stop
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import TYPE_CHECKING

from . import server as _server
from .session import ConversationSession, TurnHandler, TurnResult
from .voice_pipeline import ASRProvider, TTSProvider, SilentTTS

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class ConversationApp:
    """Wire together a TurnHandler, ASR, TTS, and the FastAPI server.

    Parameters
    ----------
    handler:
        Your TurnHandler subclass. Controls all LLM / agent logic.
    asr:
        ASRProvider for speech-to-text (default: no-op — transcript is empty).
    tts:
        TTSProvider for text-to-speech (default: SilentTTS — no audio played).
    port:
        HTTP port for the browser canvas (default 8765).
    open_browser:
        Auto-open the canvas in the default browser on startup.
    html_path:
        Custom HTML file to serve at / (overrides built-in app.html).
    """

    def __init__(
        self,
        handler:      TurnHandler,
        asr:          ASRProvider  | None = None,
        tts:          TTSProvider  | None = None,
        port:         int                 = 8765,
        open_browser: bool                = True,
        html_path:    "str | Path | None" = None,
    ):
        self.handler      = handler
        self.asr          = asr
        self.tts          = tts or SilentTTS()
        self.port         = port
        self.open_browser = open_browser
        self.html_path    = html_path
        self._session: ConversationSession | None = None

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the server and block until Ctrl-C."""
        asyncio.run(self._main())

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _main(self) -> None:
        loop = asyncio.get_running_loop()

        def _sigint():
            logger.info("SIGINT — shutting down")
            loop.create_task(self._shutdown())

        loop.add_signal_handler(signal.SIGINT,  _sigint)
        loop.add_signal_handler(signal.SIGTERM, _sigint)

        _server.start_server(
            port         = self.port,
            open_browser = self.open_browser,
            html_path    = self.html_path,
        )
        _server.register_voice_handler(self._handle_voice_turn)

        logger.info(f"Live Conversation ready at http://127.0.0.1:{self.port}")
        print(f"  Canvas → http://127.0.0.1:{self.port}")
        print("  Ctrl-C to stop\n")

        while True:
            await asyncio.sleep(0.5)
            cmd = _server.poll_command()
            if cmd == "start_session" and self._session is None:
                await self._start_session()
            elif cmd == "end_session" and self._session is not None:
                await self._end_session()
            elif cmd == "tab_closed" and self._session is not None:
                logger.info("Tab closed — ending session")
                await self._end_session()

    async def _start_session(self) -> None:
        _server.push_event("orb", {"state": "thinking"})
        _server.push_event("session_start", {})
        self._session = ConversationSession(self.handler)
        result = await self._session.start()
        if result:
            await self._speak_result(result)
        _server.push_event("orb", {"state": "listening"})

    async def _end_session(self) -> None:
        if self._session:
            await self._session.end()
            self._session = None
        _server.push_event("session_end", {})
        _server.push_event("orb", {"state": "idle"})

    async def _shutdown(self) -> None:
        await self._end_session()
        asyncio.get_running_loop().stop()

    async def _handle_voice_turn(self, pcm: bytes, cancel_event: asyncio.Event) -> dict:
        """Called by server.py for each completed user utterance.

        Returns a dict consumed by the WebSocket handler:
          transcript:    text the user said
          spoken_text:   what the assistant will say
          audio_chunks:  list[bytes] WAV blobs
          events:        list[{type, data}] SSE events
        """
        transcript = ""
        if self.asr:
            try:
                transcript = await self.asr.transcribe(pcm)
            except Exception as e:
                logger.error(f"ASR error: {e}")

        if not transcript or cancel_event.is_set():
            return {"transcript": transcript, "spoken_text": "", "audio_chunks": [], "events": []}

        if self._session is None:
            return {"transcript": transcript, "spoken_text": "No active session.", "audio_chunks": [], "events": []}

        result: TurnResult = await self._session.handle(transcript)

        if result.end_session:
            asyncio.get_running_loop().create_task(self._end_session())

        audio_chunks: list[bytes] = []
        if result.spoken_text and not cancel_event.is_set():
            try:
                audio_chunks = await self.tts.synthesize(result.spoken_text)
            except Exception as e:
                logger.error(f"TTS error: {e}")

        return {
            "transcript":   transcript,
            "spoken_text":  result.spoken_text,
            "audio_chunks": audio_chunks,
            "events":       result.events,
        }

    async def _speak_result(self, result: TurnResult) -> None:
        """Broadcast a TurnResult via SSE + synthesise audio if TTS is available.

        Used for the opening greeting (before any WebSocket is connected).
        """
        if not result.spoken_text:
            return
        _server.push_event("agent_speaking", {})
        for ev in result.events:
            _server.push_event(ev["type"], ev.get("data", {}))
        # Opening audio goes through the WS once connected; nothing to do here.
        _server.push_event("agent_done", {})
