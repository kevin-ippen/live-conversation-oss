"""Echo example — zero external dependencies.

Demonstrates the minimal wiring:
  - EchoASR: always "transcribes" the audio as a fixed string (no API call)
  - SilentTTS: no audio playback
  - EchoHandler: mirrors what the user said back as the reply text

Run:
    cd examples/echo
    python main.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import liveconv
from liveconv.voice_pipeline import EchoASR, SilentTTS


class EchoHandler(liveconv.TurnHandler):
    async def on_session_start(self, session: liveconv.ConversationSession):
        return liveconv.TurnResult(spoken_text="Echo mode started. Say anything.")

    async def on_turn(
        self,
        transcript: str,
        session: liveconv.ConversationSession,
    ) -> liveconv.TurnResult:
        reply = f"You said: {transcript}"
        return liveconv.TurnResult(spoken_text=reply)

    async def on_session_end(self, session: liveconv.ConversationSession):
        pass


if __name__ == "__main__":
    liveconv.ConversationApp(
        handler = EchoHandler(),
        asr     = EchoASR(response="[audio received — no real ASR in echo mode]"),
        tts     = SilentTTS(),
        port    = 8765,
    ).run()
