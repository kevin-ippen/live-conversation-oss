"""OpenAI GPT-4o voice chat example.

Requires:
    pip install openai
    export OPENAI_API_KEY=sk-...

Run:
    cd examples/openai_chat
    python main.py

Every spoken turn is forwarded to GPT-4o as a user message.
The model's reply is spoken back via OpenAI TTS (tts-1, nova voice).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import liveconv
from liveconv.voice_pipeline import WhisperASR, OpenAITTS

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Keep your responses short — 1 to 3 sentences max — "
    "since they will be spoken aloud."
)


class GPT4oHandler(liveconv.TurnHandler):

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self._system = system_prompt

    def _client(self):
        try:
            import openai
        except ImportError:
            raise RuntimeError("openai package required: pip install openai")
        key = os.environ.get("OPENAI_API_KEY", "")
        return openai.AsyncOpenAI(api_key=key)

    async def on_session_start(
        self, session: liveconv.ConversationSession
    ) -> liveconv.TurnResult | None:
        return liveconv.TurnResult(
            spoken_text="Hello! I'm ready. How can I help you?"
        )

    async def on_turn(
        self,
        transcript: str,
        session: liveconv.ConversationSession,
    ) -> liveconv.TurnResult:
        client = self._client()
        messages = [{"role": "system", "content": self._system}]
        for msg in session.history:
            messages.append({"role": msg.role, "content": msg.content})

        try:
            resp = await client.chat.completions.create(
                model    = "gpt-4o",
                messages = messages,
            )
            reply = resp.choices[0].message.content.strip()
        except Exception as e:
            reply = f"Sorry, I encountered an error: {e}"

        return liveconv.TurnResult(spoken_text=reply)

    async def on_session_end(
        self, session: liveconv.ConversationSession
    ) -> None:
        pass


if __name__ == "__main__":
    liveconv.ConversationApp(
        handler = GPT4oHandler(),
        asr     = WhisperASR(),
        tts     = OpenAITTS(voice="nova"),
        port    = 8765,
    ).run()
