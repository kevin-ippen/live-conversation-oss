"""Databricks voice chat example.

Uses Databricks Model Serving endpoints for both ASR and TTS:
  - ASR:  Parakeet TDT  (parakeet-tdt-asr-endpoint)
  - TTS:  Kokoro TTS    (kokoro-tts)
  - LLM:  Any FMAPI-compatible endpoint (default: databricks-claude-sonnet-4-6)

Required env vars::

    export DATABRICKS_HOST=https://<your-workspace>.azuredatabricks.net
    export DATABRICKS_TOKEN=<your-pat>

Optional overrides::

    export LIVECONV_ASR_ENDPOINT=parakeet-tdt-asr-endpoint   # default shown
    export LIVECONV_TTS_ENDPOINT=kokoro-tts                   # default shown
    export LIVECONV_TTS_SPEAKER=am_michael                    # default shown
    export LIVECONV_LLM_ENDPOINT=databricks-claude-sonnet-4-6 # default shown

Run::

    cd examples/databricks_chat
    python main.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import httpx
import liveconv
from liveconv.voice_pipeline import DatabricksASR, DatabricksTTS

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Keep your responses short — 1 to 3 sentences max — "
    "since they will be spoken aloud."
)

_HOST     = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
_TOKEN    = os.environ.get("DATABRICKS_TOKEN", "")
_LLM      = os.environ.get("LIVECONV_LLM_ENDPOINT", "databricks-claude-sonnet-4-6")
_ASR      = os.environ.get("LIVECONV_ASR_ENDPOINT", "parakeet-tdt-asr-endpoint")
_TTS      = os.environ.get("LIVECONV_TTS_ENDPOINT", "kokoro-tts")
_SPEAKER  = os.environ.get("LIVECONV_TTS_SPEAKER", "am_michael")


class DatabricksHandler(liveconv.TurnHandler):

    async def on_session_start(
        self, session: liveconv.ConversationSession
    ) -> liveconv.TurnResult | None:
        return liveconv.TurnResult(spoken_text="Hello! I'm ready. How can I help you?")

    async def on_turn(
        self,
        transcript: str,
        session: liveconv.ConversationSession,
    ) -> liveconv.TurnResult:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in session.history:
            messages.append({"role": msg.role, "content": msg.content})

        url = f"{_HOST}/serving-endpoints/{_LLM}/invocations"
        headers = {"Authorization": f"Bearer {_TOKEN}", "Content-Type": "application/json"}
        payload = {"messages": messages, "max_tokens": 256}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            reply = f"Sorry, I encountered an error: {e}"

        return liveconv.TurnResult(spoken_text=reply)

    async def on_session_end(self, session: liveconv.ConversationSession) -> None:
        pass


if __name__ == "__main__":
    if not _HOST or not _TOKEN:
        sys.exit("Set DATABRICKS_HOST and DATABRICKS_TOKEN before running.")

    liveconv.ConversationApp(
        handler = DatabricksHandler(),
        asr     = DatabricksASR(endpoint=_ASR),
        tts     = DatabricksTTS(endpoint=_TTS, speaker=_SPEAKER),
        port    = 8765,
    ).run()
