"""Agent-with-tools example for live-conversation-oss.

Demonstrates:
- OpenAI tool-calling loop (get_weather, lookup_fact, calculate)
- Custom SSE events pushed to the browser UI during reasoning:
    thinking     → orb shows "thinking" label + tool pill
    tool_call    → shows tool name + args in the UI
    tool_result  → shows the result
    final        → shows the spoken reply before TTS plays
- Custom HTML canvas (canvas.html) that renders those events live

Usage:
    export OPENAI_API_KEY=sk-...
    python main.py
"""

import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import liveconv
from liveconv.voice_pipeline import WhisperASR, OpenAITTS
from tools import TOOLS, dispatch

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a helpful voice assistant with access to tools.
Keep spoken replies short (1-3 sentences) — they will be read aloud.
Think out loud only if the user asks; otherwise just use tools and answer.
"""


class ToolCallHandler(liveconv.TurnHandler):

    def _client(self):
        try:
            import openai
        except ImportError:
            raise RuntimeError("pip install openai")
        return openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    async def on_session_start(self, session) -> liveconv.TurnResult:
        return liveconv.TurnResult(
            spoken_text="Ready. Ask me anything — I can check the weather, look up facts, or do math.",
            events=[{"type": "agent_ready", "data": {}}],
        )

    async def on_turn(
        self, transcript: str, session: liveconv.ConversationSession
    ) -> liveconv.TurnResult:
        client   = self._client()
        events   = []
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for msg in session.history:
            messages.append({"role": msg.role, "content": msg.content})

        events.append({"type": "thinking", "data": {"text": "Deciding what to do..."}})

        # Tool-calling loop (max 5 rounds to avoid runaway)
        for _ in range(5):
            response = await client.chat.completions.create(
                model    = "gpt-4o",
                messages = messages,
                tools    = TOOLS,
            )
            choice = response.choices[0]

            # No tool calls → final answer
            if choice.finish_reason != "tool_calls":
                break

            # Process every tool call in this round
            tool_calls = choice.message.tool_calls or []
            messages.append(choice.message)  # assistant message with tool_calls

            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                events.append({
                    "type": "tool_call",
                    "data": {"name": fn_name, "args": fn_args},
                })

                result = await dispatch(fn_name, fn_args)

                events.append({
                    "type": "tool_result",
                    "data": {"name": fn_name, "result": result},
                })

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result,
                })

        spoken = choice.message.content or ""
        events.append({"type": "final", "data": {"text": spoken}})

        return liveconv.TurnResult(spoken_text=spoken, events=events)

    async def on_session_end(self, session) -> None:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    here = os.path.dirname(os.path.abspath(__file__))

    liveconv.ConversationApp(
        handler   = ToolCallHandler(),
        asr       = WhisperASR(),
        tts       = OpenAITTS(voice="nova"),
        port      = 8765,
        html_path = os.path.join(here, "canvas.html"),
    ).run()
