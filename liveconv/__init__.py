"""live-conversation-oss — provider-agnostic real-time voice conversation framework.

Quick start::

    import liveconv
    from liveconv.voice_pipeline import WhisperASR, OpenAITTS

    class EchoHandler(liveconv.TurnHandler):
        async def on_turn(self, transcript, session):
            return liveconv.TurnResult(spoken_text=f"You said: {transcript}")

    liveconv.ConversationApp(
        handler = EchoHandler(),
        asr     = WhisperASR(),
        tts     = OpenAITTS(),
    ).run()
"""

from .app     import ConversationApp
from .session import ConversationSession, TurnHandler, TurnResult, Message

__all__ = [
    "ConversationApp",
    "ConversationSession",
    "TurnHandler",
    "TurnResult",
    "Message",
]
