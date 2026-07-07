"""voice_pipeline.py — provider-agnostic ASR and TTS contracts.

Provides:
  - ASRProvider / TTSProvider  abstract bases
  - pcm_to_wav()               utility used by every ASR adapter
  - Built-in providers:
      WhisperASR          openai.Audio.transcriptions  (requires openai)
      OpenAITTS           openai.Audio.speech          (requires openai)
      EchoASR             returns the literal bytes as hex — useful for smoke tests
      SilentTTS           returns empty audio — useful for text-only sessions
"""

from __future__ import annotations

import logging
import struct
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ── WAV helper ────────────────────────────────────────────────────────────────

def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000,
               channels: int = 1, bits: int = 16) -> bytes:
    """Wrap raw PCM bytes in a minimal RIFF/WAV header."""
    data_size  = len(pcm_bytes)
    byte_rate  = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, channels,
        sample_rate, byte_rate, block_align, bits,
        b"data", data_size,
    )
    return header + pcm_bytes


# ── Abstract bases ────────────────────────────────────────────────────────────

class ASRProvider(ABC):
    """Transcribe audio → text."""

    @abstractmethod
    async def transcribe(self, audio_pcm: bytes, sample_rate: int = 16000) -> str:
        """Accept raw 16-bit mono PCM, return transcript text (empty string on failure)."""


class TTSProvider(ABC):
    """Synthesise text → audio chunks."""

    @abstractmethod
    async def synthesize(self, text: str) -> list[bytes]:
        """Return a list of WAV byte blobs (one per sentence / chunk).
        Returns an empty list if synthesis fails — callers should fall back to silence.
        """


# ── Built-in: OpenAI Whisper ASR ──────────────────────────────────────────────

class WhisperASR(ASRProvider):
    """OpenAI Whisper via the openai Python SDK.

    Args:
        api_key:  OpenAI API key (or reads OPENAI_API_KEY env var).
        model:    Whisper model name, e.g. "whisper-1".
        language: BCP-47 language code hint, e.g. "en".
    """

    def __init__(self, api_key: str | None = None,
                 model: str = "whisper-1", language: str = "en"):
        self._model    = model
        self._language = language
        self._api_key  = api_key

    def _client(self):
        try:
            import openai
        except ImportError:
            raise RuntimeError("openai package required: pip install openai")
        import os
        key = self._api_key or os.environ.get("OPENAI_API_KEY", "")
        return openai.AsyncOpenAI(api_key=key)

    async def transcribe(self, audio_pcm: bytes, sample_rate: int = 16000) -> str:
        import io
        wav = pcm_to_wav(audio_pcm, sample_rate)
        try:
            client = self._client()
            result = await client.audio.transcriptions.create(
                model    = self._model,
                file     = ("audio.wav", io.BytesIO(wav), "audio/wav"),
                language = self._language,
            )
            return result.text.strip()
        except Exception as e:
            logger.error(f"WhisperASR error: {e}")
            return ""


# ── Built-in: OpenAI TTS ──────────────────────────────────────────────────────

class OpenAITTS(TTSProvider):
    """OpenAI text-to-speech (tts-1 / tts-1-hd).

    Returns MP3 bytes wrapped in a single-item list.
    The browser decodeAudioData() handles MP3 natively in Chrome/Safari.

    Args:
        api_key: OpenAI API key (or reads OPENAI_API_KEY env var).
        model:   "tts-1" (fast) or "tts-1-hd" (better quality).
        voice:   alloy | echo | fable | onyx | nova | shimmer.
    """

    def __init__(self, api_key: str | None = None,
                 model: str = "tts-1", voice: str = "alloy"):
        self._model   = model
        self._voice   = voice
        self._api_key = api_key

    def _client(self):
        try:
            import openai
        except ImportError:
            raise RuntimeError("openai package required: pip install openai")
        import os
        key = self._api_key or os.environ.get("OPENAI_API_KEY", "")
        return openai.AsyncOpenAI(api_key=key)

    async def synthesize(self, text: str) -> list[bytes]:
        if not text.strip():
            return []
        try:
            client   = self._client()
            response = await client.audio.speech.create(
                model          = self._model,
                voice          = self._voice,
                input          = text,
                response_format= "wav",
            )
            audio = b""
            async for chunk in response.iter_bytes():
                audio += chunk
            return [audio] if audio else []
        except Exception as e:
            logger.error(f"OpenAITTS error: {e}")
            return []


# ── Built-in: Echo (testing) ──────────────────────────────────────────────────

class EchoASR(ASRProvider):
    """Returns a fixed string — useful for unit testing the pipeline without audio."""

    def __init__(self, response: str = "[audio received]"):
        self._response = response

    async def transcribe(self, audio_pcm: bytes, sample_rate: int = 16000) -> str:
        return self._response


class SilentTTS(TTSProvider):
    """Returns empty audio — useful for text-only or test sessions."""

    async def synthesize(self, text: str) -> list[bytes]:
        return []
