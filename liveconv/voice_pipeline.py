"""voice_pipeline.py — provider-agnostic ASR and TTS contracts.

Provides:
  - ASRProvider / TTSProvider  abstract bases
  - pcm_to_wav()               utility used by every ASR adapter
  - Built-in providers:
      WhisperASR          openai.Audio.transcriptions  (requires openai)
      OpenAITTS           openai.Audio.speech          (requires openai)
      DatabricksASR       Databricks Model Serving endpoint (e.g. Parakeet TDT)
      DatabricksTTS       Databricks Model Serving endpoint (e.g. Kokoro TTS)
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


# ── Built-in: Databricks Model Serving ASR ───────────────────────────────────

class DatabricksASR(ASRProvider):
    """ASR via a Databricks Model Serving endpoint (e.g. Parakeet TDT).

    The endpoint must accept::

        {"inputs": [{"audio_b64": "<base64-WAV>", "language": "en"}]}

    and return::

        {"predictions": [{"text": "<transcript>"}]}

    Args:
        endpoint:     Serving endpoint name, e.g. ``"parakeet-tdt-asr-endpoint"``.
        host:         Databricks workspace URL. Reads ``DATABRICKS_HOST`` if omitted.
        token:        Personal access token. Reads ``DATABRICKS_TOKEN`` if omitted.
                      When running inside a Databricks App, leave both as ``None`` —
                      the runtime injects ``DATABRICKS_CLIENT_ID`` / ``DATABRICKS_CLIENT_SECRET``
                      and the SDK resolves auth automatically via ``httpx`` + OAuth M2M.
        language:     BCP-47 language hint forwarded to the endpoint.
        timeout:      HTTP timeout in seconds.
    """

    def __init__(
        self,
        endpoint: str = "parakeet-tdt-asr-endpoint",
        host: str | None = None,
        token: str | None = None,
        language: str = "en",
        timeout: float = 20.0,
    ):
        self._endpoint = endpoint
        self._host     = host
        self._token    = token
        self._language = language
        self._timeout  = timeout

    def _url_and_headers(self) -> tuple[str, dict]:
        import os
        host  = (self._host or os.environ.get("DATABRICKS_HOST", "")).rstrip("/")
        token = self._token or os.environ.get("DATABRICKS_TOKEN", "")
        if not host:
            raise RuntimeError("DatabricksASR: set DATABRICKS_HOST or pass host=")
        if not token:
            raise RuntimeError("DatabricksASR: set DATABRICKS_TOKEN or pass token=")
        url = f"{host}/serving-endpoints/{self._endpoint}/invocations"
        return url, {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async def transcribe(self, audio_pcm: bytes, sample_rate: int = 16000) -> str:
        import base64
        import httpx
        wav = pcm_to_wav(audio_pcm, sample_rate)
        audio_b64 = base64.b64encode(wav).decode()
        payload = {"inputs": [{"audio_b64": audio_b64, "language": self._language}]}
        try:
            url, headers = self._url_and_headers()
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                preds = resp.json().get("predictions", [])
                return preds[0].get("text", "").strip() if preds else ""
            logger.error("DatabricksASR HTTP %s: %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.error("DatabricksASR error: %s", e)
        return ""


# ── Built-in: Databricks Model Serving TTS ────────────────────────────────────

class DatabricksTTS(TTSProvider):
    """TTS via a Databricks Model Serving endpoint (e.g. Kokoro TTS).

    The endpoint must accept::

        {"inputs": [{"text": "...", "speaker": "am_michael", "language": "en", "instruct": ""}]}

    and return::

        {"predictions": [{"audio_b64": "<base64-WAV>"}]}

    Args:
        endpoint: Serving endpoint name, e.g. ``"kokoro-tts"``.
        host:     Databricks workspace URL. Reads ``DATABRICKS_HOST`` if omitted.
        token:    Personal access token. Reads ``DATABRICKS_TOKEN`` if omitted.
        speaker:  Speaker voice ID supported by the endpoint (e.g. ``"am_michael"``).
        language: BCP-47 language code forwarded to the endpoint.
        timeout:  HTTP timeout per sentence chunk in seconds.
    """

    def __init__(
        self,
        endpoint: str = "kokoro-tts",
        host: str | None = None,
        token: str | None = None,
        speaker: str = "am_michael",
        language: str = "en",
        timeout: float = 15.0,
    ):
        self._endpoint = endpoint
        self._host     = host
        self._token    = token
        self._speaker  = speaker
        self._language = language
        self._timeout  = timeout

    def _url_and_headers(self) -> tuple[str, dict]:
        import os
        host  = (self._host or os.environ.get("DATABRICKS_HOST", "")).rstrip("/")
        token = self._token or os.environ.get("DATABRICKS_TOKEN", "")
        if not host:
            raise RuntimeError("DatabricksTTS: set DATABRICKS_HOST or pass host=")
        if not token:
            raise RuntimeError("DatabricksTTS: set DATABRICKS_TOKEN or pass token=")
        url = f"{host}/serving-endpoints/{self._endpoint}/invocations"
        return url, {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async def _synthesize_chunk(self, text: str) -> bytes | None:
        import base64
        import httpx
        payload = {"inputs": [{"text": text, "speaker": self._speaker,
                                "language": self._language, "instruct": ""}]}
        try:
            url, headers = self._url_and_headers()
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                preds = resp.json().get("predictions", [])
                if preds:
                    audio_b64 = preds[0].get("audio_b64") or preds[0].get("audio", "")
                    if audio_b64:
                        return base64.b64decode(audio_b64)
            else:
                logger.error("DatabricksTTS HTTP %s: %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.error("DatabricksTTS error: %s", e)
        return None

    async def synthesize(self, text: str) -> list[bytes]:
        if not text.strip():
            return []
        import asyncio
        import re

        # Split into sentence-sized chunks and synthesize concurrently.
        parts = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
        # Merge very short fragments to avoid tiny requests.
        merged: list[str] = []
        for p in parts:
            if merged and len(merged[-1]) < 20:
                merged[-1] += " " + p
            else:
                merged.append(p)

        tasks = [asyncio.create_task(self._synthesize_chunk(s)) for s in merged]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, bytes) and r]


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
