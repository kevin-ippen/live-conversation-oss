"""Tests for voice_pipeline utilities and built-in providers."""

import asyncio
import struct
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from liveconv.voice_pipeline import (
    pcm_to_wav,
    ASRProvider,
    TTSProvider,
    EchoASR,
    SilentTTS,
)


# ── pcm_to_wav ─────────────────────────────────────────────────────────────────

def test_pcm_to_wav_produces_valid_riff_header():
    pcm   = b"\x00\x01" * 100  # 100 samples of silence
    wav   = pcm_to_wav(pcm)
    # RIFF magic
    assert wav[:4]  == b"RIFF"
    # WAVE magic
    assert wav[8:12] == b"WAVE"
    # fmt  chunk
    assert wav[12:16] == b"fmt "
    # data chunk
    assert wav[36:40] == b"data"


def test_pcm_to_wav_file_size():
    pcm = b"\x00" * 3200   # 100ms of silence at 16kHz mono 16-bit
    wav = pcm_to_wav(pcm)
    # RIFF chunk size = 36 + data_size
    riff_size = struct.unpack_from("<I", wav, 4)[0]
    assert riff_size == 36 + len(pcm)
    # data chunk size
    data_size = struct.unpack_from("<I", wav, 40)[0]
    assert data_size == len(pcm)


def test_pcm_to_wav_sample_rate_encoded():
    pcm = b"\x00" * 100
    wav = pcm_to_wav(pcm, sample_rate=22050)
    sample_rate = struct.unpack_from("<I", wav, 24)[0]
    assert sample_rate == 22050


def test_pcm_to_wav_custom_channels():
    pcm = b"\x00" * 100
    wav = pcm_to_wav(pcm, channels=2)
    channels = struct.unpack_from("<H", wav, 22)[0]
    assert channels == 2


def test_pcm_to_wav_total_length():
    pcm = b"\x00" * 3200
    wav = pcm_to_wav(pcm)
    assert len(wav) == 44 + len(pcm)


def test_pcm_to_wav_empty_audio():
    wav = pcm_to_wav(b"")
    assert len(wav) == 44
    data_size = struct.unpack_from("<I", wav, 40)[0]
    assert data_size == 0


# ── EchoASR ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_echo_asr_returns_fixed_response():
    asr = EchoASR(response="test response")
    result = await asr.transcribe(b"\x00" * 100)
    assert result == "test response"


@pytest.mark.asyncio
async def test_echo_asr_default_response():
    asr = EchoASR()
    result = await asr.transcribe(b"\x00" * 100)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_echo_asr_ignores_audio_content():
    asr = EchoASR(response="fixed")
    r1  = await asr.transcribe(b"\x00" * 100)
    r2  = await asr.transcribe(b"\xFF" * 100)
    assert r1 == r2 == "fixed"


# ── SilentTTS ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_silent_tts_returns_empty_list():
    tts    = SilentTTS()
    result = await tts.synthesize("hello")
    assert result == []


@pytest.mark.asyncio
async def test_silent_tts_on_empty_text():
    tts    = SilentTTS()
    result = await tts.synthesize("")
    assert result == []


# ── Abstract base compliance ──────────────────────────────────────────────────

def test_asr_provider_is_abstract():
    with pytest.raises(TypeError):
        ASRProvider()  # type: ignore


def test_tts_provider_is_abstract():
    with pytest.raises(TypeError):
        TTSProvider()  # type: ignore


def test_echo_asr_is_asr_provider():
    assert isinstance(EchoASR(), ASRProvider)


def test_silent_tts_is_tts_provider():
    assert isinstance(SilentTTS(), TTSProvider)
