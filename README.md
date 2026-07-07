# live-conversation-oss

A provider-agnostic, self-hosted real-time voice conversation framework.
Drop in any ASR, TTS, or LLM — the framework handles mic capture, VAD, transport, and audio playback.

> **Scope / limitations**
> This is a **local, single-user, turn-based** runtime.
> - Designed for one active session at a time on localhost.
> - Turn-based: audio is streamed to the server only after end-of-speech is detected in the browser; the server transcribes, calls your handler, synthesizes, and replies before accepting the next turn.
> - Not a full-duplex streaming pipeline (no Realtime API / WebRTC).
> - No auth, no multi-user support, no production deployment story yet.
>
> Good for: local voice agents, prototypes, internal tools, demos.
> Not (yet) for: multi-user apps, low-latency streaming, public deployments.

---

## Architecture

```
Browser                              Python server
───────────────────────────────────  ──────────────────────────────────
AudioWorklet (pcm-processor.js)
  ↓ 20ms PCM chunks
Silero VAD (vad_core.js, ONNX)   →  WebSocket /voice
  ↓ end_of_speech event              ↓
                                   ASRProvider.transcribe(pcm)
                                     ↓
                                   TurnHandler.on_turn(transcript, session)
                                     ↓
                                   TTSProvider.synthesize(text)
                                     ↓
                                   base64 WAV chunks →
AudioContext playback              ←  WebSocket agent.audio

SSE /events  ← ─ ─ ─ ─ ─ ─ ─ ─ ─ push_event() — orb state, transcripts
```

**VAD** runs entirely in the browser via [Silero VAD v5 ONNX](https://github.com/ricky0123/vad-web) loaded from jsDelivr CDN.
No audio is ever streamed to the server until end-of-speech is detected.

---

## Quick start

> **PyPI package coming soon.** Install directly from GitHub for now:

```bash
pip install git+https://github.com/kevin-ippen/live-conversation-oss.git
# or for OpenAI support:
pip install "git+https://github.com/kevin-ippen/live-conversation-oss.git#egg=liveconv[openai]"
```

### Echo example (zero API keys)

```python
import liveconv
from liveconv.voice_pipeline import EchoASR, SilentTTS

class EchoHandler(liveconv.TurnHandler):
    async def on_turn(self, transcript, session):
        return liveconv.TurnResult(spoken_text=f"You said: {transcript}")

liveconv.ConversationApp(
    handler = EchoHandler(),
    asr     = EchoASR(),
    tts     = SilentTTS(),
).run()
```

Open [http://localhost:8765](http://localhost:8765), click **Start**, and speak.

### GPT-4o example

```bash
export OPENAI_API_KEY=sk-...
python examples/openai_chat/main.py
```

---

## Implementing a custom TurnHandler

```python
import liveconv

class MyHandler(liveconv.TurnHandler):

    async def on_session_start(self, session) -> liveconv.TurnResult | None:
        """Called once when a session begins. Return a greeting or None."""
        return liveconv.TurnResult(spoken_text="Hello, how can I help?")

    async def on_turn(self, transcript: str, session) -> liveconv.TurnResult:
        """Called for every completed user utterance.

        `session.history` is a list of Message(role, content) objects.
        Append to it if you want the handler to maintain its own context,
        or use it as a read-only view of what liveconv has recorded.
        """
        reply = f"You said: {transcript}"
        return liveconv.TurnResult(
            spoken_text = reply,
            events      = [],        # optional list of SSE events to broadcast
            end_session = False,     # set True to close the session
        )

    async def on_session_end(self, session) -> None:
        """Called when the session ends (user clicked End or server shutdown)."""
        pass
```

### Plugging in your own ASR / TTS

```python
from liveconv.voice_pipeline import ASRProvider, TTSProvider

class MyASR(ASRProvider):
    async def transcribe(self, audio_pcm: bytes, sample_rate: int = 16000) -> str:
        # audio_pcm is raw 16-bit mono PCM
        return my_stt_service(audio_pcm)

class MyTTS(TTSProvider):
    async def synthesize(self, text: str) -> list[bytes]:
        # return list of WAV byte blobs
        return [my_tts_service(text)]
```

Pass them to `ConversationApp`:

```python
liveconv.ConversationApp(
    handler = MyHandler(),
    asr     = MyASR(),
    tts     = MyTTS(),
    port    = 9000,
).run()
```

### Custom UI

Serve your own HTML instead of the default canvas:

```python
liveconv.ConversationApp(
    handler   = MyHandler(),
    asr       = MyASR(),
    tts       = MyTTS(),
    html_path = "my_app.html",
).run()
```

The HTML file has access to `/pcm-processor.js` and `/vad_core.js`.
See [liveconv/app.html](liveconv/app.html) for the reference implementation.

---

## Built-in providers

| Class | Type | Notes |
|-------|------|-------|
| `WhisperASR` | ASR | OpenAI Whisper-1 via `openai` SDK. Requires `pip install openai`. |
| `OpenAITTS` | TTS | OpenAI tts-1 / tts-1-hd. Voices: alloy, echo, fable, onyx, nova, shimmer. |
| `EchoASR` | ASR | Returns a fixed string. No API call. Useful for testing. |
| `SilentTTS` | TTS | Returns empty audio. No API call. Useful for text-only sessions. |

---

## SSE events

The server broadcasts these events to all connected browser tabs:

| Event | Data | Description |
|-------|------|-------------|
| `session_start` | `{}` | New session began |
| `session_end` | `{}` | Session ended |
| `orb` | `{state: "idle\|listening\|thinking\|speaking"}` | Orb state change |
| `transcript` | `{text, live}` | Heard transcript |
| `agent_speaking` | `{}` | TTS audio starting |
| `agent_done` | `{}` | TTS audio finished |
| `vad_pause` | `{}` | Suppress mic (e.g. during TTS) |
| `vad_resume` | `{}` | Resume mic |

Custom events from `TurnResult.events` are forwarded as-is.

---

## Requirements

- Python 3.10+
- Modern browser with AudioWorklet + WebSocket support (Chrome 66+, Safari 16+, Firefox 76+)
- `fastapi`, `uvicorn`, `httpx` (installed automatically)

---

## License

MIT
