/**
 * vad_core.js — VoiceVAD class for UCO Voice Canvas
 *
 * Encapsulates all VAD logic (Silero ML + RMS fallback) so it can be:
 *  - loaded in the browser:  <script src="/vad_core.js"></script>
 *  - imported in Node tests: import { VoiceVAD } from './vad_core.js'
 *
 * Dependencies are injected so the caller controls ORT and the WebSocket.
 * The class itself has no side-effects at construction time.
 *
 * Usage (browser):
 *   const vad = new VoiceVAD({ ort: window.ort, onEndOfSpeech, onSpeechStart,
 *                               onBargeIn, onMeter });
 *   await vad.init();
 *   vad.push({ pcm16, float32, rms });   // called per AudioWorklet chunk
 *   vad.pause(); vad.resume();           // echo suppression
 *   vad.reset();                         // session teardown
 */

// ── Silero constants ────────────────────────────────────────────────────────
const SILERO_SAMPLE_RATE    = 16000;
const SILERO_WINDOW_SAMPLES = 512;    // 32ms — Silero v5 / @ricky0123/vad-web
const SILERO_START_THRESHOLD  = 0.5;
const SILERO_END_THRESHOLD    = 0.35;
const SILERO_SILENCE_FRAMES   = 47;   // ~1500 ms
const SILERO_MIN_SPEECH_FRAMES = 9;   // ~300 ms

// ── RMS fallback constants ──────────────────────────────────────────────────
const RMS_START_THRESHOLD  = 0.022;
const RMS_MIN_STOP         = 0.012;
const RMS_NOISE_MULTIPLIER = 2.2;
const RMS_SPEECH_DROP      = 0.38;
const RMS_SILENCE_MS       = 1500;
const RMS_MIN_SPEECH_MS    = 300;
const RMS_MAX_TURN_MS      = 75000;

// ── Barge-in (always RMS, no inference latency) ─────────────────────────────
const BARGE_IN_THRESHOLD   = 0.06;


class VoiceVAD {
  /**
   * @param {object} opts
   * @param {object}   opts.ort             - ORT global (window.ort or a mock)
   * @param {string}   [opts.modelUrl]      - override Silero model URL
   * @param {function} opts.onEndOfSpeech   - called with (reason:string)
   * @param {function} [opts.onSpeechStart] - called when speech begins
   * @param {function} [opts.onBargeIn]     - called when user interrupts TTS
   * @param {function} [opts.onMeter]       - called with (rms:number) for UI
   */
  constructor(opts = {}) {
    this._ort            = opts.ort   || null;
    this._modelUrl       = opts.modelUrl
      || 'https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.19/dist/silero_vad.onnx';
    this._onEndOfSpeech  = opts.onEndOfSpeech  || (() => {});
    this._onSpeechStart  = opts.onSpeechStart  || (() => {});
    this._onBargeIn      = opts.onBargeIn      || (() => {});
    this._onMeter        = opts.onMeter        || (() => {});

    // Silero session state
    this._session      = null;
    this.ready         = false;  // true = ML VAD active, false = RMS fallback

    // Silero LSTM tensors (reset per session)
    this._h = null; this._c = null; this._sr = null;

    // Silero turn counters
    this._sileroSpeaking      = false;
    this._sileroSpeechFrames  = 0;
    this._sileroSilenceFrames = 0;
    this._sileroWindow        = new Float32Array(0);

    // RMS turn state
    this._speechStart    = 0;
    this._silenceStart   = 0;
    this._speechPeak     = 0;
    this._noiseFloor     = 0.006;
    this._longTurnHinted = false;

    // Shared control flags
    this.paused          = false;   // true = daemon playing audio, suppress mic
    this._ttsPlaying     = false;   // true = TTS audio playing in browser
    this._speaking       = false;   // true = user turn in progress

    // Async queue for WASM inference
    this._queue    = [];
    this._draining = false;
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /** Load the Silero model. Safe to call before mic is ready. */
  async init() {
    if (!this._ort) {
      this._warn('ORT not provided — using RMS fallback');
      return;
    }
    try {
      if (this._ort.env && this._ort.env.wasm) {
        this._ort.env.wasm.wasmPaths =
          'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
      }
      this._session = await this._ort.InferenceSession.create(this._modelUrl, {
        executionProviders: ['wasm'],
      });
      this._resetSileroState();
      this.ready = true;
      this._log('model loaded — ML VAD active');
    } catch (e) {
      this._warn('load failed, using RMS fallback:', e.message || e);
      this.ready = false;
    }
  }

  /**
   * Feed one AudioWorklet chunk. Thread-safe (queues for async drain).
   * @param {object} chunk - { pcm16: ArrayBuffer|Int16Array, float32: ArrayBuffer|Float32Array, rms: number }
   * @param {boolean} [isTtsPlaying] - current TTS playback state from caller
   */
  push(chunk, isTtsPlaying = false) {
    if (this.paused) {
      this._onMeter(0);
      return;
    }
    this._ttsPlaying = isTtsPlaying;
    this._queue.push(chunk);
    if (!this._draining) this._drain();
  }

  /** Signal that the daemon is about to play audio via afplay (suppress mic echo). */
  pause()  { this.paused = true;  this._queue = []; this._resetRmsState(); }
  /** Resume after daemon finishes playing. */
  resume() { this.paused = false; }

  /** Notify VAD that TTS playback state changed. */
  setTtsPlaying(val) { this._ttsPlaying = val; }

  /** Full reset — call on session teardown or _stopVoiceCapture. */
  reset() {
    this._queue    = [];
    this._draining = false;
    this.paused    = false;
    this._ttsPlaying = false;
    this._speaking = false;
    this._resetRmsState();
    if (this.ready) this._resetSileroState();
  }

  // ── Private: async drain ──────────────────────────────────────────────────

  async _drain() {
    this._draining = true;
    while (this._queue.length > 0) {
      const chunk = this._queue.shift();
      await this._processChunk(chunk);
    }
    this._draining = false;
  }

  async _processChunk(raw) {
    if (this.paused) return;

    const rms    = raw.rms || 0;
    const pcm16  = raw.pcm16  instanceof ArrayBuffer ? new Int16Array(raw.pcm16)  : raw.pcm16;
    const float32= raw.float32 instanceof ArrayBuffer ? new Float32Array(raw.float32) : raw.float32;

    this._onMeter(rms);

    // ── Barge-in check (always RMS — no inference latency) ──────────────────
    if (this._ttsPlaying) {
      if (rms > BARGE_IN_THRESHOLD) {
        this._resetRmsState();
        if (this.ready) this._resetSileroState();
        this._ttsPlaying = false;
        this._onBargeIn();
        // fall through so VAD can pick up the new speech turn
      } else {
        return;
      }
    }

    // ── ML path ──────────────────────────────────────────────────────────────
    if (this.ready && float32) {
      const result = await this._sieroInfer(float32);
      if (result !== null) {
        const now = Date.now();
        if (result.speech) {
          if (!this._speechStart) {
            this._speechStart = now;
            this._onSpeechStart();
          }
          this._silenceStart = 0;
          this._emitPcm(pcm16);
        }
        if (result.endOfSpeech) {
          this._speechStart = 0;
          this._onEndOfSpeech('silero');
        }
        if (this._speechStart && (now - this._speechStart) > RMS_MAX_TURN_MS) {
          this._speechStart = 0;
          this._onEndOfSpeech('max_turn');
        }
        return;
      }
      // Silero inference failed → fall through to RMS
    }

    // ── RMS fallback ─────────────────────────────────────────────────────────
    const now = Date.now();
    this._updateNoiseFloor(rms);

    const isSpeech = this._speechStart
      ? rms > this._getStopThreshold()
      : rms > this._getStartThreshold();

    if (isSpeech) {
      if (!this._speechStart) {
        this._speechStart = now;
        this._onSpeechStart();
      }
      this._speechPeak = Math.max(this._speechPeak, rms);
      this._silenceStart = 0;
    } else {
      if (this._speechStart && !this._silenceStart) this._silenceStart = now;
      if (this._speechStart && this._silenceStart &&
          (now - this._silenceStart) > RMS_SILENCE_MS) {
        if ((this._silenceStart - this._speechStart) > RMS_MIN_SPEECH_MS) {
          this._speechStart = 0;
          this._onEndOfSpeech('rms_silence');
          return;
        }
        this._resetRmsState();
      }
    }

    if (this._speechStart && (now - this._speechStart) > RMS_MAX_TURN_MS) {
      this._speechStart = 0;
      this._onEndOfSpeech('max_turn');
      return;
    }

    if (this._speechStart) this._emitPcm(pcm16);
  }

  _emitPcm(pcm16) {
    // The caller receives PCM via this hook; canvas.html sends it over the WS.
    // We don't touch the WebSocket directly — keep this class side-effect-free
    // so tests can spy on emitted frames without a WebSocket.
    if (this._onPcm) this._onPcm(pcm16);
  }

  // ── Silero inference ──────────────────────────────────────────────────────

  _resetSileroState() {
    const T = this._ort.Tensor;
    this._h  = new T('float32', new Float32Array(2 * 1 * 64), [2, 1, 64]);
    this._c  = new T('float32', new Float32Array(2 * 1 * 64), [2, 1, 64]);
    this._sr = new T('int64', BigInt64Array.from([BigInt(SILERO_SAMPLE_RATE)]), [1]);
    this._sileroSpeaking      = false;
    this._sileroSpeechFrames  = 0;
    this._sileroSilenceFrames = 0;
    this._sileroWindow        = new Float32Array(0);
  }

  // Returns {speech:bool, endOfSpeech:bool} or null on error.
  async _sieroInfer(float32Chunk) {
    const merged = new Float32Array(this._sileroWindow.length + float32Chunk.length);
    merged.set(this._sileroWindow);
    merged.set(float32Chunk, this._sileroWindow.length);
    this._sileroWindow = merged;

    let endOfSpeech = false;

    while (this._sileroWindow.length >= SILERO_WINDOW_SAMPLES) {
      const frame = this._sileroWindow.slice(0, SILERO_WINDOW_SAMPLES);
      this._sileroWindow = this._sileroWindow.slice(SILERO_WINDOW_SAMPLES);

      let prob = 0;
      try {
        const input = new this._ort.Tensor('float32', frame, [1, SILERO_WINDOW_SAMPLES]);
        const out   = await this._session.run({
          input, h: this._h, c: this._c, sr: this._sr,
        });
        prob     = out.output.data[0];
        this._h  = out.hn;
        this._c  = out.cn;
      } catch (e) {
        this._warn('inference error:', e.message || e);
        this.ready = false;
        return null;
      }

      if (prob >= SILERO_START_THRESHOLD) {
        if (!this._sileroSpeaking) this._sileroSpeaking = true;
        this._sileroSpeechFrames++;
        this._sileroSilenceFrames = 0;
      } else if (prob < SILERO_END_THRESHOLD) {
        if (this._sileroSpeaking) {
          this._sileroSilenceFrames++;
          if (this._sileroSilenceFrames >= SILERO_SILENCE_FRAMES &&
              this._sileroSpeechFrames  >= SILERO_MIN_SPEECH_FRAMES) {
            this._sileroSpeaking      = false;
            this._sileroSpeechFrames  = 0;
            this._sileroSilenceFrames = 0;
            endOfSpeech = true;
          }
        }
      }
    }

    return { speech: this._sileroSpeaking, endOfSpeech };
  }

  // ── RMS helpers ───────────────────────────────────────────────────────────

  _resetRmsState() {
    this._speechStart    = 0;
    this._silenceStart   = 0;
    this._speechPeak     = 0;
    this._longTurnHinted = false;
  }

  _updateNoiseFloor(rms) {
    if (this._speechStart || !Number.isFinite(rms)) return;
    const clamped = Math.max(0.002, Math.min(0.05, rms));
    this._noiseFloor = Math.max(
      0.004, Math.min(0.04, this._noiseFloor * 0.96 + clamped * 0.04)
    );
  }

  _getStartThreshold() {
    return Math.max(RMS_START_THRESHOLD, this._noiseFloor * RMS_NOISE_MULTIPLIER);
  }

  _getStopThreshold() {
    if (!this._speechPeak)
      return Math.max(RMS_MIN_STOP, this._noiseFloor * 1.7);
    return Math.max(
      RMS_MIN_STOP,
      this._noiseFloor * 1.8,
      Math.min(0.055, this._speechPeak * RMS_SPEECH_DROP)
    );
  }

  _log(...args)  { if (typeof console !== 'undefined') console.log('[vad]', ...args); }
  _warn(...args) { if (typeof console !== 'undefined') console.warn('[vad]', ...args); }
}

// ── Export ────────────────────────────────────────────────────────────────────
// Universal: works as CommonJS (Node), ES module (test file), and browser global.
if (typeof module !== 'undefined' && module.exports) {
  // CommonJS (Node require / node:test)
  module.exports = { VoiceVAD,
    SILERO_START_THRESHOLD, SILERO_END_THRESHOLD,
    SILERO_SILENCE_FRAMES, SILERO_MIN_SPEECH_FRAMES,
    SILERO_WINDOW_SAMPLES,
    RMS_START_THRESHOLD, RMS_MIN_STOP, RMS_NOISE_MULTIPLIER, RMS_SPEECH_DROP,
    RMS_SILENCE_MS, RMS_MIN_SPEECH_MS, RMS_MAX_TURN_MS,
    BARGE_IN_THRESHOLD,
  };
} else if (typeof globalThis !== 'undefined') {
  // Browser global
  globalThis.VoiceVAD = VoiceVAD;
}
