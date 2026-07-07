// pcm-processor.js — AudioWorklet for 16kHz PCM16 capture + energy VAD
// Runs in audio thread, posts PCM chunks + RMS to main thread

class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Float32Array(0);
    this._chunkSize = 320; // 20ms at 16kHz
  }

  process(inputs, outputs, params) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const samples = input[0]; // Float32, sampleRate of context (16kHz)

    // Append to buffer
    const newBuf = new Float32Array(this._buffer.length + samples.length);
    newBuf.set(this._buffer);
    newBuf.set(samples, this._buffer.length);
    this._buffer = newBuf;

    // Emit 20ms chunks
    while (this._buffer.length >= this._chunkSize) {
      const chunk = this._buffer.slice(0, this._chunkSize);
      this._buffer = this._buffer.slice(this._chunkSize);

      // Compute RMS for energy-based VAD
      let sum = 0;
      for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i];
      const rms = Math.sqrt(sum / chunk.length);

      // Convert to Int16
      const pcm16 = new Int16Array(chunk.length);
      for (let i = 0; i < chunk.length; i++) {
        pcm16[i] = Math.max(-32768, Math.min(32767, Math.round(chunk[i] * 32767)));
      }

      // Keep a float32 copy for Silero VAD (needs un-quantized samples)
      const float32 = chunk.slice();

      this.port.postMessage({ pcm16: pcm16.buffer, float32: float32.buffer, rms },
                            [pcm16.buffer, float32.buffer]);
    }
    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);
