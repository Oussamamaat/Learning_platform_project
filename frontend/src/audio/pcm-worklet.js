// AudioWorkletProcessor: converts the mic's float32 samples to 16-bit PCM at
// 16 kHz and posts fixed-size frames to the main thread over its MessagePort.
//
// Plain JS, not TypeScript -- AudioWorklet modules run in a separate
// realm loaded via audioContext.audioWorklet.addModule(url), which does
// not go through Vite's TS pipeline the way a normal module import does;
// keeping this file untranspiled avoids a separate build step for one
// tiny file. See frontend/src/hooks/useVoiceSession.ts for the loader.
//
// Frame size matches app/services/vad.py's FRAME_BYTES contract: 20ms of
// 16-bit mono PCM = 320 samples = 640 bytes.
//
// RESAMPLING: useVoiceSession.ts asks for a 16 kHz AudioContext, but that is
// a REQUEST, not a guarantee -- browsers and OS drivers routinely hand back
// 44100 or 48000 instead (Chrome on Windows with echoCancellation on is a
// common case). The server cannot detect that: buffering by SAMPLE count
// keeps every frame exactly 640 bytes on the wire, so voice.py's frame-size
// check passes while the frame really holds ~6.7ms of 48 kHz audio -- VAD
// timings silently mean the wrong thing and STT transcribes garbage.
// Refusing to start is worse than fixing it, so convert here instead: the
// worklet is the only place that both knows the true input rate
// (`sampleRate`, a global in AudioWorkletGlobalScope) and sees every sample.

const TARGET_RATE = 16000;
const FRAME_SAMPLES = 320; // 20ms @ 16kHz

class PcmWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Int16Array(FRAME_SAMPLES);
    this._offset = 0;

    // Input samples consumed per output sample. 1 when the browser honoured
    // the 16 kHz request (the fast path below), 3 at 48 kHz, 2.75625 at 44.1.
    this._ratio = sampleRate / TARGET_RATE;
    // Fractional read cursor, carried across process() calls so resampling
    // does not restart -- and therefore does not click -- every 128-sample
    // render quantum.
    this._pos = 0;
    // Last sample of the previous quantum, so interpolation spanning the
    // block boundary has a real left-hand value instead of a zero.
    this._prev = 0;

    this.port.postMessage({ type: "rate", inputSampleRate: sampleRate, resampling: this._ratio !== 1 });
  }

  _push(sample) {
    const s = sample < -1 ? -1 : sample > 1 ? 1 : sample;
    this._buffer[this._offset++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    if (this._offset === FRAME_SAMPLES) {
      // Transfer ownership of the underlying buffer -- avoids a copy on
      // every 20ms frame for the lifetime of an open-mic session.
      this.port.postMessage(this._buffer.buffer, [this._buffer.buffer]);
      this._buffer = new Int16Array(FRAME_SAMPLES);
      this._offset = 0;
    }
  }

  process(inputs) {
    const input = inputs[0];
    const channel = input && input[0];
    if (!channel || channel.length === 0) return true;

    if (this._ratio === 1) {
      for (let i = 0; i < channel.length; i++) this._push(channel[i]);
      return true;
    }

    // Linear interpolation over the virtual array v = [prev, ...channel]
    // (length n + 1), read at fractional positions `pos`, `pos + ratio`, ...
    // Reading v[i] and v[i+1] requires i <= n - 1, hence the `pos < n` bound;
    // whatever overshoot is left becomes the next block's starting cursor,
    // which is exact because the next block's v[0] IS this block's last
    // sample.
    const n = channel.length;
    let pos = this._pos;
    while (pos < n) {
      const i = pos | 0;
      const frac = pos - i;
      const a = i === 0 ? this._prev : channel[i - 1];
      const b = channel[i];
      this._push(a + (b - a) * frac);
      pos += this._ratio;
    }
    this._prev = channel[n - 1];
    this._pos = pos - n;

    return true;
  }
}

registerProcessor("pcm-worklet-processor", PcmWorkletProcessor);
