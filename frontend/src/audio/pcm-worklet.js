// AudioWorkletProcessor: converts the mic's float32 samples to 16-bit PCM
// and posts fixed-size frames to the main thread over its MessagePort.
//
// Plain JS, not TypeScript -- AudioWorklet modules run in a separate
// realm loaded via audioContext.audioWorklet.addModule(url), which does
// not go through Vite's TS pipeline the way a normal module import does;
// keeping this file untranspiled avoids a separate build step for one
// tiny file. See frontend/src/hooks/useVoiceSession.ts for the loader.
//
// Frame size matches app/services/vad.py's FRAME_BYTES contract: 20ms of
// 16-bit mono PCM. The AudioContext this runs in is created with
// { sampleRate: 16000 } (useVoiceSession.ts), so no resampling happens
// here -- one AudioWorklet render quantum (128 samples) is simply
// accumulated until a full 20ms (320 samples) frame is ready.

const FRAME_SAMPLES = 320; // 20ms @ 16kHz

class PcmWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Int16Array(FRAME_SAMPLES);
    this._offset = 0;
  }

  process(inputs) {
    const input = inputs[0];
    const channel = input && input[0];
    if (!channel) return true;

    for (let i = 0; i < channel.length; i++) {
      const sample = Math.max(-1, Math.min(1, channel[i]));
      this._buffer[this._offset++] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      if (this._offset === FRAME_SAMPLES) {
        // Transfer ownership of the underlying buffer -- avoids a copy on
        // every 20ms frame for the lifetime of an open-mic session.
        this.port.postMessage(this._buffer.buffer, [this._buffer.buffer]);
        this._buffer = new Int16Array(FRAME_SAMPLES);
        this._offset = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm-worklet-processor", PcmWorkletProcessor);
