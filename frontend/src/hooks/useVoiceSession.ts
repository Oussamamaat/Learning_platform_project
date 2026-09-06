import { useCallback, useRef, useState } from "react";
import { API_BASE } from "../services/api";

/**
 * Open-mic voice session -- connects to WS /api/v1/voice/session
 * (app/routers/voice.py), captures the mic via an AudioWorklet
 * (frontend/src/audio/pcm-worklet.js), and plays back the server's
 * synthesized audio through a WebAudio scheduled buffer queue (needed for
 * sample-accurate playback and for barge-in to flush cleanly -- an
 * <audio> element cannot do either).
 *
 * NOT exercised against a real STT/TTS vendor or a live microphone in
 * this codebase yet: settings.stt_engine/tts_engine both default to
 * "none" server-side (see app/services/stt.py, app/services/tts.py's
 * module docstrings -- the Phase 0 bake-off has not run). Connecting
 * still works and the full message protocol is real; only the two engine
 * calls the server makes in response to a transcribed utterance fail
 * loudly today (surfaced here as an "error" state), and the UI is written
 * to handle that gracefully rather than assume a vendor is live.
 */

export type VoiceStatus = "idle" | "connecting" | "listening" | "thinking" | "speaking" | "error";

interface VoiceSessionOptions {
  tenantId?: string;
  sessionId?: string;
  domain?: string;
}

export interface UseVoiceSessionResult {
  status: VoiceStatus;
  transcript: string;
  answerText: string;
  citations: string[];
  errorMessage: string | null;
  start: () => Promise<void>;
  stop: () => void;
}

type ServerMessage =
  | { type: "transcript.partial"; text: string }
  | { type: "transcript.final"; text: string }
  | { type: "citations"; sources: string[] }
  | { type: "audio.start"; sample_rate: number }
  | { type: "answer.delta"; text: string }
  | { type: "audio.end" }
  | { type: "barge_in" }
  | { type: "error"; code: string; detail: string };

function wsUrl(path: string): string {
  return `${API_BASE.replace(/^http/, "ws")}${path}`;
}

export function useVoiceSession(options: VoiceSessionOptions = {}): UseVoiceSessionResult {
  const [status, setStatus] = useState<VoiceStatus>("idle");
  const [transcript, setTranscript] = useState("");
  const [answerText, setAnswerText] = useState("");
  const [citations, setCitations] = useState<string[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const micContextRef = useRef<AudioContext | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const playbackContextRef = useRef<AudioContext | null>(null);
  const nextPlayTimeRef = useRef(0);
  const activeSourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const playbackSampleRateRef = useRef(22050);

  const flushPlayback = useCallback(() => {
    for (const src of activeSourcesRef.current) {
      try {
        src.stop();
      } catch {
        // Already finished -- stopping a finished source throws, harmless.
      }
    }
    activeSourcesRef.current = [];
    if (playbackContextRef.current) {
      nextPlayTimeRef.current = playbackContextRef.current.currentTime;
    }
  }, []);

  const playChunk = useCallback((data: ArrayBuffer) => {
    const ctx = playbackContextRef.current;
    if (!ctx) return;

    const int16 = new Int16Array(data);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 0x8000;

    const buffer = ctx.createBuffer(1, float32.length, playbackSampleRateRef.current);
    buffer.copyToChannel(float32, 0);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.onended = () => {
      activeSourcesRef.current = activeSourcesRef.current.filter((s) => s !== source);
    };

    // Scheduled queue, not immediate playback -- sentence N+1's audio
    // arrives while sentence N is still playing (that overlap is the
    // whole point of sentence-chunked TTS streaming); scheduling each
    // chunk to start exactly when the previous one ends gives gapless
    // playback instead of racing/overlapping audio.
    const startAt = Math.max(nextPlayTimeRef.current, ctx.currentTime);
    source.start(startAt);
    nextPlayTimeRef.current = startAt + buffer.duration;
    activeSourcesRef.current.push(source);
  }, []);

  // Release the mic, socket and audio graph WITHOUT touching `status`.
  // Kept separate from stop() because the failure paths below need to tear
  // everything down and then report "error" -- when teardown owned the
  // status it unconditionally overwrote that with "idle", so every failure
  // looked identical to a normal hang-up: "Connecting..." for an instant,
  // then the panel vanishes with nothing said. Whatever actually broke has
  // to survive the cleanup that follows it.
  const teardown = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;

    workletNodeRef.current?.disconnect();
    workletNodeRef.current = null;

    micStreamRef.current?.getTracks().forEach((t) => t.stop());
    micStreamRef.current = null;

    void micContextRef.current?.close();
    micContextRef.current = null;

    flushPlayback();
    void playbackContextRef.current?.close();
    playbackContextRef.current = null;
  }, [flushPlayback]);

  const stop = useCallback(() => {
    teardown();
    setStatus("idle");
  }, [teardown]);

  const start = useCallback(async () => {
    setStatus("connecting");
    setErrorMessage(null);
    setTranscript("");
    setAnswerText("");
    setCitations([]);

    try {
      const micStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      micStreamRef.current = micStream;

      // Ask for 16kHz to match app/services/vad.py's SAMPLE_RATE /
      // FRAME_BYTES contract, which avoids resampling entirely when the
      // browser obliges. It often does not: `sampleRate` is a REQUEST, and
      // Chrome on Windows (especially with echoCancellation on) commonly
      // hands back the device's own 44100/48000 instead -- and some
      // browser/driver combinations reject the option outright by THROWING
      // from the constructor. Neither is a reason to refuse the session:
      // pcm-worklet.js resamples to 16kHz from whatever rate it is handed,
      // so both cases are handled rather than fatal.
      let micContext: AudioContext;
      try {
        micContext = new AudioContext({ sampleRate: 16000 });
      } catch {
        micContext = new AudioContext();
      }
      micContextRef.current = micContext;
      if (micContext.sampleRate !== 16000) {
        // Worth knowing about -- it is the difference between a no-op and a
        // real resample on every 20ms frame -- but not worth blocking on.
        console.info(
          `[voice] microphone AudioContext is ${micContext.sampleRate}Hz; ` +
            `pcm-worklet.js will resample to 16000Hz for the server.`,
        );
      }

      await micContext.audioWorklet.addModule(new URL("../audio/pcm-worklet.js", import.meta.url));

      const source = micContext.createMediaStreamSource(micStream);
      const worklet = new AudioWorkletNode(micContext, "pcm-worklet-processor");
      workletNodeRef.current = worklet;
      source.connect(worklet);

      // Playback runs in its own AudioContext at the browser's default
      // rate -- WebAudio resamples an AudioBuffer created at a different
      // sampleRate automatically on playback, so this does not need to
      // match whatever settings.tts_voice_fr/tts_voice_ar's engine emits;
      // playbackSampleRateRef just needs to be correct at buffer-creation
      // time (set from the server's own "audio.start" message).
      const playbackContext = new AudioContext();
      playbackContextRef.current = playbackContext;
      nextPlayTimeRef.current = playbackContext.currentTime;
      // The getUserMedia await above (plus its permission prompt) can break
      // the browser's user-activation chain, leaving this context stuck in
      // "suspended" -- source.start() on a suspended context throws nothing
      // and plays nothing, so this must be forced explicitly rather than
      // relying on autoplay-on-creation.
      if (playbackContext.state === "suspended") {
        await playbackContext.resume();
      }

      const params = new URLSearchParams();
      if (options.tenantId) params.set("tenant_id", options.tenantId);
      if (options.sessionId) params.set("session_id", options.sessionId);
      if (options.domain) params.set("domain", options.domain);
      const query = params.toString();
      const ws = new WebSocket(wsUrl(`/api/v1/voice/session${query ? `?${query}` : ""}`));
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      // Set before the socket opens, not inside onopen: the worklet posts its
      // one-off rate report from its constructor, which has already run by
      // now. Frames captured before the socket is OPEN are dropped on
      // purpose -- the server is not listening for them yet.
      worklet.port.onmessage = (event: MessageEvent<ArrayBuffer | { type: string }>) => {
        // The port carries two things: PCM frames (ArrayBuffer) and the
        // worklet's rate report (a plain object). Forwarding the latter
        // would put the string "[object Object]" on a socket whose text
        // channel is strict JSON, so discriminate rather than assume.
        if (!(event.data instanceof ArrayBuffer)) return;
        if (ws.readyState === WebSocket.OPEN) ws.send(event.data);
      };

      ws.onopen = () => {
        setStatus("listening");
      };

      ws.onmessage = (event: MessageEvent<string | ArrayBuffer>) => {
        if (typeof event.data === "string") {
          const msg = JSON.parse(event.data) as ServerMessage;
          switch (msg.type) {
            case "transcript.partial":
              setStatus("thinking");
              break;
            case "transcript.final":
              setTranscript(msg.text);
              setAnswerText("");
              setCitations([]);
              break;
            case "citations":
              setCitations(msg.sources);
              break;
            case "audio.start":
              playbackSampleRateRef.current = msg.sample_rate;
              setStatus("speaking");
              break;
            case "answer.delta":
              setAnswerText((prev) => prev + msg.text);
              break;
            case "audio.end":
              setStatus("listening");
              break;
            case "barge_in":
              flushPlayback();
              setStatus("listening");
              break;
            case "error":
              setErrorMessage(msg.detail);
              setStatus((prev) => (prev === "thinking" || prev === "speaking" ? "listening" : prev));
              break;
          }
        } else {
          playChunk(event.data);
        }
      };

      ws.onerror = () => {
        setErrorMessage("Voice connection failed.");
        setStatus("error");
      };
      ws.onclose = (event) => {
        if (wsRef.current !== ws) return;
        teardown();
        // A close that nobody asked for is a failure, and saying "idle"
        // about it hides it. wasClean covers a normal server-side close;
        // code 1000/1005 covers our own stop() and a plain browser close.
        if (event.wasClean || event.code === 1000 || event.code === 1005) {
          setStatus((prev) => (prev === "error" ? prev : "idle"));
          return;
        }
        setErrorMessage(
          `Voice connection closed unexpectedly (code ${event.code}${event.reason ? `: ${event.reason}` : ""}).`,
        );
        setStatus("error");
      };
    } catch (err) {
      // teardown() BEFORE setStatus, and never stop(): stop() ends with
      // setStatus("idle"), which is what previously swallowed every startup
      // failure into a silent return to the idle button.
      console.error("[voice] could not start the session:", err);
      teardown();
      setErrorMessage(
        err instanceof Error
          ? `Could not start the voice session: ${err.message}`
          : "Could not access the microphone.",
      );
      setStatus("error");
    }
  }, [flushPlayback, options.domain, options.sessionId, options.tenantId, playChunk, teardown]);

  return { status, transcript, answerText, citations, errorMessage, start, stop };
}
