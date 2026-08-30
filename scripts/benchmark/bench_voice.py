"""
Voice pipeline latency benchmark (WS /api/v1/voice/session).
    pip install websockets
    python scripts/benchmark/bench_voice.py --base-url http://<host>:<port> --wav utterance.wav

Streams a 16 kHz mono 16-bit PCM WAV as binary frames, appends trailing silence
to trigger the server's VAD endpoint (speech_end), then times:
    end-of-speech  →  first audio byte back      (target ≈1.2–2.2 s, voice doc §4)

Requires the server running with STT_ENGINE and TTS_ENGINE OFF "none" — otherwise
the session returns {"code":"voice_unavailable"} and this reports that cleanly
(which still confirms the WS wiring end-to-end).

--wav must be 16 kHz mono 16-bit PCM. With no --wav, a 1.5 s 220 Hz tone is sent
as a WIRING check only (STT will likely return empty text → no answer; use a real
speech WAV to measure the full path).
"""
import argparse
import asyncio
import json
import math
import struct
import time
import wave

SAMPLE_RATE = 16000
FRAME_SAMPLES = 320          # 20 ms — matches app/services/vad.py FRAME_BYTES=640
FRAME_BYTES = FRAME_SAMPLES * 2


def _load_pcm(path: str | None) -> bytes:
    if path is None:
        n = int(SAMPLE_RATE * 1.5)
        return b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * 220 * i / SAMPLE_RATE)))
                        for i in range(n))
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != SAMPLE_RATE:
            raise SystemExit(f"{path} must be 16 kHz mono 16-bit PCM "
                             f"(got {w.getframerate()}Hz, {w.getnchannels()}ch, "
                             f"{w.getsampwidth()*8}-bit)")
        return w.readframes(w.getnframes())


def _ws_url(base_url: str) -> str:
    u = base_url.rstrip("/")
    u = u.replace("https://", "wss://").replace("http://", "ws://")
    return u + "/api/v1/voice/session"


async def run(base_url: str, wav: str | None, timeout: float) -> dict:
    import websockets  # imported here so --help works without the dep

    pcm = _load_pcm(wav)
    silence = b"\x00" * FRAME_BYTES
    url = _ws_url(base_url)
    print(f"connecting: {url}")
    result: dict = {"wav": wav, "url": url}

    async with websockets.connect(url, max_size=None) as ws:
        # Stream speech frames in real time-ish, then ~1 s of silence to endpoint.
        speech_frames = [pcm[i:i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]
        for fr in speech_frames:
            if len(fr) < FRAME_BYTES:
                fr = fr + b"\x00" * (FRAME_BYTES - len(fr))
            await ws.send(fr)
            await asyncio.sleep(0.02)
        end_of_speech = time.perf_counter()
        for _ in range(50):  # 1 s of silence
            await ws.send(silence)
            await asyncio.sleep(0.02)

        first_audio = None
        transcript = None
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                if isinstance(msg, bytes):
                    if first_audio is None:
                        first_audio = time.perf_counter()
                        print(f"  first audio byte: {first_audio - end_of_speech:.2f}s after speech end")
                    continue
                evt = json.loads(msg)
                t = evt.get("type")
                if t == "transcript.final":
                    transcript = evt.get("text")
                    print(f"  transcript: {transcript!r}")
                elif t == "error":
                    print(f"  server error: {evt.get('code')} — {evt.get('detail')}")
                    result["error"] = evt
                    if evt.get("code") in ("voice_unavailable", "stt_unavailable"):
                        break
                elif t == "audio.end":
                    break
        except asyncio.TimeoutError:
            print("  timed out waiting for audio")
        try:
            await ws.send(json.dumps({"type": "end"}))
        except Exception:
            pass

    result["transcript"] = transcript
    if first_audio is not None:
        result["end_of_speech_to_first_audio_s"] = round(first_audio - end_of_speech, 2)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--wav", default=None, help="16 kHz mono 16-bit PCM WAV of a real utterance")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--out", default="benchmark_voice.json")
    args = ap.parse_args()

    result = asyncio.run(run(args.base_url, args.wav, args.timeout))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
