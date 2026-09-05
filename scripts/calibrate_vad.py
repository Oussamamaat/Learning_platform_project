"""
Offline VAD threshold calibration -- sweeps app.services.vad.EnergyEndpointer
against the REAL speech recordings already committed at
tests/data/voice_eval/*.wav (16kHz mono 16-bit PCM, 30 files: 16 Darija, 8
French, 6 code-switched). No mic, no GPU, no model -- pure signal
processing over committed fixtures, so this is free and reproducible.

Why this exists (POST_LEASE_MVP_SPRINT_PLAN.md item 2): the 2026-09-04
Akash lease found the live voice interface did not reliably detect speech
or barge-in, and a prior planning pass hypothesized the fixed RMS
threshold (500.0) was too high. This script tests that hypothesis against
the only real-speech evidence in the repo, and the answer is: no --
500.0 fires speech_start on 30/30 files. See this script's own output and
docs/architecture/rectified/adr/0005-vad-calibration.md for the full
writeup and what's checked instead (browser AGC/noise suppression, sample
rate, frame-size mismatches -- upstream of the threshold).

Usage:
    .gguf_venv/Scripts/python.exe scripts/calibrate_vad.py
"""
import statistics
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.services.vad import EnergyEndpointer, FRAME_BYTES  # noqa: E402

EVAL_DIR = REPO_ROOT / "tests" / "data" / "voice_eval"
THRESHOLDS = (500.0, 300.0, 200.0, 100.0, 60.0)


def _frames(path: Path) -> list[bytes]:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == 16000, (
            f"{path.name}: expected mono/16-bit/16kHz, got "
            f"{w.getnchannels()}ch/{w.getsampwidth() * 8}bit/{w.getframerate()}Hz"
        )
        data = w.readframes(w.getnframes())
    return [data[i:i + FRAME_BYTES] for i in range(0, len(data) - FRAME_BYTES + 1, FRAME_BYTES)]


def main() -> None:
    files = sorted(EVAL_DIR.glob("*.wav"))
    if not files:
        print(f"No .wav files found under {EVAL_DIR}")
        sys.exit(1)

    print(f"Calibrating against {len(files)} real utterances in {EVAL_DIR}\n")

    # Frame-RMS distribution across the whole corpus, threshold-independent.
    all_rms = []
    for path in files:
        for frame in _frames(path):
            all_rms.append(EnergyEndpointer._rms(frame))
    all_rms.sort()
    print(f"Frame-RMS distribution over {len(all_rms)} frames (all files):")
    for q in (1, 5,10,25,50,75,90,95, 99):
        idx = int(len(all_rms) * q / 100)
        print(f"  p{q:02d} = {all_rms[idx]:7.1f}")
    print()

    print(f"{'threshold':>10} {'files_detected':>15} {'mean_segments':>14} {'mean_start_ms':>14}")
    for threshold in THRESHOLDS:
        detected = 0
        segment_counts = []
        start_frames = []
        for path in files:
            frames = _frames(path)
            ep = EnergyEndpointer(threshold=threshold)
            starts = 0
            first_start = None
            for i, frame in enumerate(frames):
                event = ep.push(frame)
                if event == "speech_start":
                    starts += 1
                    if first_start is None:
                        first_start = i
            if starts:
                detected += 1
            segment_counts.append(starts)
            if first_start is not None:
                start_frames.append(first_start)
        mean_segments = sum(segment_counts) / len(segment_counts)
        mean_start_ms = (sum(start_frames) / len(start_frames)) * 20 if start_frames else float("nan")
        marker = " (current default)" if threshold == 500.0 else ""
        print(f"{threshold:>10.1f} {detected:>10d}/{len(files):<4d} {mean_segments:>14.2f} "
              f"{mean_start_ms:>11.0f} ms{marker}")

    print(
        "\nConclusion: if every threshold above detects speech in "
        f"{len(files)}/{len(files)} files, the threshold is not the "
        "bottleneck for THIS corpus -- a live-mic failure is more likely "
        "upstream (browser AGC/noise suppression, sample-rate mismatch, "
        "frame-size mismatch) than in EnergyEndpointer's threshold. See "
        "app/routers/voice.py's settings.vad_debug_log for live capture."
    )


if __name__ == "__main__":
    main()
