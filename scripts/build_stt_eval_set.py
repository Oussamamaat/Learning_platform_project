"""
Build tests/data/voice_eval/ from atlasia/DODa-audio-dataset (plan doc Part 3.1).

Run locally (not in the lease -- this is free prep, no GPU needed):
    HF_TOKEN=hf_xxx .gguf_venv/Scripts/python.exe scripts/build_stt_eval_set.py

Prerequisite: accept the dataset's gated terms once at
https://huggingface.co/datasets/atlasia/DODa-audio-dataset (logged into the
account HF_TOKEN belongs to) -- the API 403s until that's done, no way around
it programmatically.

This dataset's exact column names aren't independently verified here (the gate
blocked inspecting it without a token) -- the first run prints the raw schema
of one example before writing anything, specifically so a wrong guess about
which column is "the audio" vs "the transcript" gets caught immediately
instead of producing 25 silently-mislabeled eval files.

Only pulls Darija (`ary`) and French utterances from the dataset -- the ~6
code-switched utterances the plan calls for have no public source; record
those yourself (a phone voice memo + `ffmpeg -i in.m4a -ar 16000 -ac 1 out.wav`
is enough) and drop them in tests/data/voice_eval/ alongside this script's
output using the same <id>.wav/<id>.txt/<id>.lang naming.
"""
import os
import sys
import wave
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "data" / "voice_eval"
N_DARIJA = 8
N_FRENCH = 8

# Column-name candidates, tried in order -- HF audio datasets aren't
# consistent about naming. Adjust here if the printed schema (see main())
# shows something not in these lists.
AUDIO_COL_CANDIDATES = ["audio", "wav", "speech"]
TEXT_COL_CANDIDATES = ["sentence", "text", "transcription", "transcript"]
LANG_COL_CANDIDATES = ["lang", "language", "locale"]


def _write_wav(path: Path, array, sampling_rate: int) -> None:
    import numpy as np
    pcm16 = (np.clip(array, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sampling_rate)
        f.writeframes(pcm16.tobytes())


def _pick_column(row: dict, candidates: list[str], kind: str) -> str:
    for c in candidates:
        if c in row:
            return c
    sys.exit(
        f"Could not find a {kind} column among {list(row.keys())}. "
        f"Add the real column name to {kind.upper()}_COL_CANDIDATES at the top "
        f"of this script and re-run."
    )


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN to an account that has accepted the dataset's gated terms.")

    from datasets import load_dataset

    print("Loading atlasia/DODa-audio-dataset (first access after accepting the "
          "gate may take a minute to authorize) ...")
    ds = load_dataset("atlasia/DODa-audio-dataset", split="train", token=token, streaming=True)
    it = iter(ds)
    first = next(it)

    print("\n=== first example's raw schema -- verify before this writes anything ===")
    for k, v in first.items():
        preview = v if not isinstance(v, (bytes, dict)) else type(v).__name__
        text = str(preview)
        if isinstance(preview, str) and len(text) > 80:
            text = repr(text[:80] + "...")
        else:
            text = repr(preview)
        print(f"  {k!r}: {text}")
    print()

    audio_col = _pick_column(first, AUDIO_COL_CANDIDATES, "audio")
    text_col = _pick_column(first, TEXT_COL_CANDIDATES, "text")
    lang_col = None
    for c in LANG_COL_CANDIDATES:
        if c in first:
            lang_col = c
            break
    lang_col_desc = repr(lang_col) if lang_col else "(dataset is Darija-only, defaulting to ary)"
    print(f"Using audio={audio_col!r} text={text_col!r} lang={lang_col_desc}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"ary": 0, "fr": 0}
    written = 0

    def rows():
        yield first
        yield from it

    for row in rows():
        if written >= N_DARIJA + N_FRENCH:
            break
        lang = row[lang_col] if lang_col else "ary"
        lang = "ary" if lang.lower().startswith(("ar", "dar")) else "fr" if lang.lower().startswith("fr") else None
        if lang is None or counts[lang] >= (N_DARIJA if lang == "ary" else N_FRENCH):
            continue

        audio = row[audio_col]
        text = row[text_col]
        if not text or not text.strip():
            continue

        idx = counts[lang]
        stem = f"doda_{lang}_{idx:02d}"
        _write_wav(OUT_DIR / f"{stem}.wav", audio["array"], audio["sampling_rate"])
        (OUT_DIR / f"{stem}.txt").write_text(text.strip(), encoding="utf-8")
        (OUT_DIR / f"{stem}.lang").write_text(lang, encoding="utf-8")
        counts[lang] += 1
        written += 1
        print(f"  wrote {stem} ({lang}): {text.strip()[:60]}")

    print(f"\nWrote {written} triples to {OUT_DIR} (ary={counts['ary']} fr={counts['fr']}).")
    print("Still needed: ~6 code-switched utterances -- see this script's module docstring.")
    print("Commit + push tests/data/voice_eval/ so lease-00-seed.sh's git-clone step can pull it.")


if __name__ == "__main__":
    main()
