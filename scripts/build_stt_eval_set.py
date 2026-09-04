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
before writing anything, specifically so a wrong guess about which column is
"the audio" vs "the transcript" gets caught immediately instead of producing
25 silently-mislabeled eval files.

Decodes audio with `soundfile`, not `datasets`' own auto-decode: recent
`datasets` versions decode the Audio column via torchcodec, which dynamically
loads FFmpeg's shared libraries at import time -- not installed on a bare
Windows dev box, and a heavier, more environment-specific dependency than this
one-off script warrants. The Audio column is cast to decode=False (raw bytes,
no decoding attempted by `datasets` at all) and decoded here instead;
`soundfile` bundles libsndfile in its wheel, so it needs nothing else on the
system.

atlasia/DODa-audio-dataset turns out to be Darija-audio-only (confirmed live,
not assumed -- see the printed schema): four transcript variants per row
(darija_Latn, darija_Arab_new, darija_Arab_old, english) and no French text or
audio at all. So this script only produces the Darija half of the eval set --
`darija_Arab_new` is the transcript column used, never darija_Latn, which is
Arabizi and exactly what CLAUDE.md's data-generation invariants say Darija
output must never be (Arabic script only). French utterances and the ~6
code-switched ones need a separate source entirely -- French from something
like Common Voice fr, code-switched from your own recordings (a phone voice
memo + `ffmpeg -i in.m4a -ar 16000 -ac 1 out.wav` is enough) -- and drop them
in tests/data/voice_eval/ alongside this script's output using the same
<id>.wav/<id>.txt/<id>.lang naming.
"""
import os
import sys
import wave

# Windows' default console codepage (cp1252) can't encode Arabic script, and
# every transcript printed below is Arabic -- reconfigure stdout to UTF-8 so
# progress output doesn't crash mid-run. The written .txt files were already
# unaffected (write_text(..., encoding="utf-8") below is explicit); this only
# fixes what gets echoed to the terminal.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "data" / "voice_eval"
# This dataset has no French (see module docstring) -- pull up to 16 Darija
# utterances from it instead of splitting 8/8; get French from elsewhere.
N_DARIJA = 16
N_FRENCH = 0

# Column-name candidates, tried in order -- HF audio datasets aren't
# consistent about naming, and this list is deliberately checked BEFORE
# darija_Latn (confirmed present in atlasia/DODa-audio-dataset -- Arabizi,
# never the right choice here, see module docstring). Adjust here if the
# printed schema (see main()) shows something not in this list.
AUDIO_COL_CANDIDATES = ["audio", "wav", "speech"]
TEXT_COL_CANDIDATES = [
    "darija_Arab_new", "darija_Arab_old",  # atlasia/DODa-audio-dataset, Arabic script
    "sentence", "text", "transcription", "transcript",
]
LANG_COL_CANDIDATES = ["lang", "language", "locale"]


def _decode_and_write_wav(path: Path, audio_bytes: bytes) -> int:
    """Decode with soundfile (not datasets' own torchcodec-based auto-decode
    -- see module docstring) and write 16-bit mono PCM. Returns the sample
    rate actually written, for the caller to print/sanity-check."""
    import io
    import numpy as np
    import soundfile as sf

    array, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if array.ndim > 1:
        array = array.mean(axis=1)  # downmix to mono if the source isn't already
    pcm16 = (np.clip(array, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sampling_rate)
        f.writeframes(pcm16.tobytes())
    return sampling_rate


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

    from datasets import Audio, load_dataset

    print("Loading atlasia/DODa-audio-dataset (first access after accepting the "
          "gate may take a minute to authorize) ...")
    ds = load_dataset("atlasia/DODa-audio-dataset", split="train", token=token, streaming=True)

    # Schema only, from dataset metadata -- no row is fetched or decoded here,
    # so this is safe to print even before the Audio column is recast below.
    print("\n=== column schema -- verify before this writes anything ===")
    for name, feature in ds.features.items():
        print(f"  {name!r}: {feature}")
    print()

    columns = list(ds.features.keys())
    audio_col = next(
        (n for n, f in ds.features.items() if isinstance(f, Audio)),
        None,
    ) or _pick_column({c: None for c in columns}, AUDIO_COL_CANDIDATES, "audio")
    text_col = _pick_column({c: None for c in columns}, TEXT_COL_CANDIDATES, "text")
    lang_col = next((c for c in LANG_COL_CANDIDATES if c in columns), None)
    lang_col_desc = repr(lang_col) if lang_col else "(dataset is Darija-only, defaulting to ary)"
    print(f"Using audio={audio_col!r} text={text_col!r} lang={lang_col_desc}\n")

    # decode=False: datasets returns {"bytes": ..., "path": ...} for this
    # column instead of trying to auto-decode it (which needs torchcodec) --
    # _decode_and_write_wav does the actual decoding, with soundfile.
    ds = ds.cast_column(audio_col, Audio(decode=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"ary": 0, "fr": 0}
    written = 0

    for row in ds:
        if written >= N_DARIJA + N_FRENCH:
            break
        lang = row[lang_col] if lang_col else "ary"
        lang = "ary" if lang.lower().startswith(("ar", "dar")) else "fr" if lang.lower().startswith("fr") else None
        if lang is None or counts[lang] >= (N_DARIJA if lang == "ary" else N_FRENCH):
            continue

        text = row[text_col]
        if not text or not text.strip():
            continue
        audio_bytes = row[audio_col]["bytes"]
        if not audio_bytes:
            continue  # decode=False with no inline bytes (path-only row) -- skip rather than fetch separately

        idx = counts[lang]
        stem = f"doda_{lang}_{idx:02d}"
        sr = _decode_and_write_wav(OUT_DIR / f"{stem}.wav", audio_bytes)
        (OUT_DIR / f"{stem}.txt").write_text(text.strip(), encoding="utf-8")
        (OUT_DIR / f"{stem}.lang").write_text(lang, encoding="utf-8")
        counts[lang] += 1
        written += 1
        print(f"  wrote {stem} ({lang}, {sr}Hz): {text.strip()[:60]}")

    print(f"\nWrote {written} triples to {OUT_DIR} (ary={counts['ary']}).")
    print("Still needed: ~8 French utterances (this dataset has none -- try Common Voice fr,")
    print("or your own recordings) and ~6 code-switched -- see this script's module docstring.")
    print("Commit + push tests/data/voice_eval/ so lease-00-seed.sh's git-clone step can pull it.")


if __name__ == "__main__":
    main()
