# STT/TTS bake-off eval set (plan doc Part 3.1)

Not committed yet — this directory is scaffolding. Once populated, it's fetched
into the lease container by `docs/deploy/lease-00-seed.sh`'s `git clone` step
(the Dockerfile does not `COPY tests/`, so it isn't baked into the image).

## Format

One triple per utterance, same stem:

```
<id>.wav    16 kHz mono PCM
<id>.txt    reference transcript, UTF-8
<id>.lang   "fr" or "ary" (nothing else — matches app/services/stt.py's
            language_hint contract)
```

`scripts/eval_stt.py` (run via `scripts/benchmark/run_bakeoffs.sh` in the
lease) globs `*.wav` here and scores WER + RTF per STT engine against the
matching `.txt`.

## How to fill it

1. `HF_TOKEN=hf_xxx .gguf_venv/Scripts/python.exe scripts/build_stt_eval_set.py`
   — pulls 8 Darija + 8 French utterances from `atlasia/DODa-audio-dataset`.
   Requires accepting that dataset's gated terms on huggingface.co first,
   logged into the account `HF_TOKEN` belongs to.
2. Record ~6 **code-switched** utterances yourself (French/Darija mixed
   mid-sentence) — the generic dataset has none, and `eval_stt.py`'s own
   docstring calls these the case ad hoc testing misses. A phone voice memo +
   `ffmpeg -i in.m4a -ar 16000 -ac 1 out.wav` is enough; name them
   `codeswitch_00.wav` etc. following the same triple format.
3. `git add tests/data/voice_eval/ && git commit && git push` — this repo is
   public, so treat what goes in here as public. (`arabic_test.pdf` for phase 3
   went through the same path — see `.gitignore`'s `*.pdf` exception.)
