# Vendored: Chatterbox multilingual TTS runtime

This directory is a **vendored, inference-only slice** of
[gokhaneraslan/chatterbox-finetuning](https://github.com/gokhaneraslan/chatterbox-finetuning)
(Apache 2.0, see `LICENSE`), commit `fac31c4` (2026-06-01), plus the speed patches and text
normalizer written for this project's own Darija fine-tune (`tts_test/remote/chatterbox-finetuning/`
on the dev machine, not part of this repo).

Vendored here, unmodified:
- `src/__init__.py`, `src/chatterbox_/` -- the runtime model package (T3, S3Gen, tokenizers,
  voice encoder). `ChatterboxMultilingualTTS.from_local()` in `src/chatterbox_/mtl_tts.py` is the
  loader `scripts/tts_worker_chatterbox.py` calls.
- `fast_t3.py`, `fast_s3gen.py` -- CUDA-graph decode paths (bit-identical / near-identical to
  stock; see `docs/` memory `chatterbox-t3-is-cpu-launch-bound.md` for the numbers).
- `text_normalize.py` -- `prepare_for_tts()`: sentence split, French-run tagging
  (`[fr]...[ar]`), number/unit expansion to MSA words.

**Deliberately NOT vendored** (training-only, lives in `tts_test/` on the dev machine, not
needed to run inference): `finetune_multilingual.py`, `train.py`, `src/dataset.py`,
`src/model.py`, `src/config.py`, `src/config_multilingual.py`, `src/preprocess_*.py`,
`src/inference_callback.py`, `src/utils.py`, and all model weights/adapters (fetched at
deploy/runtime from Hugging Face -- see `scripts/tts_worker_chatterbox.py` and
`scripts/docker/entrypoint.sh`).

**Do not hand-edit these files.** If a fix is needed, make it in `tts_test/remote/
chatterbox-finetuning/` first (where it can be tested against the full toolkit and the
approved-by-ear adapter), then re-copy here and update the commit reference above. An edit made
only here will silently diverge from what was actually validated.

Model weights license: the base checkpoint (`ResembleAI/chatterbox`) is MIT. This code's own
license (Apache 2.0) is independent of that. The `cs-run1` LoRA adapter's training-data license
is unresolved (YouTube-sourced, no stated license) -- see
`tts_test/docs/CURRENT_MODEL.md` and this project's memory `darija-tts-current-working-version.md`.
