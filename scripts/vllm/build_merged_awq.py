"""
Build one fp16 merge + AWQ-INT4 quantized model for a production tutor.
Plan rev 2, Phase A7 (written) / B3 (run on the lease) -- run TWICE, once
per language.

WHERE THIS RUNS: an Akash rtx5090 lease (or any >=32GB-VRAM box), in its
OWN venv per scripts/vllm/requirements-build.txt -- NEVER this repo's
.gguf_venv/.ocr_venv/.speech_venv/.tts_venv (llmcompressor 0.13.0 requires
transformers>=5.9, and those venvs pin transformers 4.x for
sentence-transformers/coqui-tts compatibility; installing this here would
force-upgrade transformers across the whole app). A full merge peaks
around 2x the base model's fp16 size (~18.5GB transient for a 9B model)
plus AWQ calibration activations; this repo's own 8GB laptop cannot run
it AT REAL SCALE, which is why rev 1 of this script shipped without a
runtime test against real weights -- and shipped broken (see
CAVEAT-RESOLVED below). rev 2's local gate instead runs
merge->AWQ->save->reload against a TINY random model
(scripts/vllm/dry_run_awq.py) -- easily small enough for an 8GB card --
before spending lease time on the real 9B run.

Note this gate needs an actual CUDA GPU, even for the tiny model: the
first CPU-only attempt at this dry run failed with `RuntimeError: Cannot
access accelerator device when none is available` inside
AWQModifier._apply_smoothing's cache.pin_memory() call --
llmcompressor's AWQ smoothing step is CUDA-only by construction, not
merely slow on CPU. A laptop with no NVIDIA GPU at all would need to run
dry_run_awq.py on some other small CUDA box instead.

Unlike rev 1, this script does NOT import this repo's `app` package --
calibration text is PRE-RENDERED by scripts/vllm/make_fixtures.py
(calib_darija.jsonl / calib_fr.jsonl, one {"text": ...} object per line,
already run through render_conversation()) and fetched from HF alongside
the adapter, so the lease job never needs this repo's source tree at all,
only the published kit (plan rev 2, Phase A9/B).

    pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
    pip install -r scripts/vllm/requirements-build.txt

Usage -- Darija:
    python build_merged_awq.py \\
        --adapter-repo Oussamamaat/iblog-tutor-adapters --adapter-subdir darija_v11/lora_model \\
        --base MBZUAI-Paris/Atlas-Chat-9B \\
        --calib-repo Oussamamaat/iblog-vllm-lease --calib-file kit/<sha>/fixtures/calib_darija.jsonl \\
        --out-repo Oussamamaat/iblog-tutor-awq --out-subdir darija \\
        --hf-token "$HF_TOKEN" --work-dir /work/build

Usage -- French (note --base is the full-precision repo, NOT what
french_v1/lora_model/adapter_config.json's own base_model_name_or_path
says -- see OVERRIDE below):
    python build_merged_awq.py \\
        --adapter-repo Oussamamaat/iblog-tutor-adapters --adapter-subdir french_v1/lora_model \\
        --base unsloth/gemma-2-9b \\
        --calib-repo Oussamamaat/iblog-vllm-lease --calib-file kit/<sha>/fixtures/calib_fr.jsonl \\
        --out-repo Oussamamaat/iblog-tutor-awq --out-subdir french \\
        --hf-token "$HF_TOKEN" --work-dir /work/build

OVERRIDE (load-bearing, do not remove): the French adapter's own
adapter_config.json names `unsloth/gemma-2-9b-bnb-4bit` as
base_model_name_or_path -- a 4-bit quantized repo. TRAINING_REPORT.json
confirms the model was actually trained against the full-precision
`unsloth/gemma-2-9b`. This script NEVER reads base_model_name_or_path off
the adapter config; it always loads exactly `--base` as passed on the
command line. Merging against the config's literal value would silently
produce a model merged onto degraded 4-bit weights. See
docs/architecture/model-artifacts.md for the full writeup of this finding.

CAVEAT-RESOLVED (was open in rev 1, fixed in rev 2): rev 1 called
`from llmcompressor.modifiers.awq import AWQModifier` with a single
`AWQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])` and
`oneshot(..., dataset=[{"text": p} for p in prompts], ...)`. Neither
matches the current (0.13.0, checked 2026-09-10 against
github.com/vllm-project/llm-compressor's own examples/awq/) API:
  - the import is `llmcompressor.modifiers.transform.awq.AWQModifier`
    (scale-computation step) run ALONGSIDE a SEPARATE
    `llmcompressor.modifiers.quantization.QuantizationModifier(
     scheme="W4A16_ASYM", targets="Linear", ignore=["lm_head"])`
    (the actual quantization step) -- a two-modifier recipe, not one;
  - the calibration dataset must be a tokenized HF `datasets.Dataset`
    (input_ids/attention_mask columns), not a list of raw {"text": ...}
    dicts -- oneshot's dataset= expects pre-tokenization has already run;
  - saving requires `model.save_pretrained(path, save_compressed=True)`,
    not a bare oneshot(output_dir=...) call.
This script's _quantize_awq() below reflects the corrected shape. Before
a real Phase B run: `pip show llmcompressor` on the lease and diff against
https://github.com/vllm-project/llm-compressor/tree/main/examples/awq --
the plan's own "Skills and tools" section flags this exact library as
having moved before, and it already has, once, against this exact script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _done_marker(work_dir: Path, phase: str) -> Path:
    return work_dir / f".done.{phase}"


def _phase_done(work_dir: Path, phase: str) -> bool:
    """Idempotency (plan rev 2, assumption 4): a phase that already ran
    (marker present) is skipped, so a mid-build restart resumes instead of
    redoing GPU-hours of work from zero."""
    return _done_marker(work_dir, phase).exists()


def _mark_done(work_dir: Path, phase: str, info: dict) -> None:
    _done_marker(work_dir, phase).write_text(json.dumps(info, indent=2), encoding="utf-8")


def download_calibration_file(calib_repo: str, calib_file: str, token: str, dest: Path) -> Path:
    """Pull the pre-rendered calibration JSONL from HF -- no `app` import,
    no local repo checkout needed on the lease."""
    from huggingface_hub import hf_hub_download

    print(f"Downloading calibration file {calib_repo}/{calib_file} ...")
    path = hf_hub_download(repo_id=calib_repo, repo_type="dataset", filename=calib_file, token=token)
    return Path(path)


def load_calibration_texts(calib_path: Path, n: int) -> list[str]:
    texts = []
    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(texts) >= n:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(row["text"])
    if not texts:
        sys.exit(f"ERROR: no usable rows in {calib_path}")
    return texts


def merge_adapter(base: str, adapter_dir: Path, merged_out: Path) -> None:
    """Load `base` in bf16, apply the LoRA adapter at `adapter_dir`, and
    save a standalone merged model. `base` is trusted exactly as passed --
    NEVER read from adapter_dir/adapter_config.json's
    base_model_name_or_path. See the module docstring's OVERRIDE section."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base {base} (bf16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base)

    print(f"Applying adapter {adapter_dir} ...")
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    print("Merging (merge_and_unload) ...")
    model = model.merge_and_unload()

    merged_out.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged fp16 model to {merged_out} ...")
    model.save_pretrained(str(merged_out), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_out))
    del model
    torch.cuda.empty_cache()


def _build_calibration_dataset(calib_texts: list[str], tokenizer, max_seq_length: int):
    """oneshot()'s dataset= must already be tokenized (input_ids /
    attention_mask columns) -- the current llmcompressor API, unlike rev
    1's assumption that a list of {"text": ...} dicts would be tokenized
    internally. add_special_tokens=True (the OPPOSITE of
    llm-compressor's own custom_dataset_example.py, which sets False):
    that example's text already had special tokens baked in by
    apply_chat_template; our calibration text is rendered by
    render_conversation() with NO literal <bos> by design (the tokenizer
    adds exactly one), matching real inference -- see
    app/services/llm.py's render_conversation docstring and the
    chat-template-bos-inconsistency memory this migration exists partly
    to avoid repeating."""
    from datasets import Dataset

    ds = Dataset.from_dict({"text": calib_texts})

    def _tokenize(sample):
        return tokenizer(
            sample["text"], padding=False, max_length=max_seq_length,
            truncation=True, add_special_tokens=True,
        )

    return ds.map(_tokenize, remove_columns=["text"])


def quantize_awq(merged_dir: Path, calib_texts: list[str], awq_out: Path, *, max_seq_length: int) -> None:
    """AWQ-INT4 quantize the merged model.

    Two-modifier recipe (see the module docstring's CAVEAT-RESOLVED):
    AWQModifier computes activation-aware per-channel scales;
    QuantizationModifier(scheme="W4A16_ASYM") does the actual weight
    quantization using those scales. Running AWQModifier alone (rev 1's
    mistake) computes scales and quantizes nothing."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier

    print(f"Loading merged model from {merged_dir} for quantization ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir), dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))

    print(f"Tokenizing {len(calib_texts)} calibration rows (max_seq_length={max_seq_length}) ...")
    calib_dataset = _build_calibration_dataset(calib_texts, tokenizer, max_seq_length)

    recipe = [
        AWQModifier(),
        QuantizationModifier(scheme="W4A16_ASYM", targets="Linear", ignore=["lm_head"]),
    ]

    print(f"Running AWQ oneshot quantization ({len(calib_dataset)} calibration rows) ...")
    oneshot(
        model=model,
        dataset=calib_dataset,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=len(calib_dataset),
    )

    awq_out.mkdir(parents=True, exist_ok=True)
    print(f"Saving compressed AWQ weights to {awq_out} ...")
    model.save_pretrained(str(awq_out), save_compressed=True)
    tokenizer.save_pretrained(str(awq_out))
    print(f"AWQ weights written to {awq_out}")


def upload_awq(awq_out: Path, out_repo: str, out_subdir: str, token: str, manifest_path: Path) -> dict:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=out_repo, repo_type="model", private=True, exist_ok=True)
    print(f"Uploading {awq_out} -> {out_repo}/{out_subdir}/ ...")
    api.upload_folder(folder_path=str(awq_out), path_in_repo=out_subdir,
                       repo_id=out_repo, repo_type="model")

    manifest = {"repo": out_repo, "subdir": out_subdir, "files": {}}
    print("\nsha256 of uploaded files:")
    for f in sorted(awq_out.rglob("*")):
        if f.is_file():
            digest = _sha256(f)
            manifest["files"][f.name] = {"sha256": digest, "size_bytes": f.stat().st_size}
            print(f"  {f.name}: {digest}")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest written to {manifest_path}")
    print(f"done -> https://huggingface.co/{out_repo}/tree/main/{out_subdir}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-repo", required=True)
    ap.add_argument("--adapter-subdir", required=True, help="e.g. darija_v11/lora_model")
    ap.add_argument("--base", required=True, help="Public HF base model id -- see OVERRIDE in the docstring")
    ap.add_argument("--calib-repo", help="HF dataset repo holding the pre-rendered calibration JSONL "
                                          "(ignored if --calib-local-file is given)")
    ap.add_argument("--calib-file", help="Path within --calib-repo, e.g. kit/<sha>/fixtures/calib_darija.jsonl")
    ap.add_argument("--calib-local-file", type=Path,
                     help="Local path to an already-checked-out calibration JSONL (e.g. from "
                          "scripts/lease/job.sh's local kit checkout) -- skips the HF download "
                          "entirely. Takes priority over --calib-repo/--calib-file when given.")
    ap.add_argument("--calib-n", type=int, default=256)
    ap.add_argument("--calib-max-seq-length", type=int, default=2048,
                     help="Matches scripts/vllm/make_fixtures.py's own calibration row cap")
    ap.add_argument("--out-repo", required=True)
    ap.add_argument("--out-subdir", required=True, help="e.g. darija or french")
    ap.add_argument("--hf-token", required=True)
    ap.add_argument("--work-dir", type=Path, default=Path("/work/build"),
                     help="Persistent volume path (plan rev 2, R3) -- survives a container restart, "
                          "unlike rev 1's /tmp-or-cwd scratch dir")
    ap.add_argument("--delete-base-cache", action="store_true",
                     help="Delete the merged fp16 copy (and torch's HF cache for --base) after "
                          "quantizing, to reclaim disk before the next language's run (R3/Known risk: "
                          "140Gi ephemeral, ~43GB peak per model)")
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    adapter_local = args.work_dir / "adapter"
    merged_local = args.work_dir / "merged_fp16"
    awq_local = args.work_dir / "awq"
    manifest_path = args.work_dir / f"awq_manifest_{args.out_subdir}.json"

    from huggingface_hub import snapshot_download

    if _phase_done(args.work_dir, f"merge_{args.out_subdir}"):
        print(f"Merge phase already done (marker present) -- skipping to quantize.")
    else:
        print(f"Downloading adapter {args.adapter_repo}/{args.adapter_subdir} ...")
        snapshot_dir = snapshot_download(
            repo_id=args.adapter_repo, repo_type="model", token=args.hf_token,
            allow_patterns=f"{args.adapter_subdir}/*",
        )
        adapter_local = Path(snapshot_dir) / args.adapter_subdir
        merge_adapter(args.base, adapter_local, merged_local)
        _mark_done(args.work_dir, f"merge_{args.out_subdir}", {"base": args.base, "adapter": args.adapter_subdir})

    if _phase_done(args.work_dir, f"quantize_{args.out_subdir}"):
        print(f"Quantize phase already done (marker present) -- skipping to upload.")
    else:
        if args.calib_local_file is not None:
            calib_path = args.calib_local_file
            if not calib_path.exists():
                sys.exit(f"ERROR: --calib-local-file {calib_path} does not exist.")
        elif args.calib_repo and args.calib_file:
            calib_path = download_calibration_file(args.calib_repo, args.calib_file, args.hf_token,
                                                     args.work_dir / "calib_download")
        else:
            sys.exit("ERROR: give either --calib-local-file or both --calib-repo and --calib-file.")
        calib_texts = load_calibration_texts(calib_path, args.calib_n)
        print(f"Loaded {len(calib_texts)} calibration rows from {calib_path}")
        quantize_awq(merged_local, calib_texts, awq_local, max_seq_length=args.calib_max_seq_length)
        _mark_done(args.work_dir, f"quantize_{args.out_subdir}", {"calib_n": len(calib_texts)})

    manifest = upload_awq(awq_local, args.out_repo, args.out_subdir, args.hf_token, manifest_path)
    _mark_done(args.work_dir, f"upload_{args.out_subdir}", manifest)

    if args.delete_base_cache:
        print(f"Cleaning up {merged_local} (18GB+ transient) ...")
        shutil.rmtree(merged_local, ignore_errors=True)
        import os
        hf_cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
        base_cache_name = "models--" + args.base.replace("/", "--")
        base_cache_dir = hf_cache / base_cache_name
        if base_cache_dir.exists():
            print(f"Cleaning up base model HF cache {base_cache_dir} ...")
            shutil.rmtree(base_cache_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
