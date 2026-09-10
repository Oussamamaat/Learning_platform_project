"""
Build one fp16 merge + AWQ-INT4 quantized model for a production tutor.
Plan Step 0 — run this TWICE, once per language.

WHERE THIS RUNS: an Akash rtx5090 lease (or any ≥32GB-VRAM box), in a
throwaway venv — never this repo's .gguf_venv/.ocr_venv/.speech_venv/
.tts_eval_venv. A full merge peaks around 2× the base model's fp16 size
(~18.5GB transient for a 9B model) plus AWQ calibration activations; this
repo's own 8GB laptop cannot run it, which is why it was written without a
runtime test against real weights — see the CAVEAT below before using it.

    pip install "torch>=2.5" --index-url https://download.pytorch.org/whl/cu128
    pip install transformers peft accelerate huggingface_hub llmcompressor

Usage — Darija:
    python build_merged_awq.py \\
        --adapter-repo Oussamamaat/iblog-tutor-adapters --adapter-subdir darija_v11/lora_model \\
        --base MBZUAI-Paris/Atlas-Chat-9B \\
        --calib-file data/v11_merged/train.jsonl \\
        --out-repo Oussamamaat/iblog-tutor-awq --out-subdir darija \\
        --hf-token "$HF_TOKEN"

Usage — French (note --base is the full-precision repo, NOT what
french_v1/lora_model/adapter_config.json's own base_model_name_or_path
says — see OVERRIDE below):
    python build_merged_awq.py \\
        --adapter-repo Oussamamaat/iblog-tutor-adapters --adapter-subdir french_v1/lora_model \\
        --base unsloth/gemma-2-9b \\
        --calib-file data/fr_v3_merged/train.jsonl \\
        --out-repo Oussamamaat/iblog-tutor-awq --out-subdir french \\
        --hf-token "$HF_TOKEN"

OVERRIDE (load-bearing, do not remove): the French adapter's own
adapter_config.json names `unsloth/gemma-2-9b-bnb-4bit` as
base_model_name_or_path — a 4-bit quantized repo. TRAINING_REPORT.json
confirms the model was actually trained against the full-precision
`unsloth/gemma-2-9b`. This script NEVER reads base_model_name_or_path off
the adapter config; it always loads exactly `--base` as passed on the
command line. Merging against the config's literal value would silently
produce a model merged onto degraded 4-bit weights. See
docs/architecture/model-artifacts.md for the full writeup of this finding.

CAVEAT — not yet runtime-verified. Written against llmcompressor's oneshot
AWQ API as documented at authoring time; this library's API has moved
before (the plan's own "Skills and tools" section flags exactly this for
vLLM's CLI surface, and the same caution applies here). Before running on a
real lease: `pip show llmcompressor` for the installed version, then check
https://github.com/vllm-project/llm-compressor's current README/examples
for the AWQModifier / oneshot() call shape and adjust _quantize_awq below
if it has changed. Everything else here (the merge, the base-override, the
calibration-rendering-through-render_conversation) has no such external API
dependency and needs no such check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # so `from app.services.llm import render_conversation` works

SCRATCH = Path("/tmp/vllm_build") if Path("/tmp").exists() else REPO_ROOT / "_scratch_vllm_build"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_calibration_prompts(calib_file: Path, n: int) -> list[str]:
    """Render `n` real training rows through THIS repo's own
    render_conversation() (app/services/llm.py) — the same byte-exact,
    test-locked prompt shape the model was actually trained and served on
    (tests/test_prompt_format.py). Calibrating AWQ's activation scales on
    generic web text instead of this would fit the quantization to the
    wrong distribution entirely: Darija/French code-switched, Socratic,
    citation-heavy prose looks nothing like it.
    """
    from app.services.llm import render_conversation

    prompts = []
    with open(calib_file, "r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= n:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(render_conversation(row["messages"], add_generation_prompt=False))
    if not prompts:
        sys.exit(f"ERROR: no usable rows in {calib_file}")
    return prompts


def merge_adapter(base: str, adapter_dir: Path, merged_out: Path) -> None:
    """Load `base` in bf16, apply the LoRA adapter at `adapter_dir`, and
    save a standalone merged model. `base` is trusted exactly as passed —
    NEVER read from adapter_dir/adapter_config.json's
    base_model_name_or_path. See the module docstring's OVERRIDE section."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base {base} (bf16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto"
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


def quantize_awq(merged_dir: Path, calib_prompts: list[str], awq_out: Path) -> None:
    """AWQ-INT4 quantize the merged model. See the module docstring's
    CAVEAT — verify this call shape against the installed llmcompressor
    version before relying on it."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier

    print(f"Loading merged model from {merged_dir} for quantization ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir), torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))

    calib_dataset = [{"text": p} for p in calib_prompts]

    recipe = [AWQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])]

    awq_out.mkdir(parents=True, exist_ok=True)
    print(f"Running AWQ oneshot quantization ({len(calib_dataset)} calibration prompts) ...")
    oneshot(
        model=model,
        dataset=calib_dataset,
        recipe=recipe,
        output_dir=str(awq_out),
        max_seq_length=8192,  # matches settings.ollama_num_ctx / --max-model-len
        num_calibration_samples=len(calib_dataset),
    )
    tokenizer.save_pretrained(str(awq_out))
    print(f"AWQ weights written to {awq_out}")


def upload_awq(awq_out: Path, out_repo: str, out_subdir: str, token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=out_repo, repo_type="model", private=True, exist_ok=True)
    print(f"Uploading {awq_out} -> {out_repo}/{out_subdir}/ ...")
    api.upload_folder(folder_path=str(awq_out), path_in_repo=out_subdir,
                       repo_id=out_repo, repo_type="model")

    print("\nsha256 of uploaded files (record these in docs/architecture/model-artifacts.md):")
    for f in sorted(awq_out.rglob("*")):
        if f.is_file():
            print(f"  {f.name}: {_sha256(f)}")
    print(f"\ndone -> https://huggingface.co/{out_repo}/tree/main/{out_subdir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-repo", required=True)
    ap.add_argument("--adapter-subdir", required=True, help="e.g. darija_v11/lora_model")
    ap.add_argument("--base", required=True, help="Public HF base model id — see OVERRIDE in the docstring")
    ap.add_argument("--calib-file", required=True, type=Path, help="JSONL with a 'messages' field per row")
    ap.add_argument("--calib-n", type=int, default=128)
    ap.add_argument("--out-repo", required=True)
    ap.add_argument("--out-subdir", required=True, help="e.g. darija or french")
    ap.add_argument("--hf-token", required=True)
    ap.add_argument("--keep-scratch", action="store_true", help="Don't delete the local merged fp16 copy after quantizing")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    SCRATCH.mkdir(parents=True, exist_ok=True)
    adapter_local = SCRATCH / "adapter"
    merged_local = SCRATCH / "merged_fp16"
    awq_local = SCRATCH / "awq"

    print(f"Downloading adapter {args.adapter_repo}/{args.adapter_subdir} ...")
    snapshot_dir = snapshot_download(
        repo_id=args.adapter_repo, repo_type="model", token=args.hf_token,
        allow_patterns=f"{args.adapter_subdir}/*",
    )
    adapter_local = Path(snapshot_dir) / args.adapter_subdir

    calib_prompts = load_calibration_prompts(args.calib_file, args.calib_n)
    print(f"Loaded {len(calib_prompts)} calibration prompts from {args.calib_file}")

    merge_adapter(args.base, adapter_local, merged_local)
    quantize_awq(merged_local, calib_prompts, awq_local)
    upload_awq(awq_local, args.out_repo, args.out_subdir, args.hf_token)

    if not args.keep_scratch:
        print(f"Cleaning up {merged_local} (18GB+ transient) ...")
        shutil.rmtree(merged_local, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
