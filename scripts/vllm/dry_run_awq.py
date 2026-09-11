"""
Dry run of build_merged_awq.py's merge -> AWQ -> save -> reload path,
against a TINY random Gemma-2 model + a tiny random LoRA adapter -- not a
real 9B, but STILL needs a real CUDA GPU (any size -- an 8GB laptop card
is plenty for this tiny model). Plan rev 2, Phase A7's gate.

Originally written as a CPU-only check; the first real run of this exact
script proved that wrong -- llmcompressor's AWQModifier calls
cache.pin_memory() unconditionally during its smoothing step, which
raises `RuntimeError: Cannot access accelerator device when none is
available` with no CUDA device present. That failure is a genuine
environment requirement of the library, not a bug in this script or in
build_merged_awq.py -- see scripts/vllm/requirements-build.txt for the
full note. A machine with no NVIDIA GPU at all cannot run this dry run.

This validates the llmcompressor/peft/transformers CALL SHAPE actually
works against the pinned versions in requirements-build.txt (rev 1
shipped this same script against an API that had already moved --
build_merged_awq.py's module docstring has the full story), so a real
Phase B lease run doesn't discover an import error or an argument
mismatch after already paying for a 32GB-VRAM box and an 18GB base-model
download. It does NOT validate model quality -- a random model produces
random output; only exercising these library entry points without an
exception counts as a pass.

Usage (from a scratch venv with CUDA torch installed -- see
scripts/vllm/requirements-build.txt, and NEVER run this in .gguf_venv):
    python scripts/vllm/dry_run_awq.py [--work-dir <path>] [--keep]

Steps:
    1. Build a tiny Gemma2Config model (real vocab_size, matching the real
       tokenizer -- so calibration token ids are in-range -- but tiny
       hidden_size/layers/heads so this runs in seconds even on a small GPU).
    2. Download ONLY the real unsloth/gemma-2-9b tokenizer (a few MB, not
       the 9B model weights) so tokenization behavior (BOS handling,
       special tokens) matches what production actually sees.
    3. Wrap a tiny random LoRA adapter around it (peft), save the adapter
       alone (not the base) -- mimics the shape of a real adapter dir.
    4. Call build_merged_awq.py's own merge_adapter() unmodified.
    5. Call build_merged_awq.py's own quantize_awq() unmodified, with a
       handful of short calibration strings.
    6. Reload the saved AWQ output with a fresh AutoModelForCausalLM.
       from_pretrained() call and run one forward pass -- confirms the
       compressed checkpoint transformers/vLLM would load is actually
       loadable, not just writable.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_merged_awq import merge_adapter, quantize_awq  # noqa: E402

TOKENIZER_REPO = "unsloth/gemma-2-9b"  # tokenizer-only download, not the 9B weights

CALIB_TEXTS = [
    "<start_of_turn>user\nShnu hiya mou3adat al himaya?<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nQuels sont les equipements de protection ?<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWhy must equipment match the work being done?<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nA short third calibration sentence for testing.<end_of_turn>\n<start_of_turn>model\n",
]


def build_tiny_base(work_dir: Path) -> Path:
    import torch
    from transformers import AutoTokenizer, Gemma2Config, Gemma2ForCausalLM

    print(f"Downloading tokenizer only from {TOKENIZER_REPO} ...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)

    print("Building tiny random Gemma2ForCausalLM (real vocab_size, tiny everything else) ...")
    # Every Linear layer's in/out width below is a multiple of 128
    # (hidden_size=128, intermediate_size=256, head_dim=128) -- AWQ's
    # W4A16_ASYM scheme group-quantizes in blocks of 128 by default,
    # and the first attempt at this dry run (hidden_size=32) failed with
    # `unflatten: Provided sizes [-1, 128] don't multiply up to the size
    # of dim 1 (32)` -- a real constraint of the quantization scheme, not
    # a llmcompressor API-shape bug (real 9B dims are all multiples of
    # 128 already, so this never bites the actual Phase B run).
    config = Gemma2Config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=128,
    )
    model = Gemma2ForCausalLM(config)
    model = model.to(torch.bfloat16)

    base_dir = work_dir / "fake_base"
    base_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(base_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(base_dir))
    print(f"  wrote tiny base model + real tokenizer to {base_dir}")
    return base_dir


def build_tiny_adapter(base_dir: Path, work_dir: Path) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    print("Building a tiny random LoRA adapter over the tiny base ...")
    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16)
    lora_config = LoraConfig(
        r=4, lora_alpha=4, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)

    adapter_dir = work_dir / "fake_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_dir))
    print(f"  wrote adapter-only dir to {adapter_dir}")
    return adapter_dir


def reload_and_forward(awq_out: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Reloading AWQ output from {awq_out} (fresh process would do this; "
          f"here it's the same process, but a separate from_pretrained call) ...")
    model = AutoModelForCausalLM.from_pretrained(str(awq_out), dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(str(awq_out))
    inputs = tokenizer("Test forward pass.", return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    print(f"  forward pass OK, logits shape = {tuple(out.logits.shape)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", type=Path, default=None,
                     help="default: a fresh temp dir, cleaned up unless --keep")
    ap.add_argument("--keep", action="store_true", help="don't delete --work-dir when done")
    args = ap.parse_args()

    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="vllm_dry_run_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Work dir: {work_dir}\n")

    try:
        base_dir = build_tiny_base(work_dir)
        adapter_dir = build_tiny_adapter(base_dir, work_dir)

        print("\n--- merge_adapter() (from build_merged_awq.py, unmodified) ---")
        merged_dir = work_dir / "merged"
        merge_adapter(str(base_dir), adapter_dir, merged_dir)

        print("\n--- quantize_awq() (from build_merged_awq.py, unmodified) ---")
        awq_dir = work_dir / "awq"
        quantize_awq(merged_dir, CALIB_TEXTS, awq_dir, max_seq_length=32)

        print("\n--- reload + forward pass ---")
        reload_and_forward(awq_dir)

        print("\n" + "=" * 60)
        print("DRY RUN PASSED: merge -> AWQ -> save -> reload all completed "
              "with no exception, against the pinned llmcompressor/peft/"
              "transformers versions.")
        return 0
    except Exception:
        print("\n" + "=" * 60)
        print("DRY RUN FAILED -- fix this before spending lease time on a real run:")
        traceback.print_exc()
        return 1
    finally:
        if not args.keep:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"\n--keep set: work dir left at {work_dir}")


if __name__ == "__main__":
    sys.exit(main())
