"""
Extract the tutor GGUFs + Modelfiles for baking into the GPU Docker image.
────────────────────────────────────────────────────────────────────────────
Run this LOCALLY on the machine that already has the models registered in
Ollama (the dev laptop), BEFORE `docker build -f config/Dockerfile.gpu`.

Why this exists: the two production models live only inside Ollama's private
blob store (`~/.ollama/models/blobs/sha256-...`) -- there is no `.gguf` on
disk in the repo, and nothing in `app/` or the containers registers them
(see docs/deploy/akash-rtx5090-runbook.md). This script pulls each model's
GGUF blob out of that store and writes a self-contained Modelfile next to it,
so `config/Dockerfile.gpu` can `COPY config/models/` into the image and the
container entrypoint can `ollama create` from it with zero network access.

It does NOT talk to Ollama's HTTP API or re-download anything -- it parses
`ollama show <model> --modelfile` (which prints the absolute blob path on the
`FROM` line and the exact TEMPLATE/PARAMETER block the model was built with),
copies that blob, and rewrites only the FROM line to a relative path.

Output (git-ignored -- large binaries, build input only):
    config/models/IBLOG_TUTOR.gguf        + IBLOG_TUTOR.Modelfile
    config/models/iblog-tutor-fr.gguf     + iblog-tutor-fr.Modelfile
    config/models/manifest.json           (sha256 + sizes, for verification)

Usage (from repo root, any Python 3.9+, no venv needed -- stdlib only):
    python scripts/docker/prepare_models.py
    python scripts/docker/prepare_models.py --model IBLOG_TUTOR:latest --as IBLOG_TUTOR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "config" / "models"

# The two tags the running app actually requests (app/config.py:11-12,
# read as settings.ollama_model / ollama_model_fr). `--as` is the filename
# stem the entrypoint's `ollama create <stem>` will use, so it must match
# OLLAMA_MODEL / OLLAMA_MODEL_FR (minus the :latest tag) in deploy/akash.env.
DEFAULT_MODELS = [
    ("IBLOG_TUTOR:latest", "IBLOG_TUTOR"),
    ("iblog-tutor-fr:latest", "iblog-tutor-fr"),
]

_FROM_RE = re.compile(r"^\s*FROM\s+(.+?)\s*$", re.MULTILINE)


def _run_ollama_show(model: str) -> str:
    """Return `ollama show <model> --modelfile` stdout, or raise with a
    readable message if Ollama isn't on PATH or the model isn't registered."""
    try:
        proc = subprocess.run(
            ["ollama", "show", model, "--modelfile"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        sys.exit(
            "ERROR: `ollama` is not on PATH. Run this on the machine that has "
            "the models registered (the dev laptop), with Ollama installed."
        )
    if proc.returncode != 0:
        sys.exit(
            f"ERROR: `ollama show {model} --modelfile` failed "
            f"(exit {proc.returncode}):\n{proc.stderr.strip()}\n"
            f"Is the model registered? Check `ollama list`."
        )
    return proc.stdout


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _copy_with_progress(src: Path, dst: Path) -> None:
    total = src.stat().st_size
    copied = 0
    step = max(total // 20, 1)
    next_mark = step
    with src.open("rb") as fin, dst.open("wb") as fout:
        while True:
            chunk = fin.read(4 * 1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)
            copied += len(chunk)
            if copied >= next_mark:
                pct = 100 * copied / total
                print(f"    ...{pct:4.0f}%  ({copied/1e9:.2f} GB)", end="\r", flush=True)
                next_mark += step
    print(f"    ...100%  ({total/1e9:.2f} GB)          ")


def export_model(model: str, stem: str) -> dict:
    print(f"\n[{model}] -> {stem}.gguf")
    modelfile_text = _run_ollama_show(model)

    m = _FROM_RE.search(modelfile_text)
    if not m:
        sys.exit(f"ERROR: no FROM line in `ollama show {model} --modelfile` output.")
    blob_path = Path(m.group(1).strip().strip('"'))
    if not blob_path.exists():
        sys.exit(
            f"ERROR: GGUF blob not found at {blob_path}\n"
            f"(parsed from the FROM line of `ollama show {model}`)."
        )

    gguf_dst = OUT_DIR / f"{stem}.gguf"
    print(f"  blob: {blob_path}")
    print(f"  copying -> {gguf_dst.relative_to(REPO_ROOT)}")
    _copy_with_progress(blob_path, gguf_dst)

    # Rewrite only the FROM line to a path relative to the Modelfile itself,
    # so `ollama create` works from inside the image regardless of cwd.
    rewritten = _FROM_RE.sub(f"FROM ./{stem}.gguf", modelfile_text, count=1)
    modelfile_dst = OUT_DIR / f"{stem}.Modelfile"
    modelfile_dst.write_text(rewritten, encoding="utf-8")
    print(f"  wrote  -> {modelfile_dst.relative_to(REPO_ROOT)}")

    digest = _sha256(gguf_dst)
    return {
        "tag": model,
        "stem": stem,
        "gguf": f"{stem}.gguf",
        "modelfile": f"{stem}.Modelfile",
        "size_bytes": gguf_dst.stat().st_size,
        "sha256": digest,
        "source_blob": str(blob_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="Ollama tag to export (repeatable with --as).")
    ap.add_argument("--as", dest="stem", help="Output filename stem for --model.")
    args = ap.parse_args()

    if bool(args.model) ^ bool(args.stem):
        sys.exit("ERROR: --model and --as must be given together.")
    models = [(args.model, args.stem)] if args.model else DEFAULT_MODELS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    manifest = [export_model(model, stem) for model, stem in models]
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_gb = sum(e["size_bytes"] for e in manifest) / 1e9
    print(f"\nDone. {len(manifest)} model(s), {total_gb:.2f} GB total in config/models/.")
    print("Next: docker build -f config/Dockerfile.gpu -t <registry>/iblog-tutor:gpu .")


if __name__ == "__main__":
    main()
