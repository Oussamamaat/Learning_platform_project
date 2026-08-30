"""
Upload the tutor GGUFs to a HuggingFace repo, for fetch-on-startup on Akash.
────────────────────────────────────────────────────────────────────────────
The GPU image no longer bakes the 11 GB of weights (too large for the build
host's disk); the container downloads them at boot from a URL onto a persistent
/models volume (see scripts/docker/entrypoint.sh). This script hosts them on
HuggingFace and prints the exact env lines to paste into deploy/akash-deploy.yaml.

Run AFTER scripts/docker/prepare_models.py has populated config/models/*.gguf.

Prereqs:
    - A HuggingFace account + a WRITE access token (https://huggingface.co/settings/tokens)
    - huggingface_hub (already in .gguf_venv via sentence-transformers)

Usage (from repo root):
    setx HF_TOKEN hf_xxx           # or pass --token; PowerShell: $env:HF_TOKEN="hf_xxx"
    .gguf_venv/Scripts/python.exe scripts/docker/upload_models_hf.py --repo <user>/iblog-tutor-gguf
    # public instead of private:
    .gguf_venv/Scripts/python.exe scripts/docker/upload_models_hf.py --repo <user>/iblog-tutor-gguf --public

Private repo (default) keeps your fine-tunes non-public: the download then needs
a token, so this prints an HF_TOKEN line for the SDL too and entrypoint.sh sends
it as a Bearer header. Public repo needs no token but exposes the weights.

Tip: `pip install hf_transfer` and set HF_HUB_ENABLE_HF_TRANSFER=1 for a much
faster multi-GB upload.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "config" / "models"


def _env_var_for(stem: str) -> str:
    # Same mapping entrypoint.sh uses: uppercase, '-' -> '_', append _GGUF_URL.
    return stem.upper().replace("-", "_") + "_GGUF_URL"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. youruser/iblog-tutor-gguf")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF write token (or set HF_TOKEN)")
    ap.add_argument("--public", action="store_true", help="Make the repo public (default: private)")
    args = ap.parse_args()

    ggufs = sorted(MODELS_DIR.glob("*.gguf"))
    if not ggufs:
        sys.exit(f"ERROR: no *.gguf in {MODELS_DIR}. Run scripts/docker/prepare_models.py first.")

    try:
        from huggingface_hub import HfApi, get_token
    except ImportError:
        sys.exit("ERROR: huggingface_hub not installed. Use .gguf_venv's python, "
                 "or: pip install huggingface_hub")

    # Prefer an explicit token, then HF_TOKEN, then the `hf auth login` cache.
    token = args.token or get_token()
    if not token:
        sys.exit("ERROR: no token. Run `hf auth login`, set HF_TOKEN, or pass --token "
                 "(WRITE scope: https://huggingface.co/settings/tokens).")

    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        print("hf_transfer enabled (faster upload).")

    api = HfApi(token=args.token)
    print(f"Creating repo {args.repo} (private={not args.public}) if needed ...")
    api.create_repo(repo_id=args.repo, repo_type="model", private=not args.public, exist_ok=True)

    base = f"https://huggingface.co/{args.repo}/resolve/main"
    env_lines = []
    for gguf in ggufs:
        print(f"\nUploading {gguf.name} ({gguf.stat().st_size/1e9:.2f} GB) — this is the slow part ...")
        api.upload_file(
            path_or_fileobj=str(gguf), path_in_repo=gguf.name,
            repo_id=args.repo, repo_type="model",
        )
        url = f"{base}/{gguf.name}"
        env_lines.append(f"{_env_var_for(gguf.stem)}={url}")
        print(f"  done -> {url}")

    print("\n" + "=" * 70)
    print("Paste these into deploy/akash-deploy.yaml (app service env:) as `- KEY=VALUE`:")
    for line in env_lines:
        print(f"  - {line}")
    if not args.public:
        print("  - HF_TOKEN=<a READ token>   # private repo: entrypoint sends it as a Bearer header")
        print("\nNOTE: private repo — create a READ token for the SDL, do NOT reuse your write token.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
