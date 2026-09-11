"""
Publish everything the benchmark/build lease needs -- scripts + fixtures --
to a private HF dataset repo, so the lease job (scripts/lease/job.sh) never
needs a git checkout of this repo (plan rev 2, Phase A9). Run this from the
laptop whenever a lease-side script changes, before starting a lease.

Kit layout on HF (Oussamamaat/iblog-vllm-lease, dataset repo):
    kit/<git-sha>/
        scripts/vllm/{build_merged_awq.py, serve_pair.sh, parity_probe.py,
                      quality_sample.py, requirements-build.txt}
        scripts/lease/job.sh
        scripts/benchmark/bench_concurrency.py
        fixtures/{bench_prompts.json, parity_prompts.json,
                  quality_prompts.json, calib_darija.jsonl, calib_fr.jsonl}
    kit/latest -> a text file containing the git-sha the lease should use
                  (job.sh reads this; avoids hand-editing the SDL's env
                  every time a script changes)

Never prints the token. Reads it from HF_TOKEN only, same convention as
scripts/docker/upload_models_hf.py and the earlier mirror_adapters_hf.py.

Usage (from the vLLM worktree, .gguf_venv python which already has
huggingface_hub via sentence-transformers):
    HF_TOKEN=hf_xxx .gguf_venv/Scripts/python.exe scripts/lease/publish_kit.py \\
        --repo Oussamamaat/iblog-vllm-lease
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

KIT_FILES = [
    "scripts/vllm/build_merged_awq.py",
    "scripts/vllm/serve_pair.sh",
    "scripts/vllm/parity_probe.py",
    "scripts/vllm/quality_sample.py",
    "scripts/vllm/requirements-build.txt",
    "scripts/lease/job.sh",
    "scripts/benchmark/bench_concurrency.py",
]

FIXTURE_FILES = [
    "scripts/vllm/fixtures/bench_prompts.json",
    "scripts/vllm/fixtures/parity_prompts.json",
    "scripts/vllm/fixtures/quality_prompts.json",
    "scripts/vllm/fixtures/calib_darija.jsonl",
    "scripts/vllm/fixtures/calib_fr.jsonl",
]


def _git_sha() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _git_dirty() -> bool:
    out = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True)
    return bool(out.stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true",
                     help="publish even with uncommitted changes (they still get "
                          "uploaded -- this just controls whether the script refuses "
                          "by default, since 'sha' would then not describe what's on HF)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN not set.")

    sha = _git_sha()
    if _git_dirty():
        if not args.allow_dirty:
            sys.exit(
                f"ERROR: uncommitted changes present. The kit would be tagged "
                f"'{sha}' but not match what that commit actually contains. "
                f"Commit first, or pass --allow-dirty to publish anyway."
            )
        print("WARNING: publishing with uncommitted changes (--allow-dirty).")

    missing = [f for f in KIT_FILES + FIXTURE_FILES if not (REPO_ROOT / f).exists()]
    if missing:
        sys.exit(
            "ERROR: missing files -- run scripts/vllm/make_fixtures.py first if "
            f"these are fixtures:\n  " + "\n  ".join(missing)
        )

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"Creating repo {args.repo} (private={not args.public}) if needed ...")
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=not args.public, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for rel in KIT_FILES:
            dest = tmp_path / "scripts_and_fixtures" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((REPO_ROOT / rel).read_bytes())
        for rel in FIXTURE_FILES:
            dest = tmp_path / "scripts_and_fixtures" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((REPO_ROOT / rel).read_bytes())

        print(f"Uploading kit -> {args.repo}/kit/{sha}/ ...")
        api.upload_folder(
            folder_path=str(tmp_path / "scripts_and_fixtures"),
            path_in_repo=f"kit/{sha}",
            repo_id=args.repo, repo_type="dataset",
        )

        latest_path = tmp_path / "latest.txt"
        latest_path.write_text(sha, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(latest_path), path_in_repo="kit/latest.txt",
            repo_id=args.repo, repo_type="dataset",
        )

    print("\n" + "=" * 70)
    print(f"Published. Repo: https://huggingface.co/datasets/{args.repo}")
    print(f"Kit sha: {sha}  (kit/latest.txt now points here)")
    print("The lease SDL's KIT_REPO env should be set to this repo id;")
    print("job.sh reads kit/latest.txt at boot to know which sha to fetch.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
