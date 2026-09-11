"""
Token-count parity: does vLLM tokenize the EXACT same rendered prompt to
the same length Ollama does? Plan rev 2, Phase A8 / Verification V3.

Extends probe_history_parity.py's own method (tashkeel-eval worktree,
Ollama /api/chat vs /api/generate raw:true) one step further: instead of
comparing two OLLAMA transports, this compares OLLAMA vs VLLM on the
SAME pre-rendered text (scripts/vllm/fixtures/parity_prompts.json's
`rendered` field -- built by make_fixtures.py via render_conversation(),
so this script never imports render_conversation or this repo's `app`
package itself; it only sends already-rendered strings).

Gate: prompt token counts must match EXACTLY.
  - Ollama's prompt_eval_count (POST /api/generate, raw:true)
  - vLLM's usage.prompt_tokens (POST /v1/completions)
A mismatch of exactly 1 is the double-BOS bug this migration was designed
around from the start (render_conversation() deliberately emits no
literal <bos>, relying on the tokenizer to add exactly one -- see
app/services/llm.py's render_conversation docstring and this project's
chat-template-bos-inconsistency memory). Any other mismatch is something
new and must be investigated before trusting anything else on the lease.

Stdlib only (urllib), no Postgres/app import -- runs standalone on the
lease as part of job.sh's Phase B2 (Ollama side) + B4 (vLLM side).

Usage:
    python scripts/vllm/parity_probe.py \\
        --prompts scripts/vllm/fixtures/parity_prompts.json \\
        --ollama-url http://127.0.0.1:11434 \\
        --ollama-model-darija IBLOG_TUTOR:latest --ollama-model-fr iblog-tutor-fr:latest \\
        --vllm-url-darija http://127.0.0.1:8101 --vllm-url-fr http://127.0.0.1:8102 \\
        --vllm-model-darija iblog-tutor-darija-awq --vllm-model-fr iblog-tutor-fr-awq \\
        --out parity_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def ollama_prompt_eval_count(base_url: str, model: str, rendered: str, timeout: int) -> int:
    payload = {
        "model": model, "prompt": rendered, "raw": True, "stream": False,
        "options": {"temperature": 0.0, "num_predict": 1},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    count = body.get("prompt_eval_count")
    if count is None:
        raise RuntimeError(f"Ollama response had no prompt_eval_count: {body}")
    return count


def vllm_prompt_tokens(base_url: str, model: str, rendered: str, timeout: int) -> int:
    payload = {
        "model": model, "prompt": rendered, "temperature": 0.0, "max_tokens": 1, "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    usage = body.get("usage") or {}
    count = usage.get("prompt_tokens")
    if count is None:
        raise RuntimeError(f"vLLM response had no usage.prompt_tokens: {body}")
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--ollama-model-darija", default="IBLOG_TUTOR:latest")
    ap.add_argument("--ollama-model-fr", default="iblog-tutor-fr:latest")
    ap.add_argument("--vllm-url-darija", default="http://127.0.0.1:8101")
    ap.add_argument("--vllm-url-fr", default="http://127.0.0.1:8102")
    ap.add_argument("--vllm-model-darija", default="iblog-tutor-darija-awq")
    ap.add_argument("--vllm-model-fr", default="iblog-tutor-fr-awq")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", default="parity_report.json")
    args = ap.parse_args()

    with open(args.prompts, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    rows = []
    all_match = True
    for conv in conversations:
        name = conv["name"]
        language = conv["language"]
        rendered = conv["rendered"]
        ollama_model = args.ollama_model_fr if language == "fr" else args.ollama_model_darija
        vllm_url = args.vllm_url_fr if language == "fr" else args.vllm_url_darija
        vllm_model = args.vllm_model_fr if language == "fr" else args.vllm_model_darija

        print(f"\n=== {name} ({language}) ===")
        row = {"name": name, "language": language}
        try:
            ollama_count = ollama_prompt_eval_count(args.ollama_url, ollama_model, rendered, args.timeout)
            row["ollama_prompt_eval_count"] = ollama_count
            print(f"  Ollama ({ollama_model})  prompt_eval_count = {ollama_count}")
        except Exception as e:
            print(f"  Ollama FAILED: {type(e).__name__}: {e}")
            row["ollama_error"] = str(e)
            all_match = False
            rows.append(row)
            continue

        try:
            vllm_count = vllm_prompt_tokens(vllm_url, vllm_model, rendered, args.timeout)
            row["vllm_prompt_tokens"] = vllm_count
            print(f"  vLLM   ({vllm_model})  usage.prompt_tokens = {vllm_count}")
        except Exception as e:
            print(f"  vLLM FAILED: {type(e).__name__}: {e}")
            row["vllm_error"] = str(e)
            all_match = False
            rows.append(row)
            continue

        match = ollama_count == vllm_count
        row["match"] = match
        all_match = all_match and match
        print(f"  MATCH: {match}")
        if not match:
            diff = vllm_count - ollama_count
            note = " (diff == 1 -> classic double-BOS)" if abs(diff) == 1 else " (unexpected -- investigate)"
            print(f"  DIFF: {diff}{note}")
            row["diff"] = diff
        rows.append(row)

    report = {"all_match": all_match, "rows": rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    if all_match:
        print("VERDICT: token-count parity holds for every conversation. Trust the transport.")
    else:
        print("VERDICT: MISMATCH on at least one conversation -- do NOT trust downstream "
              "measurements (quality/bench) until this is root-caused. See the row detail above.")
    print(f"Wrote {args.out}")
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
