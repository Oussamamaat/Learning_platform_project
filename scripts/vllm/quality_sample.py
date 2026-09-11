"""
Generate side-by-side answers (Ollama Q4_K_M vs vLLM AWQ) for the
green_light_model.md quality cases, for a human to read and grade against
that checklist. Plan rev 2, Phase A8 / B4/B5's quality-output step /
Verification criterion 4.

This script does NOT grade anything itself -- ADR 0008's own standard
("inventing a number and presenting it as measured would misrepresent the
evidence") applies just as much to quality as to latency, and
green_light_model.md's checklist (§2 behavioral items, §3 red flags) is
written for a human reader, not a regex. What this script automates is
production: pulling scripts/vllm/fixtures/quality_prompts.json's cases
(each already tagged with the checklist item it targets, e.g.
"D2/RF9 insufficient-context refusal"), generating BOTH backends' answers
to the SAME question with the SAME retrieved context, and writing them
side by side so Phase C's review is "read N transcripts", not "also go
build the transcripts by hand under time pressure".

Stdlib only (urllib), no Postgres/app import -- the retrieved `context`
already lives in each fixture row (built once by make_fixtures.py against
the real corpus), so this script only needs the two serving endpoints.

Usage:
    python scripts/vllm/quality_sample.py \\
        --prompts scripts/vllm/fixtures/quality_prompts.json \\
        --ollama-url http://127.0.0.1:11434 \\
        --ollama-model-darija IBLOG_TUTOR:latest --ollama-model-fr iblog-tutor-fr:latest \\
        --vllm-url-darija http://127.0.0.1:8101 --vllm-url-fr http://127.0.0.1:8102 \\
        --vllm-model-darija iblog-tutor-darija-awq --vllm-model-fr iblog-tutor-fr-awq \\
        --out quality_transcripts.md
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

_VLLM_STOP = ["<end_of_turn>", "<start_of_turn>"]
# vLLM matches stop strings after stripping special tokens, so they never fire; stop on the ids.
_VLLM_STOP_TOKEN_IDS = [106, 107]  # Gemma-2 <start_of_turn>, <end_of_turn>


def ollama_generate(base_url: str, model: str, rendered: str, max_tokens: int, timeout: int) -> str:
    payload = {
        "model": model, "prompt": rendered, "raw": True, "stream": False,
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "").strip()


def vllm_generate(base_url: str, model: str, rendered: str, max_tokens: int, timeout: int) -> str:
    payload = {
        "model": model, "prompt": rendered, "temperature": 0.0, "max_tokens": max_tokens,
        "stop": _VLLM_STOP, "stop_token_ids": _VLLM_STOP_TOKEN_IDS, "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    choices = body.get("choices") or []
    return (choices[0].get("text", "") if choices else "").strip()


def _cell(row: dict, side: str) -> str:
    if f"{side}_answer" in row:
        return row[f"{side}_answer"]
    if f"{side}_error" in row:
        return "ERROR: " + str(row[f"{side}_error"])
    return "(not run)"


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
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--backends", choices=["both", "ollama", "vllm"], default="both",
                     help="run one side only -- Ollama and vLLM must not share the GPU")
    ap.add_argument("--ollama-answers", default=None,
                     help="--out-json of an earlier --backends ollama run, used for the Ollama column")
    ap.add_argument("--out", default="quality_transcripts.md")
    ap.add_argument("--out-json", default=None,
                     help="optional: also write the raw rows as JSON (for a scripted re-check later)")
    args = ap.parse_args()

    with open(args.prompts, "r", encoding="utf-8") as f:
        cases = json.load(f)

    prior = {}
    if args.ollama_answers:
        with open(args.ollama_answers, "r", encoding="utf-8") as f:
            prior = {(r["tag"], r["language"]): r for r in json.load(f)}

    md_lines = ["# Quality transcripts -- Ollama Q4_K_M vs vLLM AWQ\n",
                "Generated by scripts/vllm/quality_sample.py. Grade each pair against "
                "green_light_model.md §2 (behavioral checklist) / §3 (red flags) -- "
                "this file does not self-grade.\n"]
    json_rows = []

    for case in cases:
        tag = case["tag"]
        language = case["language"]
        question = case["question"]
        prompt = case["prompt"]
        ollama_model = args.ollama_model_fr if language == "fr" else args.ollama_model_darija
        vllm_url = args.vllm_url_fr if language == "fr" else args.vllm_url_darija
        vllm_model = args.vllm_model_fr if language == "fr" else args.vllm_model_darija

        print(f"\n=== {tag} ({language}): {question[:60]} ===")
        row = {"tag": tag, "language": language, "question": question, "has_context": case.get("has_context")}

        if args.backends in ("both", "ollama"):
            try:
                ollama_answer = ollama_generate(args.ollama_url, ollama_model, prompt, args.max_tokens, args.timeout)
                row["ollama_answer"] = ollama_answer
                print(f"  Ollama: {ollama_answer[:150]}")
            except Exception as e:
                row["ollama_error"] = str(e)
                print(f"  Ollama FAILED: {e}")
        else:
            row.update({k: v for k, v in prior.get((tag, language), {}).items()
                        if k in ("ollama_answer", "ollama_error")})

        if args.backends in ("both", "vllm"):
            try:
                vllm_answer = vllm_generate(vllm_url, vllm_model, prompt, args.max_tokens, args.timeout)
                row["vllm_answer"] = vllm_answer
                print(f"  vLLM:   {vllm_answer[:150]}")
            except Exception as e:
                row["vllm_error"] = str(e)
                print(f"  vLLM FAILED: {e}")

        json_rows.append(row)
        md_lines.append(f"## {tag} ({language})\n")
        md_lines.append(f"**Question:** {question}\n")
        md_lines.append(f"**Context present in retrieval:** {case.get('has_context')}\n")
        md_lines.append("**Ollama (Q4_K_M):**\n")
        md_lines.append(f"> {_cell(row, 'ollama')}\n")
        md_lines.append("**vLLM (AWQ-INT4):**\n")
        md_lines.append(f"> {_cell(row, 'vllm')}\n")
        md_lines.append("---\n")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"\nWrote {args.out}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(json_rows, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
