"""
Concurrency sweep -- ADR 0008's deferred Phase B measurement, extended
(plan rev 2, Phase A6) to drive three backends with the SAME prompts, so
Ollama-vs-vLLM columns are actually comparable (rev 1's laptop "unset"
run and the RTX 5090 "parallel4" run were NOT -- different GPUs and
different prompt shapes; see the plan's "What rev 1 got wrong" table).

    # Ollama, raw text completion (matches production's real transport --
    # _call_ollama_generate sends already-rendered text with raw:true, so
    # this script must too, or it would double-template):
    python scripts/benchmark/bench_concurrency.py --backend ollama \\
        --base-url http://127.0.0.1:11434 --model IBLOG_TUTOR:latest \\
        --prompts scripts/vllm/fixtures/bench_prompts.json --language darija \\
        --num-parallel-note unset --out benchmark_concurrency_ollama_unset.json

    # vLLM, OpenAI-compatible /v1/completions:
    python scripts/benchmark/bench_concurrency.py --backend vllm \\
        --base-url http://127.0.0.1:8101 --model iblog-tutor-darija-awq \\
        --prompts scripts/vllm/fixtures/bench_prompts.json --language darija \\
        --out benchmark_concurrency_vllm.json

    # Full pipeline (retrieval + grounding + generation) through this app's
    # own /api/v1/chat/ -- used for Phase D5's staging verification, not
    # Phase B (which measures the serving layer in isolation on purpose).
    python scripts/benchmark/bench_concurrency.py --backend app \\
        --base-url http://127.0.0.1:8000 \\
        --prompts scripts/vllm/fixtures/bench_prompts.json \\
        --out benchmark_concurrency_app.json

Sweeps N = 1, 2, 4, 8, 16, 32 concurrent requests, reporting per N:
p50/p95/p99 latency, TTFT, throughput (req/s, tok/s), error rate, and peak
VRAM (1 Hz nvidia-smi sample). Stdlib only (urllib, threading, subprocess).

Per-request rows are kept in the output JSON (not just percentiles) so a
later script (e.g. a token-parity check) can re-derive anything from the
raw data without re-running the sweep.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# Windows' console defaults to cp1252, which can't encode the box-drawing
# headers or the Arabic question text this script prints. UTF-8 stdout is
# safe everywhere else this runs (Linux/the lease default to it already).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Fallback prompts when --prompts is omitted (a quick smoke run with no
# fixtures file, e.g. against a tiny stand-in model where real retrieval
# context would be meaningless anyway). Real runs pass
# scripts/vllm/fixtures/bench_prompts.json (plan Phase A5).
_FALLBACK_PROMPTS = [
    {"language": "darija", "prompt": "شنو هي معدات الحماية الشخصية الإجبارية فورشة الخدمة؟"},
    {"language": "fr", "prompt": "Quels sont les équipements de protection individuelle obligatoires ?"},
]

_VLLM_STOP = ["<end_of_turn>", "<start_of_turn>"]
# vLLM matches stop strings after stripping special tokens, so they never fire; stop on the ids.
_VLLM_STOP_TOKEN_IDS = [106, 107]  # Gemma-2 <start_of_turn>, <end_of_turn>


def _load_prompts(path: str, language: str) -> list[dict]:
    if path is None:
        rows = _FALLBACK_PROMPTS
    else:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    if language != "both":
        rows = [r for r in rows if r["language"] == language]
    if not rows:
        sys.exit(f"ERROR: no prompts left after filtering to language={language!r} (from {path!r}).")
    return rows


def _one_request_ollama(base_url: str, model: str, prompt: str, max_tokens: int,
                         temperature: float, timeout: int) -> dict:
    """POST /api/generate with raw:true -- the prompt is already a fully
    rendered ChatML string (render_conversation output), so raw:true
    bypasses Ollama's own Modelfile TEMPLATE instead of double-templating
    it. Mirrors app.services.llm._call_ollama_generate's real transport,
    not a simplified approximation of it."""
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "raw": True,
        "stream": True,
        "keep_alive": "30m",
        "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 8192},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    prompt_eval_count = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("response"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    tokens += 1
                if obj.get("done"):
                    prompt_eval_count = obj.get("prompt_eval_count")
                    return {"ok": True, "total_s": time.perf_counter() - t0,
                            "ttft_s": ttft, "tokens": tokens,
                            "prompt_eval_count": prompt_eval_count}
        return {"ok": False, "error": "stream ended without a done:true message",
                "total_s": time.perf_counter() - t0}
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read()[:200]!r}",
                "total_s": time.perf_counter() - t0}
    except Exception as e:
        return {"ok": False, "error": str(e), "total_s": time.perf_counter() - t0}


def _one_request_vllm(base_url: str, model: str, prompt: str, max_tokens: int,
                       temperature: float, timeout: int) -> dict:
    """POST /v1/completions with stream_options.include_usage -- the final
    SSE chunk before [DONE] carries usage.prompt_tokens (the vLLM analogue
    of Ollama's prompt_eval_count, for the parity check) without a second
    non-streaming request."""
    url = base_url.rstrip("/") + "/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": _VLLM_STOP,
        "stop_token_ids": _VLLM_STOP_TOKEN_IDS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    prompt_tokens = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw_line in r:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices and choices[0].get("text"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    tokens += 1
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
            return {"ok": True, "total_s": time.perf_counter() - t0,
                    "ttft_s": ttft, "tokens": tokens, "prompt_tokens": prompt_tokens}
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read()[:200]!r}",
                "total_s": time.perf_counter() - t0}
    except Exception as e:
        return {"ok": False, "error": str(e), "total_s": time.perf_counter() - t0}


def _one_request_app(base_url: str, prompt_row: dict, timeout: int) -> dict:
    """POST /api/v1/chat/ -- the FULL pipeline (retrieval + grounding +
    generation + citation injection), not the raw serving layer. Used for
    Phase D5's staging verification, where the question is "does the
    deployed app work end to end on vLLM", not "how fast is the serving
    layer alone" (that's the ollama/vllm backends' job -- mixing the two
    would make Phase B's numbers include retrieval latency that
    OLLAMA_NUM_PARALLEL/vLLM's batching has no effect on).

    prompt_row is a scripts/vllm/fixtures/bench_prompts.json entry:
    {"domain", "language", "question", ...} -- this backend re-asks the
    ORIGINAL question through the real endpoint (which does its own
    retrieval) rather than replaying the already-rendered "prompt" field,
    since /api/v1/chat/ takes a message, not a raw prompt.
    """
    url = base_url.rstrip("/") + "/api/v1/chat/"
    lang = "fr" if prompt_row["language"] == "fr" else "darija"
    payload = {
        "message": prompt_row.get("question") or prompt_row.get("prompt", "")[:500],
        "domain": prompt_row.get("domain"),
        "language": lang,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8"))
        return {"ok": True, "total_s": time.perf_counter() - t0,
                "ttft_s": None, "tokens": len(body.get("answer", "").split())}
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read()[:200]!r}",
                "total_s": time.perf_counter() - t0}
    except Exception as e:
        return {"ok": False, "error": str(e), "total_s": time.perf_counter() - t0}


def one_request(backend: str, base_url: str, model: str, prompt_row: dict,
                 max_tokens: int, temperature: float, timeout: int) -> dict:
    if backend == "ollama":
        return _one_request_ollama(base_url, model, prompt_row["prompt"], max_tokens, temperature, timeout)
    if backend == "vllm":
        return _one_request_vllm(base_url, model, prompt_row["prompt"], max_tokens, temperature, timeout)
    if backend == "app":
        return _one_request_app(base_url, prompt_row, timeout)
    raise ValueError(f"Unknown backend: {backend!r}")


def _vram_sampler(stop_event: threading.Event, samples: list) -> None:
    while not stop_event.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2)
            if out.returncode == 0 and out.stdout.strip():
                samples.append(int(out.stdout.strip().splitlines()[0]))
        except Exception:
            pass
        stop_event.wait(1.0)


def _percentile(sorted_data: list, p: float):
    if not sorted_data:
        return None
    k = (len(sorted_data) - 1) * p
    f, c = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return round(sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f), 2)


def run_one_n(backend: str, base_url: str, model: str, prompts: list[dict], n: int,
              requests_per_worker: int, max_tokens: int, temperature: float, timeout: int) -> dict:
    print(f"\n── N={n} concurrent ({backend}) ──")
    stop_event = threading.Event()
    vram_samples: list = []
    sampler = threading.Thread(target=_vram_sampler, args=(stop_event, vram_samples), daemon=True)
    sampler.start()

    jobs = [prompts[i % len(prompts)] for i in range(n * requests_per_worker)]
    results_list: list = []
    lock = threading.Lock()
    sem = threading.Semaphore(n)  # caps in-flight requests at exactly N

    def bounded_worker(prompt_row: dict) -> None:
        with sem:
            r = one_request(backend, base_url, model, prompt_row, max_tokens, temperature, timeout)
        with lock:
            results_list.append(r)

    t_start = time.perf_counter()
    threads = [threading.Thread(target=bounded_worker, args=(p,)) for p in jobs]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t_start

    stop_event.set()
    sampler.join(timeout=2)

    oks = [r for r in results_list if r.get("ok")]
    errs = [r for r in results_list if not r.get("ok")]
    latencies = sorted(r["total_s"] for r in oks)
    ttfts = sorted(r["ttft_s"] for r in oks if r.get("ttft_s") is not None)
    total_tokens = sum(r.get("tokens", 0) for r in oks)

    summary = {
        "n": n,
        "requests": len(jobs),
        "errors": len(errs),
        "error_rate": round(len(errs) / len(jobs), 3) if jobs else None,
        "wall_seconds": round(wall, 2),
        "throughput_req_s": round(len(oks) / wall, 3) if wall else None,
        "throughput_tok_s": round(total_tokens / wall, 1) if wall else None,
        "latency_p50_s": _percentile(latencies, 0.50),
        "latency_p95_s": _percentile(latencies, 0.95),
        "latency_p99_s": _percentile(latencies, 0.99),
        "ttft_p50_s": _percentile(ttfts, 0.50),
        "ttft_p95_s": _percentile(ttfts, 0.95),
        "peak_vram_mib": max(vram_samples) if vram_samples else None,
        "errors_sample": [e.get("error") for e in errs[:3]],
        "requests_raw": results_list,
    }
    print(f"  requests={summary['requests']}  errors={summary['errors']}  "
          f"p50={summary['latency_p50_s']}s  p95={summary['latency_p95_s']}s  "
          f"ttft_p50={summary['ttft_p50_s']}s  "
          f"throughput={summary['throughput_req_s']}req/s  "
          f"peak_vram={summary['peak_vram_mib']}MiB")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["ollama", "vllm", "app"], default="ollama")
    ap.add_argument("--base-url", default=None,
                     help="default: http://127.0.0.1:11434 (ollama) / :8101 (vllm) / :8000 (app)")
    ap.add_argument("--ollama-url", dest="base_url",
                     help="deprecated alias for --base-url, kept for the existing "
                          "benchmark_concurrency_unset.json invocation")
    ap.add_argument("--model", default="IBLOG_TUTOR:latest",
                     help="served model name (ollama/vllm only -- ignored for --backend app, "
                          "which picks the model itself from each prompt's language)")
    ap.add_argument("--prompts", default=None,
                     help="path to a fixtures JSON file, e.g. scripts/vllm/fixtures/bench_prompts.json "
                          "(plan Phase A5). Omit for a 2-question smoke fallback.")
    ap.add_argument("--language", choices=["darija", "fr", "both"], default="both")
    ap.add_argument("--n-values", default="1,2,4,8,16,32")
    ap.add_argument("--requests-per-worker", type=int, default=2,
                     help="requests each of the N concurrent slots sends before the sweep "
                          "step ends (cycles through --prompts in order)")
    ap.add_argument("--max-tokens", type=int, default=300,
                     help="num_predict/max_tokens cap -- keeps sweep runtime bounded; this "
                          "measures queueing/throughput, not answer completeness")
    ap.add_argument("--temperature", type=float, default=0.0,
                     help="0.0 by default (plan rev 2 assumption 11): bench/parity/quality runs "
                          "use greedy decoding for reproducibility; the live app itself uses 0.2")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default="benchmark_concurrency.json")
    ap.add_argument("--num-parallel-note", default="n/a",
                     help="label only -- records what OLLAMA_NUM_PARALLEL was set to for this "
                          "run (ollama backend) or the vLLM launch flags used, in the output JSON. "
                          "This script does NOT set it.")
    args = ap.parse_args()

    if args.base_url is None:
        args.base_url = {"ollama": "http://127.0.0.1:11434",
                          "vllm": "http://127.0.0.1:8101",
                          "app": "http://127.0.0.1:8000"}[args.backend]

    n_values = [int(x) for x in args.n_values.split(",")]
    prompts = _load_prompts(args.prompts, args.language)

    print(f"Concurrency sweep -- backend={args.backend}  model={args.model}  "
          f"base_url={args.base_url}  prompts={len(prompts)} "
          f"({args.language})  note={args.num_parallel_note}")
    results = {}
    for n in n_values:
        results[str(n)] = run_one_n(args.backend, args.base_url, args.model, prompts, n,
                                     args.requests_per_worker, args.max_tokens,
                                     args.temperature, args.timeout)

    report = {
        "backend": args.backend,
        "base_url": args.base_url,
        "model": args.model,
        "language": args.language,
        "num_parallel_note": args.num_parallel_note,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
