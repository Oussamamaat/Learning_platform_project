"""
Ollama concurrency sweep — ADR 0008's deferred Phase B measurement.
    python scripts/benchmark/bench_concurrency.py --ollama-url http://127.0.0.1:11434 \
        --num-parallel-note unset --out benchmark_concurrency_unset.json

Answers the question this migration actually turns on: with OLLAMA_NUM_PARALLEL
unset (today's real deployment — Ollama defaults this to 1), does raising it
alone fix concurrent-load latency well enough that vLLM isn't needed for
concurrency? Per this repo's standard (ADR 0008: prefer the measured cheap
fix), this script's job is to produce that measurement, not assume the answer.

Sweeps N = 1, 2, 4, 8, 16, 32 concurrent /api/generate requests directly
against Ollama (not the full /api/v1/chat/ pipeline — that adds retrieval +
grounding, which OLLAMA_NUM_PARALLEL does not affect, so hitting Ollama
directly isolates the thing being measured), fixed question set in both
languages, reporting per N: p50/p95/p99 latency, TTFT, throughput
(req/s, tok/s), error rate, and peak VRAM (1 Hz nvidia-smi sample, same
pattern as bench_all.sh).

Stdlib only (urllib, threading, subprocess) — no extra deps, mirrors
bench_llm.py's shape so bench_all.sh could call this too.

VRAM NOTE — this laptop has 8 GB VRAM; one resident 9B Q4_K_M model is
~5.8 GB. Only the OLLAMA_NUM_PARALLEL-unset run (today's real deployment,
which serializes regardless of N) is meaningful here — that's the default.
The NUM_PARALLEL=4/8 comparison run needs multiple concurrent model copies
resident and must happen on the Akash rtx5090 lease alongside Step 0's
builds (see the plan's Step 1) — this script is unchanged there, just point
--ollama-url at the lease and set OLLAMA_NUM_PARALLEL on its Ollama process
before starting the sweep, then pass --num-parallel-note accordingly.
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

DARIJA_Q = "شنو هي معدات الحماية الشخصية الإجبارية فورشة الخدمة؟"
FRENCH_Q = "Quels sont les équipements de protection individuelle obligatoires ?"
QUESTIONS = [("darija", DARIJA_Q), ("french", FRENCH_Q)]


def _one_request(ollama_url: str, model: str, question: str, max_tokens: int,
                  timeout: int) -> dict:
    """Streaming /api/generate call. Returns latency/TTFT/token-count, or an error."""
    url = ollama_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": question,
        "stream": True,
        "keep_alive": "30m",
        "options": {"temperature": 0.2, "num_predict": max_tokens},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
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
                    return {"ok": True, "total_s": time.perf_counter() - t0,
                            "ttft_s": ttft, "tokens": tokens}
        return {"ok": False, "error": "stream ended without a done:true message",
                "total_s": time.perf_counter() - t0}
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read()[:200]!r}",
                "total_s": time.perf_counter() - t0}
    except Exception as e:
        return {"ok": False, "error": str(e), "total_s": time.perf_counter() - t0}


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


def run_one_n(ollama_url: str, model: str, n: int, requests_per_worker: int,
              max_tokens: int, timeout: int) -> dict:
    print(f"\n── N={n} concurrent ──")
    stop_event = threading.Event()
    vram_samples: list = []
    sampler = threading.Thread(target=_vram_sampler, args=(stop_event, vram_samples), daemon=True)
    sampler.start()

    jobs = [QUESTIONS[i % 2][1] for i in range(n * requests_per_worker)]
    results_list: list = []
    lock = threading.Lock()
    sem = threading.Semaphore(n)  # caps in-flight requests at exactly N

    def bounded_worker(question: str) -> None:
        with sem:
            r = _one_request(ollama_url, model, question, max_tokens, timeout)
        with lock:
            results_list.append(r)

    t_start = time.perf_counter()
    threads = [threading.Thread(target=bounded_worker, args=(q,)) for q in jobs]
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
    }
    print(f"  requests={summary['requests']}  errors={summary['errors']}  "
          f"p50={summary['latency_p50_s']}s  p95={summary['latency_p95_s']}s  "
          f"throughput={summary['throughput_req_s']}req/s  "
          f"peak_vram={summary['peak_vram_mib']}MiB")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="IBLOG_TUTOR:latest")
    ap.add_argument("--n-values", default="1,2,4,8,16,32")
    ap.add_argument("--requests-per-worker", type=int, default=2,
                     help="requests each of the N concurrent slots sends before the sweep "
                          "step ends (alternates darija/french)")
    ap.add_argument("--max-tokens", type=int, default=200,
                     help="num_predict cap — keeps sweep runtime bounded; this measures "
                          "queueing/throughput, not answer completeness")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default="benchmark_concurrency.json")
    ap.add_argument("--num-parallel-note", default="unset",
                     help="label only — records what OLLAMA_NUM_PARALLEL was set to for this "
                          "run in the output JSON. This script does NOT set it; that's an env "
                          "var on the Ollama server process itself, set before it starts.")
    args = ap.parse_args()

    n_values = [int(x) for x in args.n_values.split(",")]

    print(f"Ollama concurrency sweep — model={args.model}  "
          f"OLLAMA_NUM_PARALLEL={args.num_parallel_note}  url={args.ollama_url}")
    results = {}
    for n in n_values:
        results[str(n)] = run_one_n(args.ollama_url, args.model, n,
                                     args.requests_per_worker, args.max_tokens, args.timeout)

    report = {
        "ollama_url": args.ollama_url,
        "model": args.model,
        "num_parallel_note": args.num_parallel_note,
        "max_tokens": args.max_tokens,
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
