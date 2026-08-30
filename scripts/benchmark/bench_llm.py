"""
LLM latency + throughput benchmark.
    python scripts/benchmark/bench_llm.py --base-url http://<lease-host>:<port>

Measures the two things the 5090 is supposed to fix (baselines from
docs/architecture/cloud-scaling-plan.md §4, measured on the 4060 8 GB laptop):

  1. Per-turn latency for Darija and French via the production chat path
     (POST /api/v1/chat/ — includes retrieval + grounding + generation).
     Laptop: Darija 17–24 s, French 44–144 s (one 180 s timeout). Target: 2–4 s.

  2. Language-switch cost. On 8 GB the tutor is evicted and reloaded (5.7 GB)
     on every French↔Darija flip; on 32 GB both stay resident → ~0. This
     alternates languages and reports whether a switch costs extra.

Also hits Ollama's /api/generate directly (--ollama-url) for raw streaming
tokens/sec and time-to-first-token, which the non-streaming chat endpoint
can't expose.

Stdlib only (urllib) — no extra deps. Writes JSON to --out.
"""
import argparse
import json
import time
import urllib.request
import urllib.error
from statistics import mean

# (label, language, prompt) — language values are the API's Language enum
# (fr / ar-MA), per app/models/schemas.py ChatRequest.
DARIJA_Q = "شنو هي معدات الحماية الشخصية الإجبارية فورشة الخدمة؟"
FRENCH_Q = "Quels sont les équipements de protection individuelle obligatoires ?"


def _post_json(url: str, payload: dict, timeout: int) -> tuple[float, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    return time.perf_counter() - t0, body


def bench_chat(base_url: str, timeout: int) -> dict:
    url = base_url.rstrip("/") + "/api/v1/chat/"
    results = {"darija": [], "french": []}
    print("── chat turns (production path: retrieval + grounding + LLM) ──")
    # Warm each language once (not counted), then time 3 each.
    for lang_key, lang_api, q in (("darija", "ar-MA", DARIJA_Q), ("french", "fr", FRENCH_Q)):
        try:
            _post_json(url, {"message": q, "language": lang_api, "domain": "industrial"}, timeout)
        except Exception as e:
            print(f"  [{lang_key}] warmup failed: {e}")
        for i in range(3):
            try:
                dt, body = _post_json(url, {"message": q, "language": lang_api,
                                            "domain": "industrial"}, timeout)
                chars = len(body.get("response", ""))
                results[lang_key].append({"seconds": round(dt, 2), "chars": chars,
                                          "resolved_lang": body.get("language")})
                print(f"  [{lang_key}] turn {i+1}: {dt:5.2f}s  ({chars} chars)")
            except urllib.error.HTTPError as e:
                print(f"  [{lang_key}] turn {i+1}: HTTP {e.code} — {e.read()[:200]!r}")
            except Exception as e:
                print(f"  [{lang_key}] turn {i+1}: {e}")
    for k in ("darija", "french"):
        secs = [r["seconds"] for r in results[k]]
        if secs:
            print(f"  {k} mean: {mean(secs):.2f}s")
    return results


def bench_switch(base_url: str, timeout: int) -> dict:
    """Alternate languages so every turn is a switch; compare to same-language
    turns. Near-equal ⇒ both models resident (the 32 GB win)."""
    url = base_url.rstrip("/") + "/api/v1/chat/"
    print("\n── language-switch cost (alternating fr ↔ darija) ──")
    seq = [("ar-MA", DARIJA_Q), ("fr", FRENCH_Q)] * 3
    switch_times = []
    for i, (lang_api, q) in enumerate(seq):
        try:
            dt, _ = _post_json(url, {"message": q, "language": lang_api, "domain": "industrial"},
                               timeout)
            switch_times.append(dt)
            print(f"  switch turn {i+1} ({lang_api}): {dt:5.2f}s")
        except Exception as e:
            print(f"  switch turn {i+1} ({lang_api}): {e}")
    return {"alternating_seconds": [round(t, 2) for t in switch_times],
            "mean": round(mean(switch_times), 2) if switch_times else None}


def bench_ollama_raw(ollama_url: str, model: str, timeout: int) -> dict:
    """Raw streaming tokens/sec + TTFT straight from Ollama."""
    url = ollama_url.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": "Explain workplace safety in one paragraph.",
               "stream": True, "keep_alive": -1}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    print(f"\n── raw Ollama generation ({model}) ──")
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:
                obj = json.loads(line)
                if obj.get("response"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    tokens += 1
                if obj.get("done"):
                    total = time.perf_counter() - t0
                    tok_s = tokens / total if total else 0
                    print(f"  TTFT: {ttft:.2f}s   tokens: {tokens}   {tok_s:.1f} tok/s")
                    return {"ttft_s": round(ttft or 0, 2), "tokens": tokens,
                            "total_s": round(total, 2), "tokens_per_s": round(tok_s, 1)}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"error": str(e)}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000", help="FastAPI base URL")
    ap.add_argument("--ollama-url", default=None,
                    help="Ollama base URL for raw token-rate (default: skip)")
    ap.add_argument("--model", default="IBLOG_TUTOR:latest")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default="benchmark_llm.json")
    args = ap.parse_args()

    report = {
        "base_url": args.base_url,
        "chat": bench_chat(args.base_url, args.timeout),
        "language_switch": bench_switch(args.base_url, args.timeout),
    }
    if args.ollama_url:
        report["ollama_raw"] = bench_ollama_raw(args.ollama_url, args.model, args.timeout)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
