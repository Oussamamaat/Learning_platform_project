"""
End-to-end checks for the vLLM staging lease (deploy/akash-staging-vllm.yaml).

Runs from a laptop against the staging app's public address and exercises
everything the migration changed: grounded chat in both languages, an
off-topic refusal, a quiz with the exact requested question count, a diagram,
and a burst of concurrent chat requests. Seed the corpus first
(ingest_directory('raw/shared', tenant_id='company_abc') in the app container),
otherwise the grounded checks fail for the wrong reason.

Stdlib only. Request bodies are JSON-encoded here, never passed through a shell
argument, so the Arabic survives intact.

Usage:
    python scripts/vllm/staging_checks.py --base-url http://<ingress> --out staging_checks.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TENANT = "company_abc"
_RUNAWAY_TURN = re.compile(r"^\s*(user|model)\s*$", re.MULTILINE)
# Akash ingress sits behind Cloudflare, which answers the default
# "Python-urllib/3.x" agent with 403 (error 1010) while letting curl through.
_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) iblog-staging-checks/1.0",
}


def call(base_url: str, method: str, path: str, body: dict | None, timeout: int) -> tuple[int, dict | str, float]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(base_url.rstrip("/") + path, data=data, method=method,
                                 headers=_HEADERS)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, raw = resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}", time.perf_counter() - t0
    try:
        return status, json.loads(raw), time.perf_counter() - t0
    except ValueError:
        return status, raw, time.perf_counter() - t0


def check_grounded_chat(base_url, timeout, message, language, expect_lang):
    status, body, secs = call(base_url, "POST", "/api/v1/chat/",
                              {"message": message, "language": language, "tenant_id": TENANT}, timeout)
    problems = []
    if status != 200 or not isinstance(body, dict):
        return False, [f"HTTP {status}"], body, secs
    text = body.get("response", "")
    if not body.get("sources"):
        problems.append("no sources (corpus not seeded, or retrieval found nothing)")
    if body.get("language") != expect_lang:
        problems.append(f"language={body.get('language')!r}, expected {expect_lang!r}")
    if not text.strip():
        problems.append("empty response")
    if "283" not in text:
        problems.append("does not mention article 283")
    if _RUNAWAY_TURN.search(text):
        problems.append("contains an invented user/model turn (stop tokens not applied)")
    if len(text) > 2000:
        problems.append(f"response is {len(text)} chars (runaway?)")
    return not problems, problems, body, secs


def check_refusal(base_url, timeout):
    status, body, secs = call(base_url, "POST", "/api/v1/chat/",
                              {"message": "Quelle est la meilleure recette de tajine au poulet ?",
                               "language": "fr", "tenant_id": TENANT}, timeout)
    if status != 200 or not isinstance(body, dict):
        return False, [f"HTTP {status}"], body, secs
    problems = [] if not body.get("sources") else ["off-topic answer cites sources"]
    return not problems, problems, body, secs


def check_quiz(base_url, timeout, n=5):
    status, body, secs = call(base_url, "POST", "/api/v1/quiz/",
                              {"topic": "équipements de protection individuelle", "num_questions": n,
                               "language": "fr", "tenant_id": TENANT}, timeout)
    if status != 200 or not isinstance(body, dict):
        return False, [f"HTTP {status}"], body, secs
    problems = []
    questions = body.get("questions") or []
    if body.get("total_questions") != n or len(questions) != n:
        problems.append(f"got {len(questions)} questions (total_questions={body.get('total_questions')}), asked for {n}")
    for i, q in enumerate(questions):
        opts = q.get("options") or []
        if not 2 <= len(opts) <= 6:
            problems.append(f"question {i}: {len(opts)} options")
        if not 0 <= q.get("correct_index", -1) < len(opts):
            problems.append(f"question {i}: correct_index {q.get('correct_index')} out of range")
    return not problems, problems, body, secs


def check_diagram(base_url, timeout):
    status, body, secs = call(base_url, "POST", "/api/v1/chat/",
                              {"message": "Dessine un schéma des étapes pour choisir les équipements de protection individuelle",
                               "language": "fr", "tenant_id": TENANT}, timeout)
    if status != 200 or not isinstance(body, dict):
        return False, [f"HTTP {status}"], body, secs
    diagram = body.get("diagram")
    problems = []
    if not diagram:
        problems.append("no diagram in the response (fell back to plain chat)")
    elif diagram.get("kind") != "candlestick" and not (diagram.get("mermaid") or "").strip():
        problems.append(f"diagram kind={diagram.get('kind')!r} has no mermaid source")
    return not problems, problems, body, secs


def check_concurrency(base_url, timeout, n=8):
    msg = "Quelles sont les obligations de l'employeur concernant les équipements de protection individuelle ?"
    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(lambda _: call(base_url, "POST", "/api/v1/chat/",
                                               {"message": msg, "language": "fr", "tenant_id": TENANT}, timeout),
                                range(n)))
    codes = [r[0] for r in results]
    secs = [round(r[2], 1) for r in results]
    problems = [] if all(c == 200 for c in codes) else [f"status codes {codes}"]
    return not problems, problems, {"status_codes": codes, "seconds": secs}, max(secs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default="staging_checks.json")
    args = ap.parse_args()
    base = args.base_url

    status, health, _ = call(base, "GET", "/health", None, 30)
    print(f"/health -> HTTP {status}: {health}")
    if status != 200:
        print("App is not healthy; stopping.")
        return 1

    checks = [
        ("grounded chat, French", lambda: check_grounded_chat(
            base, args.timeout, "Que dit l'article 283 sur les équipements de protection individuelle ?", "fr", "fr")),
        ("grounded chat, Darija", lambda: check_grounded_chat(
            base, args.timeout, "شنو كتقول المادة 283 على معدات الوقاية الشخصية؟", "ar-MA", "darija")),
        ("off-topic refusal, French", lambda: check_refusal(base, args.timeout)),
        ("quiz, 5 questions", lambda: check_quiz(base, args.timeout)),
        ("diagram", lambda: check_diagram(base, args.timeout)),
        ("8 concurrent chats", lambda: check_concurrency(base, args.timeout)),
    ]

    report, failed = [], 0
    for name, fn in checks:
        ok, problems, body, secs = fn()
        failed += 0 if ok else 1
        print(f"\n[{'PASS' if ok else 'FAIL'}] {name} ({secs:.1f}s)")
        for p in problems:
            print(f"   - {p}")
        if isinstance(body, dict) and body.get("response"):
            print(f"   response: {body['response'][:200]!r}")
        report.append({"check": name, "ok": ok, "problems": problems, "seconds": round(secs, 1), "body": body})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"base_url": base, "health": health, "checks": report}, f, indent=2, ensure_ascii=False)
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed. Raw responses: {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
