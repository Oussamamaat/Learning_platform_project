"""
Retrieval eval runner (Stage 0). No LLM, no HTTP -- calls the retrieval
function directly and scores against tests/data/retrieval_eval.jsonl.

Usage:
    .gguf_venv/Scripts/python.exe scripts/eval_retrieval.py [--backend disk]

Currently only the "disk" backend (today's app.services.domain_context.
build_domain_context, what app/routers/chat.py actually calls) is wired up.
A "pgvector" backend is added in Stage 3 once app/services/retrieval.py
exists; both will share the same (context, sources) contract so this
runner does not change shape when that lands -- see docs/architecture/
rectified-adjacent plan for why.
"""
import argparse
import json
from typing import Optional
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EVAL_PATH = REPO_ROOT / "tests" / "data" / "retrieval_eval.jsonl"

# Same script-ratio heuristic app.services.llm.detect_query_language and
# app.services.citations.detect_target_script already use -- not a new
# language detector, just applied to a source filename's Arabic marker.
def _is_arabic_source(source_name: str) -> bool:
    return "_ar_" in source_name


def _query_is_arabic_script(query: str) -> bool:
    return sum(1 for c in query if "؀" <= c <= "ۿ") > 0


def load_eval_rows() -> list[dict]:
    rows = []
    with open(EVAL_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


EVAL_LANGUAGE_TO_UI_LANG = {"fr": "fr", "ary": "darija"}


def run_disk_backend(rows: list[dict], *, use_language_affinity: bool) -> list[dict]:
    from app.services.domain_context import build_domain_context

    results = []
    for row in rows:
        ui_lang = EVAL_LANGUAGE_TO_UI_LANG.get(row["language"]) if use_language_affinity else None
        context, sources = build_domain_context(
            query=row["query"], domain=row["domain"], top_k=4, ui_lang=ui_lang,
        )
        results.append({"row": row, "context": context, "sources": sources})
    return results


def run_pgvector_backend(
    rows: list[dict], *, use_language_affinity: bool, tenant_id: str, threshold: Optional[float] = None
) -> list[dict]:
    from app.services.retrieval import retrieve

    results = []
    for row in rows:
        ui_lang = EVAL_LANGUAGE_TO_UI_LANG.get(row["language"]) if use_language_affinity else None
        result = retrieve(
            query=row["query"], domain=row["domain"], backend="pgvector",
            tenant_id=tenant_id, ui_lang=ui_lang, top_k=4, threshold=threshold,
        )
        results.append({"row": row, "context": result.context, "sources": result.sources})
    return results


def score_auto_domain(rows: list[dict], *, tenant_id: str) -> dict:
    """Honest gate for app.services.routing's domain router (tier 2): drop
    each in-domain eval row's labelled `domain` and check whether
    resolve_domain's similarity-weighted vote picks it back correctly from
    the query alone. Out-of-domain rows (no gold_sources) are skipped --
    there is no "correct" domain to check a refusal-triggering query
    against; that's empty_context_rate_ood's job, not this one's.

    Nearly free: the 50 pairs are already labelled with their true domain,
    so this needed no new eval data, only a new metric over the existing
    set.
    """
    from app.services.routing import resolve_domain

    in_domain_rows = [r for r in rows if r["gold_sources"]]
    correct = 0
    misses = []
    for row in in_domain_rows:
        domain, source = resolve_domain(row["query"], tenant_id=tenant_id, backend="pgvector")
        if domain == row["domain"]:
            correct += 1
        else:
            misses.append(f"{row['id']}: expected {row['domain']!r}, routed to {domain!r} ({source})")

    return {
        "n": len(in_domain_rows),
        "accuracy": correct / max(1, len(in_domain_rows)),
        "misses": misses,
    }


def score(results: list[dict]) -> dict:
    n = len(results)
    recall_hits = 0
    mrr_sum = 0.0
    substring_hits = 0
    substring_total = 0
    cross_language_hits = 0
    cross_language_total = 0
    ood_empty = 0
    ood_total = 0
    misses = []

    for r in results:
        row, context, sources = r["row"], r["context"], r["sources"]
        gold = set(row["gold_sources"])

        if gold:
            # recall@4 / MRR over the (already top_k-limited) sources list
            hit_rank = next((i for i, s in enumerate(sources) if s in gold), None)
            if hit_rank is not None:
                recall_hits += 1
                mrr_sum += 1.0 / (hit_rank + 1)
            else:
                misses.append(row["id"])

            if row.get("gold_substring"):
                substring_total += 1
                if row["gold_substring"] in context:
                    substring_hits += 1

            query_is_ar = _query_is_arabic_script(row["query"]) or row["language"] == "ary"
            if sources:
                cross_language_total += 1
                top_is_ar = _is_arabic_source(sources[0])
                if query_is_ar != top_is_ar:
                    # Arabizi queries legitimately want Arabic-script sources
                    # (Arabizi routes to the Darija template); only flag a
                    # true script mismatch, not the arabizi->arabic case.
                    if not (row["language"] == "ary" and not _query_is_arabic_script(row["query"]) and top_is_ar):
                        cross_language_hits += 1
        else:
            ood_total += 1
            if not context.strip():
                ood_empty += 1

    return {
        "n": n,
        "recall_at_4": recall_hits / max(1, n - ood_total),
        "mrr": mrr_sum / max(1, n - ood_total),
        "gold_substring_present_in_context": substring_hits / max(1, substring_total),
        "cross_language_rate": cross_language_hits / max(1, cross_language_total),
        "empty_context_rate_ood": ood_empty / max(1, ood_total),
        "ood_total": ood_total,
        "misses": misses,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="disk", choices=["disk", "pgvector"])
    parser.add_argument(
        "--language-affinity", action="store_true",
        help="Pass ui_lang through (ADR 0002 decision 5) instead of plain top-k by similarity.",
    )
    parser.add_argument(
        "--tenant-id", default="company_abc",
        help="pgvector backend only: which tenant's ingested documents to search.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="pgvector backend only: override the similarity cutoff (default settings.similarity_threshold).",
    )
    parser.add_argument(
        "--auto-domain", action="store_true",
        help=(
            "Score app.services.routing.resolve_domain's tier-2 vote instead "
            "of retrieval quality: drop each row's labelled domain and check "
            "the router picks it back from the query alone. pgvector only."
        ),
    )
    args = parser.parse_args()

    rows = load_eval_rows()
    print(f"Loaded {len(rows)} eval rows from {EVAL_PATH}")

    if args.auto_domain:
        metrics = score_auto_domain(rows, tenant_id=args.tenant_id)
        print(f"\n=== Domain auto-routing eval (tenant={args.tenant_id}) ===")
        print(f"  n (in-domain rows)  = {metrics['n']}")
        print(f"  accuracy            = {metrics['accuracy']:.3f}")
        if metrics["misses"]:
            print("  misses:")
            for m in metrics["misses"]:
                print(f"    - {m}")
        return

    if args.backend == "disk":
        results = run_disk_backend(rows, use_language_affinity=args.language_affinity)
    elif args.backend == "pgvector":
        results = run_pgvector_backend(
            rows, use_language_affinity=args.language_affinity, tenant_id=args.tenant_id,
            threshold=args.threshold,
        )
    else:
        raise SystemExit(f"backend {args.backend!r} not wired up yet")

    metrics = score(results)
    mode = "language-affinity ON" if args.language_affinity else "language-affinity OFF"
    print(f"\n=== Retrieval eval: backend={args.backend}, {mode} ===")
    print(f"  n                                = {metrics['n']} ({metrics['ood_total']} out-of-domain)")
    print(f"  recall@4                         = {metrics['recall_at_4']:.3f}")
    print(f"  MRR                               = {metrics['mrr']:.3f}")
    print(f"  gold_substring_present_in_context = {metrics['gold_substring_present_in_context']:.3f}")
    print(f"  cross_language_rate               = {metrics['cross_language_rate']:.3f}")
    print(f"  empty_context_rate (ood, want 1.0)= {metrics['empty_context_rate_ood']:.3f}")
    if metrics["misses"]:
        print(f"  misses: {', '.join(metrics['misses'])}")


if __name__ == "__main__":
    main()
