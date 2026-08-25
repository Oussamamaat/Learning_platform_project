"""
Out-of-domain refusal eval. Measures the SERVING DECISION, not retrieval.

Why this exists alongside scripts/eval_retrieval.py: that harness reports
`empty_context_rate_ood`, which measures only whether retrieval came back
empty -- the OLD refusal trigger, and the exact thing
progress_report_2026-08-18 recorded as insufficient ("out-of-domain queries
currently always retrieve a top-4 context and never trigger the
deterministic refusal path"). So the harness that was used to justify the
threshold sweep structurally cannot score the fix that followed it: a query
that retrieves four irrelevant chunks and is then refused via
domain_source == "no_match" counts as a FAILURE under empty_context_rate_ood
and a SUCCESS in production.

This runner reproduces app/routers/chat.py's actual gate --

    refuse  <=>  domain_source == "no_match" OR not context.strip()

-- over both classes of query, and reports which trigger fired, so the
marginal contribution of the new trigger over the old one is visible rather
than assumed. No LLM is called: the decision is made before the model, and
adding a generation round trip per query would make this too slow to run
often for no extra signal.

Two rates, and both matter -- a gate is only interesting if it is scored
against its own cost:

  ood_refusal_rate    (want 1.0) -- out-of-corpus questions that got refused.
  false_refusal_rate  (want 0.0) -- in-corpus questions that got refused
                                    anyway. A gate that refuses everything
                                    scores a perfect 1.0 above.

In-corpus queries come from tests/data/retrieval_eval.jsonl (the 45 labeled
rows, reused so this is scored against the same content the retrieval
numbers in the report were). Out-of-corpus queries come from that file's 5
`ood-*` rows PLUS the expanded set below -- 5 was too thin to distinguish
0.8 from 1.0 with any confidence, and the classes it missed (English,
adjacent-but-outside legal domains) are the ones most likely to fail.

Requires live Postgres/pgvector with the tenant corpus ingested. Run:

    .gguf_venv/Scripts/python.exe scripts/eval_refusal.py
    .gguf_venv/Scripts/python.exe scripts/eval_refusal.py --json out.json
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EVAL_PATH = REPO_ROOT / "tests" / "data" / "retrieval_eval.jsonl"

# Out-of-corpus queries beyond the 5 in retrieval_eval.jsonl. Grouped by the
# KIND of out-of-domain-ness, because they are not equally hard and a single
# blended rate hides which kind is failing:
#
#   everyday   -- plainly unrelated to any tenant document. The easy case;
#                 if these fail, the gate is not working at all.
#   adjacent   -- real regulatory/professional questions from a domain this
#                 tenant has NO documents for (tax, immigration, medicine).
#                 The hard case, and the one that matters commercially: the
#                 embedding space puts "legal obligation" language close to
#                 the corpus even when the subject is entirely different, so
#                 these are exactly the queries that retrieve a confident-
#                 looking top-4 and get answered as if grounded.
#   english    -- in scope per the project brief but with no corpus behind
#                 it yet; must refuse rather than answer from French chunks.
EXTRA_OOD = [
    ("ood-ex-01", "fr", "everyday", "Quel temps fera-t-il demain à Marrakech ?"),
    ("ood-ex-02", "fr", "everyday", "Comment changer la courroie de distribution d'une Dacia Logan ?"),
    ("ood-ex-03", "fr", "everyday", "Quels sont les meilleurs restaurants de Rabat ?"),
    ("ood-ex-04", "ary", "everyday", "شنو هي أحسن طريقة باش نطيب الكسكس؟"),
    ("ood-ex-05", "ary", "everyday", "فين نقدر نشري تيليفون رخيص فالدار البيضاء؟"),

    ("ood-ex-06", "fr", "adjacent", "Quel est le taux de l'impôt sur les sociétés au Maroc ?"),
    ("ood-ex-07", "fr", "adjacent", "Quelles sont les conditions d'obtention d'un visa Schengen ?"),
    ("ood-ex-08", "fr", "adjacent", "Quelle est la posologie recommandée de l'amoxicilline chez l'adulte ?"),
    ("ood-ex-09", "fr", "adjacent", "Quelles sont les obligations du bailleur dans un bail d'habitation ?"),
    ("ood-ex-10", "ary", "adjacent", "شنو هي الوثائق اللي خاصني باش نسجل شركة جديدة؟"),
    ("ood-ex-11", "ary", "adjacent", "شحال كايخلص الضريبة على الدخل فالمغرب؟"),

    ("ood-ex-12", "en", "english", "What is the capital of Australia?"),
    ("ood-ex-13", "en", "english", "How do I file a patent application in the United States?"),
    ("ood-ex-14", "en", "english", "What are the symptoms of vitamin D deficiency?"),
]


def load_rows():
    """(in_corpus, out_of_corpus) -- both as {id, language, group, query}."""
    in_corpus, ood = [], []
    for line in EVAL_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        row = {
            "id": r["id"],
            "language": r.get("language", "fr"),
            "query": r["query"],
            "group": "labeled",
        }
        (ood if r["id"].startswith("ood-") else in_corpus).append(row)

    for rid, lang, group, query in EXTRA_OOD:
        ood.append({"id": rid, "language": lang, "group": group, "query": query})

    return in_corpus, ood


def decide(query: str, *, tenant_id: str, top_k: int) -> dict:
    """Reproduce app/routers/chat.py's refusal gate for one query, with no
    explicit domain (an explicit domain short-circuits the vote to
    "page_context" and could never reach the OOD path -- see
    tests/test_roadmap_2026_08_18.py's explicit-domain test).

    Deliberately calls the SAME two functions chat.py calls, in the same
    order, rather than reimplementing the decision: resolve_domain for the
    tier-2 corpus vote, then _retrieve_context filtered to whatever domain
    that returned.
    """
    from app.config import get_settings
    from app.routers.chat import _retrieve_context
    from app.services.routing import resolve_domain

    settings = get_settings()
    domain, domain_source = resolve_domain(
        query, tenant_id=tenant_id, backend=settings.retrieval_backend
    )
    context, sources, degraded = _retrieve_context(
        query, domain=domain, top_k=top_k, ui_lang="fr", tenant_id=tenant_id
    )

    empty = not context.strip()
    no_match = domain_source == "no_match"
    return {
        "domain": domain,
        "domain_source": domain_source,
        "n_sources": len(sources),
        "context_chars": len(context),
        "refused": no_match or empty,
        # Which trigger fired. "no_match_only" is the fix's marginal
        # contribution: these are the queries the old empty-context gate
        # would have answered.
        "trigger": (
            "both" if (no_match and empty)
            else "no_match_only" if no_match
            else "empty_only" if empty
            else "none"
        ),
        "degraded": degraded,
    }


def sweep(rows, *, tenant_id: str, top_k: int, thresholds) -> list[dict]:
    """Re-score the refusal gate across candidate `domain_vote_threshold`
    values, holding similarity_threshold (and therefore retrieval) fixed.

    Exists because of what the default run shows: at the shipped config,
    domain_vote_threshold == similarity_threshold == 0.4, and under that
    equality the vote can only return "no_match" when NOTHING in the corpus
    clears 0.4 -- which is exactly when domain-filtered retrieval comes back
    empty anyway. So `no_match` is strictly subsumed by the empty-context
    trigger it was added to supplement, and contributes nothing. The vote
    only becomes an independent signal once its threshold is strictly
    HIGHER than the retrieval threshold, and this sweep is how the gap gets
    picked from measurement rather than guessed.

    One embedding + one unfiltered search per query, reused for every
    threshold; filtered retrieval is cached per (query, domain) because a
    changing threshold can change which domain wins.
    """
    from app.config import get_settings
    from app.routers.chat import _retrieve_context
    from app.services.routing import vote_domain
    from app.services.search import search_similar_chunks

    settings = get_settings()
    cache: dict[tuple[str, str], bool] = {}

    def context_is_empty(query: str, domain: str) -> bool:
        key = (query, domain)
        if key not in cache:
            context, _, _ = _retrieve_context(
                query, domain=domain, top_k=top_k, ui_lang="fr", tenant_id=tenant_id
            )
            cache[key] = not context.strip()
        return cache[key]

    candidates = {
        row["id"]: search_similar_chunks(
            query=row["query"], tenant_id=tenant_id, top_k=20, domain=None
        )
        for row in rows
    }

    out = []
    for t in thresholds:
        refused_ood = refused_in = 0
        n_ood = n_in = 0
        for row in rows:
            voted = vote_domain(candidates[row["id"]], threshold=t)
            domain = voted or settings.default_domain
            refused = voted is None or context_is_empty(row["query"], domain)
            if row["expected_refusal"]:
                n_ood += 1
                refused_ood += refused
            else:
                n_in += 1
                refused_in += refused
        out.append({
            "domain_vote_threshold": round(t, 3),
            "ood_refusal_rate": refused_ood / max(1, n_ood),
            "false_refusal_rate": refused_in / max(1, n_in),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant-id", default=None, help="defaults to settings.default_tenant_id")
    ap.add_argument("--top-k", type=int, default=4, help="matches the report's top-4 framing")
    ap.add_argument("--json", dest="json_out", default=None, help="write full per-query results here")
    ap.add_argument("--sweep", action="store_true",
                    help="also sweep domain_vote_threshold to find a value where the "
                         "vote is an independent signal rather than a no-op")
    args = ap.parse_args()

    from app.config import get_settings

    settings = get_settings()
    tenant_id = args.tenant_id or getattr(settings, "default_tenant_id", "default")

    if settings.retrieval_backend != "pgvector":
        print(
            f"[warn] retrieval_backend={settings.retrieval_backend!r}: the disk backend "
            "never returns domain_source == 'no_match', so the OOD gate under test "
            "cannot fire at all. Results below measure only the empty-context path."
        )

    in_corpus, ood = load_rows()
    print(f"tenant={tenant_id} backend={settings.retrieval_backend} "
          f"top_k={args.top_k} threshold={settings.domain_vote_threshold}")
    print(f"in-corpus queries: {len(in_corpus)}   out-of-corpus queries: {len(ood)}\n")

    results = []

    print("=== out-of-corpus (want: refused) ===")
    for row in ood:
        d = decide(row["query"], tenant_id=tenant_id, top_k=args.top_k)
        results.append({**row, "expected_refusal": True, **d})
        mark = "OK  " if d["refused"] else "MISS"
        print(f"  [{mark}] {row['id']:<12} {row['group']:<8} trigger={d['trigger']:<14} "
              f"src={d['n_sources']} ctx={d['context_chars']:>5}  {row['query'][:52]}")

    print("\n=== in-corpus (want: answered) ===")
    for row in in_corpus:
        d = decide(row["query"], tenant_id=tenant_id, top_k=args.top_k)
        results.append({**row, "expected_refusal": False, **d})
        if d["refused"]:
            print(f"  [FALSE REFUSAL] {row['id']:<12} trigger={d['trigger']:<14} "
                  f"{row['query'][:52]}")
    n_false = sum(1 for r in results if not r["expected_refusal"] and r["refused"])
    if not n_false:
        print("  (none)")

    ood_res = [r for r in results if r["expected_refusal"]]
    ind_res = [r for r in results if not r["expected_refusal"]]
    n_ood_ok = sum(1 for r in ood_res if r["refused"])

    # The number the old harness could not see: OOD queries refused ONLY
    # because of the new trigger. Under empty_context_rate_ood these all
    # scored as failures.
    marginal = [r for r in ood_res if r["trigger"] == "no_match_only"]
    empty_only = [r for r in ood_res if r["trigger"] in ("empty_only", "both")]

    metrics = {
        "tenant_id": tenant_id,
        "backend": settings.retrieval_backend,
        "top_k": args.top_k,
        "domain_vote_threshold": settings.domain_vote_threshold,
        "similarity_threshold": settings.similarity_threshold,
        "n_ood": len(ood_res),
        "n_in_corpus": len(ind_res),
        "ood_refusal_rate": n_ood_ok / max(1, len(ood_res)),
        "false_refusal_rate": n_false / max(1, len(ind_res)),
        "refused_by_new_trigger_only": len(marginal),
        "refused_by_empty_context": len(empty_only),
        "ood_refusal_rate_by_group": {
            g: (
                sum(1 for r in ood_res if r["group"] == g and r["refused"])
                / max(1, sum(1 for r in ood_res if r["group"] == g))
            )
            for g in sorted({r["group"] for r in ood_res})
        },
    }

    print("\n" + "=" * 62)
    print(f"  ood_refusal_rate        (want 1.0) = {metrics['ood_refusal_rate']:.3f}"
          f"  ({n_ood_ok}/{len(ood_res)})")
    print(f"  false_refusal_rate      (want 0.0) = {metrics['false_refusal_rate']:.3f}"
          f"  ({n_false}/{len(ind_res)})")
    print(f"  refused by NEW trigger only        = {metrics['refused_by_new_trigger_only']}"
          f"   <- invisible to eval_retrieval.py's empty_context_rate_ood")
    print(f"  refused by empty context           = {metrics['refused_by_empty_context']}")
    print("  by group: " + ", ".join(
        f"{g}={v:.3f}" for g, v in metrics["ood_refusal_rate_by_group"].items()
    ))
    print("=" * 62)

    sweep_rows = None
    if args.sweep:
        print("\n=== domain_vote_threshold sweep "
              f"(similarity_threshold fixed at {settings.similarity_threshold}) ===")
        sweep_rows = sweep(
            results, tenant_id=tenant_id, top_k=args.top_k,
            thresholds=[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        )
        print(f"  {'thr':>5}  {'ood_refusal':>12}  {'false_refusal':>14}")
        for r in sweep_rows:
            print(f"  {r['domain_vote_threshold']:>5.2f}  {r['ood_refusal_rate']:>12.3f}  "
                  f"{r['false_refusal_rate']:>14.3f}")
        metrics["sweep"] = sweep_rows

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"metrics": metrics, "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.json_out}")

    # Never exits nonzero on a rate: this is a measurement tool, and a rate
    # is a judgement call against a threshold that is still being tuned.
    # The pass/fail gate for this behaviour lives in
    # tests/test_roadmap_2026_08_18.py.


if __name__ == "__main__":
    main()
