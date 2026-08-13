"""
CI floor for the retrieval eval (Stage 0). Not the full 50-pair eval --
that needs the embedding model loaded and takes a few seconds, so it's a
manual/CI step (scripts/eval_retrieval.py), not part of the fast unit
suite. This runs a small fixed subset against the disk backend so a
retrieval regression shows up in `pytest`, not just when someone
remembers to run the eval script by hand.

Known flakiness risk: real sentence-transformer embeddings on CPU can have
tiny floating-point nondeterminism (thread-scheduling-dependent summation
order in the underlying matrix ops) that occasionally flips a genuinely
borderline top-4 ranking. Observed once in this session (both tests here
failed together in a full-suite run, then passed cleanly both standalone
and on a full-suite rerun) -- not reproducible, not a logic bug. If this
becomes a recurring flake, widen the margin (pick a query with a clearer
top-4 gap) rather than retrying blindly.
"""
from app.services.domain_context import build_domain_context

# 8-pair subset of tests/data/retrieval_eval.jsonl: one clear in-domain hit
# per domain/script combination, plus 2 out-of-domain probes.
SMOKE_CASES = [
    ("industrial", "Quelle est l'obligation du travailleur en matière de sécurité au travail ?",
     "1.1_code_du_travail_health_safety.md"),
    ("industrial", "شنو هي التزامات المشغل فيما يخص نظافة أماكن الشغل؟",
     "1.11_ar_code_travail_salama.md"),
    ("securite", "Quelles sont les sanctions prévues par la loi 27.06 sur le gardiennage ?",
     "2.1_loi_27_06_regulation.md"),
    ("securite", "شنو كايقول الباب الثاني ديال القانون 27.06 على شروط مزاولة المهنة؟",
     "2.6_ar_loi_27_06.md"),
    ("blockchain", "Comment le projet de loi 42.25 définit-il les actifs numériques ?",
     "3.1_bill_42_25_draft.md"),
    ("blockchain", "شنو هوما العقوبات المنصوص عليها فمشروع القانون ديال الأصول المشفرة؟",
     "3.7_ar_projet_loi_42_25.md"),
]

OOD_CASES = [
    ("industrial", "Quelle est la recette traditionnelle du tajine aux pruneaux ?"),
    ("blockchain", "Quel a été le score du dernier match du Raja Casablanca ?"),
]


def test_in_domain_queries_retrieve_gold_source():
    for domain, query, gold_source in SMOKE_CASES:
        _, sources = build_domain_context(query=query, domain=domain, top_k=4)
        assert gold_source in sources, (
            f"{domain!r}/{query!r} did not retrieve {gold_source!r}, got {sources!r}"
        )


def test_out_of_domain_queries_still_retrieve_something_today():
    """Documents the measured baseline gap (data-and-retrieval.md): the disk
    backend has no similarity threshold, so out-of-domain queries do NOT hit
    empty context today. This is a known-open gap, not a passing guarantee --
    the test exists so fixing it (Stage 3) is a deliberate, visible diff
    here, not a silent behavior change no one notices."""
    for domain, query in OOD_CASES:
        context, _ = build_domain_context(query=query, domain=domain, top_k=4)
        assert context.strip(), (
            "Out-of-domain query unexpectedly got empty context -- if this "
            "starts failing, retrieval thresholding has improved; update "
            "this test (and data-and-retrieval.md's baseline) to assert "
            "empty context instead."
        )
