"""
Supervised RAG ground-truth check for company_efg's two documents.

Two tiers of ground truth, both established INDEPENDENTLY of the pipeline
under test (this is what makes it supervised rather than self-confirming):

  Tier 1 -- hand-verified. Pages 15 and 51 of arabic_test.pdf were
  rendered and OCR'd directly, outside the ingestion pipeline, and the
  exact strings below were read off that output by hand. These are the
  strongest signal: page 51 was classified EMPTY and dropped entirely by
  the pre-fix classifier (so its formula constants were absent from the
  corpus at all), and page 15 was classified NATIVE with only its sparse
  caption text kept (so its CAD layer table was absent).

  Tier 2 -- from the tenant's own benchmark report (TC-01..TC-07), which
  cites page numbers in arabic_test.pdf for each value.

Two things are checked, and the distinction matters because they fail for
different reasons and have different fixes:

  1. STORED  -- is the value present in any chunk in Postgres at all?
     A miss here is an INGESTION defect (parsing/OCR/classification).
  2. RETRIEVED -- does the app's own retrieval actually surface a chunk
     containing it for a natural question? A miss here with STORED=yes is
     a RETRIEVAL defect (embedding/threshold/language-affinity), not an
     ingestion one. The original benchmark conflated these two and blamed
     the model for both.

Run AFTER both documents reach ready/partial under company_efg:
    .gguf_venv/Scripts/python.exe probe_rag_groundtruth_company_efg.py

Writes probe_rag_groundtruth_results.json. Prints a table; exits 0 always
(report generator, not a gate).
"""
import io
import json
import os
import sys

import psycopg2

# A Windows console is cp1252 and raises UnicodeEncodeError on any Arabic
# diagnostic, killing the run AFTER the work is done -- the same defect
# already fixed once in scripts/verify_ocr_arabic.py (see
# docs/architecture/data-and-retrieval.md's OCR gate run notes). Reconfigure
# before printing anything.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="backslashreplace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import get_settings  # noqa: E402

TENANT = os.environ.get("PROBE_TENANT", "company_efg")

# Which uploaded file the arabic needles should be looked for in. Defaults
# to the full 80-page arabic_test.pdf; set PROBE_ARABIC_DOC=arabic_probe.pdf
# to run the same checks against tests/data/real_pdfs/arabic_probe.pdf, the
# 10-page structural subset (see that fixture's README for the page map).
# Every needle below lives on a page the subset keeps, so the checks are
# identical in strength -- only the OCR wall-clock differs.
ARABIC_DOC = os.environ.get("PROBE_ARABIC_DOC", "arabic_test.pdf")

# (label, needle, tier, source_doc, why_it_matters)
STORED_CHECKS = [
    # --- Tier 1: hand-verified, the pages the pre-fix pipeline lost ---
    ("p51 DPF constant 1.5765", "1.5765", 1, ARABIC_DOC, "page was classified EMPTY and dropped entirely pre-fix"),
    ("p51 DPF exponent 0.158", "0.158", 1, ARABIC_DOC, "same page"),
    ("p51 seconds/day 86400", "86400", 1, ARABIC_DOC, "same page"),
    ("p15 CAD layer حدود القسيمة", "حدود القسيمة", 1, ARABIC_DOC, "table lost pre-fix (page kept as sparse NATIVE)"),
    ("p15 CAD layer التقسيمات", "التقسيمات", 1, ARABIC_DOC, "same table"),
    ("p15 CAD layer نص التقسيمات", "نص التقسيمات", 1, ARABIC_DOC, "same table"),
    ("p15 datum 1970", "1970", 1, ARABIC_DOC, "geodetic datum, native layer"),
    ("p15 scale factor 0.999600", "0.999600", 1, ARABIC_DOC, "native-only value -- proves native+OCR MERGE, not replace"),
    # --- Tier 2: tenant benchmark TC-01..TC-07 ---
    ("TC-06 substation threshold 12,000", "12,000", 2, ARABIC_DOC, "was stored pre-fix but never retrieved"),
    ("TC-05 reclamation line خط الدفان", "خط الدفان", 2, ARABIC_DOC, "waterfront setback"),
    ("TC-07 telecom zone BB1", "BB1", 2, ARABIC_DOC, "tower exclusion zone list, p80"),
    # --- french_test.pdf: needs ZERO OCR, pure native-parse control ---
    ("FR safety component rule", "composant de sécurité", 1, "french_test.pdf", "p13 rule used by the tutoring eval"),
    ("FR CE marking", "marquage CE", 1, "french_test.pdf", "p12/p13"),
]

# Natural questions -> the value their answer must contain.
RETRIEVAL_CHECKS = [
    ("ما هو الحد الأدنى لمجموع الأحمال الكهربائية الذي يوجب توفير محطة كهرباء رئيسية؟", "12,000", "darija", "TC-06"),
    ("ما هي الطبقات المطلوبة في ملف CAD لطلب تقسيم الأراضي؟", "التقسيمات", "darija", "TC-02"),
    ("كيف يتم حساب عامل التصميم في وقت الذروة؟", "1.5765", "darija", "TC-04"),
    ("Dans quel cas le remplacement d'un composant de securite est-il une modification ?", "composant de sécurité", "fr", "FR-p13"),
]


def main() -> None:
    settings = get_settings()
    conn = psycopg2.connect(settings.database_url)
    cur = conn.cursor()

    cur.execute(
        "SELECT source_name, count(*), sum(length(content)) FROM documents "
        "WHERE tenant_id=%s GROUP BY source_name ORDER BY source_name;",
        (TENANT,),
    )
    corpus = [{"source": r[0], "chunks": r[1], "chars": int(r[2] or 0)} for r in cur.fetchall()]
    print(f"=== corpus for tenant {TENANT} ===")
    for c in corpus:
        print(f"  {c['source']:<24} chunks={c['chunks']:<5} chars={c['chars']}")
    if not corpus:
        print("  (empty -- ingest has not completed)")
        return

    print("\n=== TIER 1/2 STORED checks (is the value in the corpus at all?) ===")
    stored_results = []
    for label, needle, tier, doc, why in STORED_CHECKS:
        cur.execute(
            "SELECT count(*) FROM documents WHERE tenant_id=%s AND source_name=%s AND content LIKE %s;",
            (TENANT, doc, f"%{needle}%"),
        )
        hits = cur.fetchone()[0]
        ok = hits > 0
        stored_results.append({
            "label": label, "needle": needle, "tier": tier, "doc": doc,
            "hits": hits, "stored": ok, "why": why,
        })
        print(f"  [{'PASS' if ok else 'FAIL'}] T{tier} {label:<38} hits={hits}")

    print("\n=== RETRIEVED checks (does retrieval actually surface it?) ===")
    from app.services import sources as source_service
    from app.services.search import build_rag_context

    active = source_service.active_source_ids(TENANT, requested=None)
    retrieval_results = []
    for question, needle, ui_lang, tc in RETRIEVAL_CHECKS:
        try:
            ctx, srcs = build_rag_context(
                query=question, tenant_id=TENANT, top_k=4,
                domain="industrial", ui_lang=ui_lang, source_ids=active,
            )
            found = needle in ctx
            retrieval_results.append({
                "test": tc, "question": question, "needle": needle, "ui_lang": ui_lang,
                "retrieved": found, "sources": srcs, "context_chars": len(ctx),
            })
            print(f"  [{'PASS' if found else 'FAIL'}] {tc:<8} needle={needle!r:<22} sources={srcs}")
        except Exception as e:
            retrieval_results.append({"test": tc, "error": f"{type(e).__name__}: {e}"})
            print(f"  [ERR ] {tc}: {type(e).__name__}: {e}")

    t1 = [r for r in stored_results if r["tier"] == 1]
    summary = {
        "tenant": TENANT,
        "corpus": corpus,
        "tier1_stored_pass": sum(1 for r in t1 if r["stored"]),
        "tier1_stored_total": len(t1),
        "stored_pass": sum(1 for r in stored_results if r["stored"]),
        "stored_total": len(stored_results),
        "retrieved_pass": sum(1 for r in retrieval_results if r.get("retrieved")),
        "retrieved_total": len(retrieval_results),
        "stored_results": stored_results,
        "retrieval_results": retrieval_results,
    }
    with open("probe_rag_groundtruth_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"\nSTORED   {summary['stored_pass']}/{summary['stored_total']} "
        f"(Tier-1 hand-verified: {summary['tier1_stored_pass']}/{summary['tier1_stored_total']})"
    )
    print(f"RETRIEVED {summary['retrieved_pass']}/{summary['retrieved_total']}")
    print("Wrote probe_rag_groundtruth_results.json")
    conn.close()


if __name__ == "__main__":
    main()
