"""
Batch-generate grounded, non-Socratic explanations for the video-generation
partner, gate-validated and written in the exact VideoJobOut contract shape
(docs/PARTNER_VIDEO_ONBOARDING.md "Locked payload").

Retrieval: app.services.retrieval.retrieve (pgvector). Generation:
app.services.llm.generate_llm_response with the explanatory system-prompt
override (app.services.llm.build_explanatory_prompt) -- the Socratic
production prompts are never used here, a video viewer has no way to answer
a question posed to them.

Seeds are sorted by language before generation: Ollama loads/evicts models
per name with no explicit swap function, so alternating languages would pay
a model reload every single sample instead of two loads total for the run.

Usage:
    .gguf_venv/Scripts/python.exe scripts/generate_video_samples.py [--dry-run]
    .gguf_venv/Scripts/python.exe scripts/generate_video_samples.py --only-lang fr
"""
import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

if sys.stdout.encoding is None or "utf-8" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = REPO_ROOT / "out"

# Drawn from tests/data/retrieval_eval.jsonl -- the 5 out-of-domain rows
# (empty gold_sources) are excluded on purpose, they're built to trigger a
# refusal and would be useless as video content. `title` is hand-written in
# the same language as the eventual explanation, matching the contract's
# "title ... same language as input_text" rule -- one extra model call per
# sample isn't worth it for 20 short labels.
SEEDS = [
    # -- French --
    {"id": "ind-fr-01", "domain": "industrial", "lang": "fr",
     "query": "Quelle est l'obligation du travailleur en matière de sécurité au travail ?",
     "title": "Obligations du travailleur en sécurité"},
    {"id": "ind-fr-02", "domain": "industrial", "lang": "fr",
     "query": "Que fait le comité de sécurité et d'hygiène dans une entreprise ?",
     "title": "Rôle du comité de sécurité et d'hygiène"},
    {"id": "ind-fr-04", "domain": "industrial", "lang": "fr",
     "query": "C'est quoi une norme NM chez IMANOR ?",
     "title": "La norme NM chez IMANOR"},
    {"id": "ind-fr-05", "domain": "industrial", "lang": "fr",
     "query": "Qu'est-ce que le cycle PDCA dans la norme ISO 45001 ?",
     "title": "Le cycle PDCA en ISO 45001"},
    {"id": "sec-fr-01", "domain": "securite", "lang": "fr",
     "query": "Quelles sont les sanctions prévues par la loi 27.06 sur le gardiennage ?",
     "title": "Sanctions de la loi 27.06 sur le gardiennage"},
    {"id": "sec-fr-02", "domain": "securite", "lang": "fr",
     "query": "Quelles sont les conditions pour exercer comme agent de gardiennage ?",
     "title": "Conditions pour devenir agent de gardiennage"},
    {"id": "sec-fr-05", "domain": "securite", "lang": "fr",
     "query": "Quels sont les différents niveaux de contrôle d'accès dans un site ?",
     "title": "Niveaux de contrôle d'accès sur site"},
    {"id": "bc-fr-01", "domain": "blockchain", "lang": "fr",
     "query": "Comment le projet de loi 42.25 définit-il les actifs numériques ?",
     "title": "Définition des actifs numériques (loi 42.25)"},
    {"id": "bc-fr-03", "domain": "blockchain", "lang": "fr",
     "query": "Qu'est-ce que la Règle du Voyage (Travel Rule) du GAFI ?",
     "title": "La Règle du Voyage du GAFI"},
    {"id": "bc-fr-05", "domain": "blockchain", "lang": "fr",
     "query": "Quelles sont les caractéristiques fondamentales d'une blockchain ?",
     "title": "Caractéristiques fondamentales d'une blockchain"},
    # -- Darija (ar-MA) --
    {"id": "ind-ar-01", "domain": "industrial", "lang": "ar-MA",
     "query": "شنو هي التزامات المشغل فيما يخص نظافة أماكن الشغل؟",
     "title": "التزامات المشغل فنظافة أماكن الشغل"},
    {"id": "ind-ar-02", "domain": "industrial", "lang": "ar-MA",
     "query": "واش المشغل خاصو يوفر معدات الوقاية الشخصية للعمال؟",
     "title": "معدات الوقاية الشخصية ديال العمال"},
    {"id": "ind-ar-03", "domain": "industrial", "lang": "ar-MA",
     "query": "شنو هي المراحل الخمس ديال فصل مصادر الطاقة؟",
     "title": "المراحل الخمس ديال فصل الطاقة"},
    {"id": "ind-arz-01", "domain": "industrial", "lang": "ar-MA",
     "query": "chno howa dyal les EPI li khass el 3amel ylabss?",
     "title": "les EPI اللي خاص العامل يلبس"},
    {"id": "sec-ar-01", "domain": "securite", "lang": "ar-MA",
     "query": "شنو كايقول الباب الثاني ديال القانون 27.06 على شروط مزاولة المهنة؟",
     "title": "شروط مزاولة مهنة الحراسة (القانون 27.06)"},
    {"id": "sec-ar-02", "domain": "securite", "lang": "ar-MA",
     "query": "شنو هوما حدود الصلاحيات ديال عون الحراسة؟",
     "title": "حدود صلاحيات عون الحراسة"},
    {"id": "sec-ar-03", "domain": "securite", "lang": "ar-MA",
     "query": "كيفاش كيتصرح بالحوادث الأمنية فالمسطرة الداخلية؟",
     "title": "كيفاش كيتصرح بالحوادث الأمنية"},
    {"id": "bc-ar-01", "domain": "blockchain", "lang": "ar-MA",
     "query": "شنو كايهضر عليه الباب الثاني ديال القانون 42-25 على الترخيص؟",
     "title": "الترخيص فالباب الثاني ديال القانون 42-25"},
    {"id": "bc-ar-02", "domain": "blockchain", "lang": "ar-MA",
     "query": "شنو هوما العقوبات المنصوص عليها فمشروع القانون ديال الأصول المشفرة؟",
     "title": "العقوبات فمشروع قانون الأصول المشفرة"},
    {"id": "bc-arz-01", "domain": "blockchain", "lang": "ar-MA",
     "query": "chno khass ndiro bach ntsna b l'identite dyal l'client (KYC)?",
     "title": "KYC : التحقق من هوية الزبون"},
]

# Calibrated against real output (2026-08-18 smoke test), not guessed: a
# single-topic factual question under the explanatory prompt legitimately
# produces a complete, grounded 30-55 word answer -- an initial 60-word
# floor rejected several genuinely good samples. 20 still catches an
# actually-broken/truncated generation without penalizing a correct, concise
# one.
MIN_WORDS = 20
MAX_WORDS = 300
MAX_ATTEMPTS = 3


def _ui_lang(lang: str) -> str:
    return "fr" if lang == "fr" else "darija"


def gate(text: str, lang: str) -> str | None:
    """Return None if `text` passes every check, else the name of the first
    gate it failed. All checks below are standalone on a plain string (see
    app/services/generate_training_data.py) -- no training-row wrapping."""
    from app.services.generate_training_data import (
        _CJK,
        _QUESTION_MARK,
        _REFUSAL_MARKERS,
        darija_marker_count,
        english_marker_count,
        french_marker_count,
        has_arabic_outside_citations,
        has_arabic_script,
    )

    if not text or not text.strip():
        return "empty"
    if _QUESTION_MARK.search(text):
        return "socratic_leak"
    if _REFUSAL_MARKERS.search(text):
        return "refusal"
    if _CJK.search(text):
        return "cjk_contamination"

    n_words = len(text.split())
    if not (MIN_WORDS <= n_words <= MAX_WORDS):
        return f"length_out_of_band({n_words}w)"

    if lang == "fr":
        if has_arabic_outside_citations(text):
            return "arabic_outside_citation"
        if french_marker_count(text) < 2:
            return "insufficient_french_markers"
        if english_marker_count(text) > 1:
            return "too_much_english"
    else:
        if not has_arabic_script(text):
            return "no_arabic_script"
        if darija_marker_count(text) < 2:
            return "insufficient_darija_markers"
        # No minimum french_term_count here: the system prompt requires
        # preserving a French term in Latin letters *if one appears* --
        # it never requires one to appear. Measured 2026-08-18: a correct,
        # grounded answer about guarding-license conditions (Loi 27.06)
        # legitimately contains zero French loanwords.

    return None


def generate_one(seed: dict, *, tenant_id: str) -> tuple[str | None, str | None, dict]:
    """Returns (accepted_text, reject_reason, retrieval_meta). Exactly one
    of accepted_text / reject_reason is non-None on return."""
    from app.services.llm import build_explanatory_prompt, generate_llm_response
    from app.services.retrieval import retrieve

    r = retrieve(
        query=seed["query"], domain=seed["domain"], backend="pgvector",
        tenant_id=tenant_id, ui_lang=_ui_lang(seed["lang"]), top_k=4,
    )
    meta = {"n_sources": len(r.sources), "sources": r.sources}
    if not r.context.strip():
        return None, "empty_context", meta

    for attempt in range(1, MAX_ATTEMPTS + 1):
        text = generate_llm_response(
            query=seed["query"], context=r.context, domain=seed["domain"],
            language=_ui_lang(seed["lang"]),
            system_prompt_override=build_explanatory_prompt(
                seed["domain"], r.context, seed["lang"] if seed["lang"] == "fr" else "darija"
            ),
        )
        reason = gate(text, seed["lang"])
        if reason is None:
            return text, None, meta
        meta[f"attempt_{attempt}_rejected"] = reason
    return None, reason, meta


def make_job(seed: dict, text: str, *, tenant_id: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "tenant_id": tenant_id,
        "session_id": None,
        "input_text": text,
        "title": seed["title"],
        "language": seed["lang"],
        "status": "pending",
        "video_url": None,
        "error_message": None,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", default="company_abc")
    parser.add_argument("--out", default=None, help="Output dir (default: ./out)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Retrieval only -- confirms every seed has grounded context, no generation.")
    parser.add_argument("--only-lang", choices=["fr", "ar-MA"], default=None)
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    clean_path = out_dir / f"video_samples_{date_tag}.jsonl"
    raw_path = out_dir / f"video_samples_{date_tag}.raw.jsonl"

    seeds = [s for s in SEEDS if args.only_lang is None or s["lang"] == args.only_lang]
    # Language-sorted so Ollama loads each model exactly once for the run.
    seeds = sorted(seeds, key=lambda s: s["lang"])

    if args.dry_run:
        from app.services.retrieval import retrieve

        ok = 0
        for s in seeds:
            r = retrieve(query=s["query"], domain=s["domain"], backend="pgvector",
                          tenant_id=args.tenant_id, ui_lang=_ui_lang(s["lang"]), top_k=4)
            has_ctx = bool(r.context.strip())
            ok += has_ctx
            print(f"[{'OK' if has_ctx else 'EMPTY'}] {s['id']} ({s['domain']}/{s['lang']}) "
                  f"-> {len(r.sources)} sources")
        print(f"\n{ok}/{len(seeds)} seeds have grounded context.")
        return

    accepted, rejected = 0, 0
    with open(clean_path, "w", encoding="utf-8") as clean_fh, \
         open(raw_path, "w", encoding="utf-8") as raw_fh:
        for s in seeds:
            print(f"[{s['id']}] {s['domain']}/{s['lang']} ...", end=" ", flush=True)
            text, reason, meta = generate_one(s, tenant_id=args.tenant_id)
            raw_fh.write(json.dumps(
                {"seed_id": s["id"], "domain": s["domain"], "lang": s["lang"],
                 "accepted": reason is None, "reject_reason": reason,
                 "meta": meta, "text": text},
                ensure_ascii=False,
            ) + "\n")
            raw_fh.flush()

            if reason is not None:
                rejected += 1
                print(f"REJECTED ({reason})")
                continue

            job = make_job(s, text, tenant_id=args.tenant_id)
            clean_fh.write(json.dumps(job, ensure_ascii=False) + "\n")
            clean_fh.flush()
            accepted += 1
            print(f"ok ({len(text.split())}w)")

    print(f"\n{accepted} accepted, {rejected} rejected.")
    print(f"Clean output: {clean_path}")
    print(f"Raw log:      {raw_path}")


if __name__ == "__main__":
    main()
