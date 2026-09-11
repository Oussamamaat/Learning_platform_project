"""
Build the fixture files Phase A/B scripts read: benchmark prompts, the
parity-probe conversations, quality-review cases, and AWQ calibration text.
Plan rev 2, Phase A5.

Runs on the laptop, against the SHARED Postgres (read-only queries only --
see the worktree's W3 rule: never write outside a vllm_dev tenant). Reuses
this repo's own retrieval and prompt-rendering code so every fixture is
built from the EXACT path the app takes, not a hand-rolled approximation:
    - app.routers.chat._retrieve_context   (same top_k=4, same fail-open
      behavior chat.py itself relies on)
    - app.services.llm._build_system_prompt / render_conversation

Usage:
    .gguf_venv/Scripts/python.exe scripts/vllm/make_fixtures.py

Writes scripts/vllm/fixtures/:
    bench_prompts.json     20 real RAG-grounded prompts per language
    parity_prompts.json    the 4 fixed conversations probe_history_parity.py
                            (tashkeel-eval worktree) already uses, copied
                            verbatim -- same content, so a parity mismatch
                            here is comparable to that probe's own history
    quality_prompts.json   green_light_model.md §2/§3 cases with real
                            retrieved context, tagged by checklist item
    calib_darija.jsonl      256 rendered rows from data/v11_merged/train.jsonl
    calib_fr.jsonl          256 rendered rows from data/fr_v3_merged/train.jsonl

Gate (Assumption 1): every bench/quality prompt's token count, under BOTH
tokenizers (Atlas-Chat-9B for Darija, unsloth/gemma-2-9b for French), plus
llm_max_tokens (1024), must stay <= 8192. A prompt that fails this needs
history/context trimming fixed upstream before Phase B, not a workaround
here -- printed as an error, not silently dropped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# -- Real RAG-grounded bench questions -------------------------------------
# No query log exists to sample from (chat_messages has 0 real rows -- this
# is a pre-launch corpus), so these are hand-written but domain-realistic,
# same convention as probe_history_parity.py's own DARIJA_SYSTEM/FRENCH_
# content and bench_llm.py's DARIJA_Q/FRENCH_Q -- and unlike those, run
# through REAL retrieval against the real ingested corpus (37 chunks across
# industrial/blockchain/securite, company_abc), not sent as a bare string.
BENCH_QUESTIONS = {
    "industrial": {
        "darija": [
            "شنو هي معدات الحماية الشخصية الإجبارية فورشة الخدمة؟",
            "شكون المسؤول على صيانة معدات الوقاية؟",
            "واش خاصني نلبس القفازات فالخدمة ديال الصيانة؟",
            "شنو كايقول القانون على تكوين العمال على السلامة؟",
            "علاش خاص معدات الوقاية تكون ملائمة للشغل؟",
        ],
        "fr": [
            "Quels sont les équipements de protection individuelle obligatoires ?",
            "Qui est responsable de l'entretien des équipements de protection ?",
            "Dois-je porter des gants pour les travaux de maintenance ?",
            "Que dit la réglementation sur la formation des travailleurs à la sécurité ?",
            "Pourquoi les équipements de protection doivent-ils être adaptés au travail effectué ?",
        ],
    },
    "blockchain": {
        "darija": [
            "شنو كايقول القانون 27.06 على العملات المشفرة؟",
            "واش الشركة خاصها ترخيص باش تخدم بالبلوكشاين؟",
            "شنو هوما الالتزامات ديال المشغل فمجال البلوكشاين؟",
        ],
        "fr": [
            "Que dit la loi 27.06 sur les crypto-actifs ?",
            "Une entreprise a-t-elle besoin d'une licence pour opérer avec la blockchain ?",
            "Quelles sont les obligations de l'employeur dans le domaine blockchain ?",
        ],
    },
    "securite": {
        "darija": [
            "شنو كايقول القانون على السلامة الصحية فالخدمة؟",
            "واش كاين تكوين إجباري على السلامة؟",
        ],
        "fr": [
            "Que dit la loi sur la santé et sécurité au travail ?",
            "Une formation sur la sécurité est-elle obligatoire ?",
        ],
    },
}

# -- Quality-review cases (green_light_model.md §2/§3, stable folder) ------
# Tagged by checklist item so Phase C's manual review can walk them in
# order. D2/D3/RF9/RF10 (insufficient-context refusal, no fabricated
# citations) is the highest-value case: this repo's own memory
# (finetune-degrades-citation-grounding) is exactly this failure mode, so
# it gets one real in-corpus question and one deliberately OUT-of-corpus
# question that must trigger a refusal, not a guess.
QUALITY_CASES = [
    {"tag": "A5/D3 grounded-with-citation", "domain": "industrial", "language": "fr",
     "question": "Que dit l'article 283 sur les équipements de protection individuelle ?"},
    {"tag": "A5/D3 grounded-with-citation", "domain": "industrial", "language": "darija",
     "question": "شنو كتقول المادة 283 على معدات الوقاية الشخصية؟"},
    {"tag": "D2/RF9 insufficient-context refusal", "domain": "industrial", "language": "fr",
     "question": "Quel est le salaire minimum légal pour un ouvrier du bâtiment ?"},
    {"tag": "D2/RF9 insufficient-context refusal", "domain": "industrial", "language": "darija",
     "question": "شحال هو الأجر الأدنى القانوني لعامل البناء؟"},
    {"tag": "D1/RF11 off-topic refusal", "domain": "industrial", "language": "fr",
     "question": "Quelle est la meilleure recette de tajine au poulet ?"},
    {"tag": "B1 explain-then-question", "domain": "industrial", "language": "darija",
     "question": "علاش خاصني نلبس القفازات؟"},
    {"tag": "E2 domain isolation", "domain": "securite", "language": "fr",
     "question": "Que dit le document sur les crypto-actifs ?"},
]


def load_settings():
    from app.config import get_settings
    return get_settings()


def retrieve(question: str, *, domain: str, language: str, tenant_id: str) -> tuple[str, list[str]]:
    """Exactly chat.py's own retrieval call (app/routers/chat.py:252),
    top_k=4, so bench/quality prompts are the same shape a real request
    would build -- not a hand-approximated RAG context."""
    from app.routers.chat import _retrieve_context

    ui_lang = "fr" if language == "fr" else "ar-MA"
    context, sources, degraded = _retrieve_context(
        question, domain=domain, top_k=4, ui_lang=ui_lang, tenant_id=tenant_id,
    )
    return context, sources


def render_prompt(question: str, *, domain: str, language: str, context: str) -> str:
    from app.services.llm import _build_system_prompt, render_conversation

    system = _build_system_prompt(domain, context, language)
    return render_conversation(
        [{"role": "system", "content": system}, {"role": "user", "content": question}]
    )


def make_bench_prompts(tenant_id: str, tokenizers: dict) -> list[dict]:
    rows = []
    for domain, by_lang in BENCH_QUESTIONS.items():
        for language, questions in by_lang.items():
            for q in questions:
                context, sources = retrieve(q, domain=domain, language=language, tenant_id=tenant_id)
                prompt = render_prompt(q, domain=domain, language=language, context=context)
                tok = tokenizers["fr" if language == "fr" else "darija"]
                token_count = len(tok(prompt)["input_ids"]) if tok else None
                rows.append({
                    "domain": domain, "language": language, "question": q,
                    "prompt": prompt, "n_sources": len(sources), "token_count": token_count,
                })
    return rows


def make_quality_prompts(tenant_id: str, tokenizers: dict) -> list[dict]:
    rows = []
    for case in QUALITY_CASES:
        context, sources = retrieve(
            case["question"], domain=case["domain"], language=case["language"], tenant_id=tenant_id,
        )
        prompt = render_prompt(
            case["question"], domain=case["domain"], language=case["language"], context=context,
        )
        tok = tokenizers["fr" if case["language"] == "fr" else "darija"]
        token_count = len(tok(prompt)["input_ids"]) if tok else None
        rows.append({**case, "prompt": prompt, "n_sources": len(sources),
                     "has_context": bool(context.strip()), "token_count": token_count})
    return rows


def make_parity_prompts() -> list[dict]:
    """The 4 fixed conversations probe_history_parity.py already uses
    (tashkeel-eval worktree), copied verbatim -- not regenerated -- so a
    parity check against these is directly comparable to that probe's own
    prior Ollama-vs-Ollama result, now extended to Ollama-vs-vLLM.

    Each row carries BOTH the raw `messages` (for reference/debugging) and
    a pre-`rendered` text (render_conversation(messages,
    add_generation_prompt=True) -- the generation-ready shape, matching a
    real live call). scripts/vllm/parity_probe.py sends `rendered` to
    Ollama's /api/generate raw:true AND vLLM's /v1/completions and
    compares each side's own reported prompt token count -- it never
    imports render_conversation itself, so it can run standalone on the
    lease without this repo's `app` package (same reasoning as
    build_merged_awq.py's calibration file: one render, reused everywhere,
    instead of a second implementation that could silently drift)."""
    from app.services.llm import render_conversation
    DARIJA_SYSTEM = (
        "You are an expert bilingual enterprise tutor specializing in "
        "industrial safety and workplace protocols.\n"
        "Answer in Moroccan Darija written in Arabic script, using a Socratic method.\n"
        "Ground all answers strictly in the provided context.\n\n"
        "CONTEXTE :\n"
        "المادة 283: يجب على المشغل أن يوفر معدات الوقاية الشخصية الملائمة لطبيعة "
        "الأشغال المنجزة، وأن يضمن صيانتها في حالة جيدة."
    )
    FRENCH_SYSTEM = (
        "Tu es un tuteur d'entreprise expert, specialise en securite industrielle et "
        "protocoles de travail.\n"
        "Reponds en francais, avec une methode socratique.\n"
        "Fonde toutes tes reponses strictement sur le contexte fourni.\n\n"
        "CONTEXTE :\n"
        "Article 283 : L'employeur doit fournir des equipements de protection "
        "individuelle adaptes a la nature des travaux effectues, et garantir leur "
        "maintenance en bon etat."
    )
    conversations = [
        {
            "name": "trained-shape 4-message (french)", "language": "fr",
            "messages": [
                {"role": "system", "content": FRENCH_SYSTEM},
                {"role": "user", "content": "Que doit fournir l'employeur aux travailleurs ?"},
                {"role": "assistant", "content": "Il doit fournir des equipements de protection individuelle. Sais-tu a quoi ils servent ?"},
                {"role": "user", "content": "Pourquoi doivent-ils etre adaptes au travail effectue ?"},
            ],
        },
        {
            "name": "single-turn (darija)", "language": "darija",
            "messages": [
                {"role": "system", "content": DARIJA_SYSTEM},
                {"role": "user", "content": "شنو خاص المشغل يوفر للعمال؟"},
            ],
        },
        {
            "name": "trained-shape 4-message (darija)", "language": "darija",
            "messages": [
                {"role": "system", "content": DARIJA_SYSTEM},
                {"role": "user", "content": "شنو خاص المشغل يوفر للعمال؟"},
                {"role": "assistant", "content": "خاصو يوفر معدات الوقاية الشخصية. شنو كتعرف على هاد المعدات؟"},
                {"role": "user", "content": "علاش خاصها تكون ملائمة للشغل؟"},
            ],
        },
        {
            "name": "one-past-trained 6-message (darija)", "language": "darija",
            "messages": [
                {"role": "system", "content": DARIJA_SYSTEM},
                {"role": "user", "content": "شنو خاص المشغل يوفر للعمال؟"},
                {"role": "assistant", "content": "خاصو يوفر معدات الوقاية الشخصية. شنو كتعرف على هاد المعدات؟"},
                {"role": "user", "content": "علاش خاصها تكون ملائمة للشغل؟"},
                {"role": "assistant", "content": "لأن كل شغل عندو مخاطر مختلفة. واش عندك مثال على شغل فيه خطر؟"},
                {"role": "user", "content": "شكون المسؤول على الصيانة ديال هاد المعدات؟"},
            ],
        },
    ]
    for conv in conversations:
        conv["rendered"] = render_conversation(conv["messages"], add_generation_prompt=True)
    return conversations


def make_calibration_file(train_jsonl: Path, out_path: Path, *, n: int, max_tokens: int, tokenizer) -> int:
    """Render `n` real training rows through render_conversation() (the
    same byte-exact, test-locked prompt shape the model was trained and
    served on) and write them PRE-RENDERED, one JSON object per line, so
    the lease-side AWQ build (scripts/vllm/build_merged_awq.py) never
    needs to import this repo's `app` package -- see plan rev 2's A7."""
    from app.services.llm import render_conversation

    written = 0
    with open(train_jsonl, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if written >= n:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = render_conversation(row["messages"], add_generation_prompt=False)
            if tokenizer is not None:
                token_count = len(tokenizer(text)["input_ids"])
                if token_count > max_tokens:
                    continue
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
    return written


def _load_tokenizers() -> dict:
    """Best-effort: AutoTokenizer.from_pretrained needs network access to
    the (public, non-gated -- verified in the plan's research pass)
    Atlas-Chat-9B / unsloth/gemma-2-9b repos. If offline, token counts are
    written as null and the A5 gate is skipped with a loud warning rather
    than silently passing."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("WARNING: transformers not importable -- token counts will be null.", file=sys.stderr)
        return {"darija": None, "fr": None}

    tokenizers = {}
    for key, repo_id in (("darija", "MBZUAI-Paris/Atlas-Chat-9B"), ("fr", "unsloth/gemma-2-9b")):
        try:
            tokenizers[key] = AutoTokenizer.from_pretrained(repo_id)
        except Exception as e:
            print(f"WARNING: could not load tokenizer for {repo_id}: {e} -- "
                  f"token counts for {key} will be null.", file=sys.stderr)
            tokenizers[key] = None
    return tokenizers


def main() -> int:
    settings = load_settings()
    tenant_id = settings.default_tenant_id
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizers (Atlas-Chat-9B, unsloth/gemma-2-9b) ...")
    tokenizers = _load_tokenizers()

    print(f"Retrieving bench prompts against tenant={tenant_id!r} (real corpus, top_k=4) ...")
    bench_rows = make_bench_prompts(tenant_id, tokenizers)
    (FIXTURES_DIR / "bench_prompts.json").write_text(
        json.dumps(bench_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  wrote {len(bench_rows)} bench prompts")

    print("Retrieving quality-review cases ...")
    quality_rows = make_quality_prompts(tenant_id, tokenizers)
    (FIXTURES_DIR / "quality_prompts.json").write_text(
        json.dumps(quality_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  wrote {len(quality_rows)} quality-review cases")

    print("Writing parity-probe conversations (copied verbatim from probe_history_parity.py) ...")
    parity_rows = make_parity_prompts()
    (FIXTURES_DIR / "parity_prompts.json").write_text(
        json.dumps(parity_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  wrote {len(parity_rows)} parity conversations")

    print("Rendering AWQ calibration files ...")
    darija_calib = REPO_ROOT / "data" / "v11_merged" / "train.jsonl"
    fr_calib = REPO_ROOT / "data" / "fr_v3_merged" / "train.jsonl"
    n_darija = make_calibration_file(
        darija_calib, FIXTURES_DIR / "calib_darija.jsonl",
        n=256, max_tokens=2048, tokenizer=tokenizers.get("darija"),
    )
    n_fr = make_calibration_file(
        fr_calib, FIXTURES_DIR / "calib_fr.jsonl",
        n=256, max_tokens=2048, tokenizer=tokenizers.get("fr"),
    )
    print(f"  wrote {n_darija} Darija + {n_fr} French calibration rows")

    # -- Assumption 1 gate: max prompt tokens + llm_max_tokens <= 8192 -----
    all_counts = [r["token_count"] for r in bench_rows + quality_rows if r["token_count"] is not None]
    if not all_counts:
        print("\nWARNING: no token counts computed (tokenizers unavailable) -- "
              "the A5 gate could NOT be checked. Re-run with network access before Phase B.")
        return 0

    max_prompt_tokens = max(all_counts)
    budget = 8192 - settings.llm_max_tokens
    print(f"\nMax prompt tokens across bench+quality fixtures: {max_prompt_tokens}")
    print(f"Budget (8192 - llm_max_tokens={settings.llm_max_tokens}): {budget}")
    if max_prompt_tokens > budget:
        print(
            f"GATE FAILED: a fixture prompt ({max_prompt_tokens} tokens) exceeds the "
            f"budget ({budget} tokens). Add history/context trimming before Phase B -- "
            f"do not silently proceed (plan rev 2, Assumption 1)."
        )
        return 1
    print("GATE PASSED: all fixture prompts fit within budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
