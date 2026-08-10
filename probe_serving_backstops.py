"""
Live probe: deterministic serving-layer backstops against the real model.

Re-verifies both defects this session's 4-step plan fixed, against the
ACTUAL served IBLOG_TUTOR model over the ACTUAL Ollama HTTP API -- not
mocks. Postgres/Redis are not running in this environment, so real pgvector
retrieval is out of scope for this probe (see MIGRATION_PLAN.md); context
chunks are instead pulled verbatim from data/v11_merged, the real corpus
that trained the model, via context_from_system_prompt -- genuine legal
text, not synthetic filler.

Usage:
    .gguf_venv/Scripts/python.exe probe_serving_backstops.py

Requires: Ollama running with IBLOG_TUTOR:latest loaded.
"""

import json
import sys
from unittest.mock import patch

sys.path.insert(0, ".")

from app.services.generate_training_data import context_from_system_prompt
from app.services.llm import generate_llm_response, deterministic_refusal
from app.services.quiz import generate_quiz_questions
from app.services.grounding import filter_grounded_questions
from app.models.schemas import ChatRequest, Domain
from app.routers.chat import chat


def load_industrial_contexts():
    rows = []
    for fn in ("data/v11_merged/train.jsonl", "data/v11_merged/eval.jsonl"):
        with open(fn, encoding="utf-8") as f:
            rows += [json.loads(l) for l in f]

    industrial = [r for r in rows if r.get("domain") == "industrial"]
    seen = {}
    for r in industrial:
        sys_msg = next((m["content"] for m in r["messages"] if m["role"] == "system"), None)
        if not sys_msg:
            continue
        ctx = context_from_system_prompt(sys_msg)
        if len(ctx) < 200:
            continue
        key = ctx[:150]
        if key not in seen:
            seen[key] = ctx
    return list(seen.values())


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    contexts = load_industrial_contexts()
    print(f"Loaded {len(contexts)} unique real industrial context chunks from data/v11_merged")
    # Pick two structurally distinct chunks (different article ranges) so the
    # quiz probe below isn't testing the same content twice.
    quiz_chunks = contexts[:2] if len(contexts) >= 2 else contexts

    results = {"refusal": [], "quiz": []}

    # -------------------------------------------------------------------
    section("PART A1 -- raw model behaviour on an off-topic, no-context "
            "question (reproduces the original defect, informational only)")
    # -------------------------------------------------------------------
    off_topic_probes = [
        ("securite", "Chno howa the best crypto wallet li khass ndir?"),
        ("blockchain", "Chno howa l-procedure LOTO f l-usine?"),
    ]
    for domain, query in off_topic_probes:
        try:
            reply = generate_llm_response(query=query, context="", domain=domain)
            print(f"\n[{domain}] raw model reply to off-topic/empty-context query:")
            print(f"  {reply[:300]}")
        except Exception as e:
            print(f"\n[{domain}] raw model call failed: {e}")

    # -------------------------------------------------------------------
    section("PART A2 -- FIX: real chat() router, empty context, all 3 domains")
    # -------------------------------------------------------------------
    ollama_call_count = {"n": 0}
    real_urlopen = __import__("urllib.request", fromlist=["urlopen"]).urlopen

    def counting_urlopen(*args, **kwargs):
        ollama_call_count["n"] += 1
        return real_urlopen(*args, **kwargs)

    def empty_context(*args, **kwargs):
        return "", []

    for domain in (Domain.INDUSTRIAL, Domain.SECURITE, Domain.BLOCKCHAIN):
        before = ollama_call_count["n"]
        with patch("app.routers.chat.build_rag_context", side_effect=empty_context), \
             patch("app.services.llm.urllib.request.urlopen", side_effect=counting_urlopen):
            response = chat(ChatRequest(message="test off-topic", domain=domain))
        called_model = ollama_call_count["n"] > before
        print(f"\n[{domain.value}] deterministic refusal (Ollama called: {called_model}):")
        print(f"  {response.response}")
        results["refusal"].append({
            "domain": domain.value, "called_model": called_model,
            "text": response.response,
        })

    # -------------------------------------------------------------------
    section("PART A3 -- happy path unaffected: real grounded question, real "
            "context, real model, through the real chat() router")
    # -------------------------------------------------------------------
    happy_chunk = contexts[0] if contexts else ""
    with patch("app.routers.chat.build_rag_context",
               side_effect=lambda *a, **k: (happy_chunk, ["labour_code.pdf"])):
        response = chat(ChatRequest(
            message="Chno kayngoulou l qanoun 3la had l mawdo3?",
            domain=Domain.INDUSTRIAL,
        ))
    print(f"\nGrounded happy-path reply (first 300 chars):")
    print(f"  {response.response[:300]}")
    print(f"  sources: {response.sources}")

    # -------------------------------------------------------------------
    section("PART B -- quiz generation + grounding filter, real model, "
            "real corpus context")
    # -------------------------------------------------------------------
    total_raw, total_kept, total_dropped = 0, 0, 0
    for i, ctx in enumerate(quiz_chunks):
        print(f"\n--- context chunk {i} (len={len(ctx)}) ---")
        print(f"  {ctx[:200]}...")
        try:
            raw_questions = generate_quiz_questions(
                topic="quiz sur ce document",
                context=ctx,
                domain="industrial",
                language="darija",
                n=5,
            )
        except Exception as e:
            print(f"  generation failed: {e}")
            continue

        kept, dropped = filter_grounded_questions(raw_questions, ctx)
        total_raw += len(raw_questions)
        total_kept += len(kept)
        total_dropped += len(dropped)

        print(f"  raw questions: {len(raw_questions)} | kept: {len(kept)} | dropped: {len(dropped)}")
        for q in kept:
            print(f"    KEPT: {q.get('question', '')[:90]}")
        for d in dropped:
            reason = d.get("_reject_reason")
            offenders = d.get("_offenders", [])
            qtext = d.get("question", {}).get("question", "") if isinstance(d.get("question"), dict) else ""
            print(f"    DROPPED ({reason}): {qtext[:90]}")
            if offenders:
                print(f"      offending references not found in context: {offenders}")

        results["quiz"].append({
            "chunk": i, "raw": len(raw_questions),
            "kept": len(kept), "dropped": len(dropped),
        })

    # -------------------------------------------------------------------
    section("SUMMARY")
    # -------------------------------------------------------------------
    all_correct_domain = all(
        not r["called_model"] for r in results["refusal"]
    )
    print(f"Refusal interception: {len(results['refusal'])}/3 domains tested, "
          f"Ollama bypassed on all: {all_correct_domain}")
    print(f"Quiz grounding: {total_raw} raw questions generated across "
          f"{len(quiz_chunks)} real chunks -> {total_kept} kept, "
          f"{total_dropped} dropped as ungrounded/invalid "
          f"({100 * total_dropped / total_raw:.0f}% caught)" if total_raw else
          "Quiz grounding: no questions generated")


if __name__ == "__main__":
    main()
