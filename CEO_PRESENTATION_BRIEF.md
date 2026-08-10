# IBLOG Adaptive Tutor — CEO Demonstration Brief
**Prepared:** 2026-08-05 · **Status:** Dual-language tutor, LIVE AND DEMO READY · **Read time:** ~10 min

---

## 1. Executive Summary

**Two production-grade AI tutors, language-specific, deployed and running:**

- **🇫🇷 French model** (`iblog-tutor-fr:latest`) — Gemma-2-9B fine-tuned on 1,542 French regulatory rows
- **🇲🇦 Darija model** (`IBLOG_TUTOR:latest`) — Atlas-Chat-9B with LoRA adapter, 3,064 Darija rows
- **Auto language routing** — UI sends language parameter, backend chooses the correct model
- **Demo ready** — Pre-loaded multi-turn conversations showcasing adaptive Socratic teaching

```
┌────────────────────────────────────────────────┐
│       LIVE DUAL-LANGUAGE TUTOR                │
├──────────────┬──────────────┬────────────────┤
│ Language     │ Model        │ Status         │
├──────────────┼──────────────┼────────────────┤
│ 🇫🇷 French   │ iblog-tutor-fr  │ ✅ LIVE     │
│ 🇲🇦 Darija   │ IBLOG_TUTOR     │ ✅ LIVE     │
│ Routing      │ Auto-select     │ ✅ Working  │
└──────────────┴──────────────┴────────────────┘
```

---

## 2. Why Two Models? (The Architecture Decision)

### The Problem We Solved

A single model trained on mixed French/Darija data has to learn *both* languages simultaneously, creating a parameter constraint where "French mode" competes with "Darija mode" for the same weights.

**Real issue caught in testing:** When asked in French about Moroccan law, the base model confidently misclassified Arabic-script regulations as French law. It was never trained on French output for regulatory questions, so it invented wrong answers with high confidence.

### The Solution: Language-Specific Fine-Tuning

Two specialized models, each trained on its language's domain data:

```
DARIJA PATH              FRENCH PATH
    │                        │
    ▼                        ▼
Atlas-Chat-9B          Gemma-2-9B
(450K Darija)          (strong French)
    │                        │
    ▼                        ▼
LoRA fine-tune         Full fine-tune
3,064 rows             1,542 rows
    │                        │
    ▼                        ▼
IBLOG_TUTOR:latest     iblog-tutor-fr:latest
    │                        │
    └──── AUTO ROUTING ──────┘
         (language param)
```

### Why This Architecture Wins

1. **No parameter conflict** — each model focuses on one language, both parameters optimized
2. **Better accuracy** — French model doesn't waste capacity on Darija phonetics
3. **Tenant scalability** — future clients add their own language/domain as a new fine-tune
4. **Same cost** — one model per request, language detected client-side, no double inference

---

## 3. Data Preparation: The Real Story

### The Pipeline
Every tutor needs training data. Ours comes from self-distillation (the base model generates candidates, automated gates filter them):

```
Base model generates → Automated 8-gate check → Accepted / Rejected
candidate Q&A rows       (per-gate thresholds)    training row
```

### The 8 Quality Gates (What Each Does)

| # | Gate | Why It Matters | Threshold |
|---|------|----------------|-----------|
| 1 | **Citation grounding** | Model must never cite law/article absent from source | 0 fabrications allowed |
| 2 | **Arabic-script enforcement** | Darija must be written in Arabic script, never Latin (Arabizi) | ≥90% for Darija |
| 3 | **Code-switch discipline** | French technical terms (les EPI, la procedure) kept as-is, never translated | ≥60% correct |
| 4 | **Socratic progression** | Step-by-step discovery teaching, not answer-dumping | ≥40% multi-turn rows |
| 5 | **Quiz JSON validity** | Generated quizzes must parse, correct answer must be actually justified | 0 invalid |
| 6 | **Script contamination** | Zero CJK, Hangul, or other stray Unicode characters | 0 allowed |
| 7 | **French refusal language** | French refusals stay in French (not leak into Darija) | ≤50% leakage |
| 8 | **Base vs. adapter regression** | Compare fine-tuned model to untouched base — block if adapter *adds* fabrication | 0 regressions |

### It Took 3 Tries (The Honest Part)

Every gap was root-caused to code, not smoothed over with a better prompt:

| Date | Verdict | What Was Wrong |
|---|---|---|
| 07-31 | ❌ Not ready | Citation recall 53% (metric scoping bug + real leak) |
| 08-01 | ❌ Not ready | 2 components had **no citation gate at all** |
| 08-02 | ✅ **GREEN** | Root-caused to missing gates, fixed in code, re-measured |

### Final Results (Both Models)

**Darija dataset: 3,064 rows (2,757 train / 307 eval)**
- Arabic-script output: **100%**
- Code-switch accuracy: **94.4%**
- Socratic multi-turn share: **50.0%**
- Citation recall: **78.9%**
- Fabricated references: **0**
- Script contamination: **0**
- Quiz JSON errors: **0**

**French dataset: 1,542 rows (cleared for fine-tuning)**
- Passed all 8 gates before training
- Same rigor, language-specific thresholds applied

---

## 4. The Two Fine-Tuned Models

### 🇫🇷 French Model: `iblog-tutor-fr:latest`
- **Base:** Gemma-2-9B (strong multilingual instruction-following)
- **Training:** 1,542 French regulatory rows, all 8 gates passed
- **Deployment:** Quantized GGUF (Q4_K_M, 5.4GB), Ollama-served
- **Behavior:** Answers in polished French; cites French law correctly; refuses off-topic questions in French

### 🇲🇦 Darija Model: `IBLOG_TUTOR:latest`
- **Base:** Atlas-Chat-9B (450K Darija examples, native fluency)
- **Training:** 3,064 Darija rows, LoRA adapter (54M params)
- **Deployment:** Merged + quantized (5.8GB), Ollama-served
- **Behavior:** Native Darija fluency; handles Arabizi input; production-grade refusal behavior

---

## 5. Production Bugs Found & Fixed This Week

### Bug 1: Domain Mismatch in Refusals
**Problem:** Blockchain tutor asked off-topic question → says "I'm a safety specialist"

**Fix:** Moved refusal generation to serving layer. Backend creates refusal with *correct* domain before model is called.

**Verified:** All 3 domains tested live, zero cross-domain leakage.

### Bug 2: Quiz Generation Invents Citations
**Problem:** Generated quiz cites article not in retrieved context

**Fix:** Grounding filter checks every quiz. Fabricated citations blocked, real ones pass through.

**Verified:** Live against real documents.

---

## 6. What You'll See in the Demo

### Pre-loaded Multi-Turn Conversations

**Session 1: French Tutor** — 3 turns on workplace safety
- Turn 1: "Why is EPI mandatory?" → Model cites article 281, explains
- Turn 2: "How do I choose equipment?" → Model asks Socratic questions, builds understanding
- Turn 3: "What about chemical hazards?" → Model gives specialized guidance

**Session 2: Darija Tutor** — Same progression, native Darija fluency

### Live Interaction
- Select language → correct model chosen automatically
- Select domain → context loaded
- Send message → grounded, cite-verified response
- Multi-turn conversation maintains prior context

---

## 7. The Architecture: What Changed, What Stayed

**What changed:**
- `app/config.py`: Added `ollama_model_fr` setting
- `app/services/llm.py`: Route French→iblog-tutor-fr, else→IBLOG_TUTOR (3 lines)
- Frontend: Pre-existing language selector now drives model selection

**What stays the same:**
- RAG retrieval pipeline
- Domain/tenant isolation
- All 8 quality gates still enforced
- Socratic teaching methodology
- 242/242 tests passing (220 old + 22 new)

---

## 8. Bottom Line

**Production-grade bilingual tutor, deployed and ready to demo:**

✅ Two fine-tuned models, language-optimized  
✅ 3,064 + 1,542 training rows, all 8 gates passed  
✅ Zero fabricated citations enforced  
✅ Auto language routing (no manual switching)  
✅ Socratic teaching in both languages  
✅ Two production bugs caught and fixed  
✅ Demo sessions pre-loaded and ready  

**Why this scales:** Future clients add language/domain via new fine-tune, not full retrain. Tenant #1 (Moroccan safety, French & Darija) is live and production-ready.
