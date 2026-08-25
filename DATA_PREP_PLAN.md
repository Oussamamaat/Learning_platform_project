# IBLOG AI Assistant — Complete Data Prep & Fine-Tuning Plan

**Date:** July 26, 2026
**Deadline:** August 30, 2026 (5 weeks)
**Status:** Data generation ready to start

---

## 1. Project Overview

Enterprise AI e-learning assistant for iBlog Services with:
- **RAG** (pgvector) for tenant-specific document retrieval
- **LoRA fine-tuned** LLM for persona, tone, and Darija/French code-switching
- **Three domains:** Industrial safety, Physical security (sécurité physique), Blockchain compliance
- **Languages:** Moroccan Darija (Arabizi) + French + Mixed

---

## 2. Hardware

| Component | Spec |
|-----------|------|
| GPU | **NVIDIA RTX 4060** (8GB VRAM) |
| Ollama | v0.32.4, local server on port 11434 |
| Available models | `qwen2.5-coder:latest` (7.6B, Q4_K_M), `llama3.1:latest` (8B, Q4_K_M) |
| Training platform | **Kaggle** (free T4 GPU, 16GB VRAM) |

---

## 3. Model Strategy

### 3.1 Data Generation Model
- **Model:** `qwen2.5-coder:latest` (7.6B) via local Ollama
- **Why:** Already pulled, instruction-tuned, can output structured JSON
- **Speed:** ~2-3 seconds/row on RTX 4060
- **Total time for 7,500 rows:** ~6 hours (overnight feasible)

### 3.2 Fine-Tuning Base Model
- **Primary:** `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` (Kaggle)
- **Fallback:** `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
- **Why Qwen2.5 over Llama:** Better multilingual support, smaller footprint, same quality

### 3.3 LoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `r` (rank) | 32 | Balance capacity vs VRAM |
| `alpha` | 32 | Scaling factor = 1x (alpha/r) |
| `target_modules` | all linear layers | Maximum adaptation |
| `dropout` | 0.05 | Prevent overfitting |
| `bias` | none | Standard for LoRA |
| `epochs` | 3 | Enough to learn patterns, not overfit |
| `learning_rate` | 2e-4 | Standard for Qwen2.5 LoRA |
| `lr_scheduler` | cosine | Smooth convergence |
| `warmup_ratio` | 0.05 | 5% warmup steps |
| `max_seq_length` | 2048 | Covers multi-turn conversations |
| `batch_size` | 4 | Fits in T4 16GB VRAM |
| `gradient_accumulation` | 4 | Effective batch = 16 |
| `fp16` | true | T4 supports FP16 |
| `quantization` | 4-bit NF4 | VRAM efficiency |

### 3.4 Export Configuration

| Parameter | Value |
|-----------|-------|
| Format | GGUF |
| Quantization | `q4_k_m` |
| Ollama model name | `iblog-finetuned:latest` |

---

## 4. Dataset Specification

### 4.1 Composition

| Component | Rows | Purpose | Source |
|-----------|------|---------|--------|
| 1. Socratic Pedagogy | 2,500 | Tutor asks follow-ups, guides discovery | Generated via Ollama |
| 2. Code-Switching Fluency | 2,500 | Natural French/Darija mixing | Generated via Ollama |
| 3. Grounded Context + Refusal | 2,000 | Answer from context, refuse when insufficient | Generated via Ollama + raw/ corpus |
| 4. General Capability Preservation | 500 | Prevent catastrophic forgetting | Filter from Darija-SFT-Mixture |
| **TOTAL** | **7,500** | | |

### 4.2 Language Split

| Language | % | Rows |
|----------|---|------|
| Darija Arabizi | 40% | 3,000 |
| French | 30% | 2,250 |
| Mixed (French + Darija) | 30% | 2,250 |

### 4.3 Turn Format

| Format | % | Rows |
|--------|---|------|
| Single-turn (Q → A) | 40% | 3,000 |
| Multi-turn (3-5 turns) | 60% | 4,500 |

### 4.4 Train/Eval Split

| Set | Rows | Purpose |
|-----|------|---------|
| Train | ~6,750 (90%) | LoRA training |
| Eval | ~750 (10%) | Loss monitoring, early stopping |

---

## 5. Orthography & Language Rules (LOCKED IN)

### 5.1 Arabizi Numeral Mapping

| Numeral | Arabic | Example |
|---------|--------|---------|
| 2 | ء / أ | so2al |
| 3 | ع | 3afak, m3a |
| 5 | خ | khass (prefer kh, allow 5) |
| 7 | ح | 7ta |
| 8 | غ | 8ir (prefer gh, allow 8) |
| 9 | ق | 9bl |

### 5.2 Code-Switching Rules

1. **Technical nouns → French** (with Darija article `l-`, `d-`, `f-`)
   - `l-casque de sécurité`, `la verification`, `les equipements`
2. **Complete French clauses** allowed for regulatory citations
3. **Conversational anchor → Darija** (connectors, questions, Socratic prompts)
4. **Random word alternation → BANNED** (switch at phrase boundaries only)

### 5.3 Script Rules

- **Arabizi only** in training data (Latin + numerals)
- No Arabic script (كيفاش → kifach)
- Formal capitalization and punctuation
- Acronyms stay uppercase (LOTO, EPI, ATEX, HSE, ISO)

---

## 6. Generation Pipeline

### 6.1 Foundation Files (COMPLETE)

| File | Status | Content |
|------|--------|---------|
| `data/ortho_guide.md` | DONE | Numeral table, 42 domain terms, capitalization rules |
| `data/code_switching_rules.md` | DONE | 4 syntax rules, anti-pattern, 4 good/bad pairs |
| `data/refusal_templates.md` | DONE | 10 templates (direct, warm, partial, mixed) |
| `data/few_shot_examples.md` | DONE | 5 ChatML examples (3 multi-turn, 1 refusal, 1 grounded) |

### 6.2 Raw Corpus (COMPLETE)

| Domain | Files | Content |
|--------|-------|---------|
| Industrial | 10 | Code du travail, CNSS, ISO 45001, LOTO, PPE, etc. |
| Sécurité physique | 6 | Loi 27-06, Loi 032-26, guarding, incident reporting |
| Blockchain | 6 | Loi 42-25, FATF AML, BAM/AMMC, smart contracts |
| **Total** | **22** | All in `raw/` directory |

### 6.3 Script

- **File:** `app/services/generate_training_data.py`
- **Usage:** `python -m app.services.generate_training_data --model qwen2.5-coder:latest --data-dir data/ --raw-dir raw/`
- **Output:** `data/training/train.jsonl` + `eval.jsonl` + `component_stats.json`

### 6.4 Row Generation Process

```
For each row:
1. Pick random component (weighted by target counts)
2. Pick random domain (industrial, securite, blockchain)
3. Build prompt with:
   - Orthography rules (injected)
   - Code-switching rules (injected)
   - Few-shot examples (injected)
   - For Component 3: random raw/ document as context
4. Call Ollama API (qwen2.5-coder, temperature=0.7, max_tokens=512)
5. Parse JSON response
6. Validate ChatML structure
7. Apply orthography normalization
8. Add to dataset

Post-processing:
9. Deduplication (cosine similarity > 0.95)
10. Split 90/10 train/eval per component
11. Export JSONL
```

---

## 7. Contamination Prevention

- Component 3 uses **real `raw/` corpus** documents as context (not fictional)
- Unanswerable rows: real context + question the context CAN'T answer
- No overlap between raw/ training context and actual tenant data at inference

---

## 8. Quality Gates

### 8.1 Row-Level Checklist

- [ ] Follows orthography guide (numerals, capitalization)
- [ ] Code-switching follows rules (technical=French, verbs=Darija)
- [ ] Assistant turn length 50-200 tokens
- [ ] If context provided: answer uses ONLY context
- [ ] If refusal: distinct from last 5 refusal rows
- [ ] Not a near-duplicate (cosine < 0.95)
- [ ] System prompt matches fixed template

### 8.2 Native-Speaker QA

- Oussama reviews 15% sample (~1,125 rows)
- Focus: Darija naturalness, code-switching quality, refusal appropriateness
- Reject/fix rows that don't pass

### 8.3 Evaluation Metrics

| Eval Set | Rows | Metric |
|----------|------|--------|
| Socratic | 200 | Human rubric: Socratic-ness, tone, language |
| Code-Switching | 200 | Native-speaker naturalness (1-3 scale) |
| Grounded/Refusal | 200 | Zero-hallucination + refusal diversity |

---

## 9. Licensing (LOCKED IN)

| Asset | License | Usage |
|-------|---------|-------|
| Atlas-Chat-9B | Apache 2.0 | Generator model (no restrictions) |
| Darija-SFT-Mixture | CC BY-NC 4.0 | Training input only (not redistributed) |
| facat/Socratic | Research-only | SKIPPED |
| Multilingual Alpaca | Research-only | SKIPPED |
| Our training data | Proprietary | iBlog Services internal use |

---

## 10. Production Audit (July 26, 2026)

**Score: 52/100, Risky**

### What's Good
- ✅ Parameterized SQL (no injection)
- ✅ Multi-tenant isolation via tenant_id
- ✅ Typed error hierarchy + global exception handler
- ✅ Pydantic input validation
- ✅ Health check endpoint
- ✅ Docker with healthchecks
- ✅ Proper logging (no print statements)

### Blockers (fix before deploy, not before training)
- ❌ No authentication on endpoints
- ❌ CORS allow_origins=["*"]
- ❌ No rate limiting
- ❌ No env var validation at startup
- ❌ No test suite

---

## 11. Execution Timeline

| Date | Task | Status |
|------|------|--------|
| **Jul 26 (Sat)** | Foundation files, raw corpus, generation script | DONE |
| **Jul 26 (Sat)** | Production audit, error handling, verification | DONE |
| **Jul 26 (Sat) ~22:00** | Start data generation (7,500 rows, ~6 hours) | NEXT |
| **Jul 27 (Sun) ~04:00** | Generation complete, review sample | PENDING |
| **Jul 27 (Sun)** | Upload to Kaggle, run LoRA training (~1-2 hours) | PENDING |
| **Jul 27 (Sun)** | Export GGUF, test in Ollama | PENDING |
| **Jul 28 (Mon)** | Show working model to CEO | PENDING |
| **Jul 28 - Aug 3** | Week 2: Backend completion, auth, rate limiting | PENDING |
| **Aug 4 - Aug 10** | Week 3: Frontend development | PENDING |
| **Aug 11 - Aug 17** | Week 4: Integration testing | PENDING |
| **Aug 18 - Aug 24** | Week 5: Polish, security hardening | PENDING |
| **Aug 25 - Aug 30** | Final delivery | PENDING |

---

## 12. Files Inventory

### Application Code
```
app/
├── main.py                    # FastAPI app, CORS, error handlers
├── config.py                  # Settings (DATABASE_URL, OLLAMA_*, REDIS_*)
├── errors.py                  # Typed error hierarchy
├── models/
│   ├── schemas.py             # Pydantic models (ChatRequest, Domain enum, etc.)
│   └── db_init.py             # Database initialization
├── routers/
│   ├── chat.py                # /api/v1/chat — RAG + LLM
│   ├── audio.py               # /api/v1/audio — Whisper placeholder
│   └── quiz.py                # /api/v1/quiz — Quiz generation placeholder
└── services/
    ├── llm.py                 # Ollama client (multi-domain, typed errors)
    ├── search.py              # Semantic search via pgvector
    ├── ingestion.py           # Document ingestion + chunking
    ├── generate_training_data.py  # Dataset generation script
    ├── generate_corpus.py     # Raw corpus generator (21 files)
    ├── generate_corpus_index.py   # CORPUS_INDEX.csv tracker
    ├── generate_data.py       # Old road safety generator (deprecated)
    └── export_openapi.py      # OpenAPI schema export
```

### Data Files
```
data/
├── ortho_guide.md             # Orthography rules + 42 domain terms
├── code_switching_rules.md    # 4 rules + good/bad examples
├── refusal_templates.md       # 10 refusal templates
└── few_shot_examples.md       # 5 ChatML few-shot examples

raw/
├── shared/industrial/         # 8 files (labor law, ISO, LOTO, PPE...)
├── shared/securite_physique/  # 4 files (Loi 27-06, guarding...)
├── shared/blockchain/         # 6 files (Loi 42-25, FATF, BAM...)
├── tenant_placeholder/        # 2 synthetic tenant files
└── CORPUS_INDEX.csv           # Metadata tracker
```

### Infrastructure
```
config/docker-compose.yml      # pgvector + Ollama + FastAPI
Dockerfile                     # Python 3.11-slim + uvicorn
requirements.txt               # Dependencies
```

---

## 13. What's Next (Immediate)

1. **Start data generation** — run overnight on RTX 4060
2. **Morning QA** — review 15% sample for quality
3. **Upload to Kaggle** — train LoRA on free T4 GPU
4. **Export + test** — GGUF → Ollama → verify output quality
5. **Show CEO** — working model by Monday
