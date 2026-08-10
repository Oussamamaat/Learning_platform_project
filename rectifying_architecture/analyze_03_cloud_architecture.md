# Cloud Dual-Path Architecture — IBLOG_SERVICE (self-hosted, both paths fine-tuned)

> **Changed 2026-08-03 — the laptop MVP is a PHASE before this, not a fork.**
> Owner decision: MVP + CEO demo run on the laptop (RTX 4060, 8 GB, Ollama) with **serial**
> one-model-at-a-time routing. Cloud remains the destination; this document becomes **Phase 1**.
>
> | | Phase 0 — Laptop MVP | Phase 1 — Cloud (this doc) |
> |---|---|---|
> | Runtime | Ollama, **serial** (one model resident) | vLLM, **co-resident** on one L4 |
> | French model | **Gemma-2-9B + French LoRA** (GGUF Q4_K_M) | Gemma-2-9B + French LoRA (adapter) |
> | Darija model | `IBLOG_TUTOR` | `IBLOG_TUTOR` (unchanged in both) |
> | Structured output | Ollama `format` | vLLM `guided_json` |
> | Language switch | **~30 s model swap** | free (both resident) |
>
> **Transfers to Phase 0 now:** model matrix logic (Gemma FR / Atlas ARY), the
> same-architecture/same-recipe insight, the guardrail design, the Kaggle T4 LoRA recipe, and the
> schema + RLS *design* (implementation deferred — see ADR 0001's demo-deferred table).
> **Deferred to Phase 1:** L4 cost model, scale-to-zero, business-hours warm scheduling, prefix
> caching, TTFT-under-load targets, PgBouncer.
> **⚠️ Changed below:** §1/§2 assumed vLLM serving a LoRA adapter. On the laptop the French model
> must instead follow the **identical Atlas GGUF path** — see the amendment inside §1.
> **⚠️ Also amended:** §5 step 3's "build the French SFT set" is **confirmed as a near-term item**,
> but **rescoped**: ~**1,800 rows over 8 components**, not the full 3 k 11-component build —
> exploiting the facts that 30 of 36 corpus documents are already French and `citations.py` already
> parses French citation forms. Full specification in **`analyze_05_french_finetune_plan.md`**.
> (An intermediate revision of this note claimed stock Gemma needed no LoRA for the demo; that was
> reversed — the MVP promises *tutoring*, not merely French.)
>
> **New, concrete reason to migrate to Phase 1:** co-residency **eliminates the ~30 s
> language-switch stall** Phase 0 cannot avoid. A user-visible UX win, not just scale.

**Supersedes** the local-hardware plan. History: `analyze_01.md` (challenge report),
`analyze_02_test_results.md` (empirical French-drift test), `analyze_04_corrected_drift.md`
(corrected measurement + laptop-MVP amendment).

---

## Context

Deployment moves to **cloud**; the frontend gains an **explicit language dropdown**. User decisions
this round: **fully self-hosted** (no third-party API for tenant data) and the **French path must
also be fine-tuned** for adaptive-learning tutoring. ADRs: **write 0002 only**, hold 0001 until the
French model is measured.

### ⚠️ I am reversing my previous rejection of dual-model routing
Two of my three objections were environment-specific and are now void (VRAM → cloud; probabilistic
router → UI dropdown). The third — SFT behaviour living only in the Atlas adapter — is resolved by
the user's decision to fine-tune the French path too.

> 🔴 **Correction 2026-08-03 — the supporting number below is RETRACTED.**
> The original text read: *"the measurement now argues for dual-path: `IBLOG_TUTOR` produced French
> 0/12 … 100% French precision is unreachable from a model that emits French 0% of the time."*
> **That 0/12 was a harness artifact** — it bypassed `generate_llm_response()`. Through the real
> serving path `IBLOG_TUTOR` produces **clean French 9/9** with matching context
> (`analyze_04_corrected_drift.md`).
>
> **The dual-path conclusion still holds, but for different and better reasons:** the owner's UI
> dropdown decision, and stock Gemma's measured **12/12 `PASS_FR`** with correct refusal and
> Arabic-context handling. Dual-path is a *product* choice about using the best model per language —
> **not** a rescue of a broken French generator. Nothing downstream in this document depends on the
> retracted figure.

### ⚠️ The dropdown fixes *intent*, not *compliance*
It tells the system which language to use; it does not make a model obey — exactly what the test
disproved. The **post-generation validator stays mandatory**. What the dropdown buys is a *simpler*
validator: expected language is known, so it is an **assertion**, not detection. No fastText anywhere.

---

## 1. The constraint that picks the French model

**Your proven training rig is Kaggle dual-T4 (fp16, no bf16), which has successfully QLoRA'd a 9B.**
A 24B QLoRA does not fit that rig. So Mistral Small 24B is out — not on quality, on trainability.
The French base must be **≤14B, ideally 9B-class**.

### Recommendation: **Gemma-2-9B** for the French path

| Reason | Detail |
|---|---|
| **Same architecture as Atlas-Chat** | Atlas-Chat-9B *is* Gemma-2-9B + Darija tuning. Identical tokenizer, chat template, vLLM config, and **your exact proven LoRA recipe** (r=16/α=16, 7 linear targets, fp16, custom template merge) transfers unchanged. |
| **One rig, one recipe, two adapters** | No second training pipeline, no second serving config, no second set of template-parity bugs. |
| **Both fit one GPU** | 2 × 9B at 4-bit ≈ 12 GB → one **L4 24 GB** holds both with KV headroom. |
| **It doubles as the decisive experiment** | `analyze_02.md` already named "test stock `gemma2:9b` on French" as the highest-information measurement. That test now *also* selects the French base. One experiment, two answers. |
| License | Gemma Terms of Use — commercial use permitted. |

**Alternatives if Gemma-2-9B underperforms on French:**
- **Qwen2.5-14B** (Apache 2.0) — likely the quality ceiling of the three; QLoRA on T4 is tight but feasible.
- **Mistral-7B-Instruct-v0.3** (Apache 2.0) — French vendor, smallest training cost, different arch.
- ⚠️ **Avoid Ministral-8B** — Mistral Research License; commercial use requires a separate paid
  licence. A licensing trap for a B2B product.

> ✅ **Resolved 2026-08-03:** Gemma-2-9B did **not** underperform — 12/12 `PASS_FR` zero-shot
> (`analyze_04_corrected_drift.md`). The alternatives above are retained for the record but are not
> live options; the "decisive experiment" row in the table above has now been run and returned in
> Gemma's favour.

### ⚠️ AMENDED 2026-08-03 — export path on the laptop (Phase 0)

The table above says the LoRA recipe "transfers unchanged," which is true — but this section
assumed **vLLM serving a LoRA adapter against an fp16 base**. Phase 0 runs Ollama, so the French
model must follow the **identical path Atlas already took**:

```
LoRA (Kaggle T4: r=16/α=16, 7 linear targets, fp16)
  → merge to fp16  (~18 GB transient)
  → convert to GGUF
  → quantize Q4_K_M  (~5.4 GB)
  → ollama create  (Modelfile)
```

**Reuse win:** because Gemma-2-9B and Atlas-Chat-9B are the same architecture, the existing
convert/quantize toolchain and Modelfile conventions work unchanged — no second export pipeline.

**Two gotchas:**
- ⚠️ **Chat-template parity.** Reuse the same `.System`-folded-into-the-user-turn convention the
  Atlas Modelfiles use (`FINETUNE_AND_DEPLOY.md §5.2`). Gemma-2 has no native system role, and this
  project has already lost time to a template mismatch — train and serve must be byte-identical.
- ⚠️ **Transient disk: ~18–25 GB** for the merged fp16 stage. Note the Atlas fp16 export was
  previously deleted to reclaim space, so this room must be made again before starting.

**Phase 1 (vLLM) keeps the adapter-based path** described above — the GGUF merge is a Phase 0
requirement, not a permanent change.

## 2. Model matrix

| Path | Model | Serving |
|---|---|---|
| **French** chat / Socratic / RAG | **Gemma-2-9B + new French LoRA** (pending §5 test) | vLLM, self-hosted |
| **Darija** chat / Socratic | **`IBLOG_TUTOR`** (Atlas-Chat-9B + existing LoRA) | vLLM, self-hosted, same GPU |
| **Quiz, both languages** | Same models + vLLM `guided_json` | Batched, offline |
| Embeddings | **BGE-M3** or multilingual-e5-large | CPU-viable |

**No alternative supplier exists for Darija.** Atlas-Chat (2B/9B/27B) is still the only Darija
instruction-tuned family; AtlasIA's 2025–26 **Terjman v2** releases (77M–3.3B) are **translation-only**.
Qwen/Llama as a Darija base stays rejected (0.92 vs 28.08 DODa BLEU); P5 confirms Darija works today.

## 3. Cost — self-hosted, and cheaper than expected

Because the French model is 9B rather than 24B, **both models share one L4** — so self-hosting
French costs almost nothing beyond the GPU you already need for Darija.

| Item | Monthly |
|---|---|
| 1 × L4 24 GB, warm business hours (11 h × 22 d = 242 h) @ $0.44–0.80/hr | **$107–194** |
| Postgres + Redis (managed, small) | ~$40–70 |
| **Total** | **≈ $150–265 / mo** |

24/7 warm instead: ~$320–580/mo GPU. Off-hours cold start on a 9B is ~30–60 s — acceptable outside
business hours, not during. (Measured locally: 34–38 s cold vs 5–9 s warm.)

## 4. Latency — honest numbers
- Warm 9B on L4 with ~2 k tokens of RAG prefill: **~0.6–1.5 s TTFT**. Sub-500 ms is *not* reliable.
- Mitigations, biggest first: **vLLM automatic prefix caching** (the system prompt and repeated
  tenant chunks are identical across requests), retrieve top-3 not top-5, stream immediately.
  Realistic target **~400–800 ms**.
- Do not promise sub-500 ms in a contract until measured under load.

## 5. Work plan

1. **Measure Gemma-2-9B on French** — extend `scratchpad/fr_drift_test.py` with `gemma2:9b`, same 5
   prompts, same deterministic scoring. **Gate: ≥95% `PASS_FR` on P1–P4.** This selects the French
   base *and* settles the open question from `analyze_02.md`. **Nothing else is decided first.**
2. **Ship the language invariant** (path-agnostic, independent of step 1): `ui_lang` wired end-to-end,
   output validator, deterministic templated refusal on retrieval miss.
3. **Build the French SFT set.** ⚠️ Honest scope: this is a real workstream, not a sprint — the Darija
   set took months. **Reuse, don't rebuild**: `generate_training_data.py` already has French-required
   component gating (`:50`, `:123`, `:132`) and the ~10-gate reject-and-retry architecture. Extend the
   existing 11 components to emit French rows; keep the corpus ceiling discipline (~3 k rows).
4. **Train the French LoRA** on the same Kaggle T4 recipe. Base-vs-adapter comparison is a standing
   requirement — run it.
5. **Stand up the dual vLLM node**: both 9B models 4-bit on one L4, `--gpu-memory-utilization ~0.45`
   each, prefix caching on, business-hours warm schedule. Re-run P5 for Darija regression.
6. **RLS hardening** — see §6.
7. **Quiz path**: `guided_json` both sides + `_explanation_supports_answer()` as cache-admission gate.
8. **Confirm the embedder is multilingual** (open gap — blocks French RAG quality).

## 6. Guardrails detail

**JWT → RLS:**
- Verify JWT **signature** at the gateway; take `tenant_id` from the **claim**, never a header.
- `BEGIN; SELECT set_config('app.tenant_id', $1, true); … COMMIT;` — the `true` makes it txn-local.
- `POLICY USING (tenant_id = current_setting('app.tenant_id')::uuid)`
- ⚠️ **`ALTER TABLE knowledge_chunks FORCE ROW LEVEL SECURITY`** — the table **owner bypasses RLS**
  by default. Without this the policy is decorative.
- App role must not be owner/superuser.
- ⚠️ **PgBouncer: transaction pooling only.** Session pooling leaks `SET` state across clients.
- Per-tenant **partial HNSW index** — a plain `WHERE tenant_id` post-filters and can silently return
  fewer than `LIMIT` rows for small tenants.

**Validator placement:** in the gateway, after generation, before the client. One implementation,
both paths, driven by `ui_lang`:
- `fr` → Arabic chars **outside citation spans** must be 0 (French answers legitimately quote
  `المادة 12` — a naive "any Arabic = fail" rejects correct answers)
- `ary` → Arabic-script dominance ≥ 0.6 (French technical terms are the feature, not a defect)
- streaming: check first ~20 tokens → abort + fallback; **terminal deterministic fallback** is what
  makes this an invariant rather than a high pass-rate
- emit `validator_fire_rate{path}` — on `fr` it should approach 0; a rise is the regression alarm

**Reuse:** script detection `app/services/citations.py:262`, citation spans `citations.py:126`,
Darija scoring `generate_training_data.py:709`, script/CJK `:1677`. Currently generation-time only —
**extract into one shared module called by both the pipeline and the serving path.**

**Quiz across paths:** precomputed, so latency-insensitive. `_explanation_supports_answer()` is
path-agnostic Python over parsed JSON — **one implementation, cache-admission gate on both paths**.
It, not constrained decoding, is the fix for the measured 16% answer-key defect. Cold path on cache
miss = deterministic Postgres question-bank query, no model call.

## 7. ADRs

- ✅ **Write now:** `adr/0002-ui-driven-language-routing.md` — `ui_lang` as authoritative routing
  input; records explicitly that the dropdown fixes intent **not** compliance, and that the output
  validator is therefore non-optional. Independent of which French model wins.
- ⏸ **Hold:** `adr/0001-cloud-dual-model-architecture.md` — blocked on step 1. Writing it now would
  name a French model on reputation rather than measurement, which is the exact error that produced
  the (since-retracted) 0/12 result.
  > ✅ **Resolved 2026-08-03:** step 1 ran, and **both** ADRs are now written. ADR 0001 exists in
  > amended form covering laptop MVP → cloud. The discipline this bullet argued for held: the base
  > model was named only after measurement.

## Verification

- Re-run `fr_drift_test.py` per path. **Gate: `arabic_outside_citations` = 0 on 100% of French turns**
  (enforceable — terminal fallback is deterministic). P5 Darija must not regress.
- Cross-tenant RLS test: two tenants, assert tenant B's rows are unreachable **with the `WHERE`
  clause deliberately removed** — RLS must fail closed.
- TTFT p50/p95 per path, warm and cold, under load.
- `.gguf_venv/Scripts/python.exe -m pytest` green after the shared-module extraction.

**Sources:** [Atlas-Chat paper](https://arxiv.org/abs/2409.17912) ·
[AtlasIA Terjman v2](https://www.middleeastainews.com/p/atlasia-releases-smarter-moroccan-llms) ·
[L4 pricing](https://getdeploying.com/gpus/nvidia-l4) ·
[Mistral pricing](https://pricepertoken.com/pricing-page/model/mistral-ai-mistral-small)
