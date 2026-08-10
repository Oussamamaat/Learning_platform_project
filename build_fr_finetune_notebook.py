"""
One-off generator for kaggle_finetune_fr_v1.ipynb from kaggle_finetune_v11.ipynb.

Not a reusable tool -- run once, inspect the output, delete or keep for the
record. Cell-by-cell diffs are commented at each edit site so the rationale
survives even if this script doesn't get run again.
"""
import copy
import json
from pathlib import Path

SRC = json.loads(Path("kaggle_finetune_v11.ipynb").read_text(encoding="utf-8"))
nb = copy.deepcopy(SRC)
cells = nb["cells"]


def set_source(idx, text):
    cells[idx]["source"] = text.splitlines(keepends=True)


def insert_cells(after_idx, new_cells):
    for i, c in enumerate(new_cells):
        cells.insert(after_idx + 1 + i, c)


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


# ---------------------------------------------------------------------------
# Cell 0: title / intro. New dataset stats, new model, note on base-vs-
# adapter being built in (LESSONS_LEARNED.md #1 -- non-negotiable per
# MIGRATION_PLAN.md P9).
# ---------------------------------------------------------------------------
set_source(0, """# French Enterprise Tutor — LoRA Fine-Tune of Gemma-2-9B (Unsloth)

Trains one LoRA adapter on the **1,542-row** `fr_v3_merged` dataset (French
path -- see `docs/architecture/rectified/analyze_05_french_finetune_plan.md`)
and exports it as a LoRA adapter (primary) plus a Q4_K_M GGUF for Ollama
(secondary). Read this cell before running anything.

**This notebook is adapted from `kaggle_finetune_v11.ipynb`** (the Darija /
Atlas-Chat-9B fine-tune). Sections 1, 4, 5, 6, 7 are reused unmodified where
the underlying mechanism is language-agnostic or Gemma-2-native (chat
template, masking, trainer, LoRA config) — see the per-section notes below
for exactly what changed and why.

## Run this with "Save & Run All (Commit)", NOT interactively

Same rule as the generation runs: an interactive session has a 60-minute idle
timeout that kills every process in the container. A committed run is
headless, has no idle timeout, and persists `/kaggle/working` as version
output.

**Settings:** Accelerator: **GPU T4 x2** (or P100) · Internet: **On** ·
Persistence: Variables and Files.

**Dataset upload must contain:** `fr_v3_merged/` (with `train.jsonl` +
`eval.jsonl`) and `app/` (used for the train/serve prompt parity assertion
against `SYSTEM_PROMPT_TEMPLATE_FR` — the run still works without it, with
that check skipped).

---

## What changed from the Darija notebook, and why

### Base model: `unsloth/gemma-2-9b`, not Atlas-Chat-9B

Stock Gemma-2-9B, not a Darija fine-tune of it — this is the French path's
own base per `analyze_05_french_finetune_plan.md` §"LoRA recipe": "Gemma-2-9B
is Atlas-Chat's base" — same architecture, same softcapping, same chat
template mechanics, different (absent) prior fine-tuning. `unsloth/gemma-2-9b`
is Unsloth's own un-gated mirror (confirmed 2026-08-04: no license-gate
requiring `HF_TOKEN`), matching the same "no token needed" constraint the
Atlas-Chat notebook relied on.

### Chat template (Sections 4-5): reused BYTE-IDENTICAL, not re-derived

`LESSONS_LEARNED.md` #4 / `MIGRATION_PLAN.md` P9: "reuse its chat-template +
byte-parity assertion cell UNMODIFIED — Gemma-2 has no system-role turn,
already solved there." Atlas-Chat-9B IS a Gemma-2-9B fine-tune, so its
system-merge-into-first-user-turn template, `<start_of_turn>`/`<end_of_turn>`
tokens, and the parity-proof mechanism apply to stock Gemma-2-9B without
modification. Do not touch Sections 4-5 without re-deriving the reason first.

### Dataset: `fr_v3_merged` (1,542 rows: 1,387 train / 155 eval), 8 components

`code_switching`, `darija_preservation`, `reasoning_preservation` are absent
by design (`FRENCH_COMPONENT_CONFIG` — no French analogue, or deliberately
dropped, see `analyze_05` §1 "Risk 1"). Train/serve parity check now compares
against `SYSTEM_PROMPT_TEMPLATE_FR`, not the Darija template.

### Script-gate direction is INVERTED for the generation smoke test

Darija's smoke test checks for Arabic-script *presence*. French's checks for
Arabic-script *absence outside citation spans* —
`has_arabic_outside_citations()` (`app/services/generate_training_data.py`),
the same function this session's audit used to verify F1's fix, reused here
against the fine-tuned model's own generations. This directly implements
`analyze_05` §4's acceptance gate: "`arabic_outside_citations` = 0 on 100% of
French turns."

### NEW — Section 10.5: base-vs-adapter comparison, built into this run

Not an afterthought. `docs/LESSONS_LEARNED.md` #1: "the exact discipline that
caught the original Darija citation-fabrication defect" — the trained
adapter must not fall below stock Gemma's own grounding floor on the same
prompts. Uses `model.disable_adapter()` (no second model load — same PEFT
wrapper, LoRA off vs. on) to generate from both on real grounded eval rows,
then checks for the specific worst-failure-mode pattern: the adapter citing
a reference number the source context does not contain, on a prompt where
the un-adapted base model did not fabricate one. See Section 10.5 below for
what it does and does not assert.

### Hyperparameters: UNCHANGED

`MIGRATION_PLAN.md` P9: "Do not change the LoRA hyperparameters without
cause." r=16/alpha=16/dropout=0, same 7 target modules, 2 epochs, batch 1 x
accum 16, lr 2e-4 cosine 3% warmup, `adamw_8bit`. See the original notebook's
own hyperparameter table for the reasoning — none of it was Darija-specific.

---

## Fail-fast order (unchanged from the Darija notebook)

| § | Step | Cost |
|---|---|---|
| 1 | env + single-GPU pin + install | ~5 min |
| 2 | dataset located, counted, structurally validated, parrot rows dropped | ~10 s |
| 3 | model download + 4-bit quantise | ~15-20 min (9B, smaller download than Atlas's bf16-only repo) |
| 4 | chat template installed + **parity assertion** | ~10 s |
| 5 | tokenise, length report, overlong rows dropped | ~1 min |
| 6 | LoRA attached, masking **verified and asserted** | ~1 min |
| 7 | trainer built + one-batch finite-loss smoke test | ~2 min |
| 8 | train | **~1-1.5 h** (1,387 rows vs. Darija's 2,757 — roughly half the steps) |
| 9 | per-component eval loss | ~3 min |
| 10 | generation smoke test | ~3 min |
| 10.5 | **base-vs-adapter comparison** | ~3 min |
| 11 | save, patch, package, GGUF | ~45 min |

**Expected wall clock ≈ 3-3.5 h.** `TRAIN_TIME_BUDGET_H` in §7 hard-stops
training so §11 is always reached.
""")

# ---------------------------------------------------------------------------
# Cell 7: dataset discovery marker + DATA_DIR.
# ---------------------------------------------------------------------------
src7 = "".join(cells[7]["source"])
src7 = src7.replace('marker="v11_merged/train.jsonl"', 'marker="fr_v3_merged/train.jsonl"')
src7 = src7.replace('DATA_DIR = SRC / "v11_merged"', 'DATA_DIR = SRC / "fr_v3_merged"')
set_source(7, src7)

# ---------------------------------------------------------------------------
# Cell 9 (parrot filter): mechanism is language-agnostic, keep as-is except
# the docstring's Darija-specific provenance claim (0.34%/9-row measurement
# was against v11_merged, not this dataset) -- re-scope the comment, not the
# logic, and don't assume the same rate holds.
# ---------------------------------------------------------------------------
src9 = "".join(cells[9]["source"])
src9 = src9.replace(
    "# Parrot filter -- standing safety net, not a fix for a currently-known\n"
    "# defect. QUALITY_FLAGS.md section 7's \"few-shot parroting\" failure mode\n"
    "# (assistant echoes a chunk of the user's turn verbatim before answering)\n"
    "# hit 9 rows (0.34%) in the dataset this notebook was originally written\n"
    "# against; v11_merged already measures at 2/3,064 (0.07%), well under the\n"
    "# 5% kill line below. Kept as a regression guard, not an active cleanup.\n"
    "# The threshold is deliberately conservative: 50 consecutive characters is\n"
    "# far beyond coincidental overlap in Darija.",
    "# Parrot filter -- standing safety net carried over from the Darija\n"
    "# notebook (QUALITY_FLAGS.md section 7's \"few-shot parroting\" failure\n"
    "# mode: assistant echoes a chunk of the user's turn verbatim before\n"
    "# answering). Not measured against fr_v3_merged before this run -- this\n"
    "# cell IS that measurement, not a re-statement of a known rate. The\n"
    "# threshold is deliberately conservative: 50 consecutive characters is\n"
    "# far beyond coincidental overlap in French too.",
)
set_source(9, src9)

# ---------------------------------------------------------------------------
# Cell 10: train/serve parity against the FRENCH templates.
# ---------------------------------------------------------------------------
set_source(10, '''# Train/serve prompt parity. The system prompts baked into this dataset must be
# renderings of the same template llm.py sends in production for French
# queries; the project has held PRODUCTION_SYSTEM_PROMPT_TEMPLATE_FR ==
# SYSTEM_PROMPT_TEMPLATE_FR as an invariant since this session added the
# regression test (tests/test_generation_gates.py). This is where a silent
# drift would surface before six hours of training on a stale prompt.
import sys
sys.path.insert(0, "/kaggle/working")
PARITY_CHECKED = False
if HAVE_APP:
    try:
        from app.services.generate_training_data import PRODUCTION_SYSTEM_PROMPT_TEMPLATE_FR
        from app.services.llm import SYSTEM_PROMPT_TEMPLATE_FR
    except Exception as e:
        print(f"parity check SKIPPED ({type(e).__name__}: {e})")
    else:
        assert PRODUCTION_SYSTEM_PROMPT_TEMPLATE_FR == SYSTEM_PROMPT_TEMPLATE_FR, (
            "generate_training_data.PRODUCTION_SYSTEM_PROMPT_TEMPLATE_FR has "
            "drifted from llm.SYSTEM_PROMPT_TEMPLATE_FR — the dataset was built "
            "with one template and production serves the other."
        )
        head = PRODUCTION_SYSTEM_PROMPT_TEMPLATE_FR.split("{domain}")[0]
        bad = [r for r in all_rows if not r["messages"][0]["content"].startswith(head)]
        assert not bad, f"{len(bad)} rows carry a system prompt from a different template"
        PARITY_CHECKED = True
        print("prompt parity OK: dataset system prompts match the production")
        print("French template, and generate_training_data == llm.py")
else:
    print("parity check SKIPPED (app/ not uploaded)")
''')

# ---------------------------------------------------------------------------
# Cell 11 (markdown, Section 3 intro): base model name + gating note.
# ---------------------------------------------------------------------------
set_source(11, """## Section 3 — Load Gemma-2-9B in 4-bit

`unsloth/gemma-2-9b` is Unsloth's own un-gated mirror of stock Gemma-2-9B —
confirmed 2026-08-04 not to require a license-gate `HF_TOKEN` (unlike
`google/gemma-2-9b` on the canonical repo), matching the same "no token
needed" property the Atlas-Chat-9B base relied on. Unsloth publishes a
pre-quantized 4-bit bnb variant for this model, so the ~18.5GB bf16 download
this section historically needed for Atlas-Chat may be smaller here.

**This cell takes ~15-20 minutes.**

`dtype=None` lets Unsloth pick fp16 on T4/P100 and bf16 on Ampere+. Gemma-2's
`attn_logit_softcapping=50` / `final_logit_softcapping=30` are why this needs
Unsloth on a T4 at all: softcapping is incompatible with FlashAttention-2, and
naive fp16 softcapping overflows. Identical constraint to Atlas-Chat, since
Atlas-Chat inherits this from being a Gemma-2-9B fine-tune itself.
""")

# ---------------------------------------------------------------------------
# Cell 12: base model swap.
# ---------------------------------------------------------------------------
src12 = "".join(cells[12]["source"])
src12 = src12.replace(
    'BASE_MODEL = "MBZUAI-Paris/Atlas-Chat-9B"',
    'BASE_MODEL = "unsloth/gemma-2-9b"',
)
set_source(12, src12)

# ---------------------------------------------------------------------------
# Cells 13-15 (chat template + parity proof): UNCHANGED per LESSONS_LEARNED
# #4 / MIGRATION_PLAN.md P9 -- Gemma-2 native, not Atlas-specific. Left as-is.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cell 21 (LoRA config) and cell 24 (training args): UNCHANGED per
# MIGRATION_PLAN.md P9 ("do not change without cause"). Left as-is.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cell 28 (markdown, Section 9 intro): update component list description.
# ---------------------------------------------------------------------------
set_source(28, """## Section 9 — Per-component eval loss

Aggregate eval loss hides the thing that matters most here: this is **one**
adapter carrying eight different behaviours (`FRENCH_COMPONENT_CONFIG`).
Two of them — `structured_explanation` and `quiz_generation` — are known to
sit near a corpus-size dedup ceiling (`data/fr_v3_merged` has them at ~150
rows each against a 300-row design target, a decision accepted this session
rather than chased further) -- watch this table for whether that shortfall
shows up as measurably worse per-token loss, not just a smaller row count.

A component whose loss sits far above the others is underfit. This table is
the evidence for whether any component needs a targeted top-up before this
adapter ships.

Loss is taken from the model's own fused cross-entropy (`labels=` on the
forward pass) rather than by materialising logits: at 4096 positions × 256k
vocab an fp32 logits tensor is ~4 GB and would OOM a T4.
""")

# ---------------------------------------------------------------------------
# Cell 31: generation smoke test -- swap probed components to French's set.
# ---------------------------------------------------------------------------
src31 = "".join(cells[31]["source"])
src31 = src31.replace(
    'for comp in ("socratic", "grounded_refusal", "quiz_generation",\n'
    '             "code_switching", "darija_preservation"):',
    'for comp in ("socratic", "grounded_refusal", "quiz_generation",\n'
    '             "structured_explanation", "learner_adaptation"):',
)
# The ARABIC range constant is still used below (citation spans can legally
# contain it) but is no longer the "did it work" signal -- Section 32 checks
# absence outside citations, not presence. Keep the constant, rename nothing
# else here since it's still meaningful (counts Arabic chars for reporting).
set_source(31, src31)

# ---------------------------------------------------------------------------
# Cell 32: script-gate INVERTED -- French must be Latin-script with Arabic
# permitted only inside citation spans (analyze_05 §4's acceptance gate),
# using the same has_arabic_outside_citations() this session's F1 audit used.
# ---------------------------------------------------------------------------
set_source(32, '''# Four things the loss curve cannot see. French mode's script gate is the
# INVERSE of Darija's: correct output is Latin-script, with Arabic permitted
# only inside a verbatim legal citation (analyze_05 §4: "arabic_outside_
# citations = 0 on 100% of French turns"). has_arabic_outside_citations() is
# the exact function this session's F1 audit used to verify the citation-
# injection fix -- reused here against the fine-tuned model's own output,
# not just against training rows.
issues = []

if HAVE_APP:
    from app.services.generate_training_data import has_arabic_outside_citations
else:
    has_arabic_outside_citations = None
    issues.append("app/ not uploaded -- cannot run the arabic-outside-citations "
                   "script gate, this smoke test is incomplete")

for comp, ans in samples.items():
    if comp == "quiz_generation":
        continue
    if not ans:
        issues.append(f"{comp}: empty generation")
        continue
    if has_arabic_outside_citations is not None and has_arabic_outside_citations(ans):
        issues.append(f"{comp}: Arabic-script characters outside a citation span "
                       f"-- French-mode script gate violated")

if "quiz_generation" in samples:
    raw = samples["quiz_generation"].strip()
    for fence in ("```json", "```"):
        raw = raw.removeprefix(fence)
    raw = raw.removesuffix("```").strip()
    try:
        parsed = json.loads(raw)
        n_q = len(parsed.get("questions", []))
        print(f"quiz JSON parses: {n_q} questions")
        if n_q == 0:
            issues.append("quiz_generation: parsed but zero questions")
    except json.JSONDecodeError as e:
        issues.append(f"quiz_generation: JSON does not parse ({e})")

# Did it stop, or start role-playing the user's next turn?
for comp, ans in samples.items():
    if "<start_of_turn>" in ans or "CONTEXTE :" in ans:
        issues.append(f"{comp}: leaked a turn marker / context — stop-token problem")

CJK = list(range(0x3000, 0xA000)) + list(range(0xAC00, 0xD800))
CJK_SET = set(CJK)
for comp, ans in samples.items():
    if any(ord(ch) in CJK_SET for ch in ans):
        issues.append(f"{comp}: CJK contamination")

print("\\n" + ("SMOKE TEST CLEAN" if not issues else "SMOKE TEST ISSUES:"))
for i in issues:
    print("  -", i)
''')

nb["cells"] = cells
Path("kaggle_finetune_fr_v1_partial.ipynb").write_text(
    json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"
)
print(f"wrote kaggle_finetune_fr_v1_partial.ipynb, {len(cells)} cells "
      "(base-vs-adapter section not yet inserted -- next script)")
