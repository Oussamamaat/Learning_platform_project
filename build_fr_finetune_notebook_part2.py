"""
Second pass: insert the base-vs-adapter comparison section (Section 10.5)
into kaggle_finetune_fr_v1_partial.ipynb, between the existing Section 10
generation smoke test (ends at cell 32) and Section 11's save/export intro
(cell 33) -- writes kaggle_finetune_fr_v1.ipynb.

This is the piece MIGRATION_PLAN.md P9 and docs/LESSONS_LEARNED.md #1 require
built into the run, not bolted on after: does the trained adapter fabricate
a citation on a prompt where the un-adapted base model (LoRA disabled via
model.disable_adapter(), no second model load needed) does not. That
comparison is what caught the original Darija citation-fabrication defect,
and it must run before the adapter is treated as shippable, not after.
"""
import json
from pathlib import Path

nb = json.loads(Path("kaggle_finetune_fr_v1_partial.ipynb").read_text(encoding="utf-8"))
cells = nb["cells"]


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


intro = md_cell("""## Section 10.5 — Base-vs-adapter comparison (built into this run)

**Non-negotiable per `docs/LESSONS_LEARNED.md` #1 and `MIGRATION_PLAN.md` P9.**
This is the exact discipline that caught the original Darija
citation-fabrication defect on the Atlas-Chat side: the adapter fabricated
law numbers where the untouched base model correctly said "not in the
text." A trained adapter that fabricates MORE than its own un-adapted base
model on the same grounded prompts is a regression, not an improvement, no
matter how good its loss curve looks.

`model.disable_adapter()` is a context manager on the underlying PEFT model
Unsloth returns — it temporarily turns LoRA off without unloading or
reloading anything, so this costs a few generations, not a second ~15-20 min
model download. Both passes use the same eval rows Section 10 already
picked (grounded, citation-bearing contexts only — a hallucination on an
ungrounded prompt isn't this specific failure mode).

**What this section asserts, and what it only reports:**
- **Hard assert:** the adapter must not cite a reference number absent from
  the source context on a row where the base model did not do so either.
  This is exactly the worst-failure-mode pattern (`green_light_model.md`
  RF10-equivalent for French) — if it fires, do not ship this adapter; the
  training data has a grounding-corrupting slice that needs isolating.
- **Reported, not asserted:** general fluency/quality differences between
  base and adapter — the adapter is *expected* to differ in style, Socratic
  behaviour, citation formatting, etc. Only fabricated-reference regression
  is a hard stop here.
""")

code = code_cell('''from app.services.citations import extract_citations
from app.services.generate_training_data import context_from_system_prompt, row_cites

# Reuse Section 10's context_from_system_prompt / row_cites-style logic to
# find real eval rows whose context contains an extractable legal citation
# -- the same "citable" filter this session's audit used to measure quiz
# citation recall on data/fr_v3_merged.
citable_rows = []
for r in eval_rows:
    if r["component"] not in ("grounded_refusal", "quiz_generation", "socratic"):
        continue
    ctx = context_from_system_prompt(r["messages"][0]["content"])
    anchors = extract_citations(ctx)
    if anchors:
        citable_rows.append((r, ctx, anchors))

print(f"citable eval rows available for base-vs-adapter comparison: {len(citable_rows)}")
N_PROBES = min(6, len(citable_rows))
probes = citable_rows[:N_PROBES]
assert probes, (
    "No citable eval rows found -- cannot run the base-vs-adapter grounding "
    "check. Do not skip this silently; investigate why eval.jsonl has no "
    "citable grounded_refusal/quiz_generation/socratic rows before shipping."
)


def fabricated_refs(answer_text, anchors):
    """Reference numbers the model's own answer cites that do not appear
    among the context's real anchors -- the worst-failure-mode signal."""
    cited = extract_citations(answer_text)
    return set(cited) - set(anchors)


results = []
for r, ctx, anchors in probes:
    with model.disable_adapter():
        base_answer = ask(r)
    adapter_answer = ask(r)
    base_fab = fabricated_refs(base_answer, anchors)
    adapter_fab = fabricated_refs(adapter_answer, anchors)
    results.append({
        "component": r["component"],
        "base_answer": base_answer,
        "adapter_answer": adapter_answer,
        "base_fabricated": sorted(str(k) for k in base_fab),
        "adapter_fabricated": sorted(str(k) for k in adapter_fab),
        "regression": bool(adapter_fab) and not base_fab,
    })
    print("=" * 70)
    print(f"[{r['component']}] anchors in context: {sorted(str(k) for k in anchors)}")
    print(f"BASE   fabricated={sorted(str(k) for k in base_fab)}")
    print(f"  {base_answer[:300]}")
    print(f"ADAPTER fabricated={sorted(str(k) for k in adapter_fab)}")
    print(f"  {adapter_answer[:300]}")
''')

assert_cell = code_cell('''regressions = [r for r in results if r["regression"]]
print(f"\\nbase-vs-adapter grounding regressions: {len(regressions)} / {len(results)} probes")
for r in regressions:
    print(f"  - [{r['component']}] adapter fabricated {r['adapter_fabricated']} "
          f"where base fabricated nothing")

BASE_VS_ADAPTER_REGRESSION = bool(regressions)
assert not BASE_VS_ADAPTER_REGRESSION, (
    f"{len(regressions)} probe(s) show the adapter fabricating a citation the "
    "un-adapted base model did not, on the same grounded prompt. This is the "
    "exact citation-fabrication regression docs/LESSONS_LEARNED.md #1 warns "
    "about. DO NOT ship this adapter -- isolate which training rows taught "
    "this pattern (see this session's F1/quiz-fabrication audit methodology) "
    "and regenerate that slice before retraining."
)
print("\\nbase-vs-adapter check PASSED: adapter introduces no new citation "
      "fabrication relative to stock Gemma-2-9B on these grounded probes.")
''')

# Insert after cell 32 (end of existing Section 10 smoke test), before
# cell 33 (Section 11 markdown intro).
insert_at = 33
cells[insert_at:insert_at] = [intro, code, assert_cell]

nb["cells"] = cells
Path("kaggle_finetune_fr_v1.ipynb").write_text(
    json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"
)
print(f"wrote kaggle_finetune_fr_v1.ipynb, {len(cells)} cells total "
      f"(inserted 3 new cells at index {insert_at})")
