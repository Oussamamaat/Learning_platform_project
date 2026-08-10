# Quality Flags — What to Check Before Trusting a Generation Run

**Scope:** This gate suite covers the **tenant #1** dataset (Moroccan safety/security regulations, Arabic-script Darija + French); each tenant's generation run gets its own gate run.

A checklist for verifying a dataset generation run actually worked, not just
that it finished without crashing. Every threshold here comes from a real
number measured in this session — not a guess — and each has a "why this
number" note so a future run can tell real drift from noise.

Run these **in order**. The first two catch total failures cheaply; the
later ones catch subtler quality problems that only show up on inspection.

---

## 0. Before you even look at output: did it finish?

```bash
grep -Ei "Component:|Generated |Traceback|enforcement disabled" gen_gpu0.log gen_gpu1.log
```

**Expect:** 6 `Component:` lines and 6 `Generated` lines per GPU (one per
component), no `Traceback`.

**Red flag:** fewer than 6 `Generated` lines on either GPU — a component
crashed or the session died mid-run. Check for a `Traceback` immediately
before the gap.

**Red flag:** `enforcement disabled after N rejections` appearing for
`socratic`, `code_switching`, or `grounded_refusal`. This means a gate
exhausted its full reject budget and started accepting rows unfiltered for
the rest of that component. Not fatal — the budget is `target × 2` for
French/Darija gates specifically so a mid-run problem doesn't silently
starve the component to zero — but it means some fraction of that
component's rows skipped quality filtering. Note which component and
roughly how late in the run it happened (early = worse, most of the
component is unfiltered; late = only the tail is).

---

## 1. Script and structural integrity

```python
import json, glob
from collections import Counter

rows = []
for f in glob.glob("out_*/train.jsonl") + glob.glob("out_*/eval.jsonl"):
    rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]

print("total rows:", len(rows))
print("by component:", dict(Counter(r["component"] for r in rows)))
arabic = sum(1 for r in rows if r.get("arabic_script"))
print(f"arabic_script: {arabic}/{len(rows)} = {round(100*arabic/len(rows))}%")
```

| Check | Expect | Measured baseline | If it's off |
|---|---|---|---|
| `arabic_script` % | **~100%** | 167/167 = 100% on the 200-row test | Below ~95% means the model is drifting to a different output format — check whether `--script-policy` was set correctly and whether the system prompt template matches `llm.py` |
| Component counts | Roughly proportional to targets (§3 of `PROJECT_STATE.md`) | — | A component at 0 or far under target usually means its reject-budget was hit early — check `missing_french`/`missing_citation` in `component_stats.json` for that component |
| Total row count | Close to 3,000 (allow for dedup loss, see §2) | — | Far short means either a crash or dedup removed far more than expected |

---

## 2. Dedup loss rate

```python
import json
gen = sum(v["generated"] for f in glob.glob("out_*/component_stats.json")
          for v in json.load(open(f)).values())
print(f"generated: {gen} | survived: {len(rows)} | loss: {round(100*(gen-len(rows))/gen)}%")
```

| Baseline run | Loss rate |
|---|---|
| Earlier pilot (before the `maxItems: 2` fix) | 1.8% |
| 200-row test (`grounded_refusal` over-generating 3.8 rows/call) | **20%** |

**Expect:** back down near the 1.8–5% range now that `ROW_LIST_SCHEMA`
caps `grounded_refusal` at 2 rows per call. **If it's still near 20%**,
something is generating near-duplicate content at volume again — check
whether `grounded_refusal`'s `component_stats.json` shows
`generated > target` by a large margin (that ratio directly predicts dedup
loss).

---

## 3. Code-switching quality (`socratic`, `code_switching`)

```python
import sys; sys.path.insert(0, "/kaggle/working")  # or your local repo root
from app.services.generate_training_data import row_is_code_switched, french_term_count, darija_marker_count

cs = [r for r in rows if r["component"] in ("socratic", "code_switching")]
passing = sum(row_is_code_switched(r) for r in cs)
print(f"code-switch gate pass rate: {passing}/{len(cs)} = {round(100*passing/len(cs))}%")

multi = [r for r in cs if len([m for m in r["messages"] if m["role"]=="assistant"]) > 1]
print(f"multi-turn share: {len(multi)}/{len(cs)} = {round(100*len(multi)/len(cs))}%")
```

| Metric | Bad (pre-fix) | Good (post-fix, verified) |
|---|---|---|
| Gate pass rate at generation time | ~33% | **~88%** (live re-test) |
| Multi-turn share of accepted rows | 26% (56/76 single-turn) | Should be **majority multi-turn** now that the gate no longer structurally penalizes it |

**Red flag:** multi-turn share back down near 26% — the gate calibration
regressed, or the prompt's "EXACTLY 2-3 exchanges" instruction stopped
landing. Pull 5 raw rows and read them (§6).

**Red flag:** gate pass rate back down near 33% — check whether
`min_french_row=2` / `min_darija_per_turn=1` in `row_is_code_switched()`
still match what's in the repo (someone may have "fixed" the thresholds
back toward the median without realizing why they were set below it).

---

## 4. `grounded_refusal` — citation and register

```python
from app.services.citations import extract_citations
from app.services.generate_training_data import context_from_system_prompt, row_is_grounded_darija

gr = [r for r in rows if r["component"] == "grounded_refusal"]
darija_pass = sum(row_is_grounded_darija(r) for r in gr)
print(f"Darija register pass: {darija_pass}/{len(gr)} = {round(100*darija_pass/len(gr))}%")

citable = cited = 0
for r in gr:
    ctx = context_from_system_prompt([m["content"] for m in r["messages"] if m["role"]=="system"][0])
    c = extract_citations(ctx)
    if not c: continue
    citable += 1
    a = " ".join(m["content"] for m in r["messages"] if m["role"]=="assistant")
    if any(e["canonical"] in a or (e["arabizi"] and e["arabizi"] in a) for e in c.values()):
        cited += 1
print(f"citation recall: {cited}/{citable} = {round(100*cited/citable) if citable else 0}%")

fr_zero = sum(1 for r in gr if french_term_count(" ".join(m["content"] for m in r["messages"] if m["role"]=="assistant")) == 0)
print(f"rows with ZERO French: {fr_zero}/{len(gr)}")
```

| Metric | Measured baseline | Expect now |
|---|---|---|
| Darija register pass | 33/45 = 73% (after fixing the detector bug) | ~70%+ |
| Citation recall (of citable rows) | 72% | 70%+ |
| Rows with zero French | 45/45 = 100% (before the 50/50 source-split fix) | **Should be well under 100%** — roughly half, since only the French-source half of the 50/50 split will carry French naturally |

**Red flag:** citation recall near 0% — check for the instruction-example
leakage bug regression (was extraction scanning example citations inside
the prompt instructions again instead of just `CONTEXTE :`?).

**Red flag:** zero-French rows still at 100% — the 50/50 source split in
`pick_source_doc()` isn't landing; verify `component == "grounded_refusal"`
branch is actually being hit and not falling through to the
`ARABIC_SOURCE_COMPONENTS` (Arabic-only) branch.

---

## 5. Quiz quality

```python
quiz = [r for r in rows if r["component"] == "quiz_generation"]
print(f"quiz rows: {len(quiz)}")

bad_json = 0
for r in quiz:
    a = [m["content"] for m in r["messages"] if m["role"]=="assistant"][0]
    try:
        json.loads(a)
    except json.JSONDecodeError:
        bad_json += 1
print(f"invalid JSON: {bad_json}/{len(quiz)}")
```

**Expect:** `bad_json` at or near 0 — schema-constrained generation plus
`build_quiz_row()`'s validation should reject anything malformed before it
ever gets written.

**The dangerous failure mode is invisible to this check**: a quiz row can
be perfectly valid JSON with a **wrong answer key**. That was the actual
defect found (16% of questions, 12/76) — the `answer` index pointed at a
different option than the `explanation` supported. The fix
(`_explanation_supports_answer()`) is a self-consistency check, applied
*during generation* — it cannot be re-verified from the output alone,
because a row that failed it was never written in the first place.

**What you CAN still check post-hoc:** read 10 quiz questions by hand and
verify the marked answer is the one the explanation actually argues for.
If you find even one contradiction in a random sample of 10, the
self-consistency gate has a gap — worth a closer look at
`_explanation_supports_answer()`'s word-overlap heuristic (it's
deliberately permissive: an explanation sharing no vocabulary with any
option is allowed through rather than rejected on a technicality, so it
won't catch every possible mismatch).

---

## 6. CJK / foreign-script contamination

```python
import re
CJK = re.compile(r'[　-鿿가-힯]')
bad = [r for r in rows if any(CJK.search(m["content"]) for m in r["messages"] if m["role"] != "system")]
print(f"CJK-contaminated rows: {len(bad)}/{len(rows)}")
```

**Expect:** 0. `validate_chatml()` now rejects any row containing a CJK
character at generation time, so any that reach `train.jsonl`/`eval.jsonl`
mean that check was bypassed somehow (e.g. a row inserted outside the
normal `normalize_row()` → `validate_chatml()` path).

**Baseline it's protecting against:** 3/167 = 2% in the unfiltered
200-row test.

---

## 7. Manual read — do this regardless of what the numbers say

Numbers can't catch everything; two of the real defects found this session
(the answer-key contradiction and a Darija-detector bug that inverted a
whole conclusion) were only caught by reading actual text, not by trusting
an aggregate percentage.

Pull 8–10 full rows — not previews — spread across all 6 components, and
read them for:

- **Translate-then-bracket pattern**: `المعدات الوقاية الشخصية (les EPI)`
  instead of just `les EPI` mid-sentence. The prompts now explicitly
  forbid this, but it's a soft instruction, not a hard gate — some rate of
  recurrence is expected, just shouldn't dominate.
- **Few-shot parroting**: a generated row that's near-verbatim structure
  from `data/few_shot_examples.md` with only nouns swapped. Dedup won't
  catch this reliably since exact text differs.
- **Quiz answer sanity**: does the marked answer actually make sense given
  the source document, not just given its own explanation?
- **Register drift**: does a `grounded_refusal` or `quiz_generation` row
  read like a legal recitation (MSA, formal) rather than something a
  Moroccan tutor would actually say?

---

## Summary — go/no-go thresholds

| Check | Kill the run / regenerate if... |
|---|---|
| `arabic_script` % | < 90% |
| Dedup loss | > 15% |
| Code-switch gate pass rate | < 60% |
| Multi-turn share (socratic + code_switching) | < 40% |
| `grounded_refusal` zero-French rate | > 80% |
| CJK contamination | any rows in final output |
| Manual read of 10 rows | more than 1-2 with a clear defect from §7 |

Anything passing all of these is consistent with what was verified this
session and is reasonable to proceed to training with.
