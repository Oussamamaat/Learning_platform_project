# MIGRATION_PLAN.md — Session Handoff

Written because the previous session's quota ran out mid-task. Updated again after a follow-on
session completed the rest of Part I/Part II and launched the French fine-tune on Kaggle (§2 below).
Updated again after the deterministic-backstops frontend/backend task (§0, complete). **Latest
update (this session): the §2 fine-tune kernel actually errored — see §-1 below for the incident
and recovery, which is the current active work.**

---

## -1. INCIDENT: French fine-tune kernel crashed post-training, and recovery in progress

**Status as of this write-up: recovery kernel `darija-tutor-fr-finetune-v1-resume` is on
**version 5**, RUNNING.** Versions 1-4 each failed; **v5 carries the actual root-cause fix** — a
path bug in the notebook's `app/` discovery that made `HAVE_APP` `False` on every run this project
has ever done, including the original. See "v4 → v5" below. Check before trusting anything as final:
```
kaggle kernels status maataouioussama/darija-tutor-fr-finetune-v1-resume
```

### What happened
`darija-tutor-fr-finetune-v1` (§2's fine-tune kernel) ended in `KernelWorkerStatus.ERROR`, not
`COMPLETE`. Diagnosis (full log recovered from the Kaggle web UI after two CLI fetch attempts
returned an empty log — see below):

- **Training itself completed successfully.** `checkpoint-174/trainer_state.json`:
  `global_step: 174 == max_steps: 174`, `epoch: 2.0 == num_train_epochs: 2`, healthy loss curve,
  eval_loss improving monotonically across all 4 recorded eval points (0.636 → 0.578 → 0.577 →
  0.573 at steps 43/86/129/172).
- **The crash was `ModuleNotFoundError: No module named 'app'`** at notebook cell `In [22]`
  (`from app.services.citations import extract_citations`), one of the base-vs-adapter
  citation-fabrication safety-check cells. The notebook's own log one line earlier already flagged
  it: `"app/ not uploaded -- cannot run the arabic-outside-citations script gate, this smoke test
  is incomplete"` — cell 32 (the smoke test) has a `HAVE_APP` guard and degraded gracefully; cell
  34 (the base-vs-adapter import) did **not** have the same guard and crashed hard via
  `papermill.exceptions.PapermillExecutionError`. This cell runs **before** adapter save, GGUF
  merge/export, and `TRAINING_REPORT.json` — none of those ever ran, which is why none of those
  artifacts exist in the kernel's output.
- **Root cause of the missing `app/`**: not a code bug and not a local staging mistake — the local
  staging dir used right before this kernel was pushed (`kaggle_ds_stage_fr_finetune`) was verified
  complete (33 files, correct `app/services/citations.py` and `generate_training_data.py`
  present). The dataset (`darija-tutor-pipeline-v2`) has `app/` correctly **now**. Most likely: the
  git-bash upload path-mangling bug already documented as a standing risk in this file (§3) hit the
  specific `kaggle datasets version` push done right before this kernel launched, and Kaggle pins a
  kernel to whatever dataset version was live when its session started — so this run was stuck with
  the broken version for its full ~3-hour lifetime even though a later push fixed it.
- **CLI log-fetch note for future incidents**: `kaggle kernels output <slug> -p <dir>` returned a
  genuinely 0-byte log twice in a row (once with a Windows console `charmap` encode crash on
  non-ASCII output, once clean but still empty) even though `KaggleApi().kernels_status(...)
  ._failure_message` was also `None`. The real traceback was only visible via the Kaggle web UI,
  pasted in by the user. If this happens again, ask for the web UI log rather than trusting an
  empty CLI fetch as "no log ever existed."

### Recovery: resume from checkpoint, no retraining
`checkpoint-174` is a complete, healthy adapter — redoing the full ~3h training run would waste
GPU-hours for nothing. Instead:
1. Trimmed `checkpoint-174` to just what inference/merge needs (dropped `optimizer.pt`,
   `scheduler.pt`, `scaler.pt`, `rng_state.pth`, `training_args.bin` — training-resume-only state,
   244MB uploaded vs. ~380MB full checkpoint dir) and pushed as a **new** Kaggle dataset,
   `maataouioussama/darija-tutor-fr-checkpoint174`, via PowerShell (not Bash — same
   upload-corruption avoidance as every other dataset push in this project). Verified all 8 files
   landed via `kaggle datasets files` before proceeding.
2. Built `kaggle_finetune_fr_v1_resume.ipynb` from the original 43-cell notebook, edited
   programmatically (not by hand) so every unchanged cell is byte-identical to the original:
   - Cell 7 (base model load) → now loads base model **+ the trained adapter together** directly
     from the checkpoint174 dataset (`FastLanguageModel.from_pretrained(model_name=str(ADAPTER_SRC),
     ...)` — Unsloth auto-resolves `base_model_name_or_path` from `adapter_config.json`).
   - Cell 13 (fresh `get_peft_model` LoRA init) → replaced with a diagnostic-only cell (no new LoRA
     needed, adapter's already loaded).
   - Cells 14, 15, 16 (masking probe, `Trainer` construction, pre-train smoke forward-pass) →
     **deleted**, not needed without a live training loop.
   - Cell 17 (baseline eval + `trainer.train()` + post/pre-train assert) → replaced with a
     telemetry-reconstruction cell that reads real eval-loss/step data back out of
     `checkpoint-174/trainer_state.json`. **Honestly labeled**: the true pre-training baseline
     `trainer.evaluate()` call was never persisted to `trainer_state.json` (only the Trainer's own
     periodic logging is), so it's genuinely unrecoverable — the report uses "first-recorded" /
     "last-recorded" eval_loss (steps 43 and 172), not a fabricated "before/after" pair.
   - Cell 21 (the import that actually crashed) → **fixed**: added the missing `HAVE_APP` guard
     (now a hard `SystemExit` with a clear message if `app/` is ever missing again, instead of an
     opaque `ModuleNotFoundError` mid-run) so this exact failure mode can't silently recur.
   - Cell 25 (`TRAINING_REPORT.json`) → rebuilt to source training fields from
     `trainer_state.json` and adds `"resumed_from_checkpoint": "checkpoint-174"` +
     a `"resume_note"` explaining all of the above, so this is discoverable from the report itself
     later, not just from this file.
   - Every other cell (env setup, data loading/validation, parrot filter, train/serve parity check,
     chat template, tokenize/encode round-trip proofs, per-component eval scoring, sample
     generation, the smoke test itself, the base-vs-adapter fabrication gate and its regression
     assert, adapter save, Modelfile, zip, GGUF export, final summary) is **byte-identical** to the
     original — verified by diffing the cell list programmatically, not by eyeballing.
   - All 26 code cells syntax-checked locally (`compile()`) before pushing — 0 errors.
3. Pushed kernel `maataouioussama/darija-tutor-fr-finetune-v1-resume`, referencing both
   `darija-tutor-pipeline-v2` (app/ + fr_v3_merged data) and the new checkpoint174 dataset.
   Confirmed **RUNNING** shortly after push (not stuck queued, not immediately erroring).

### Resume attempts v1-v3: two more real bugs, both mine, both fixed

**v1 and v2 crashed with no usable Kaggle-delivered log** — `kaggle kernels output` returned a
genuinely empty log both times (same platform unreliability noted above), and
`kernels_status()._failure_message` was `None` both times. Rather than ask the user to paste the
web UI log again, built a **self-diagnosing kernel wrapper**
(`add_crash_diagnostics.py` → `kaggle_finetune_fr_v1_resume2.ipynb`): every code cell is wrapped in
try/except that appends `traceback.format_exc()` to `/kaggle/working/CRASH_TRACEBACK.txt` plus a
`_mark()` call that logs cell entry/exit/crash to `/kaggle/working/PROGRESS.log` — both reliably
fetchable via `kaggle kernels output` regardless of Kaggle's log-capture behavior for ERROR
kernels. (The one `%%capture` cell can't be wrapped — IPython cell magics must be the cell's sole
statement — so it's bracketed with plain marker cells instead.) This is now the standing technique
for any future Kaggle diagnosis in this project; don't fall back to asking for the web UI log
before trying it.

- **v2 crashed with a real bug**: `NameError: name 'math' is not defined` in the per-component
  eval-loss cell. Self-inflicted — the original notebook's only `import math` lived inside the
  Trainer-setup cell, which the resume notebook deletes (no live training loop needed), but
  `math.exp()` is reused much later for perplexity, in a cell that was otherwise kept unchanged.
  **Fixed**: moved `import math` into the diagnostic cell that replaces the deleted one. Verified
  with `compile()` on every cell plus a full AST-based sweep for other orphaned names from the same
  deletion (found none — `install_log`, flagged by the sweep, is an expected false positive from
  the `%%capture` magic, not a real bug). Pushed as v3.
- **v3 crashed on the *original* bug, still unfixed**: `HAVE_APP` was `False` again, so the guarded
  cell correctly refused to skip the base-vs-adapter safety check (`SystemExit`, not a silent
  skip — the guard from the first recovery pass did its job) — but that just re-surfaced the real
  problem, which the guard alone was never going to fix. **Root cause, finally isolated**: the
  `darija-tutor-pipeline-v2` dataset had been silently reduced to **only `app/`** (20 files, no
  `data/` at all) at some point during this recovery. Kaggle dataset versioning is a **full
  replace, not additive** — an earlier push in this recovery (done from a staging dir that only
  had `app/` in it) wiped the `fr_v3_merged` training data that a prior push had put there. The
  `git-bash path-mangling` theory from the original incident write-up above was **not actually the
  root cause of the app/-missing problem** — this was: repeatedly re-pushing a *partial* staging
  directory as a dataset *version*, not a one-off upload bug. **Fixed**: located the original,
  complete staging directory (`kaggle_ds_stage_fr_finetune/`, containing `app/` + `data/` +
  `raw/` + `tests/` together, `data/fr_v3_merged/{train,eval}.jsonl` verified byte-identical to the
  repo's current `data/fr_v3_merged/`), cleaned stale `__pycache__`, and pushed it whole as a new
  dataset version via `kaggle datasets version -p ... --dir-mode zip` (PowerShell, not git-bash) —
  confirmed via the Kaggle Python SDK (`dataset_list_files`, paged past the CLI's 20-row default)
  that all 78 files, `app/` and every `data/` variant included, landed correctly. Pushed as kernel
  **v4**, confirmed RUNNING.

**Standing correction for next time**: if `app/` (or any subset of files) goes missing from a
Kaggle dataset again, check whether the *dataset itself* currently has everything it should
(`dataset_list_files`, paged) before re-uploading anything — a dataset version push always replaces
the whole file set, so pushing from a directory that's missing something silently deletes it, even
if a previous version had it.

### v4 → v5: THE ACTUAL ROOT CAUSE — a path bug in the notebook, not the upload

v4 failed with the **same** `app/ not uploaded` `SystemExit`, on a dataset verified to contain all
78 files including `app/services/citations.py`. That ruled out every upload theory and pointed at
the notebook. Reading the discovery cell (original `code[3]`) gave the answer:

```python
def discover_src(marker="fr_v3_merged/train.jsonl"): ...   # returns the dir CONTAINING the marker
SRC = discover_src()          # -> /kaggle/input/darija-tutor-pipeline-v2/data
APP_DIR = SRC / "app"         # -> .../darija-tutor-pipeline-v2/data/app   <-- never existed
```

The marker is `fr_v3_merged/train.jsonl`, and that file lives at `<dataset_root>/data/fr_v3_merged/`,
so `SRC` resolves to `<dataset_root>/data`. But `app/` is a **sibling** of `data/`, at
`<dataset_root>/app`. `SRC / "app"` was therefore a path that **cannot exist under any upload**.

**`HAVE_APP` was `False` on every run this project has ever done — including the original ~3-hour
one.** The upload was fine the whole time. The earlier "git-bash path mangling" and "a partial
push wiped the dataset" theories were both wrong as explanations for `app/ not uploaded` (the
dataset *was* genuinely missing `data/` at one point and that repush was a real fix, but it was
never the cause of this error). Every `app/`-related failure in this incident traces to this one
line.

**Fixed in v5** (`fix_app_discovery.py`, run after `build_resume_notebook.py`): locate the package
by its own contents rather than by assumed position relative to `SRC` —
```python
def discover_app():
    for root, dirs, files in os.walk("/kaggle/input"):
        p = Path(root)
        if p.name == "app" and (p / "services" / "citations.py").is_file():
            return p
```
plus `sys.path.insert(0, "/kaggle/working")`, an `assert HAVE_APP` + live `import
app.services.citations` probe so this fails at ~2 min instead of ~8, and a guarded
`pip install pydantic-settings` (`app.services.llm` → `app.config` needs it; the parity cell
try/excepts that import, so a missing dep would have silently *skipped* the check rather than run it).

**De-risked before spending GPU time** — because `HAVE_APP=True` means the parity check (cell 6)
and the base-vs-adapter gate (cell 18) execute for the first time ever:
- Old vs. new discovery logic simulated against a byte-identical local mirror of the dataset:
  old → `False`, new → `True`.
- The parity cell's two asserts run locally against the real data:
  `PRODUCTION_SYSTEM_PROMPT_TEMPLATE_FR == SYSTEM_PROMPT_TEMPLATE_FR` → `True`, and
  **0 of 1542 rows** (1387 train + 155 eval) carry a mismatched system prompt. Both expected to pass.

**Also fixed**: the crash wrapper caught `except Exception`, but `SystemExit` inherits from
`BaseException`, so v4's `SystemExit` bypassed it entirely and `CRASH_TRACEBACK.txt` was never
written (only `PROGRESS.log` survived, which is how the stall at cell 18 was still visible). Now
`except BaseException`, re-raised as before.

*(Note: `fix_app_discovery.py` patches `kaggle_finetune_fr_v1_resume.ipynb` in place after
`build_resume_notebook.py` generates it. If the build script is ever re-run, re-run the fix script
after it or the bug comes back.)*

**Gotcha worth remembering**: `str.splitlines()` splits on Unicode separators (U+2028/U+2029/NEL),
which occur in this project's Arabic/French text — it corrupted a JSONL read during the local
simulation above. Iterate the file object or split on `"\n"` explicitly when reading these `.jsonl`
files.

### ⏭️ Active task / next immediate step
```
kaggle kernels status maataouioussama/darija-tutor-fr-finetune-v1-resume
```
Expected to be much faster than the original ~3h run (no training — model load, per-component eval
scoring, ~5 sample generations, smoke test, 6 base-vs-adapter probes, adapter save, GGUF export).
If v4 also errors, fetch `/kaggle/working/PROGRESS.log` and `CRASH_TRACEBACK.txt` first (the
self-diagnosing wrapper — see above) rather than relying on `kaggle kernels output`'s own log.
When `COMPLETE` (or `ERROR`), fetch output and check `TRAINING_REPORT.json`:
```
kaggle kernels output maataouioussama/darija-tutor-fr-finetune-v1-resume -p <clean temp dir>
```
Check, in this order:
1. **`base_vs_adapter.passed`** — the critical safety gate. If `false`, **do not ship this
   adapter**; isolate which training rows taught the fabrication pattern before doing anything else.
2. `smoke_test_issues` — should be empty now that `HAVE_APP` is genuinely true.
3. `per_component_eval_loss` — any component >1.6x the overall mean is flagged automatically by the
   notebook itself (cell 15/original cell#29's own logic, unchanged).
4. Whether the GGUF (`iblog-tutor-fr-q4_k_m.gguf`) actually exported — it's explicitly best-effort
   (needs ~18.5GB container disk); if it's missing, the adapter zip
   (`finetuned_model_export.zip`) is still the real deliverable and the GGUF can be rebuilt locally.

---

## 0. Current focus (this session): Deterministic Serving-Layer Backstops

Two defects were reproduced **live** on production `IBLOG_TUTOR`. Both were already surfaced (but
not fixed) by the prior session's audit — see §2 "Not yet actioned" — so this is a direct
continuation, not a new finding:

1. **Refusal domain-mismatch (3/3)** — off-topic refusals self-identify as a "safety" (السلامة)
   assistant regardless of the tenant's actual domain (`securite`, `blockchain`).
2. **Quiz grounding/fabrication (2–3 of 7)** — generated quiz questions invent facts
   (e.g. `المادة 15/16` guard limits) absent from the retrieved context chunk.

No compute/time budget for an unguided retraining loop, so the fix is **deterministic,
serving-layer backstops first**; dataset scrubbing is deferred to a report-only script, not
executed now, not blocking.

Full design doc — reasoning, code snippets, alternatives considered:
`C:\Users\oussa\.claude\plans\read-claude-md-docs-lessons-learned-md-a-goofy-peacock.md`
(this plan file was rewritten this session for the new task; it no longer contains the earlier
French-pipeline audit content, which is preserved in §1–§3 below instead).

### Approved 4-step execution sequence (strict order, verify after each step)

- [x] **Step 1 — Grounding verifier. DONE.** `validate_quiz_question()` (structural checks: option
      count/dedup/answer index/CJK/`_explanation_supports_answer`) extracted from `build_quiz_row`
      into `generate_training_data.py` itself (not `grounding.py` — see correction below) at
      line ~2481, and `build_quiz_row` now calls it, zero behavior change. New
      `app/services/grounding.py` imports it one-directionally plus `row_has_ungrounded_reference`
      (`:3309`) and adds `question_is_grounded()` + `filter_grounded_questions()`.
      **Two corrections made before writing code, both flagged and user-approved:**
      (1) `question_is_grounded()` wraps `row_has_ungrounded_reference` ONLY, not also
      `row_has_ungrounded_number` — that gate's own `NUMERIC_GROUNDED_COMPONENTS`
      (`:3399`) deliberately excludes `quiz_generation` because distractors are supposed to state
      plausible-but-wrong numbers; wiring it in would risk false-positive-rejecting valid
      questions live. (2) `validate_quiz_question` had to live in `generate_training_data.py`
      (where `build_quiz_row` calls it directly, no import needed) rather than in `grounding.py`
      as first drafted — putting it in `grounding.py` and having `build_quiz_row` import it back
      would have created a circular import (`grounding.py → generate_training_data.py →
      grounding.py`). Verified: `pytest tests/` — all 220 pre-existing tests still pass, plus a
      manual smoke check of `question_is_grounded`/`filter_grounded_questions` against a grounded
      vs. fabricated `Article 16` question, confirming the fabricated one is correctly rejected.
- [x] **Step 2 — Refusal interception. DONE.** `app/services/llm.py` gained `DOMAIN_LABELS_AR`,
      `REFUSAL_TEMPLATE_DARIJA`/`_FR`, `deterministic_refusal(domain, language)`,
      `UI_LANG_TO_MODEL_LANG = {"fr": "fr", "ar-MA": "darija"}` (`"en"` deliberately unmapped).
      All kept fully outside `SYSTEM_PROMPT_TEMPLATE`/`_FR` — verified by a dedicated test
      (`test_deterministic_refusal_not_derived_from_prompt_templates`). `generate_llm_response`
      gained an optional `language` param (`language or detect_query_language(query)` — backward
      compatible). `ChatRequest.language`/`QuizRequest.language` are now `Optional[Language] =
      None` in `app/models/schemas.py`. `app/routers/chat.py` resolves `ui_lang` once and returns
      `deterministic_refusal(...)` before calling Ollama when `build_rag_context` returns empty
      context. Manually verified: `securite`/`blockchain` refusals never say السلامة (the exact
      reproduced defect); French `ui_lang` gives a French refusal; omitted `language` still uses
      `detect_query_language`. `pytest tests/` — 220/220 still green.
- [x] **Step 3 — Quiz path. DONE.** New `app/services/quiz.py`:
      `generate_quiz_questions(topic, context, domain, language, n)` calls Ollama with a `format`
      JSON schema (`_quiz_format_schema`, mirrors `QUIZ_CONTENT_SCHEMA`'s question shape),
      reusing `_build_system_prompt` (train/serve parity guaranteed, same function chat uses) and
      `QUIZ_USER_FALLBACKS`/`_FR` as the user turn — matches the exact shape quiz training rows
      were built with, so the fine-tuned model stays in-distribution. Rewrote
      `app/routers/quiz.py` (was a pure `"[Placeholder] Question about {topic}?"` stub, never
      called RAG or the LLM): `build_rag_context(topic)` → empty context → deterministic refusal
      → else `generate_quiz_questions` → `filter_grounded_questions` → payload.
      `QuizResponse` gained `message: Optional[str] = None` and `sources: list[str] = []`.
      **Known limitation, not expanded beyond original stub's scope**: `QuizRequest` has no
      `domain` field, so the router hardcodes `domain="industrial"` — same single-tenant
      assumption the original stub already had. Manually verified end-to-end with a mocked
      Ollama response containing one grounded + one fabricated (`Article 99`, absent from
      context) question: exactly the grounded one survived. `pytest tests/` — still 220/220.
- [x] **Step 4 — Tests + full verification. DONE.** New `tests/test_chat.py` (10 tests) and
      `tests/test_quiz.py` (6 tests), plus 6 additions to `tests/test_llm.py` — all using
      monkeypatched `urllib.request.urlopen` and `build_rag_context`, no live Ollama/Postgres.
      Cover: empty context never calls Ollama (chat + quiz, asserted via a `urlopen` that raises
      if invoked); correct domain label per language with no cross-domain leakage, parametrized
      over all 3 domains × 2 languages; omitted `language` still routes via
      `detect_query_language`; happy-path (non-empty context) still reaches the model unaffected;
      quiz filters a fabricated question but keeps a grounded one; all-fabricated → 0 questions +
      `message`; malformed payloads (duplicate options, out-of-range answer index) dropped.
      **Full suite: 242/242 passing** (220 pre-existing + 22 new).

### Guardrails (from the user, binding for this task) — all honored
- Strict order 1 → 2 → 3 → 4 was followed; `pytest tests/` run and green after every step.
- **Zero silent improvisation** — three real deviations from the plan's literal text were found
  and reported via AskUserQuestion before writing code, not decided unilaterally:
  1. `question_is_grounded()` uses `row_has_ungrounded_reference` only, not also
     `row_has_ungrounded_number` — that gate's `NUMERIC_GROUNDED_COMPONENTS` deliberately
     excludes `quiz_generation` (distractors legitimately carry plausible-wrong numbers).
  2. `validate_quiz_question` had to be defined in `generate_training_data.py` itself, not
     `grounding.py` as first drafted, to avoid a circular import.
  3. `config/requirements.txt` was **not actually installed** in `.gguf_venv/` (only
     pydantic/sentence-transformers/pytest were present) — `fastapi`, `psycopg2-binary`,
     `sqlalchemy`, etc. were missing, blocking any import of `app.routers.*`. Ran
     `pip install -r config/requirements.txt` (the exact command `CLAUDE.md` documents as
     first-time setup) after explicit approval; this downgraded a few packages that had drifted
     newer than the pinned versions (pydantic 2.13.4→2.9.0, sentence-transformers 5.6.1→3.1.1,
     transformers 5.5.0→4.57.6) back to what's actually committed. Full suite re-verified green
     immediately after (220/220), and again at the end (242/242) — no regression from the
     downgrade.
- **Byte-identical parity is sacred** — never touched. Verified by a dedicated test
  (`test_deterministic_refusal_not_derived_from_prompt_templates`) that refusal strings are not
  substrings of either `SYSTEM_PROMPT_TEMPLATE` or `SYSTEM_PROMPT_TEMPLATE_FR`.

### Live probe — DONE, both defects re-verified against the real served model
`probe_serving_backstops.py` (repo root) run against real Ollama + `IBLOG_TUTOR:latest`.
Postgres/Redis are not running in this environment, so real pgvector retrieval is out of scope
(user-approved reduced scope) — context chunks are pulled verbatim from `data/v11_merged`
instead (genuine Moroccan Labour Code / ISO 45001 text, not synthetic).

- **Refusal defect reproduced live, unpatched.** Calling `generate_llm_response` directly (no
  fix) with an off-topic query, empty context, `domain="securite"` got back a reply naming
  "السلامة والوقاية من الحوادث" (safety/accident-prevention) — confirms the bias is real and
  still present in the raw model today, not something already stale.
- **Fix verified live, all 3 domains.** Real `chat()` router call, empty context, for
  `industrial`/`securite`/`blockchain`: Ollama never called (0 calls) in all 3, each refusal
  named the correct domain, zero cross-domain leakage.
- **Happy path unaffected.** A real grounded industrial question through the real `chat()`
  router + real model returned a correct, on-topic answer with sources attached — confirms the
  empty-context interception didn't regress the normal path.
- **Quiz grounding, 2 real corpus chunks, real model, real filter.** LOTO-procedure chunk
  (Article 1, cites real article numbers): 3/3 generated questions correctly grounded and kept.
  ISO 45001 pedagogical chunk (no article numbers to cite): every generated question was
  correctly dropped — but for a *different* reason than expected (`invalid_structure`, not
  `ungrounded_reference`). Manually verified by hand: the model marked answer index 1 correct,
  but the explanation's vocabulary overlapped most with option 3 (overlap 3 vs. 1) — a genuine
  answer/explanation self-consistency defect, exactly what `_explanation_supports_answer` exists
  to catch (its own docstring cites the same "16% of questions marked an option the explanation
  contradicted" pattern from the original audit). **Confirms the gate correctly catches real,
  observed model defects — both the originally-targeted fabrication class and this
  self-consistency class — not a synthetic pass.**

**Not yet done**:
- Results should be transcribed into `green_light_model.md` §4.2 alongside the existing 9/10
  evidence — the probe ran and produced real numbers, but nothing has been written into that
  file yet.
- **Dataset-pollution audit** (`audit_dataset_pollution.py`, report-only, `--fix` writes to
  `data/v11_scrubbed/`, never touches `v11_merged`) — still deferred, not blocking.
- **Full end-to-end HTTP/DB test** (docker-compose stack, real pgvector retrieval, real ingested
  tenant documents) was explicitly descoped this session per user decision — still open if a
  fully rigorous pre-demo check is wanted later.
- Check on the French fine-tune kernel status (§1 below) — unrelated to this task but still open.

### Task list
This session's TaskList was cleared of the prior French-pipeline items (all were already complete
per §2/§3 below) and replaced with exactly the 4 steps above (task IDs #29–#32, chained
blockedBy in order).

---

## 1. Prior session's status snapshot (French fine-tune launch — verify before trusting)

**Everything in §2/§3 below (Part I and Part II of the earlier audit) was complete as of the prior
session.** The one open item that session left was a Kaggle fine-tune run in progress. Status
unknown as of this session — check before assuming either outcome:

```
kaggle kernels status maataouioussama/darija-tutor-fr-finetune-v1
```

Key facts from that session, for context:
- **Final French dataset**: `data/fr_v3_merged/{train,eval}.jsonl` — 1,542 rows (1,387 train / 155
  eval), passed its preflight (`check_fr_dataset_preflight.py`): 100% quiz citation recall, 0 F1
  citation-corruption rows. `quiz_generation`/`structured_explanation` capped at ~150 rows each by
  a confirmed corpus-size dedup ceiling (accepted by the user, not a bug).
- **Go/no-go audit before fine-tuning**: `fr_v3_merged` was checked for the same two defect
  patterns this session (§0) is now fixing at the serving layer — refusal domain-mismatch (0/27)
  and quiz fabrication (0/30) — and found clean. **So the French dataset does not carry either
  defect; these are Darija-specific (`data/v11_merged`) / production-`IBLOG_TUTOR`-specific.**
- **Fine-tune notebook**: `kaggle_finetune_fr_v1.ipynb`, base `unsloth/gemma-2-9b`, LoRA
  hyperparameters unchanged from the Darija notebook. New Section 10.5: base-vs-adapter
  citation-fabrication comparison with a hard `assert` — if `TRAINING_REPORT.json
  ["base_vs_adapter"]["passed"]` is `false`, **do not ship the adapter**.
- Pushed as kernel `maataouioussama/darija-tutor-fr-finetune-v1`, dataset
  `maataouioussama/darija-tutor-pipeline-v2`. If `COMPLETE`, pull output and check (1) no training
  error, (2) the Section 10.5 assert passed, (3) eval loss / smoke-test results, (4) the exported
  GGUF (`iblog-tutor-fr-q4_k_m.gguf`) is present.

**Not yet actioned as of that session** — this is exactly what §0 above is now fixing: the
Darija/`IBLOG_TUTOR` quiz-fabrication and refusal domain-mismatch findings were only spot-checked
(22 rows + ~13 live probes), not quantified at scale across all of `data/v11_merged`.

---

## 2. Complete Actionable Plan & Checklist (prior session, all items complete)

Full detail preserved for audit-trail purposes.

### Part I — French pipeline (blocked the fine-tune; all done)

- [x] **P0** · Restored bare `pytest` (renamed `kaggle_fr_smoke_test.py` → `kaggle_smoketest_fr.py`,
      which matched pytest's `*_test.py` collection glob and aborted the whole suite at import).
- [x] **P1** (F1) · `normalize_row()` takes a `language` param; skips
      `inject_citations(..., target_script="arabic")` when `language == "fr"`.
- [x] **P2** (F2) · `build_quiz_prompt_fr` given a `citation_anchor_rule(...)` call.
- [x] **P3** (F3, revised) · `turn_count_reject_budget = target * 3 if component == "socratic"
      else target`. (`grounded_refusal` multi-turn work explicitly dropped — Darija's own
      `grounded_refusal` also delivers ~0.2% multi-turn and is untested by the acceptance gate, so
      it was found to be a pre-existing, language-symmetric non-issue, not a French regression.)
- [x] **P4** (F4) · Added `QUIZ_USER_FALLBACKS_FR`; `build_quiz_row` branches on `language`.
- [x] **P5** (F5, systemic) · `generate_component` computes `gates_exhausted` and logs a
      `STATUS: gate_exhausted` / `STATUS: no_gate_exhaustion` line instead of failing silently.
- [x] **P6** · Regression tests added (mutation-tested language guards, Darija template parity
      test). Suite: 190 → 205 → 220 (after P10's `test_llm.py`).
- [x] **P7** · Added `--components` CLI filter + `scale_component_targets(components=...)` param
      (replaced an originally-planned "seed 5 keepers" mechanism found to be mathematically
      unworkable). Targeted regen kernel (`darija-tutor-fr-regen-targeted`) ran to completion: 764
      rows, merged into `data/fr_v2_merged/` (1,229 rows) after stripping F1-corrupted rows.
- [x] **P7.5 (top-up)** · `fr_v2_merged` was short of the 1,800-row target (dedup ceiling on
      `quiz_generation`/`structured_explanation`, accepted by user). Topped up the other 6
      components (`darija-tutor-fr-topup-v1`, 381 rows) → merged into **`data/fr_v3_merged/`,
      1,542 rows** (final dataset used for the fine-tune).
- [x] **P8** (I-9) · `check_fr_dataset_preflight.py` built and run against `fr_v3_merged` —
      **PASSED** (100% quiz citation recall, 0 F1-corruption rows, 41.9% socratic multi-turn share
      reported).
- [x] **P9** (I-10/11/12) · Fine-tune run prep done: `kaggle_finetune_fr_v1.ipynb`, base-vs-adapter
      comparison built into Section 10.5 per `LESSONS_LEARNED.md` #1. Pushed and **RUNNING** as of
      that session — see §1 for current-status check command.

### Part II — Darija path (independent, all done)

- [x] **P10** (II-14) · Added `tests/test_llm.py` (15 tests) for `detect_query_language` and
      `_build_system_prompt`, including mutation-verified regression guards for the 2026-08-02
      hyphenated-imperative router bug.
- [x] **P11** (II-15) · `green_light_model.md` §4.2 now has a warning: script the CEO demo in
      Darija only (0/3 clean French answers measured, one mislabelling Moroccan law as French law).
- [x] **P12** (II-16) · `green_light_model.md` now documents the `atlas-darija-tutor-v11`
      (base+adapter, used for recorded demo evidence) vs. production `IBLOG_TUTOR` (merged)
      discrepancy — verified behaviourally equivalent on Darija.
- [x] **P13** (II-17) · Manually read 22 rows from `data/v11_merged` for §4.3 — this is what
      surfaced the quiz-fabrication and refusal domain-mismatch findings §0 is now addressing.

---

## 3. Critical Invariants to Remember

- **Train/serve parity is sacred.** `PRODUCTION_SYSTEM_PROMPT_TEMPLATE`/`_FR` in
  `generate_training_data.py` must stay byte-identical to `SYSTEM_PROMPT_TEMPLATE`/`_FR` in
  `app/services/llm.py`. Both have dedicated tests in `tests/test_generation_gates.py`. **This
  session's deterministic-refusal work (§0) must add new constants alongside these templates,
  never inside them.**
- **French mode skips citation injection entirely** (`normalize_row(..., language="fr")`) —
  Darija mode is unaffected and must keep injecting.
- **`scale_component_targets`** takes an optional 3rd `components` param — default `None`
  preserves every pre-existing call's behavior.
- **Kaggle CLI has no kernel-cancel command**; any run needs a self-terminating time-budget safety
  net inside the script. `kaggle datasets version` must run via PowerShell, not git-bash
  (path-mangling bug). Never upload `.gguf_venv/` or GGUF export folders to a Kaggle dataset.
- **The Darija path (`IBLOG_TUTOR`) must not be touched by French-path generation work** — every
  Darija-specific gate in `generate_component` is guarded by an explicit `language == "darija"`
  check. (Not directly relevant to §0's serving-layer work, but still binding for
  `generate_training_data.py`.)
- **Do not demo French questions on `IBLOG_TUTOR`** until the French fine-tune is validated —
  live-verified to produce incorrect jurisdiction claims 1/3 of the time.
- **`quiz_generation`/`structured_explanation` have a structural ~150-row ceiling** on the current
  36-document French corpus (confirmed twice, not a fixable generation bug).
- **Bare `pytest` works** (not just `pytest tests/`) — the colliding filename was renamed. Suite
  was at 220 tests as of the last session; re-check the count after §0 Step 1's refactor.
- **Two live-confirmed defects exist in production `IBLOG_TUTOR`** (quiz fact fabrication, refusal
  domain-mismatch) — **this is exactly what the current session (§0) is fixing.** Do not treat
  them as already resolved until §0's 4 steps and the live probe are complete.
