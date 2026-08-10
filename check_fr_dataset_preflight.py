"""
Post-merge dataset preflight for the French dataset -- MIGRATION_PLAN.md P8.

Hard, scripted numbers instead of manual eyeballing, run against the final
candidate merged dataset before it feeds a fine-tune. Three checks:

  1. Citation recall on citable quiz_generation rows (F2's fix) -- of the
     rows whose context actually contains an extractable legal reference,
     what fraction cite it. Hard gate: must be 100%, matching what was
     verified by hand on data/fr_v3_merged during this session (F2's
     citation_anchor_rule enforces this at generation time; any drop here
     means a stale/corrupted merge, not an expected shortfall).
  2. Multi-turn share on socratic (F3's fix). Reported, not hard-gated --
     this project's own green_light_model.md documents that chasing the
     per-component multi_turn_pct config value as an acceptance bar is a
     measurement-error trap (a generation knob, not a delivered-share
     metric). The number is tracked here so it's visible pre-fine-tune,
     with the known ceiling from this session's own regen run as context.
  3. F1 grammar-corruption pattern count (double-determiner: French
     determiner directly glued to an Arabic word starting with the Arabic
     definite article). Hard gate: must be exactly 0 -- this is the defect
     class that shipped uncaught in fr_v1_merged and was stripped by hand
     twice this session (regen output, top-up output); this script is what
     makes that check repeatable instead of ad hoc.

Exit code 0 = all hard gates pass. Non-zero = at least one hard gate failed.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
from app.services.citations import extract_citations
from app.services.generate_training_data import context_from_system_prompt, row_cites

DATASET_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/fr_v3_merged")

CORRUPT_RE = re.compile(
    r"\b(?:l['’]|la|le|les|du|des|au|aux|à\s+la)\s+ال\S*"
    r"|\b(?:article|loi)\s+ال\S*",
    re.IGNORECASE,
)


def load(fn):
    rows = []
    with open(fn, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_corrupted(row):
    return any(
        m["role"] == "assistant" and CORRUPT_RE.search(m["content"])
        for m in row["messages"]
    )


def main():
    train = load(DATASET_DIR / "train.jsonl")
    eval_ = load(DATASET_DIR / "eval.jsonl")
    rows = train + eval_
    print(f"STATUS: loaded dataset={DATASET_DIR} rows={len(rows)} "
          f"(train={len(train)} eval={len(eval_)})")

    failures = []

    # --- 1. Citation recall on citable quiz rows (hard gate: 100%) --------
    quiz = [r for r in rows if r["component"] == "quiz_generation"]
    citable = 0
    cited = 0
    for r in quiz:
        system_content = next(m["content"] for m in r["messages"] if m["role"] == "system")
        ctx = context_from_system_prompt(system_content)
        anchors = extract_citations(ctx)
        if anchors:
            citable += 1
            if row_cites(r, anchors):
                cited += 1
    recall = (cited / citable * 100) if citable else 100.0
    print(f"STATUS: quiz_citation_recall citable_rows={citable} cited={cited} "
          f"recall={recall:.1f}%")
    if citable and cited != citable:
        failures.append(
            f"quiz citation recall on citable rows is {recall:.1f}% (must be 100%): "
            f"{citable - cited} citable quiz row(s) don't cite their reference"
        )

    # --- 2. Multi-turn share on socratic (reported, not hard-gated) ------
    socratic = [r for r in rows if r["component"] == "socratic"]
    multi = [r for r in socratic if len([m for m in r["messages"] if m["role"] == "assistant"]) > 1]
    share = (len(multi) / len(socratic) * 100) if socratic else 0.0
    print(f"STATUS: socratic_multi_turn_share rows={len(socratic)} multi_turn={len(multi)} "
          f"share={share:.1f}% (design target 75%, this session's regen ceiling ~44%, "
          f"reported not hard-gated -- see green_light_model.md's own note on treating "
          f"multi_turn_pct as a generation knob, not an acceptance bar)")

    # --- 3. F1 corruption pattern (hard gate: 0) --------------------------
    corrupted = [r for r in rows if is_corrupted(r)]
    print(f"STATUS: f1_corruption_count={len(corrupted)}")
    if corrupted:
        by_comp = {}
        for r in corrupted:
            by_comp[r["component"]] = by_comp.get(r["component"], 0) + 1
        failures.append(f"F1 corruption pattern found in {len(corrupted)} row(s): {by_comp}")

    print()
    if failures:
        print("STATUS: preflight_FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("STATUS: preflight_PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
