"""
Cross-Shard Merge and Deduplication
───────────────────────────────────
Merges the outputs of independently-run generation processes into one dataset,
deduplicating across all of them, then re-splitting train/eval.

Why this exists
───────────────
Generation is sharded across GPUs — each process runs its own `generate_component`
loop with its own in-memory `seen_texts` set and its own final `deduplicate()`
pass. Both are process-local, so a row produced on GPU 0 is never compared
against a row produced on GPU 1.

That is not hypothetical: the 204-row two-GPU sample contained 11 cross-shard
near-duplicates that neither process could see. Same prompts, same corpus, same
temperature — independent samplers land on the same conversation more often than
the per-shard numbers suggest, and the effect scales with shard count.

Splitting train/eval per shard has the same root cause and a worse consequence:
a row in GPU 0's train set and its near-duplicate in GPU 1's eval set is train/test
leakage, which inflates eval scores for a model that has effectively memorised the
answer. Merging before the split is what removes that.

Run this after collecting shard outputs, before training:

    python -m app.services.merge_shards results/out_sample_gpu0 results/out_sample_gpu1 \
        --output data/synthetic/merged
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from app.services.generate_training_data import (
    DEFAULT_EVAL_SPLIT,
    deduplicate,
    split_train_eval,
)

logger = logging.getLogger(__name__)

SHARD_FILES = ("train.jsonl", "eval.jsonl")


def load_shard(shard_dir: Path) -> list[dict]:
    """Read every row a shard produced, ignoring its own train/eval boundary.

    The boundary is discarded deliberately: it was drawn before cross-shard
    duplicates were known, so it cannot be trusted to keep a row and its twin
    on the same side of the split.
    """
    rows = []
    for name in SHARD_FILES:
        path = shard_dir / name
        if not path.exists():
            logger.warning("%s has no %s", shard_dir, name)
            continue
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A shard killed mid-write leaves a truncated final line.
                    logger.warning("%s:%d unparseable, skipped", path, lineno)
                    continue
                row["_shard"] = shard_dir.name
                rows.append(row)
    logger.info("%s → %d rows", shard_dir, len(rows))
    return rows


def exact_dedup(rows: list[dict]) -> list[dict]:
    """Drop byte-identical conversations before the embedding pass.

    Embedding is the expensive step, and identical rows are the cheap case —
    removing them first keeps the O(n²) similarity comparison smaller.
    """
    seen = set()
    keep = []
    for row in rows:
        key = " ".join(
            m["content"] for m in row.get("messages", []) if m.get("role") != "system"
        )
        if key in seen:
            continue
        seen.add(key)
        keep.append(row)
    return keep


def report(label: str, rows: list[dict]):
    components = Counter(r.get("component") for r in rows)
    domains = Counter(r.get("domain") for r in rows)
    shards = Counter(r.get("_shard") for r in rows)
    logger.info("%s: %d rows", label, len(rows))
    logger.info("   components: %s", dict(components))
    logger.info("   domains:    %s", dict(domains))
    if len(shards) > 1:
        logger.info("   shards:     %s", dict(shards))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", help="Shard output directories")
    parser.add_argument("--output", required=True, help="Merged output directory")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--eval-split", type=float, default=DEFAULT_EVAL_SPLIT)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    rows = []
    for shard in args.shards:
        rows.extend(load_shard(Path(shard)))

    if not rows:
        raise SystemExit("No rows found in the given shard directories")

    report("Merged", rows)

    before = len(rows)
    rows = exact_dedup(rows)
    logger.info("Exact dedup: %d → %d rows", before, len(rows))

    rows = deduplicate(rows, threshold=args.threshold)
    logger.info("Cross-shard dedup removed %d of %d rows", before - len(rows), before)

    # `_shard` is bookkeeping for this script's reporting, not part of the
    # training schema — strip it so the written rows stay ChatML-clean.
    for row in rows:
        row.pop("_shard", None)

    train, evalset = split_train_eval(rows, eval_split=args.eval_split)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train.jsonl", train), ("eval.jsonl", evalset)):
        with open(out / name, "w", encoding="utf-8") as fh:
            for row in subset:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("wrote %s (%d rows)", out / name, len(subset))

    report("Final", rows)


if __name__ == "__main__":
    main()
