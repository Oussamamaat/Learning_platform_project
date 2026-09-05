"""
End-to-end regression coverage for resolve_domain() against REAL Postgres
and REAL Arabic-script (Darija) text -- the exact gap the 2026-09-04 Akash
lease found: tests/test_routing.py and tests/test_domain_routing.py mock
search_similar_chunks and use no Arabic; tests/test_chat.py's Arabic tests
all pass `domain` explicitly, so they never reach resolve_domain() at all.
No existing test exercised the unmocked resolve_domain -> search_similar_
chunks -> Postgres path with Arabic-script input before this file.

Requires: Postgres reachable (config/docker-compose.yml's `db` service) AND
the corpus seeded (see docs/deploy/lease-00-seed.sh / POST_LEASE_MVP_SPRINT_
PLAN.md item 1's Phase A reproduction steps -- `ingest_directory('raw/
shared', tenant_id='company_abc')`, expect 25 files / 37 chunks). Skips
cleanly (not a failure) when either isn't true, via the session-scoped
postgres_reachable fixture tests/conftest.py already provides.

This suite reproduces the OUTCOME of the lease's live finding (implicit-
domain Darija queries must route correctly, not fall to tenant_default with
an instant refusal); it does not and cannot reproduce the lease's specific
runtime failure locally (see POST_LEASE_MVP_SPRINT_PLAN.md item 1 -- zero
exceptions were raised across all 50 tests/data/retrieval_eval.jsonl
queries, including the exact failing lease message, against a freshly
seeded local Postgres). Green here means the routing LOGIC is correct
against this corpus; it does not by itself clear the still-open Phase B
question of what raised on the lease.
"""
import json
from pathlib import Path

import pytest

from app.services.routing import resolve_domain

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = REPO_ROOT / "tests" / "data" / "retrieval_eval.jsonl"

TENANT_ID = "company_abc"


def _load_darija_rows() -> list[dict]:
    rows = []
    with open(EVAL_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["language"] == "ary":
                rows.append(row)
    return rows


DARIJA_ROWS = _load_darija_rows()
# ood-* rows are deliberately out-of-corpus (gold_sources: [], by design
# meant to trigger a refusal -- see tests/data/retrieval_eval.jsonl and
# scripts/eval_refusal.py) -- their "domain" field is a distractor label,
# not a routing target, so they're excluded from the strict
# domain-must-match assertion below but still covered by the
# tenant_default assertion every row gets (that's the actual lease symptom
# under test, and it applies regardless of whether the query is in-corpus).
IN_CORPUS_DARIJA_ROWS = [r for r in DARIJA_ROWS if not r["id"].startswith("ood-")]


@pytest.fixture(autouse=True)
def _require_seeded_postgres(postgres_reachable):
    if not postgres_reachable:
        pytest.skip("Postgres not reachable -- see this file's docstring for setup")


@pytest.mark.parametrize("row", DARIJA_ROWS, ids=[r["id"] for r in DARIJA_ROWS])
def test_implicit_domain_darija_query_never_falls_to_tenant_default(row):
    """The exact shape of the 2026-09-04 lease bug: a real Darija query, NO
    explicit domain, unmocked resolve_domain against real Postgres. Must
    never silently fall through to "tenant_default" (the symptom: instant
    deterministic refusal with sources=[]) -- applies to every ary row,
    in-corpus or deliberately out-of-corpus (an OOD query should resolve
    via "no_match", a real positive signal, not an exception swallow that
    happens to look identical from the response alone)."""
    _domain, domain_source = resolve_domain(
        row["query"], tenant_id=TENANT_ID, backend="pgvector",
    )
    assert domain_source != "tenant_default", (
        f"{row['id']}: tier-2 domain routing fell through to the tenant "
        f"default for a real Darija query -- this is the exact swallowed-"
        f"exception (or disk-backend) symptom the 2026-09-04 lease found. "
        f"query={row['query']!r}"
    )


@pytest.mark.parametrize(
    "row", IN_CORPUS_DARIJA_ROWS, ids=[r["id"] for r in IN_CORPUS_DARIJA_ROWS]
)
def test_implicit_domain_darija_query_routes_to_labelled_domain(row):
    """Stronger than the tenant_default check above: an in-corpus Darija
    query must land on its labelled domain via a real retrieval vote, not
    just avoid the tenant-default fallback."""
    domain, domain_source = resolve_domain(
        row["query"], tenant_id=TENANT_ID, backend="pgvector",
    )
    assert domain_source == "retrieval", (
        f"{row['id']}: expected domain_source='retrieval', got {domain_source!r} "
        f"for query={row['query']!r}"
    )
    assert domain == row["domain"], (
        f"{row['id']}: expected domain={row['domain']!r}, got {domain!r} "
        f"for query={row['query']!r}"
    )


def test_darija_eval_set_is_actually_arabic_script():
    """Guards the guard: if retrieval_eval.jsonl's ary rows were ever
    replaced with uniformly toy/sanitized stand-ins, this suite would
    silently stop testing what it claims to. The set is a deliberate MIX
    of real Arabic script and Arabizi (Latin-script Darija, e.g. "chno
    howa dyal...") -- that's realistic code-switched input, not a defect
    -- so this only asserts BOTH varieties are genuinely present, not that
    every row is Arabic script."""
    assert len(DARIJA_ROWS) == 20
    has_arabic_script = [
        r for r in DARIJA_ROWS
        if sum(1 for c in r["query"] if "؀" <= c <= "ۿ") > 0
    ]
    assert has_arabic_script, "expected at least some Arabic-script ary rows"
    assert len(has_arabic_script) < len(DARIJA_ROWS), (
        "expected at least some Arabizi (Latin-script) ary rows too -- "
        "if every row is Arabic script, the code-switching realism is gone"
    )
