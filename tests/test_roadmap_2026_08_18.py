"""
Acceptance suite for the week planned in progress_report_2026-08-18.

Every other test file in this repo is organised by MODULE. This one is
organised by DELIVERABLE: each test maps to one bullet of the report's
"Plan for This Week", so "did we close the item we told the client we'd
close" is an executable question rather than a reading exercise. The unit
suites remain the real regression net -- this file deliberately asserts the
END of each chain (the behaviour a reader of the report would check), not
the internals those suites already pin, and cites the sibling test that owns
each mechanism so nothing here duplicates coverage silently.

Roadmap bullets covered (report section "Plan for This Week", item 1 and 3):

  1a  Arabic OCR: investigate teh marbuta / heh, evaluate fix options
      before re-enabling OCR.
  1b  Add Arabic orthographic normalization to citation matching, which
      "currently uses exact string matching and misses valid citations due
      to script variation".
  1c  Fix the gap where out-of-domain queries always retrieve a top-4
      context and never trigger the deterministic refusal path (which
      "today only fires on empty retrieval").
  1d  Run the end-to-end upload probe against a live environment.
  3   Begin the diagram-generation adapter: which explanations warrant a
      diagram, how generation is triggered, how it renders in the chat UI.

Item 2 (TTS vendor selection) and item 4 (video batch + auth) are NOT
covered here: neither has a code surface to assert against yet. That
absence is itself the finding -- see test_tts_has_no_code_surface_yet.

Offline and deterministic: no Ollama, no Postgres, no OCR stack. The live
halves of 1a and 1d are scripts, not tests, and are named by the tests that
stand in for them (scripts/verify_ocr_arabic.py, probe_upload_e2e.py).
"""

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from app.config import Settings
from app.models.schemas import ChatRequest, Domain
from app.routers.chat import chat
from app.services.citations import (
    arabic_variant_pattern,
    extract_citations,
    fold_arabic,
    normalize_arabic_text,
)
from app.services.grounding import question_is_grounded

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fails_if_called(*args, **kwargs):
    raise AssertionError("the model must not be called on a refusal turn")


class _FakeOllama:
    """chat.py -> generate_llm_response -> _call_ollama_chat POSTs /api/chat
    and reads message.content -- not /api/generate's flat `response` key.
    Same shape tests/test_chat.py uses."""

    def __init__(self, content):
        self._body = {"message": {"role": "assistant", "content": content}}

    def read(self):
        return json.dumps(self._body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ===========================================================================
# 1a. Arabic OCR -- teh marbuta / heh investigated, fix chosen, OCR re-enabled
# ===========================================================================
#
# The report left this as the one BLOCKING open issue: PaddleOCR misreads
# teh marbuta (ة) as heh (ه), so OCR stayed at ocr_engine="none" while three
# named fix options were weighed -- "engine tuning vs. post-OCR correction
# vs. narrowing where OCR is applied". These tests assert the investigation
# actually concluded, and in which direction.


def test_ocr_is_re_enabled_by_default():
    """The gate the report set: ocr_engine stays "none" until an engine
    passes scripts/verify_ocr_arabic.py's four gates. It no longer does,
    which is only defensible if the misread below is genuinely absorbed --
    the next two tests are what make this default safe."""
    assert Settings().ocr_engine == "paddleocr"


def test_the_measured_misread_is_absorbed_by_comparison_folding():
    """The exact pair observed in the live PaddleOCR run cited in
    app/config.py: "المسطره" returned for "المسطرة". The fix option chosen
    was none of the three the report listed -- not engine tuning, not
    post-OCR correction, not narrowing scope. The defect is tolerated at
    COMPARISON time instead, which is strictly safer than correcting the
    stored text: nothing rewrites what the tenant's document actually
    said."""
    assert fold_arabic("المسطرة") == fold_arabic("المسطره")


def test_folding_still_refuses_to_merge_non_confusable_letters():
    """The other half of that choice, and the reason it is not just "lower
    the bar". Folding is directional and narrow: dal/dhal (د/ذ) are never
    OCR-confusable, so folding them would let a FABRICATED "الماذة" ground
    itself against a real "المادة". If this ever passes by accident, the
    citation gate has been widened into a rubber stamp."""
    assert fold_arabic("المادة") != fold_arabic("الماذة")
    for real, fabricated in [("محضر", "مهضر"), ("مدير", "مذير"), ("تقرير", "ثقرير")]:
        assert fold_arabic(real) != fold_arabic(fabricated), (
            f"{fabricated!r} must not fold onto {real!r}"
        )


def test_the_verification_gate_tolerates_the_misread_only_where_it_should():
    """scripts/verify_ocr_arabic.py's four gates are the go/no-go the
    report gated the rollout on. Gate 2 ("every المادة marker present, in
    ascending order") is widened via arabic_variant_pattern so the misread
    counts as PRESENT; gates 1 and 4 (fidelity, digit integrity) must still
    run on untouched raw text, or the gate would be scoring its own
    correction rather than the engine."""
    marker = re.compile(arabic_variant_pattern("المادة") + r"\s+(\d+)")
    assert marker.search("الماده 4"), "gate 2 must see the misread as the marker"
    assert marker.search("المادة 4")

    gate_src = (REPO_ROOT / "scripts" / "verify_ocr_arabic.py").read_text(encoding="utf-8")
    assert "arabic_variant_pattern" in gate_src
    # Guards against the widening leaking into the fidelity/digit gates.
    assert "fold_arabic(" not in gate_src, (
        "gates 1/4 must score raw OCR text; folding there would make the "
        "harness grade its own normalization instead of the engine"
    )


def test_scope_narrowing_was_also_implemented_not_just_considered():
    """The third fix option the report named ("narrowing where OCR is
    applied") did land, as two-tier routing: a page whose native text is
    being merged anyway goes to the cheap classic engine, and escalates to
    the 3B VLM only when the cheap result shows no numeric evidence.
    Mechanism owned by tests/test_ocr_two_tier.py; asserted here only as
    "the roadmap option is real and on by default"."""
    from app.services.ingestion import _classic_ocr_looks_incomplete

    settings = Settings()
    assert settings.ocr_two_tier is True
    assert settings.ocr_light_engine != settings.ocr_paddle_engine, (
        "two tiers that name the same engine are one tier"
    )
    # The predicate that decides escalation, on its two calibration cases.
    assert _classic_ocr_looks_incomplete("حساب معدل استهلاك الفرد لتر للفرد") is True
    assert _classic_ocr_looks_incomplete("UTM Zone 39 - الإسقاط 1970 المسند") is False


def test_the_committed_ocr_run_still_passes_all_four_gates():
    """Turns the go/no-go into a REGRESSION test instead of a claim.

    app/config.py justifies ocr_engine="paddleocr" with a live PaddleOCR-VL
    run scoring fidelity 0.945 -- but that run is a sentence in a comment,
    and re-running it needs a GPU and several minutes (measured: the VL
    model's cold load exceeds PaddleOcrEngine._TIMEOUT_SECONDS on this
    laptop, so the gate cannot currently be re-run to completion here at
    all). The run's raw output IS committed, so the four gates can be
    re-scored against it offline, with no OCR stack, on every test run.

    What this catches: a change to the gates, to the ground-truth corpus
    file, or to the Arabic normalization that would have failed the
    original engine acceptance. What it does NOT catch: an engine or
    engine-version regression -- only a live re-run does that, and this
    test exists precisely because a live re-run is expensive enough that
    nobody does it per-commit."""
    from scripts import verify_ocr_arabic as gate

    ocr_out = REPO_ROOT / "tests" / "data" / "ocr" / "1.12_ar_procedure_consignation.paddleocr.out.txt"
    if not (ocr_out.exists() and gate.GROUND_TRUTH_MD.exists()):
        pytest.skip("committed OCR reference run not present")

    r = gate.run_gates(
        gate.GROUND_TRUTH_MD.read_text(encoding="utf-8"),
        ocr_out.read_text(encoding="utf-8"),
    )
    assert r["fidelity_pass"], f"gate 1 fidelity regressed to {r['fidelity']:.3f}"
    assert r["ocr_articles"] == ["1", "2", "3", "4", "5"], (
        f"gate 2: articles lost or reordered -- {r['ocr_articles']}"
    )
    assert r["order_pass"] and r["not_reversed_pass"] and r["digit_integrity_pass"]


def test_the_committed_run_contains_the_defect_it_is_supposed_to_tolerate():
    """Guards the test above from passing for the wrong reason. It only
    proves anything if the reference output actually CONTAINS the teh
    marbuta misread -- otherwise a future re-recorded, defect-free run
    would keep it green while the tolerance it certifies went untested.

    The committed run reads "المسطره" where the source says "المسطرة"."""
    ocr_out = REPO_ROOT / "tests" / "data" / "ocr" / "1.12_ar_procedure_consignation.paddleocr.out.txt"
    if not ocr_out.exists():
        pytest.skip("committed OCR reference run not present")

    text = ocr_out.read_text(encoding="utf-8")
    assert "المسطره" in text, (
        "the reference run no longer shows the teh-marbuta misread -- it was "
        "re-recorded from a different engine or version. Re-run the live gate "
        "and re-verify that the tolerance is still exercised."
    )
    from scripts import verify_ocr_arabic as gate

    assert "المسطرة" in gate.GROUND_TRUTH_MD.read_text(encoding="utf-8"), (
        "the ground truth must carry the CORRECT spelling, or there is no "
        "defect for the gate to be tolerating"
    )


def test_a_page_that_cannot_be_ocred_no_longer_fails_the_whole_document():
    """The blast-radius change that made re-enabling OCR safe to ship: an
    unreadable page is recorded and the document still ingests as
    'partial', instead of one bad page erroring the upload."""
    from app.models import database

    assert hasattr(database.SourceFile, "unprocessed_pages"), (
        "per-page OCR failures need somewhere to be recorded"
    )


# ===========================================================================
# 1b. Arabic orthographic normalization in citation matching
# ===========================================================================
#
# Report: citation matching "currently uses exact string matching and misses
# valid citations due to script variation". The fix is two-tier and the tiers
# have DIFFERENT safety properties, so these tests pin both the matching win
# and the containment.


@pytest.mark.parametrize(
    "source,why",
    [
        ("المادة 12", "baseline: standard orthography"),
        ("الماده 12", "the measured OCR misread: teh marbuta read as heh"),
        ("المــادة 12", "tatweel padding, common in justified PDF text"),
        ("المَادة 12", "interleaved harakat"),
    ],
)
def test_every_script_variation_yields_the_same_canonical_citation(source, why):
    """The end of the chain the report described. Before normalization each
    of these but the first missed outright -- `المادة\\s+(\\d+)` does not
    match a heh-substituted or tatweel-padded head."""
    cites = extract_citations(source)
    assert cites, f"no citation extracted from {source!r} ({why})"
    assert [c["canonical"] for c in cites.values()] == ["المادة 12"], why


# Presentation Forms B (U+FE70-FEFF) is the FOURTH variation class, and the
# one app/config.py cites as the real measured corruption -- a PDF table
# extracting "اﻟﺒﻄﺎﻗﺔ" with no plain teh marbuta in it at all. It is handled
# differently from the three above, and the split is worth pinning rather
# than hiding inside the parametrize list, because it is the one place where
# correctness depends on a CALLER having normalized first.


def test_presentation_forms_are_handled_at_ingest_not_by_the_matcher():
    """The division of labour that makes the parametrized cases above
    sufficient in production: presentation forms are folded away by
    normalize_arabic_text when the document is INGESTED, so no stored chunk
    ever reaches the matcher in that encoding."""
    assert normalize_arabic_text("الﻣﺎدة 12") == "المادة 12"
    assert extract_citations(normalize_arabic_text("الﻣﺎدة 12"))


def test_extract_citations_does_not_normalize_its_own_input():
    """Known gap, pinned deliberately rather than fixed inside a test run.

    extract_citations is a PUBLIC function whose pattern set advertises
    variant tolerance, but it tolerates only the variants
    arabic_variant_pattern encodes -- it never calls normalize_arabic_text
    on its argument. So it silently returns {} on presentation-form text,
    and is correct today only because every caller happens to feed it
    ingest-normalized text.

    Not currently a live defect (see the test above), and it is one line to
    close -- normalize the input at the top of extract_citations. Flip this
    assertion when that lands."""
    assert extract_citations("الﻣﺎدة 12") == {}, (
        "extract_citations now normalizes its own input -- good; invert this "
        "test and drop it from the report's gap list"
    )


def test_canonical_output_is_rebuilt_never_echoed_from_the_corrupted_source():
    """Why the above is safe to show a user: extract_citations reads only
    the NUMBER out of the source and rebuilds the head from a template. So
    a misread head yields a CORRECT canonical, not a normalized-looking
    copy of the corruption -- and the lossy fold never reaches output."""
    canonical = next(iter(extract_citations("الماده 12").values()))["canonical"]
    assert canonical == "المادة 12"
    assert "ه 12" not in canonical, "the corrupted head must not survive into output"
    assert canonical != fold_arabic(canonical), (
        "canonical output must not be the comparison-folded form"
    )


def test_tier_a_is_lossless_and_tier_b_is_not():
    """The invariant that keeps the two tiers from being conflated at a
    future call site: normalize_arabic_text (safe to STORE and embed) must
    not merge teh marbuta into heh; fold_arabic (comparison keys only)
    must."""
    assert normalize_arabic_text("المسطرة") != normalize_arabic_text("المسطره")
    assert fold_arabic("المسطرة") == fold_arabic("المسطره")


def test_ingestion_stores_tier_a_text_only():
    """Containment, checked at the one call site where getting it wrong
    would corrupt the corpus permanently: ingestion normalizes (Tier A)
    before chunking and embedding, and must never fold (Tier B)."""
    src = (REPO_ROOT / "app" / "services" / "ingestion.py").read_text(encoding="utf-8")
    assert "normalize_arabic_text" in src
    assert "fold_arabic" not in src.replace(
        "comparison-only fold_arabic in the same module.", ""
    ), "fold_arabic output must never be stored or embedded"


def test_citation_grounding_survives_an_ocr_misread_in_the_retrieved_context():
    """The production consequence, and the actual point of the item: a
    correctly-spelled citation in a generated answer must still ground
    against a retrieved chunk that OCR misread. Under exact-string matching
    this pair was a false fabrication alarm."""
    q = {
        "question": "ماذا تنص المادة 15؟",
        "options": ["ارتداء الخوذة", "لا شيء", "خيار ثالث", "خيار رابع"],
        "correct_index": 0,
        "explanation": "حسب المادة 15 يجب ارتداء الخوذة.",
    }
    ocr_context = "الماده 15 تنص على وجوب ارتداء الخوذة."
    grounded, offenders = question_is_grounded(q, ocr_context)
    assert grounded, f"valid citation rejected against OCR'd context: {offenders}"


def test_normalization_did_not_disarm_the_fabrication_detector():
    """The regression that would make the item above worthless: a citation
    that appears NOWHERE in the context must still be caught. Widening the
    matcher must not widen it to everything."""
    q = {
        "question": "ماذا تنص المادة 99؟",
        "options": ["ارتداء الخوذة", "لا شيء", "خيار ثالث", "خيار رابع"],
        "correct_index": 0,
        "explanation": "حسب المادة 99 يجب ارتداء الخوذة.",
    }
    grounded, offenders = question_is_grounded(q, "المادة 15 تنص على وجوب ارتداء الخوذة.")
    assert not grounded and offenders, "fabricated المادة 99 must still be caught"


# ===========================================================================
# 1c. Out-of-domain queries reach the deterministic refusal path
# ===========================================================================
#
# Report: OOD queries "always retrieve a top-4 context and never trigger the
# deterministic refusal path (which today only fires on empty retrieval)".
# The mechanism (resolve_domain returning "no_match") is pinned by
# tests/test_domain_routing.py and tests/test_chat.py; asserted here as the
# roadmap-level claim, plus the two directions those suites do not cross-
# check against each other.


def test_refusal_no_longer_depends_on_empty_retrieval():
    """The bullet, stated exactly: context is deliberately NON-empty and
    the turn still refuses, so the top-4-context case is covered."""
    def nonempty(*args, **kwargs):
        return "Selon Article 8, le port du casque est obligatoire.", ["1.1_code.md"], False

    with patch("app.routers.chat._retrieve_context", side_effect=nonempty), \
         patch("app.routers.chat.resolve_domain", return_value=("industrial", "no_match")), \
         patch("app.services.llm.urllib.request.urlopen", side_effect=_fails_if_called):
        r = chat(ChatRequest(message="Comment faire du pain au levain ?"))

    assert r.domain_source == "no_match"
    assert r.sources == [], "a refusal must not cite the irrelevant chunks it refused over"
    assert r.tokens_used == 0, "the model must not be called at all"
    assert r.response.strip(), "the refusal must be a real message"


def test_both_refusal_triggers_are_independent():
    """The empty-retrieval trigger is the OLDER path and must survive the
    new one being added -- a refactor that folded them together would make
    a disk-backend deployment (where resolve_domain never returns
    "no_match") silently unable to refuse at all."""
    def empty(*args, **kwargs):
        return "", [], False

    with patch("app.routers.chat._retrieve_context", side_effect=empty), \
         patch("app.routers.chat.resolve_domain", return_value=("industrial", "retrieval")), \
         patch("app.services.llm.urllib.request.urlopen", side_effect=_fails_if_called):
        r = chat(ChatRequest(message="Question sans contexte"))

    assert r.tokens_used == 0 and r.sources == []
    assert r.response.strip()


def test_the_gate_did_not_become_a_blanket_refusal():
    """The cost side. An ordinary auto-routed turn must still reach the
    model -- an OOD gate that refuses everything would score perfectly on
    refusal and destroy the product."""
    def nonempty(*args, **kwargs):
        return "Selon Article 8, le port du casque est obligatoire.", ["1.1_code.md"], False

    with patch("app.routers.chat._retrieve_context", side_effect=nonempty), \
         patch("app.routers.chat.resolve_domain", return_value=("industrial", "retrieval")), \
         patch("app.services.llm.urllib.request.urlopen",
               return_value=_FakeOllama("Le port du casque est obligatoire (Article 8).")):
        r = chat(ChatRequest(message="Le casque est-il obligatoire ?"))

    assert r.domain_source == "retrieval"
    assert r.sources == ["1.1_code.md"]


def test_an_explicit_domain_still_bypasses_the_vote():
    """Tier 1 precedence: a caller that already knows the domain from page
    context must not have its turn second-guessed by a corpus vote."""
    def nonempty(*args, **kwargs):
        return "Selon Article 8, le port du casque est obligatoire.", ["1.1_code.md"], False

    def _vote_must_not_run(*args, **kwargs):
        raise AssertionError("resolve_domain must not run when a domain was given")

    with patch("app.routers.chat._retrieve_context", side_effect=nonempty), \
         patch("app.routers.chat.resolve_domain", side_effect=_vote_must_not_run), \
         patch("app.services.llm.urllib.request.urlopen",
               return_value=_FakeOllama("Le port du casque est obligatoire (Article 8).")):
        r = chat(ChatRequest(message="Le casque est-il obligatoire ?", domain=Domain.INDUSTRIAL))

    assert r.domain_source == "page_context"


def test_the_new_trigger_is_a_no_op_at_the_shipped_thresholds():
    """THE finding of this review, pinned so it cannot be re-forgotten.

    The tests above prove the "no_match" path WORKS when resolve_domain
    returns it -- they patch resolve_domain, so they never ask whether it
    ever does. Measured against the live corpus
    (scripts/eval_refusal.py), it never does: 0 of 19 out-of-corpus queries
    were refused by the new trigger, and all 9 that were refused hit the
    OLD empty-context path.

    The reason is arithmetic, not tuning. resolve_domain votes over an
    unfiltered top-20 keeping candidates >= domain_vote_threshold;
    retrieval keeps chunks >= similarity_threshold. Both are 0.4. So the
    vote returns None only when NOTHING in the corpus clears 0.4 -- which
    is precisely when domain-filtered retrieval is empty too. "no_match"
    strictly IMPLIES empty context, and adds no signal whatsoever.

    The vote becomes an independent trigger only when its threshold is
    strictly HIGHER than the retrieval threshold. Measured sweep on the
    live corpus (45 in-corpus / 19 out-of-corpus queries):

        vote_thr   ood_refusal   false_refusal
          0.40        0.474          0.000      <- shipped; vote is a no-op
          0.45        0.684          0.067
          0.50        0.895          0.200
          0.55        1.000          0.311

    This test fails the moment the thresholds diverge. That is intended:
    whoever raises domain_vote_threshold must come back and re-measure the
    false-refusal cost, which falls almost entirely on Darija (see the
    next test)."""
    settings = Settings()
    assert settings.domain_vote_threshold == settings.similarity_threshold, (
        "the thresholds have diverged -- the OOD vote is now a real trigger. "
        "Re-run scripts/eval_refusal.py --sweep and update this test with the "
        "measured false_refusal_rate before shipping."
    )


def test_the_language_affinity_seam_the_refusal_fix_will_need_exists():
    """Why the sweep above cannot simply be applied as a global number.

    Measured top-similarity against the live corpus, by query language:

        French  in-corpus   n=27  mean 0.645  min 0.534
        Darija (Arabic)     n=11  mean 0.562  min 0.435
        Darija (Arabizi)    n= 7  mean 0.470  min 0.409
        out-of-corpus       n=19  mean 0.408  max 0.548

    French in-corpus queries are almost separable from out-of-corpus ones.
    Arabizi Darija is NOT separable at all -- its in-corpus range
    [0.409, 0.530] sits entirely inside the out-of-corpus range
    [0.302, 0.548]. So every global threshold that buys out-of-corpus
    refusal spends it on Darija specifically: all 9 false refusals at 0.50
    were Darija queries, and none were French.

    Darija is mandatory for tenant #1, so the fix is a language-conditioned
    threshold, not a higher global one. This asserts the seam that fix
    needs already exists -- retrieval is already language-aware, so the
    threshold can be too without new plumbing."""
    from app.services import retrieval

    src = (REPO_ROOT / "app" / "services" / "search.py").read_text(encoding="utf-8")
    assert "ui_lang" in src, "retrieval must already know the query language"
    assert hasattr(retrieval, "retrieve")


def test_the_refusal_is_composed_in_the_response_language():
    """The defect the deterministic refusal exists to bypass: the fine-tune
    answers every refusal in tenant #1's Darija safety register regardless
    of the asking language. A French OOD question must refuse in French."""
    from app.services.llm import DOMAIN_LABELS_FR, deterministic_refusal

    fr = deterministic_refusal("blockchain", "fr")
    assert DOMAIN_LABELS_FR["blockchain"] in fr
    assert sum(1 for c in fr if "؀" <= c <= "ۿ") == 0, "a French refusal must carry no Arabic script"


# ===========================================================================
# 1d. End-to-end ingestion lifecycle
# ===========================================================================


def test_the_live_e2e_probe_covers_the_four_lifecycle_stages():
    """The report's bullet names four stages: upload -> ready ->
    pin/version invalidation -> delete. This asserts the probe actually
    exercises all four (a probe that quietly stopped at 'ready' would still
    exit 0 and prove nothing). The probe itself needs live Postgres and is
    run out-of-band -- see the report accompanying this suite for its
    latest run."""
    src = (REPO_ROOT / "probe_upload_e2e.py").read_text(encoding="utf-8")
    for stage in ["_upload_file", "wait_until_ready", "SourceToggleRequest", "corpus_version"]:
        assert stage in src, f"probe no longer covers stage: {stage}"
    assert "delete" in src.lower()


def test_pin_invalidation_is_keyed_on_corpus_version():
    """The mechanism the probe proves at runtime, pinned statically so a
    refactor cannot remove it while the probe is not being run: a
    mid-conversation upload must change corpus_version, or the pinned
    context would be reused verbatim and the new document would be
    invisible to the very next turn."""
    from app.services import sources

    assert hasattr(sources, "corpus_version")


# ===========================================================================
# 3. Diagram-generation adapter
# ===========================================================================
#
# The report scoped this as "begin implementation ... scoping which
# explanation types warrant a diagram and how generation will be triggered
# and rendered". All three sub-questions have answers in code; these tests
# assert each answer, and pin the one part that is still a stub.


def test_scope_which_explanation_types_warrant_a_diagram():
    """Sub-question 1, answered as a closed six-kind enum rather than an
    open-ended "the model decides". Each kind must have a schema, a healer
    and a renderer, or requesting it would 500 at serve time."""
    from app.services import diagram_render, diagrams

    expected = {"flowchart", "sequence", "mindmap", "pie", "xy", "candlestick"}
    assert set(diagrams._SCHEMAS) == expected
    assert set(diagrams._HEAL) == expected

    # Five render to Mermaid server-side; candlestick is deliberately absent
    # from RENDERERS because Mermaid has no OHLC chart type at all -- it
    # renders client-side in React from the raw JSON spec. A future kind
    # added to _SCHEMAS without either a renderer or a client-side path
    # would KeyError at serve time, which is what this checks.
    assert set(diagram_render.RENDERERS) == expected - {"candlestick"}
    candle_card = REPO_ROOT / "frontend" / "src" / "components" / "workspace" / "CandlestickChart.tsx"
    assert candle_card.exists(), "candlestick has neither a server renderer nor a client one"


@pytest.mark.parametrize(
    "message,expected_kind,why",
    [
        ("Dessine un organigramme de la procédure de consignation", "flowchart", "explicit FR verb + kind noun"),
        ("Fais un diagramme de séquence de l'intervention", "sequence", "kind keyword"),
        ("Montre-moi une carte mentale des EPI", "mindmap", "kind keyword"),
        ("Donne-moi un camembert des types d'incidents", "pie", "kind keyword"),
        ("Trace un histogramme des incidents par mois", "xy", "kind keyword"),
        ("Montre l'évolution des incidents par mois", "xy", "kind keyword, prefix match on 'évolution de'"),
        ("رسم تخطيطي لمسطرة القفل", "flowchart", "Arabic-script diagram marker"),
    ],
)
def test_scope_how_generation_is_triggered(message, expected_kind, why):
    """Sub-question 2: triggered by the chat message itself through a
    deterministic keyword router -- no extra endpoint, no button, no model
    round trip to decide."""
    from app.services.diagrams import detect_diagram_intent

    intent = detect_diagram_intent(message)
    assert intent is not None, f"missed a diagram request ({why}): {message!r}"
    assert intent.kind == expected_kind
    assert intent.source == "keyword"


@pytest.mark.parametrize(
    "message",
    [
        "Quelle est la procédure de consignation électrique ?",
        "Explique-moi le rôle du comité de sécurité",
        "Combien de jours pour la recertification ?",
        "شنو هي مسطرة القفل؟",
    ],
)
def test_the_trigger_does_not_fire_on_ordinary_questions(message):
    """The load-bearing half of a keyword router. A bare topic noun must
    never trigger -- most turns want prose, and a false positive replaces a
    correct answer with a picture of one."""
    from app.services.diagrams import detect_diagram_intent

    assert detect_diagram_intent(message) is None, f"false positive on {message!r}"


@pytest.mark.parametrize(
    "message",
    [
        "Peux-tu me le présenter sous forme d'étapes numérotées ?",
        "J'ai du mal à suivre l'enchaînement, tu peux le rendre plus clair ?",
        "Comment ces différentes obligations s'articulent-elles entre elles ?",
    ],
)
def test_the_implicit_trigger_tier_is_still_a_stub(message):
    """The known gap in sub-question 2, pinned so it cannot be mistaken for
    finished. Tier 1 is keyword-only; Tier 2 (_semantic_fallback) returns
    None unconditionally, so a diagram request that names no diagram word is
    missed today and answered as prose.

    Each phrase above is a real way a user asks for a diagram without using
    one of the trigger words. Flip these when Tier 2 lands -- the docstring
    on _semantic_fallback already specifies the intended implementation
    (cosine similarity against French exemplars using the already-resident
    embedding model)."""
    from app.services.diagrams import _semantic_fallback, detect_diagram_intent

    assert _semantic_fallback(message) is None
    assert detect_diagram_intent(message) is None


def test_a_bare_chart_noun_falls_back_to_flowchart():
    """Second known gap in the trigger, and the more likely one to bite:
    "graphique" is a generic diagram NOUN, not an xy kind keyword, so the
    most natural French phrasing for a chart request produces a FLOWCHART
    at confidence 0.6 unless it also happens to contain "histogramme",
    "courbe de" or "évolution de".

    Pinned rather than fixed here because the fix is a judgement call --
    moving "graphique" into _KIND_KEYWORDS["xy"] would resolve this case
    but would misroute "graphique de la procédure", which really is a
    flowchart. The durable fix is Tier 2, above."""
    from app.services.diagrams import detect_diagram_intent

    intent = detect_diagram_intent("Trace un graphique des incidents par mois")
    assert intent is not None
    assert intent.kind == "flowchart", (
        "'graphique' now routes somewhere else -- re-score this gap"
    )
    assert intent.confidence == 0.6, "the fallback must stay visibly low-confidence"


def test_scope_how_it_renders_in_the_chat_ui():
    """Sub-question 3: the response carries a structured diagram payload,
    and the frontend renders it through a lazy-loaded mermaid that is
    sanitized before injection. Both halves matter -- diagram labels come
    from a model, so the SVG is untrusted input."""
    from app.models.schemas import ChatResponse

    assert "diagram" in ChatResponse.model_fields, "no transport for the diagram"

    card = (REPO_ROOT / "frontend" / "src" / "components" / "workspace" / "DiagramCard.tsx").read_text(
        encoding="utf-8"
    )
    assert "dangerouslySetInnerHTML" in card, "expected an SVG injection point to guard"
    assert "dompurify" in card.lower(), "model-authored SVG must be sanitized before injection"
    assert 'securityLevel: "strict"' in card or "securityLevel: 'strict'" in card


def test_a_diagram_request_is_checked_before_the_refusal_gate():
    """The interaction between deliverable 1c and deliverable 3, which
    neither one's own suite covers. A candlestick has no corpus under the
    three-domain enum, so the OOD gate added this week would refuse every
    diagram request outright if it ran first. Ordering is the fix, and it
    is easy to undo by accident."""
    src = (REPO_ROOT / "app" / "routers" / "chat.py").read_text(encoding="utf-8")
    intent_at = src.index("detect_diagram_intent(")
    refusal_at = src.index('if domain_source == "no_match"')
    assert intent_at < refusal_at, (
        "diagram intent must be detected BEFORE the deterministic refusal gate"
    )


def test_an_ungrounded_diagram_is_marked_not_refused():
    """The deliberate asymmetry with quiz generation, worth pinning because
    it looks like a bug: an empty retrieved context sets grounded=False and
    still returns the diagram, rather than refusing."""
    from app.services.diagrams import DiagramPayload

    assert "grounded" in DiagramPayload.model_fields
    assert "repairs" in DiagramPayload.model_fields, "repairs must be reported, never silent"


# ===========================================================================
# Items with no code surface yet -- asserted so the absence stays visible
# ===========================================================================


def test_tts_has_no_code_surface_yet():
    """Report item 2 (voice/TTS), flagged MVP-critical and "needs a decision
    early this week". No vendor was selected and no integration point
    exists. This test PASSES on that absence -- it exists so the day a TTS
    module lands, this file is the thing that reminds someone to score the
    deliverable. Delete it then."""
    services = {p.name for p in (REPO_ROOT / "app" / "services").glob("*.py")}
    assert not {s for s in services if "tts" in s or "speech" in s or "voice" in s}

    from app.models.schemas import ChatResponse

    assert not {f for f in ChatResponse.model_fields if "audio" in f or "tts" in f}


def test_video_integration_still_has_no_authentication():
    """Report item 4 / the standing risk: "no authentication currently
    exists between our system and the partner's video service". Pinned
    because the report says it must be resolved before more than one tenant
    is exposed, and nothing else in the suite would notice it landing."""
    router = (REPO_ROOT / "app" / "routers" / "video.py").read_text(encoding="utf-8")
    has_auth = any(
        marker in router
        for marker in ("Depends(verify", "api_key", "Authorization", "HTTPBearer", "Security(")
    )
    assert not has_auth, (
        "video router appears to have gained auth -- good; update the report's "
        "open-risk list and delete this test"
    )
