"""
Regression tests for deterministic citation injection.

Every case here corresponds to a defect found in generated data, not to a
hypothetical. The module rewrites text that goes straight into training rows
and into production answers, so a silent regression here is expensive: it
either corrupts citations in the dataset or, worse, makes an invented
reference look verifiable.
"""

import pytest

from app.services.citations import (
    detect_target_script,
    extract_citations,
    inject_citations,
)

ARABIC_SOURCE = """
# القانون رقم 27.06 المتعلق بأنشطة الحراسة ونقل الأموال

**المادة 1**
يخضع لهذا القانون كل شخص يقوم بأعمال الحراسة.

**المادة 8**
تحدد شروط مزاولة مهنة أعوان الحراسة.

**المادة 12**
تنظم عمليات التفتيش الجسدي.
"""

FRENCH_SOURCE = "La Loi N° 42-25 encadre les actifs numeriques. Voir ISO 45001."


@pytest.fixture
def arabic_citations():
    return extract_citations(ARABIC_SOURCE)


def test_extracts_articles_and_law_number(arabic_citations):
    canonical = {e["canonical"] for e in arabic_citations.values()}
    assert "المادة 1" in canonical
    assert "المادة 12" in canonical
    # The law's own number is the document's primary reference; it was missing
    # from extraction entirely while every article was being picked up.
    assert "القانون رقم 27.06" in canonical


def test_prefix_collision_does_not_corrupt_longer_number(arabic_citations):
    """"المادة 1" is a literal prefix of "المادة 12".

    A str.replace() backfill matched inside the longer reference and produced
    "المادة 1 (l-madda 1)2 (l-madda 12)". This hit 17 of 332 pilot rows.
    """
    result = inject_citations("7sb المادة 12 mn al9anwn", arabic_citations)
    assert result == "7sb المادة 12 (l-madda 12) mn al9anwn"


def test_model_written_pair_is_not_double_wrapped(arabic_citations):
    """The model is told to self-gloss, so it sometimes already did.

    Re-expanding its gloss produced "المادة 1 (المادة 1 (l-madda 1))".
    """
    result = inject_citations(
        "المادة 1 (l-madda 1) w المادة 12 kt7dd", arabic_citations
    )
    assert "(المادة" not in result
    assert result.count("l-madda 1)") == 1


def test_hallucinated_reference_is_left_untouched(arabic_citations):
    """The core safety property: never dress up an invented citation.

    Article 999 is not in the source, so it must survive unchanged rather than
    be rewritten into a verifiable-looking form.
    """
    result = inject_citations("7sb l-madda 999 khass", arabic_citations)
    assert result == "7sb l-madda 999 khass"


def test_repeat_mention_gets_short_form(arabic_citations):
    """First mention carries the Arabic; repeats would just be noise."""
    result = inject_citations(
        "المادة 5 ... w 3awtani l-madda 5", extract_citations("المادة 5 hna")
    )
    assert result.count("المادة 5") == 1
    assert result.endswith("l-madda 5")


def test_arabizi_only_mention_gains_arabic_source_term(arabic_citations):
    result = inject_citations("7sb l-madda 12 khass", arabic_citations)
    assert result == "7sb المادة 12 (l-madda 12) khass"


@pytest.mark.parametrize(
    "written",
    ["7sb 9anwn 27-06 khass", "al9anwn r9m 27.06 kynzm", "7sb Loi N° 27.06 khass"],
)
def test_law_recognised_across_written_forms(written, arabic_citations):
    """The model writes the law number in Darija, French, or with either
    separator. All refer to the same law and must resolve to one form."""
    result = inject_citations(written, arabic_citations)
    assert "القانون رقم 27.06 (Loi N° 27.06)" in result


def test_unknown_law_number_untouched(arabic_citations):
    assert inject_citations("7sb 9anwn 99-99", arabic_citations) == "7sb 9anwn 99-99"


def test_french_source_law_keeps_french_form():
    """A French source needs no Arabic pairing — "Loi N° 42-25" is already both
    what the document prints and what a Moroccan professional says."""
    citations = extract_citations(FRENCH_SOURCE)
    result = inject_citations("7sb Loi N° 42-25 khassek", citations)
    assert "القانون" not in result
    assert "Loi N° 42-25" in result


def test_no_citations_is_a_noop():
    assert inject_citations("chi text 3adi", {}) == "chi text 3adi"


def test_generalization_corpus_yields_no_citations():
    """Internal refs like "Protocole HYG-01" are not verifiable references.

    Extracting them would let the anchor rule demand a citation the model can
    only invent.
    """
    assert extract_citations("**Référence interne :** Protocole HYG-01") == {}


@pytest.mark.parametrize(
    "text,expected",
    [("المادة 12 و المادة 8", "arabic"), ("7sb l-madda 12 khass", "arabizi")],
)
def test_detect_target_script(text, expected):
    assert detect_target_script(text) == expected


# ---------------------------------------------------------------------------
# Structural reference shapes -- parity fix. generate_training_data.py's
# fabrication gate (_REFERENCE_SHAPES) already covered Section/Chapitre/
# Paragraphe/Annexe/Décret/EN/page references, but extract_citations() here
# -- which gates the "must cite something real" requirement AND is what
# production serving (llm.py) uses -- did not, so a context whose only
# citable material was one of these shapes was invisible to both.
# ---------------------------------------------------------------------------


def test_extract_citations_finds_section_annex_page():
    context = "حسب القسم 3 و الملحق 2 ديال هاد الدليل (صفحة 17)، خاصك تلبس EPI."
    found = extract_citations(context)
    assert ("القسم", "3") in found
    assert ("الملحق", "2") in found
    assert ("صفحة", "17") in found


def test_extract_citations_finds_latin_structural_shapes():
    context = "Voir Chapitre 4, Section 2, Décret n° 2-12-349 et EN 397, P. 7."
    found = extract_citations(context)
    heads = {h for h, _ in found}
    assert {"Chapitre", "Section", "Décret", "EN", "P."} <= heads


def test_page_reference_does_not_false_positive_on_ip_rating():
    """A prior draft of the Latin page pattern matched "P 65" inside "IP 65",
    a real and common industrial ingress-protection rating in this corpus."""
    found = extract_citations("Le boîtier doit respecter la norme IP 65.")
    assert ("P.", "65") not in found


# --- Arabic orthographic normalization (fold_arabic / _ar) -----------------
#
# Measured against a real 80-page administrative PDF: the native text layer
# preserved teh marbuta correctly (1278 occurrences), but a PaddleOCR gate
# run separately misread it as heh in 3/5 test headings, and the PDF's own
# table pages extracted as Unicode Presentation Forms B glyphs where a
# teh-marbuta-ending word literally contains no ة codepoint at all. These
# cases prove extraction survives both.

def test_extract_citations_survives_teh_marbuta_heh_confusion():
    """The exact defect that gated OCR off (ocr_engine='none'): a source
    reading "الماده" instead of "المادة" must still extract, and --
    critically -- extract_citations derives `canonical` from its own
    template (citations.py's extract_citations docstring), never from the
    matched text, so the corrupted spelling never leaks into output."""
    found = extract_citations("حسب الماده 8، يجب على العامل...")
    assert found == {("المادة", "8"): {"canonical": "المادة 8", "arabizi": "l-madda 8"}}


@pytest.mark.parametrize("source,key", [
    ("صفحه 17 توضح الاجراء.", ("صفحة", "17")),
    ("الفقره 3 ديال القسم.", ("الفقرة", "3")),
])
def test_extract_citations_survives_teh_marbuta_heh_confusion_other_heads(source, key):
    assert key in extract_citations(source)


def test_extract_citations_tolerates_harakat():
    """Harakat sit inside the U+0600-06FF block, so a naive [^\\w؀-ۿ]
    strip (as the grounding gate used before this fix) does NOT remove
    them -- "المادةُ" must still match "المادة"."""
    found = extract_citations("المادةُ 12 تنص على ذلك.")
    assert ("المادة", "12") in found


def test_extract_citations_tolerates_tatweel():
    found = extract_citations("المــادة 12 تنص على ذلك.")
    assert ("المادة", "12") in found


def test_extract_citations_arabic_indic_digit_keys_same_as_ascii():
    """\\d in Python's re already matches Arabic-Indic digits (Unicode
    category Nd), so without folding, "المادة ٥" and "المادة 5" extracted
    under two DIFFERENT keys for the same reference."""
    assert set(extract_citations("المادة ٥ تنص...").keys()) == \
        set(extract_citations("المادة 5 تنص...").keys())


def test_extract_citations_finds_decree_and_decision():
    """المرسوم (decree) and القرار (decision) -- measured on a real
    administrative guide to outnumber المادة itself (7 and 13 occurrences
    vs. 3) and were completely absent from ARABIC_REFERENCES before this."""
    found = extract_citations("المرسوم رقم 2.18.781 الصادر... والقرار 1234 كذلك")
    assert ("المرسوم", "2.18.781") in found
    assert ("القرار", "1234") in found


@pytest.mark.parametrize("source", [
    "المهاد 12 و الماذة 3",  # unrelated words, must not false-positive
    "هاد الشي مادي 12",       # "مادي" (material) is not "مادة" (article)
])
def test_widened_patterns_do_not_introduce_false_positives(source):
    """The widening only folds ة<->ه, alef variants, ya/waw-hamza -- it must
    never fold letters that distinguish genuinely different words (د/ذ,
    ر/ز, ت/ث, س/ش), or "المادة" would start matching fabricated lookalikes."""
    assert extract_citations(source) == {}


def test_inject_citations_backfills_gloss_for_corrupted_spelling(arabic_citations):
    """A model that reproduces a corrupted spelling (its own minor variance,
    or content quoted from an OCR'd source) must still get the Arabizi
    gloss backfilled, not silently skipped for not matching exactly."""
    result = inject_citations(
        "حسب الماده 1، يخضع الشخص لهذا القانون.", arabic_citations, target_script="arabizi",
    )
    assert "l-madda 1" in result


def test_inject_citations_does_not_double_wrap_corrupted_self_gloss(arabic_citations):
    """The double-wrap this normalization is required to prevent: a model
    that already self-glossed using a corrupted spelling must be recognised
    as already-cited, not re-wrapped into
    "الماده 1 (المادة 1 (l-madda 1))"."""
    already_glossed = "حسب الماده 1 (l-madda 1)، يخضع الشخص لهذا القانون."
    result = inject_citations(already_glossed, arabic_citations, target_script="arabizi")
    assert result.count("l-madda 1") == 1
    assert "المادة 1 (المادة 1" not in result
    assert "الماده 1 (المادة 1" not in result
