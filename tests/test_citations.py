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
