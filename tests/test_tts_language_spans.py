"""
Tests for scripts/tts_worker_resident.py's `_split_language_spans()` -- the
fix for XTTS mispronouncing code-switched technical terms in Darija answers
(2026-09-06, live lease listening test).

Root cause (see docs/deploy -- no ADR yet, this session's live finding):
XTTS-v2's Xtts.inference() takes exactly ONE language code per call and
prefixes the WHOLE input string with that one [lang] token before running it
through that language's cleaners/tokenizer -- there is no API-level way to
mix languages in a single call. This tenant's fine-tune is DELIBERATELY
trained (generate_training_data.py's build_code_switching_prompt /
row_is_code_switched -- a quality gate, not an edge case) to embed literal
Latin-script French/English technical terms inline in otherwise Arabic-script
Darija sentences. Synthesizing such a sentence under one language tag
mispronounces whichever script isn't the tag's own. `_split_language_spans`
segments text into consecutive same-script runs so the caller
(`_handle_synthesize`) can call `inference()` once per span with the correct
language and concatenate the resulting audio.

Loaded via importlib-by-path rather than a package import: scripts/ has no
__init__.py (tts_worker_resident.py is a standalone script, run as a
subprocess under its own dedicated venv per that module's docstring, never
imported as part of the `app` package) -- this is pure text-in/spans-out
logic, no GPU/model/torch import triggered by loading the module (those are
all deferred to _load(), never called here).
"""
import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "tts_worker_resident.py"
_spec = importlib.util.spec_from_file_location("tts_worker_resident", _MODULE_PATH)
tts_worker_resident = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tts_worker_resident)

_split_language_spans = tts_worker_resident._split_language_spans
_VOICE_EVAL_DIR = Path(__file__).resolve().parent / "data" / "voice_eval"


def _langs(spans):
    return [lang for _, lang in spans]


def _texts(spans):
    return [text for text, _ in spans]


def test_pure_darija_sentence_is_one_arabic_span():
    spans = _split_language_spans("خصك تلبس الكاسك ديال الحماية.")
    assert len(spans) == 1
    assert spans[0][1] == "ar"


def test_pure_french_sentence_is_one_french_span():
    spans = _split_language_spans("Le port du casque est obligatoire dans cette zone.")
    assert len(spans) == 1
    assert spans[0][1] == "fr"


def test_embedded_french_noun_phrase_splits_out_as_its_own_span():
    spans = _split_language_spans("واش خاصني نلبس le casque ديال sécurité قبل ما ندخل للورشة؟")
    assert _langs(spans) == ["ar", "fr", "ar", "fr", "ar"]
    assert _texts(spans) == [
        "واش خاصني نلبس", "le casque", "ديال", "sécurité", "قبل ما ندخل للورشة؟",
    ]


def test_apostrophed_french_term_with_trailing_arabic_question_mark_stays_one_span():
    """"l'incident؟" -- the Arabic '؟' glued directly onto a French word must
    not fragment the French span into two pieces of one word each; it has no
    script of its own (outside the has_arabic_script Unicode range this
    detector reuses) and attaches to whatever span it's adjacent to."""
    spans = _split_language_spans("شحال من jour خاصني نصيفط la déclaration ديال l'incident؟")
    assert _langs(spans) == ["ar", "fr", "ar", "fr", "ar", "fr"]
    assert spans[-1][0] == "l'incident؟"


def test_law_reference_digits_attach_to_the_preceding_french_span():
    """A bare number ("27-06") carries no script signal and must not force
    an extra split between "la loi" and its own citation number."""
    spans = _split_language_spans("Est-ce que la loi 27-06 كتخص جميع الشركات الصناعية؟")
    assert _langs(spans) == ["fr", "ar"]
    assert spans[0][0] == "Est-ce que la loi 27-06"


def test_french_majority_sentence_with_trailing_darija_clause():
    spans = _split_language_spans(
        "Est-ce que je dois porter le casque même si خاص ندير غير شي خدمة صغيرة؟"
    )
    assert _langs(spans) == ["fr", "ar"]


def test_empty_text_returns_no_spans():
    assert _split_language_spans("") == []
    assert _split_language_spans("   ") == []


def test_all_neutral_text_falls_back_to_default_lang():
    spans = _split_language_spans("27-06 ?!", default_lang="fr")
    assert len(spans) == 1
    assert spans[0][1] == "fr"


def test_default_lang_is_ar_unless_overridden():
    spans = _split_language_spans("123")
    assert spans[0][1] == "ar"


@pytest.mark.parametrize("filename", sorted(_VOICE_EVAL_DIR.glob("codeswitch_*.txt")), ids=lambda p: p.stem)
def test_real_codeswitch_fixtures_produce_at_least_two_spans(filename):
    """Every codeswitch_*.txt fixture (tests/data/voice_eval/) is real,
    already-collected code-switched Darija/French speech-eval data -- each
    one genuinely mixes scripts, so each must split into more than one span.
    A regression that silently stopped detecting the embedded French/English
    terms (e.g. a Unicode-range typo) would collapse these back to a single
    span and this test would catch it without needing a model or GPU."""
    text = filename.read_text(encoding="utf-8").strip()
    spans = _split_language_spans(text)
    assert len(spans) >= 2, f"{filename.name} should code-switch, got one span: {spans}"
    assert set(_langs(spans)) == {"ar", "fr"}
