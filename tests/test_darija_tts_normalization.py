"""Tests for scripts/darija_tts_normalization.py.

Loaded by path, not import: scripts/ has no __init__.py (same approach as
tests/test_tts_language_spans.py).
"""
import glob
import importlib.util
import io
import json
import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO / "scripts" / "darija_tts_normalization.py"
_VOCAB_PATH = _REPO / "data" / "tts_eval_cache" / "darija_xtts" / "vocab.json"

_spec = importlib.util.spec_from_file_location("darija_tts_normalization", _MODULE_PATH)
norm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(norm)


# ---------------------------------------------------------------------------
# The guard that matters most: nothing we emit may fall outside the
# checkpoint's tokenizer, or it is silently mangled at synthesis time.
# ---------------------------------------------------------------------------

def _all_mapping_values():
    for src, dst in norm.LEXICON.items():
        yield f"LEXICON[{src!r}]", dst
    for src, dst in norm.ACRONYM_WORDS.items():
        yield f"ACRONYM_WORDS[{src!r}]", dst
    for src, dst in norm._FR_LETTER_NAMES.items():
        yield f"_FR_LETTER_NAMES[{src!r}]", dst
    for src, dst in norm._RULES:
        yield f"_RULES[{src!r}]", dst
    for src, dst in norm._SINGLE.items():
        yield f"_SINGLE[{src!r}]", dst


@pytest.mark.parametrize("label,value", list(_all_mapping_values()))
def test_every_mapping_is_in_vocab(label, value):
    assert norm.out_of_vocab(value) == set(), (
        f"{label} = {value!r} contains characters absent from the checkpoint's "
        f"tokenizer: {norm.out_of_vocab(value)}"
    )


@pytest.mark.skipif(not _VOCAB_PATH.exists(), reason="checkpoint vocab not cached locally")
def test_in_vocab_constant_matches_the_actual_checkpoint():
    """IN_VOCAB_ARABIC is hardcoded so the module stays dependency-free; this
    asserts the hardcoded copy still matches the checkpoint it describes."""
    raw = json.load(io.open(_VOCAB_PATH, encoding="utf-8"))
    vocab = raw.get("model", raw).get("vocab", raw)
    derived = frozenset(c for c in vocab if len(c) == 1 and "؀" <= c <= "ۿ")
    assert derived == norm.IN_VOCAB_ARABIC


def test_maghrebi_letters_are_known_absent():
    """Regression guard on the specific trap: gaf/peh/veh are the natural way
    to write /g/, /p/, /v/ in Moroccan Arabic script and are NOT in this
    tokenizer. If a future checkpoint adds them, this test fails and the
    substitutions documented in the module docstring can be revisited."""
    for ch in ("گ", "پ", "ڤ"):
        assert ch not in norm.IN_VOCAB_ARABIC


# ---------------------------------------------------------------------------
# Script-majority gate
# ---------------------------------------------------------------------------

def test_french_majority_sentence_is_left_alone():
    src = "Est-ce que je dois porter le casque même si خاص ندير غير شي خدمة صغيرة؟"
    assert norm.normalize_for_tts(src) == src


def test_pure_french_is_left_alone():
    src = "Quelles sont les étapes de la consignation?"
    assert norm.normalize_for_tts(src) == src


def test_pure_darija_is_unchanged():
    src = "شنو هي الخطوات ديال السلامة قبل ما ندخل للورشة؟"
    assert norm.normalize_for_tts(src) == src


def test_darija_majority_sentence_is_transliterated():
    src = "واش خاصني نلبس le casque ديال sécurité قبل ما ندخل للورشة؟"
    out = norm.normalize_for_tts(src)
    assert "casque" not in out and "sécurité" not in out
    assert "كاسك" in out and "سيكوريتي" in out


def test_word_count_not_char_count_decides_majority():
    """5 Darija words vs 4 French, but more Latin characters than Arabic --
    a character-based vote gets this one wrong."""
    src = "شحال من jour خاصني نصيفط la déclaration ديال l'incident؟"
    assert norm.is_arabic_majority(src)
    assert "déclaration" not in norm.normalize_for_tts(src)


# ---------------------------------------------------------------------------
# Lexicon / transliteration behaviour
# ---------------------------------------------------------------------------

def test_multiword_phrase_beats_its_component_words():
    out = norm.normalize_for_tts("خاصك تفهم due diligence مزيان")
    assert out.count("ديو ديليجانس") == 1


def test_elision_is_handled():
    assert norm._render_token("l'incident") == "ل" + norm.LEXICON["incident"]
    assert norm._render_token("d'accès") == "د" + norm.LEXICON["accès"]


def test_known_acronym_read_as_word():
    assert norm._render_token("ISO") == norm.ACRONYM_WORDS["ISO"]


def test_unknown_acronym_is_spelled_out():
    out = norm._render_token("XYZ")
    assert out == " ".join(
        [norm._FR_LETTER_NAMES["x"], norm._FR_LETTER_NAMES["y"], norm._FR_LETTER_NAMES["z"]]
    )


def test_digits_and_law_references_pass_through():
    src = "واش la loi 27-06 كتطبق دابا؟"
    assert "27-06" in norm.normalize_for_tts(src)


def test_accent_insensitive_lookup():
    """Corpora are inconsistent about accents; "securite" must find "sécurité"."""
    assert norm._render_token("securite") == norm.LEXICON["sécurité"]


def test_french_infinitive_er_is_not_pronounced_er():
    assert not norm.transliterate_word("porter").endswith("ر")


def test_unknown_word_still_produces_arabic_script():
    out = norm.transliterate_word("ventilateur")
    assert out and norm.out_of_vocab(out) == set()
    assert all(not (c.isalpha() and c.isascii()) for c in out)


def test_empty_and_whitespace():
    assert norm.normalize_for_tts("") == ""
    assert norm.normalize_for_tts("   ") == "   "


# ---------------------------------------------------------------------------
# Sweep over the real eval fixtures
# ---------------------------------------------------------------------------

_FIXTURES = sorted(glob.glob(str(_REPO / "tests" / "data" / "voice_eval" / "codeswitch_*.txt")))


@pytest.mark.parametrize("path", _FIXTURES, ids=lambda p: pathlib.Path(p).stem)
def test_fixture_output_never_leaves_the_tokenizer(path):
    src = io.open(path, encoding="utf-8").read().strip()
    out = norm.normalize_for_tts(src)
    assert norm.out_of_vocab(out) == set()


@pytest.mark.parametrize("path", _FIXTURES, ids=lambda p: pathlib.Path(p).stem)
def test_darija_majority_fixtures_lose_their_latin_script(path):
    src = io.open(path, encoding="utf-8").read().strip()
    out = norm.normalize_for_tts(src)
    if norm.is_arabic_majority(src):
        assert not any(c.isalpha() and c.isascii() for c in out), (
            f"Latin script survived transliteration: {out}"
        )
    else:
        assert out == src


# ---------------------------------------------------------------------------
# Regression: a nasal digraph followed by a vowel used to hang forever.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "word",
    ["animation", "animateur", "inutile", "ananas", "unanime", "images",
     "monoxyde", "anomalie", "enumeration", "amende"],
)
def test_nasal_before_vowel_terminates(word):
    """`an`/`en`/`in`/`on`/`un` + vowel is not a nasal. The guard for that
    once `break`-ed out of the rule loop without advancing the cursor and
    without reaching the single-character fallback -- an infinite loop that
    would have hung the TTS worker on any such word."""
    out = norm.transliterate_word(word)
    assert out
    assert norm.out_of_vocab(out) == set()


def test_every_latin_letter_has_a_mapping():
    """Any ASCII letter with no entry fell through `.get(c, "")` and was
    silently deleted from the spoken word -- how a missing "s" turned
    "poste" into "بو". "c" and "g" are context-sensitive and handled
    explicitly in transliterate_word; "h" is intentionally silent."""
    import string
    special = {"c", "g"}
    missing = [c for c in string.ascii_lowercase
               if c not in norm._SINGLE and c not in special]
    assert missing == []


@pytest.mark.parametrize(
    "word,must_contain",
    [("masque", "س"), ("poste", "س"), ("stockage", "س"), ("risque", "س")],
)
def test_s_survives_transliteration(word, must_contain):
    assert must_contain in norm.transliterate_word(word)


@pytest.mark.parametrize("word", ["equipment", "intrusion", "objectif", "analyse"])
def test_word_initial_vowel_gets_a_hamza_carrier(word):
    """A bare leading ي / و / ا reads as a consonant or is unpronounceable;
    the lexicon uses hamza carriers and the fallback must match it."""
    out = norm.transliterate_word(word)
    assert out[0] in "أإآ", f"{word} -> {out} opens on a bare vowel"
