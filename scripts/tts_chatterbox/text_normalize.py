"""Number/unit normalization for the Darija+French TTS pipeline.

Root cause this fixes: the model was never trained on how to read raw digits (num2words is
listed in requirements.txt but was never actually called in preprocess_multilingual.py, so the
corpus taught it nothing about "1.2" or "95%"). Feeding raw digits produces mispronunciation and,
in at least one isolated case this session, a CUDA device-side assert crash. The fix validated by
ear across this conversation: convert the number to French words (num2words already gets this
exactly right -- 95 -> "quatre-vingt-quinze", 1500 -> "mille cinq cents") and, for units that were
tested, convert the unit into French too, wrapping both in a [fr]...[<lang>] tag pair so the
tokenizer's own per-language handling takes over for that span only.

KNOWN UNRELIABLE (confirmed by isolated ASR testing, not fixed by this module):
  - "6 بار" -> "six bar" mispronounces on every sample tested (4/4 failures).
  - "10000" -> "dix mille" is only clean ~25% of the time (2/8 attempts garbled or ran away).
  These are model-fluency gaps in the fine-tune, not something a text substitution can paper over.
  A caller that cares should retry-and-ASR-check generations containing these forms.

NOT covered here: the خ (kha) medial-mispronunciation issue. That was resolved case-by-case by
picking ال-definite noun forms (k4/k5 in the num_kha_probe), which is a grammatical rewrite of the
sentence, not a mechanical text substitution -- applying it automatically risks silently changing
meaning, so it is left as an authoring guideline, not code.
"""
import re

from num2words import num2words

# Longest / most specific patterns first -- regex alternation takes the first match, not the
# longest, so a bare "متر" before "نيوتن متر" would eat the "متر" out of "نيوتن متر" first.
_UNIT_MAP = [
    (r"نيوتن\s*متر", "newton mètre"),
    (r"دورة\s*ف[يَ]?\s*الدقيقة", "tours par minute"),
    (r"دورة\s*/\s*دقيقة", "tours par minute"),
    (r"درجة\s*حرارة", "degrés"),
    (r"درجة", "degrés"),
    (r"°\s*[كمCc]\b", "degrés"),
    (r"بالمئة|بالمائة", "pour cent"),
    (r"%", "pour cent"),
    (r"مل[يى]?\s*متر|مم\b", "millimètre"),
    (r"سم\b", "centimètre"),
    (r"كيلوغرام|كلغ\b", "kilogramme"),
    (r"غرام\b", "gramme"),
    (r"بار\b", "bar"),
    (r"لتر\b", "litre"),
    (r"متر\b", "mètre"),
]

_UNIT_RE = "|".join(f"(?:{pat})" for pat, _ in _UNIT_MAP)
_NUMBER_UNIT_RE = re.compile(rf"(\d+(?:[.,]\d+)?)(?:\s*({_UNIT_RE}))?")

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _unit_to_french(unit_text: str) -> str:
    for pat, fr in _UNIT_MAP:
        if re.fullmatch(pat, unit_text):
            return fr
    return ""


def _number_to_french(num_str: str) -> str:
    num_str = num_str.translate(_ARABIC_INDIC).replace(",", ".")
    if "." in num_str:
        int_part, frac_part = num_str.split(".", 1)
        int_words = num2words(int(int_part), lang="fr") if int_part else "zéro"
        digit_words = " ".join(num2words(int(d), lang="fr") for d in frac_part)
        return f"{int_words} virgule {digit_words}"
    return num2words(int(num_str), lang="fr")


def normalize_numbers(text: str, surrounding_lang: str = "ar") -> str:
    """Replace every digit run (optionally followed by a recognized unit) with French words,
    wrapped in [fr]...[<surrounding_lang>] so the tag returns to whatever language the rest of
    the sentence is in. A number with no recognized unit is converted alone; the following word
    (e.g. "ساعة") is left untouched, matching what was actually validated by ear this session.
    """
    def repl(m: re.Match) -> str:
        num_str, unit_str = m.group(1), m.group(2)
        french = _number_to_french(num_str)
        if unit_str:
            french += " " + _unit_to_french(unit_str)
        return f"[fr]{french}[{surrounding_lang}]"

    return _NUMBER_UNIT_RE.sub(repl, text)


# ---------------------------------------------------------------------------------------------
# normalize_numbers_msa: the scheme picked BY EAR in outputs/test_text_v2 and re-confirmed on the
# final adapter in eval/fr1-C sentences 09-11 (2026-09-26: "9-11 sounds good"):
#   numbers  -> MSA Arabic words, colloquial -in forms ("مائة وعشرين", "ستة عشر", "عشرة آلاف")
#   units    -> French, in [fr]..[ar] tags ("ستة [fr]bar[ar]")
#   decimals -> a whole number in a smaller unit (1.2 m -> "مائة وعشرين [fr]centimètres[ar]")
# This replaces normalize_numbers()'s older French-numbers scheme, whose "six bar"/"dix mille"
# were the unreliable forms; in this scheme both came out clean. normalize_numbers() is kept
# unchanged because tts_server.py still calls it.
#
# Validated by ear: bar, centimètres (from metres), and plain counts (ستة, عشرة آلاف, ستة عشر).
# NOT yet validated by ear: the other units, the other decimal conversions, and the "فاصلة"
# fallback for decimals that don't convert cleanly -- listen before relying on them.
# ---------------------------------------------------------------------------------------------

_ONES = ["صفر", "واحد", "اثنين", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة", "عشرة"]
_TENS = {2: "عشرين", 3: "ثلاثين", 4: "أربعين", 5: "خمسين", 6: "ستين", 7: "سبعين", 8: "ثمانين",
         9: "تسعين"}
_HUNDREDS = {1: "مائة", 2: "مائتين", 3: "ثلاثمائة", 4: "أربعمائة", 5: "خمسمائة", 6: "ستمائة",
             7: "سبعمائة", 8: "ثمانمائة", 9: "تسعمائة"}


def _below_1000(n: int) -> str:
    if n <= 10:
        return _ONES[n]
    if n == 11:
        return "أحد عشر"
    if n == 12:
        return "اثني عشر"
    if n < 20:
        return f"{_ONES[n - 10]} عشر"
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] if ones == 0 else f"{_ONES[ones]} و{_TENS[tens]}"
    hundreds, rest = divmod(n, 100)
    return _HUNDREDS[hundreds] if rest == 0 else f"{_HUNDREDS[hundreds]} و{_below_1000(rest)}"


def _scaled(count: int, one: str, two: str, few: str) -> str:
    """Arabic counted noun: 1 -> ألف, 2 -> ألفين, 3-10 -> ثلاثة آلاف, 11+ -> أحد عشر ألف."""
    if count == 1:
        return one
    if count == 2:
        return two
    if count <= 10:
        return f"{_below_1000(count)} {few}"
    return f"{_below_1000(count)} {one}"


def msa_number_words(n: int) -> str:
    if n < 0:
        return f"ناقص {msa_number_words(-n)}"
    if n >= 1_000_000_000:
        return str(n)  # out of range: leave the digits rather than guess
    millions, rest = divmod(n, 1_000_000)
    thousands, rest = divmod(rest, 1000)
    parts = []
    if millions:
        parts.append(_scaled(millions, "مليون", "مليونين", "ملايين"))
    if thousands:
        parts.append(_scaled(thousands, "ألف", "ألفين", "آلاف"))
    if rest or not parts:
        parts.append(_below_1000(rest))
    return " و".join(parts)


# unit key -> (singular, plural) as spoken in French. "bar" stays "bar": that form was validated.
_FR_UNITS = {
    "newton_metre": ("newton mètre", "newton mètres"),
    "rpm": ("tour par minute", "tours par minute"),
    "degree": ("degré", "degrés"),
    "percent": ("pour cent", "pour cent"),
    "micrometre": ("micromètre", "micromètres"),
    "millimetre": ("millimètre", "millimètres"),
    "centimetre": ("centimètre", "centimètres"),
    "metre": ("mètre", "mètres"),
    "gramme": ("gramme", "grammes"),
    "kilogramme": ("kilogramme", "kilogrammes"),
    "millilitre": ("millilitre", "millilitres"),
    "litre": ("litre", "litres"),
    "bar": ("bar", "bar"),
}
# Written forms in the tutor's Arabic text -> unit key. Longest/most specific first (same reason
# as _UNIT_MAP above: regex alternation takes the first match, not the longest).
_AR_UNITS = [
    (r"نيوتن\s*متر", "newton_metre"),
    (r"دورة\s*ف[يَـ]?\s*الدقيقة", "rpm"),
    (r"دورة\s*/\s*دقيقة", "rpm"),
    (r"درجة\s*حرارة", "degree"),
    (r"درجات|درجة", "degree"),
    (r"°\s*[كمCc]?", "degree"),
    (r"بالمئة|بالمائة|في\s*المائة|%", "percent"),
    (r"مل[يى]?\s*متر|مم\b", "millimetre"),
    (r"سنتيمتر|سم\b", "centimetre"),
    (r"كيلوغرام|كيلوجرام|كلغ\b|كغ\b", "kilogramme"),
    (r"غرام\b|جرام\b", "gramme"),
    (r"مل[يى]?\s*لتر", "millilitre"),
    (r"لتر\b|ليتر\b", "litre"),
    (r"بار\b", "bar"),
    (r"متر\b|أمتار\b|امتار\b", "metre"),
]
# A decimal is rescaled to the first smaller unit that makes it whole (1.2 m -> 120 cm).
_SMALLER = {
    "metre": [("centimetre", 100), ("millimetre", 1000)],
    "centimetre": [("millimetre", 10)],
    "millimetre": [("micrometre", 1000)],
    "kilogramme": [("gramme", 1000)],
    "litre": [("millilitre", 1000)],
}

_AR_UNIT_RE = "|".join(f"(?:{p})" for p, _ in _AR_UNITS)
# Numbers glued to Latin letters, hyphens, slashes or "N°" are codes/standards (CM-204,
# Loi N° 65-99, EN 397) and are left alone -- how those should be read was never tested.
_MSA_NUMBER_RE = re.compile(
    r"(?<![A-Za-z°\-/\d.,])(?<![A-Za-z°\-/] )"
    r"(\d{1,3}(?: \d{3})+|\d+(?:[.,]\d+)?)(?![\-/]\d)(?![A-Za-z])"
    rf"(?:\s*({_AR_UNIT_RE}))?"
)


def _unit_key(unit_text: str) -> str:
    for pat, key in _AR_UNITS:
        if re.fullmatch(pat, unit_text.strip()):
            return key
    raise ValueError(f"unmapped unit {unit_text!r}")


def _decimal_words(int_part: str, frac: str) -> str:
    """Fallback for a decimal with no unit to rescale: 'اثنين فاصلة خمسة'."""
    frac_words = (" ".join(_ONES[int(d)] for d in frac) if frac.startswith("0")
                  else msa_number_words(int(frac)))
    return f"{msa_number_words(int(int_part or 0))} فاصلة {frac_words}"


def normalize_numbers_msa(text: str, surrounding_lang: str = "ar") -> str:
    text = text.translate(_ARABIC_INDIC)
    # "و 1.8" -> "و1.8": the conjunction attaches to the number word it introduces ("ومائة"),
    # which is how the validated test_text_v2 sentence was written.
    text = re.sub(r"(?<!\S)و\s+(?=\d)", "و", text)

    def repl(m: re.Match) -> str:
        num, unit_text = m.group(1).replace(" ", ""), m.group(2)
        unit = _unit_key(unit_text) if unit_text else None
        int_part, _, frac = num.replace(",", ".").partition(".")
        frac = frac.rstrip("0")
        value = int(int_part or 0)
        if frac and unit:
            exact = float(f"{int_part or 0}.{frac}")
            for smaller, factor in _SMALLER.get(unit, []):
                scaled = exact * factor
                if abs(scaled - round(scaled)) < 1e-9:
                    unit, value, frac = smaller, int(round(scaled)), ""
                    break
        words = _decimal_words(int_part, frac) if frac else msa_number_words(value)
        if not unit:
            return words
        singular, plural = _FR_UNITS[unit]
        fr = singular if (value < 2 and not frac) else plural
        return f"{words} [fr]{fr}[{surrounding_lang}]"

    return _MSA_NUMBER_RE.sub(repl, text)


if __name__ == "__main__":
    tests = [
        "10 نقاط", "500 ساعة عمل", "95%", "80°C", "1500 دورة في الدقيقة",
        "6 بار", "100%", "0.05 مم", "10000 ساعة", "15 درجة", "120 نيوتن متر",
    ]
    for t in tests:
        print(f"{t!r:35s} -> {normalize_numbers(t)}")


# Validated case-by-case: the model mispronounces medial خ in the bare/indefinite noun "مخاطر"
# (k2 in num_kha_probe) but gets it right in the definite form "المخاطر" (k4/k5, same word, same
# root) -- see outputs/num_kha_probe/. This is NOT a general خ-pronunciation rule (word-initial خ
# was already fine, e.g. "خاصك"); it is a fix for this one specific word, validated by ear. Do not
# extend this dict without a fresh probe -- guessing at other words here would just be a quality
# claim with no measurement behind it.
_KHA_WORD_FIXES = {
    r"(?<!ال)\bمخاطر\b": "المخاطر",
}


def fix_known_kha_words(text: str) -> str:
    """Apply only the specific خ mispronunciation fixes that were actually validated by ear."""
    for pat, repl in _KHA_WORD_FIXES.items():
        text = re.sub(pat, repl, text)
    return text


# ---- Plain typed text -> what the cs-run1 model was trained on ------------------------------
# cs-run1 learned code-switching from MoulSot rows whose Latin runs were wrapped as [fr]...[ar]
# (corpus/ingest_moulsot_cs.py). Same word/run patterns here, so inference sees what training saw.
_LATIN_WORD = r"[A-Za-zÀ-ÖØ-öø-ÿŒœ][A-Za-zÀ-ÖØ-öø-ÿŒœ'’\-]*"
_LATIN_RUN = re.compile(rf"{_LATIN_WORD}(?:[ \t]+{_LATIN_WORD})*")
_ARABIC_LETTER = re.compile(r"[ؠ-يٱ-ۓڤگپچ]")
_LANG_TAG = re.compile(r"\[(?:fr|ar|en)\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?؟])\s+")


def tag_french_runs(text: str) -> str:
    """Wrap each Latin-script run in [fr]...[ar]. Text that already has tags is left alone."""
    if _LANG_TAG.search(text):
        return text
    return _LATIN_RUN.sub(lambda m: f"[fr]{m.group(0).replace('’', chr(39))}[ar]", text)


def prepare_for_tts(text: str, split: bool = True, tag_french: bool = True,
                    numbers: bool = True) -> list[tuple[str, str]]:
    """Typed text -> [(model_text, language_id)], one entry per sentence.

    Order matters: French runs are tagged BEFORE numbers are expanded, because the number
    normalizer inserts its own "[fr]centimètres[ar]" and tagging after would double-wrap them.
    A sentence with no Arabic letters is sent untagged with language_id "fr" (pure French was
    judged good in that mode, eval/fr2-*).
    """
    text = re.sub(r"\s+", " ", text).strip()
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()] if split else [text]
    out = []
    for p in parts:
        if not _ARABIC_LETTER.search(_LANG_TAG.sub("", p)):
            out.append((p, "fr"))
            continue
        p = fix_known_kha_words(p)
        if tag_french:
            p = tag_french_runs(p)
        if numbers:
            p = normalize_numbers_msa(p)
        out.append((p, "ar"))
    return out
