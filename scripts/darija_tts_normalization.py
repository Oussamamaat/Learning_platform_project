"""
French -> Arabic-script phonetic normalization for Darija TTS input.

WHY THIS EXISTS
---------------
This tenant's fine-tune is *deliberately* trained to leave technical, legal
and safety nouns in French, written in Latin letters, inside otherwise
Arabic-script Darija sentences (generate_training_data.py's
build_code_switching_prompt, gated by row_is_code_switched; the convention is
spelled out in data/code_switching_rules.md). Measured on the shipped corpus,
75.3% of assistant turns in data/v11_merged are mixed-script. That is the
normal case, not an edge case.

The XTTS checkpoint downstream (medmac01/darija_xtt_2.0) takes exactly ONE
language code per inference() call and prefixes the whole string with a single
[lang] token, so a mixed-script sentence necessarily mispronounces whichever
script the tag isn't. Splitting the sentence into per-language spans and
synthesizing each separately was tried and REVERTED -- it produces one
complete utterance per span, with audible hallucinated filler on short
fragments (docs/LESSONS_LEARNED.md #13).

This module takes the other route: keep exactly ONE inference() call, and
change the *text* instead -- rewrite the embedded French into Arabic script
spelled the way a Moroccan actually pronounces it ("securite" ->
"سيكوريتي"). No splice, no fragment, no prosody discontinuity. It is also
simply how these loanwords are written when Moroccans write them in Arabic
script at all.

SPOKEN TEXT ONLY. This must never touch what the learner READS -- the on-screen
caption and the chat transcript keep the French spelling, which is the correct
and legible form. Only the string handed to the TTS engine is rewritten.

VOCAB CONSTRAINT -- THE NON-OBVIOUS PART
----------------------------------------
The checkpoint's tokenizer (data/tts_eval_cache/darija_xtts/vocab.json, 6681
entries) contains only 59 characters from the Arabic block, and it is missing
precisely the three Maghrebi letters normally used to write French sounds:

    گ  U+06AF  gaf  (/g/)   -- NOT in vocab
    پ  U+067E  peh  (/p/)   -- NOT in vocab
    ڤ  U+06A4  veh  (/v/)   -- NOT in vocab

Spelling "protection" the standard way ("پروتيكسيون") would push پ through the
tokenizer as an unknown and silently mangle the word -- the exact class of
failure this whole effort is trying to stop. So every mapping below is
restricted to IN_VOCAB_ARABIC, and tests/test_darija_tts_normalization.py
asserts that mechanically for every entry. Substitutions used instead:

    /p/ -> ب   (standard Moroccan practice anyway: "protection" -> بروتيكسيون)
    /v/ -> ف   (likewise: "video" -> فيديو)
    /g/ -> ق   (a compromise. گ is the right letter and is unavailable; ق is
                widely realized as [g] in Moroccan Darija, so it is the closest
                in-vocab option. Small blast radius -- only "gants", "GAFI",
                "tagout", "guide" among the corpus's frequent terms. Worth
                listening for specifically.)

STATUS: opt-in, OFF by default (TTS_TRANSLITERATE_FR=1), pending a listening
verdict -- same rule as ADR 0006 and lessons #12/#13: TTS quality is judged by
ear, never assumed from a plausible-sounding technique.

Pure stdlib, no third-party imports: scripts/tts_worker_resident.py runs in its
own .tts_venv and must be able to import this without dragging in app deps.
"""
from __future__ import annotations

import re
import unicodedata

# The 59 Arabic-block characters actually present in the checkpoint's
# vocab.json. Anything outside this set must never reach inference().
IN_VOCAB_ARABIC = frozenset(
    "،؛؟ءآأؤإئابة"
    "تثجحخدذرزسشص"
    "ضطظعغـفقكلمن"
    "هوىيًٌٍَُِّْ"
    "ٰچڨکھیۖۗۘۚۛ"
)

_ARABIC_LO, _ARABIC_HI = "ء", "ۿ"


# ---------------------------------------------------------------------------
# Lexicon. Curated from the ACTUAL frequency distribution of Latin-script
# tokens in assistant turns of data/v11_merged (top ~150 by count), not from
# guesswork about what a French lexicon "should" contain. Keys are lowercase
# and matched longest-phrase-first, so multi-word entries win over their parts.
# ---------------------------------------------------------------------------

LEXICON: dict[str, str] = {
    # --- multi-word technical phrases (matched before their component words)
    "due diligence": "ديو ديليجانس",
    "smart contract": "سمارت كونتراكت",
    "smart contracts": "سمارت كونتراكتس",
    "crypto-actifs": "كريبتو أكتيف",
    "actifs numériques": "أكتيف نوميريك",
    "chaîne de froid": "شين دو فروا",
    "casque de sécurité": "كاسك دو سيكوريتي",
    "contrôle d'accès": "كونترول داكسي",
    "ligne de vie": "لين دو في",
    "santé au travail": "سانتي أو ترافاي",
    "bank al-maghrib": "بانك المغرب",
    "al-maghrib": "المغرب",
    # --- French grammatical words (very high frequency in the corpus)
    "le": "لو",
    "la": "لا",
    "les": "لي",
    "un": "آن",
    "une": "أون",
    "de": "دو",
    "du": "دو",
    "des": "دي",
    "et": "ي",
    "en": "ان",
    "au": "أو",
    "aux": "أو",
    "sur": "سور",
    "pour": "بور",
    "dans": "دان",
    "avec": "أفيك",
    "par": "بار",
    "sans": "سان",
    "ou": "أو",
    "que": "كو",
    "qui": "كي",
    "est": "إس",
    "est-ce": "إس",
    "est-ce que": "إس كو",
    "je": "جو",
    "dois": "دوا",
    "doit": "دوا",
    "même": "ميم",
    "si": "سي",
    "ne": "نو",
    "pas": "با",
    "plus": "بلوس",
    "tout": "تو",
    "tous": "تو",
    "être": "إتر",
    "faire": "فير",
    # --- industrial safety / occupational health
    "sécurité": "سيكوريتي",
    "santé": "سانتي",
    "travail": "ترافاي",
    "protection": "بروتيكسيون",
    "prévention": "بريفانسيون",
    "formation": "فورماسيون",
    "procédure": "بروسيدور",
    "procedure": "بروسيدور",
    "procédures": "بروسيدور",
    "consignation": "كونسيناسيون",
    "maintenance": "مانتونانس",
    "entretien": "أنتروتيان",
    "machine": "ماشين",
    "machines": "ماشين",
    "casque": "كاسك",
    "gants": "قان",
    "lunettes": "لونيت",
    "risque": "ريسك",
    "risques": "ريسك",
    "accident": "أكسيدان",
    "accidents": "أكسيدان",
    "incident": "أنسيدان",
    "zone": "زون",
    "zones": "زون",
    "agent": "أجان",
    "agents": "أجان",
    "responsable": "ريسبونسابل",
    "équipement": "إيكيبمان",
    "équipements": "إيكيبمان",
    "individuelle": "أنديفيدويل",
    "contrôle": "كونترول",
    "accès": "أكسي",
    "surveillance": "سورفييانس",
    "norme": "نورم",
    "normes": "نورم",
    "règle": "ريكل",
    "règles": "ريكل",
    "article": "أرتيكل",
    "loi": "لوا",
    "code": "كود",
    "document": "دوكيمان",
    "système": "سيستيم",
    "stockage": "ستوكاج",
    "réception": "ريسيبسيون",
    "rotation": "روتاسيون",
    "intervention": "أنتيرفانسيون",
    "gestion": "جيستيون",
    "projet": "بروجي",
    "contrat": "كونترا",
    "caisse": "كيس",
    "froid": "فروا",
    "cuve": "كوف",
    "niveau": "نيفو",
    "site": "سيت",
    "force": "فورس",
    "action": "أكسيون",
    "performance": "بيرفورمانس",
    "conditions": "كونديسيون",
    "nationale": "ناسيونال",
    "personnes": "بيرسون",
    "premier": "بروميي",
    "première": "بروميير",
    "étape": "إيتاب",
    "étapes": "إيتاب",
    "bon": "بون",
    "plaquettes": "بلاكيت",
    "utilitaires": "أوتيليتير",
    "inventaire": "أنفانتير",
    "transferts": "ترانسفير",
    "sociale": "سوسيال",
    "déclaration": "ديكلاراسيون",
    "vanne": "فان",
    "pression": "بريسيون",
    "hauteur": "أوتور",
    "atelier": "أتوليي",
    "chantier": "شانتيي",
    "danger": "دانجي",
    "urgence": "أورجانس",
    "évacuation": "إيفاكواسيون",
    "extincteur": "إكستانكتور",
    "verrouillage": "فيرويّاج",
    "énergie": "إينيرجي",
    "jour": "جور",
    "jours": "جور",
    # --- AML / CFT / fintech / blockchain
    "terrorisme": "تيروريزم",
    "recommandation": "ريكومانداسيون",
    "blanchiment": "بلانشيمان",
    "diligence": "ديليجانس",
    "chaîne": "شين",
    "détenteurs": "ديتانتور",
    "numériques": "نوميريك",
    "numérique": "نوميريك",
    "actifs": "أكتيف",
    "blockchain": "بلوكشين",
    "token": "توكن",
    "tokens": "توكنز",
    "compliance": "كومبلاينس",
    "crypto": "كريبتو",
    "stablecoin": "ستيبلكوين",
    "stablecoins": "ستيبلكوينز",
    "bitcoin": "بيتكوين",
    "ethereum": "إيثيريوم",
    "cryptocurrencies": "كريبتوكارنسيز",
    "digital": "ديجيتال",
    "security": "سيكيوريتي",
    "financial": "فاينانشال",
    "assets": "أسيتس",
    "price": "برايس",
    "stock": "ستوك",
    "stocks": "ستوك",
    "lockout": "لوكاوت",
    "tagout": "تاقاوت",
    "banque": "بانك",
    "bank": "بانك",
    "client": "كليان",
    "clients": "كليان",
    "transaction": "ترانزاكسيون",
    "transactions": "ترانزاكسيون",
    "conformité": "كونفورميتي",
    "conformite": "كونفورميتي",
    "maroc": "المغرب",
}


# Acronyms pronounced as WORDS, not spelled out letter by letter.
ACRONYM_WORDS: dict[str, str] = {
    "ISO": "إيزو",
    "GAFI": "قافي",
    "BAM": "بام",
    "LOTO": "لوطو",
    "VASP": "فاسب",
    "CNSS": "سي إن إس إس",
    "EPI": "إي بي إي",
    # Said with ENGLISH letter names even by French-speaking compliance staff
    # ("kay-why-see"); the French reading would put "i grec" in the middle.
    "KYC": "كي واي سي",
}

# French letter names, for acronyms read letter by letter. Restricted to
# in-vocab characters (so /g/ in "i grec" is ق, not گ).
_FR_LETTER_NAMES: dict[str, str] = {
    "a": "آ", "b": "بي", "c": "سي", "d": "دي", "e": "أو", "f": "إف",
    "g": "جي", "h": "آش", "i": "إي", "j": "جي", "k": "كا", "l": "إل",
    "m": "إم", "n": "إن", "o": "أو", "p": "بي", "q": "كي", "r": "إر",
    "s": "إس", "t": "تي", "u": "أو", "v": "في", "w": "دوبل في",
    "x": "إكس", "y": "إقريك", "z": "زيد",
}


# ---------------------------------------------------------------------------
# Rule-based fallback for words not in the lexicon. Deliberately a fallback:
# the lexicon covers the corpus's frequent terms exactly, and these rules only
# have to be reasonable for the long tail.
# ---------------------------------------------------------------------------

_VOWELS = set("aeiouyàâäéèêëîïôöùûüœ")

# Ordered longest-first; the first match at each position wins.
_RULES: list[tuple[str, str]] = [
    ("eaux", "و"), ("eau", "و"),
    ("ssion", "سيون"), ("tion", "سيون"), ("sion", "زيون"),
    ("aient", "ي"), ("ient", "يان"), ("ien", "يان"),
    ("ain", "ان"), ("ein", "ان"), ("oin", "وان"),
    ("ill", "يي"), ("eil", "اي"), ("ail", "اي"),
    ("gn", "ني"), ("ch", "ش"), ("ph", "ف"), ("th", "ت"), ("sh", "ش"),
    ("qu", "ك"), ("ck", "ك"),
    ("ou", "و"), ("au", "و"), ("oi", "وا"), ("ai", "ي"), ("ei", "ي"),
    ("eu", "و"), ("œu", "و"), ("oe", "و"),
    ("an", "ان"), ("am", "ام"), ("en", "ان"), ("em", "ام"),
    ("on", "ون"), ("om", "وم"), ("in", "ان"), ("im", "ام"),
    ("un", "ان"), ("um", "وم"),
    ("ss", "س"), ("ll", "ل"), ("mm", "م"), ("nn", "ن"),
    ("tt", "ت"), ("pp", "ب"), ("rr", "ر"), ("ff", "ف"),
]

_SINGLE: dict[str, str] = {
    "a": "ا", "à": "ا", "â": "ا", "ä": "ا",
    "e": "ي", "é": "ي", "è": "ي", "ê": "ي", "ë": "ي",
    "i": "ي", "î": "ي", "ï": "ي", "y": "ي",
    "o": "و", "ô": "و", "ö": "و",
    "u": "و", "û": "و", "ü": "و", "ù": "و",
    "b": "ب", "ç": "س", "d": "د", "f": "ف", "h": "",
    "j": "ج", "k": "ك", "l": "ل", "m": "م", "n": "ن",
    "p": "ب", "q": "ك", "r": "ر", "t": "ت", "v": "ف",
    # "s" is also special-cased above (intervocalic -> ز); this is the default
    # for every other position. It must exist -- without it a plain "s" fell
    # through to .get(..., "") and was silently deleted ("poste" -> "بو").
    "s": "س",
    "w": "و", "x": "كس", "z": "ز",
}

# Word-final consonants that are silent in French. "c", "r", "f", "l" are
# excluded -- the traditional CaReFuL set, which stays pronounced.
_SILENT_FINALS = set("stdxzpg")


def _strip_silent_endings(word: str) -> str:
    """French drops most word-final consonants and the final unaccented e.
    "gants" -> /gɑ̃/, "risques" -> /ʁisk/, "accidents" -> /aksidɑ̃/."""
    for _ in range(2):
        if len(word) > 2 and word.endswith("e"):
            word = word[:-1]
        elif len(word) > 2 and word[-1] in _SILENT_FINALS and word[-2] not in _VOWELS:
            word = word[:-1]
        elif len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        else:
            break
    return word


def transliterate_word(word: str) -> str:
    """Rule-based French-orthography -> Arabic-script fallback."""
    w = word.lower()
    # French infinitive/participle "-er" is /e/, not /er/ -- "porter" is
    # bor-TE. Rewrite to the accented form so the normal rules handle it.
    if len(w) > 3 and w.endswith("er"):
        w = w[:-2] + "é"
    w = _strip_silent_endings(w)
    out: list[str] = []
    i = 0
    while i < len(w):
        nxt = w[i + 1] if i + 1 < len(w) else ""
        # Context-sensitive singles have to be checked before the rule table,
        # because "c"/"g"/"s" change value based on the following letter.
        if w[i] == "c":
            if nxt in "eiyéèê":
                out.append("س")
            elif nxt == "c":
                out.append("كس" if (i + 2 < len(w) and w[i + 2] in "eiyéèê") else "ك")
                i += 2
                continue
            else:
                out.append("ك")
            i += 1
            continue
        if w[i] == "g":
            if nxt in "eiyéèê":
                out.append("ج")
                i += 1
                continue
            if w[i:i + 2] == "gu":
                out.append("ق")
                i += 2
                continue
            if w[i:i + 2] != "gn":
                out.append("ق")
                i += 1
                continue
        if w[i] == "s" and i > 0 and w[i - 1] in _VOWELS and nxt in _VOWELS:
            out.append("ز")
            i += 1
            continue
        for src, dst in _RULES:
            if w.startswith(src, i):
                # A nasal is only nasal when not followed by a vowel
                # ("machine" is ma-chine, not ma-chi-nasal). `continue`, NOT
                # `break`: breaking here would leave `i` un-incremented AND
                # skip the for/else single-character fallback, spinning
                # forever on any word where a nasal digraph precedes a vowel
                # ("animation"). Continuing lets the remaining rules try this
                # position and otherwise falls through to the single char.
                if len(src) == 2 and src[1] in "nm":
                    after = w[i + 2] if i + 2 < len(w) else ""
                    if after in _VOWELS or after == src[1]:
                        continue
                out.append(dst)
                i += len(src)
                break
        else:
            out.append(_SINGLE.get(w[i], ""))
            i += 1
    return _carry_initial_vowel("".join(out))


# Arabic script cannot open a word on a bare long vowel: a leading "ي" reads
# as the consonant /j/ ("يكيبمان" = "yakibman", not "ekibman") and a leading
# "و" as /w/. A word-initial vowel needs a hamza carrier -- which is what the
# hand-written lexicon already does ("étape" -> "إيتاب", "accès" -> "أكسي"),
# so the rule-based fallback has to agree with it.
_INITIAL_VOWEL_CARRIER = {"ي": "إي", "و": "أو", "ا": "أ"}


def _carry_initial_vowel(word: str) -> str:
    if word and word[0] in _INITIAL_VOWEL_CARRIER:
        return _INITIAL_VOWEL_CARRIER[word[0]] + word[1:]
    return word


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# A "word" for our purposes: Latin letters plus the apostrophes and hyphens
# that hold French compounds and elisions together.
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ'’\-]*")
_ELISION_RE = re.compile(r"^([ldnsjt]|qu)['’](.+)$", re.IGNORECASE)
_ELISION_PREFIX = {"l": "ل", "d": "د", "n": "ن", "s": "س", "j": "ج", "t": "ت", "qu": "ك"}

_MAX_PHRASE_WORDS = 3


def _is_arabic(text: str) -> bool:
    return any(_ARABIC_LO <= c <= _ARABIC_HI for c in text)


def is_arabic_majority(text: str) -> bool:
    """True when Arabic script carries the sentence and Latin script is only
    embedded terms -- the case transliteration is for.

    A French-MAJORITY sentence with an Arabic clause in it (real example:
    "Est-ce que la loi 27-06 كتخص جميع الشركات الصناعية؟") must NOT be
    transliterated: it is a French utterance and belongs on the French
    language tag, where XTTS pronounces it correctly already. Rewriting it
    into Arabic script would turn correct French into phonetic mush -- the
    same mistake in the opposite direction. Mirrors the whole-message
    majority vote llm.detect_query_language already uses to pick the tag.

    Counted in WORDS, not characters: French words are simply longer than
    Darija ones, so a character count calls a sentence French-majority when a
    reader plainly sees Darija carrying it (real example: "شحال من jour خاصني
    نصيفط la declaration ديال l'incident؟" -- 5 Darija words to 4 French, but
    25 Latin characters to 20 Arabic).
    """
    arabic = len([t for t in text.split() if any(_ARABIC_LO <= c <= _ARABIC_HI for c in t)])
    latin = len([t for t in text.split() if any(c.isalpha() and c.isascii() for c in t)])
    return arabic > latin


def _deaccent(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)
    )


# Accent-stripped view of the lexicon, built once at import. Corpora are
# inconsistent about accents ("securite" vs "sécurité"), so lookups have to be
# accent-blind -- but doing that by scanning every entry on each miss would put
# an O(len(LEXICON)) NFD pass in the TTS hot path, on every unknown word.
_LEXICON_DEACCENTED: dict[str, str] = {}
for _k, _v in LEXICON.items():
    _LEXICON_DEACCENTED.setdefault(_deaccent(_k), _v)
del _k, _v


def _lookup(word: str) -> str | None:
    """Lexicon lookup with the fallbacks a French word needs: accent-blind
    matching (corpora are inconsistent about them) and plural stripping."""
    key = word.lower()
    if key in LEXICON:
        return LEXICON[key]
    stripped = _deaccent(key)
    for table, cand in (
        (_LEXICON_DEACCENTED, stripped),
        (LEXICON, key.rstrip("s")),
        (_LEXICON_DEACCENTED, stripped.rstrip("s")),
    ):
        hit = table.get(cand)
        if hit is not None:
            return hit
    return None


def _render_token(token: str) -> str:
    # Acronyms: all-caps runs are read as a word if we know them, else spelled.
    bare = token.replace("-", "").replace("'", "").replace("’", "")
    if len(bare) >= 2 and bare.isupper() and bare.isalpha():
        if bare in ACRONYM_WORDS:
            return ACRONYM_WORDS[bare]
        return " ".join(_FR_LETTER_NAMES.get(c.lower(), "") for c in bare).strip()

    hit = _lookup(token)
    if hit is not None:
        return hit

    # Elision: "l'incident" -> ل + incident, "d'accès" -> د + accès.
    m = _ELISION_RE.match(token)
    if m:
        prefix = _ELISION_PREFIX[m.group(1).lower()]
        return prefix + _render_token(m.group(2))

    # Hyphenated compound: render each side independently.
    if "-" in token:
        parts = [p for p in token.split("-") if p]
        if len(parts) > 1:
            return " ".join(_render_token(p) for p in parts)

    return transliterate_word(token)


def normalize_for_tts(text: str) -> str:
    """Rewrite Latin-script (French/English) runs in `text` into Arabic-script
    phonetic spelling, leaving Arabic script, digits and punctuation untouched.

    Intended for the string handed to the TTS engine ONLY -- never for text
    the learner reads.
    """
    if not text or not is_arabic_majority(text):
        # Either pure-Latin (a French sentence) or French-majority with an
        # Arabic clause. Both belong on the French language tag as-is; see
        # is_arabic_majority.
        return text

    # Longest-phrase-first: try to match multi-word lexicon entries before
    # falling back to word-by-word, so "due diligence" beats "due" + "diligence".
    matches = list(_TOKEN_RE.finditer(text))
    out: list[str] = []
    cursor = 0
    idx = 0
    while idx < len(matches):
        consumed = 1
        replacement = None
        for size in range(min(_MAX_PHRASE_WORDS, len(matches) - idx), 1, -1):
            start, end = matches[idx].start(), matches[idx + size - 1].end()
            between = text[start:end]
            # Only a real phrase if the words are separated by plain spaces.
            if re.fullmatch(r"[A-Za-zÀ-ɏ'’\- ]+", between):
                hit = _lookup(between)
                if hit is not None:
                    replacement, consumed = hit, size
                    break
        first, last = matches[idx], matches[idx + consumed - 1]
        if replacement is None:
            replacement = _render_token(first.group(0))
        out.append(text[cursor:first.start()])
        out.append(replacement)
        cursor = last.end()
        idx += consumed
    out.append(text[cursor:])
    return re.sub(r"[ \t]{2,}", " ", "".join(out))


def out_of_vocab(text: str) -> set[str]:
    """Characters in `text` that are in the Arabic block but NOT in the
    checkpoint's tokenizer -- these would be silently mangled."""
    return {c for c in text if _ARABIC_LO <= c <= _ARABIC_HI and c not in IN_VOCAB_ARABIC}


if __name__ == "__main__":
    import glob
    import io
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    paths = sorted(glob.glob("tests/data/voice_eval/codeswitch_*.txt"))
    for p in paths:
        src = io.open(p, encoding="utf-8").read().strip()
        dst = normalize_for_tts(src)
        bad = out_of_vocab(dst)
        print(f"--- {p}")
        print(f"  in : {src}")
        print(f"  out: {dst}")
        if bad:
            print(f"  !! OUT OF VOCAB: {bad}")
