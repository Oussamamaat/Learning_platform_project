"""
Deterministic Citation Injection
────────────────────────────────
Guarantees that references to a source document appear in a verifiable,
dual-script form, without depending on the model to produce them.

Why this is post-processing and not prompting
─────────────────────────────────────────────
Three rounds of prompt and few-shot engineering moved dual-citation compliance
from 0% to ~30% and then stopped. That plateau is not specific to this project:
production RAG systems report ~74% citation accuracy, and up to 57% of model
emitted citations are *post-rationalised* — the model answers from memory and
retrofits a citation afterwards, so even the ones that appear are not evidence
of grounding.

The industry answer is post-processing correction (see CiteFix, Amazon Science
/ ACL 2025, which reported +15.46% accuracy and — the relevant part for us —
allowed replacing a large model with one ~12x cheaper). Rather than ask a 9B
model to compose a citation, this module derives citations from the source
document, which we already have.

Legal and technical references are highly regular (المادة 18, الفصل 24,
Loi N° 42-25, ISO 45001), so extraction is a matter of pattern matching, and
injection is a substitution. Both are deterministic: same input, same output,
no model call, no API cost.

The same function runs in two places, which keeps training and serving aligned:
  - dataset generation, so training rows demonstrate the correct form
  - production, so live answers are correct regardless of what the model emits
"""

import re

# ---------------------------------------------------------------------------
# Citable reference patterns
# ---------------------------------------------------------------------------
#
# Each entry: (compiled pattern over the SOURCE, canonical template,
#              arabizi gloss template or None)
#
# A None gloss means the reference is already Latin script and searchable as
# written — "Loi N° 42-25" needs no transliteration, and adding one would
# teach the model to arabize a term Moroccan professionals say in French.

ARABIC_REFERENCES = [
    (re.compile(r'المادة\s+(\d+)'), 'المادة {n}', 'l-madda {n}'),
    (re.compile(r'الفصل\s+(\d+)'), 'الفصل {n}', 'l-fasl {n}'),
    (re.compile(r'الباب\s+(\d+)'), 'الباب {n}', 'l-bab {n}'),
    (re.compile(r'الفقرة\s+(\d+)'), 'الفقرة {n}', 'l-fiqra {n}'),
    # القسم/الملحق/صفحة added alongside generate_training_data.py's
    # _REFERENCE_SHAPES widening — that gate's fabrication check already
    # covered these shapes, but extract_citations() (used for the "must cite
    # something real" requirement, and by production serving in llm.py) did
    # not, so a context whose only citable material was a section/annex/page
    # reference was invisible to that gate. صفحة confirmed live in
    # dataset_export_v3 ("صفحة 107", "صفحة 17").
    (re.compile(r'القسم\s+(\d+)'), 'القسم {n}', 'l-9ism {n}'),
    (re.compile(r'الملحق\s+(\d+)'), 'الملحق {n}', 'l-mulhaq {n}'),
    (re.compile(r'صفحة\s+(\d+)'), 'صفحة {n}', 'safha {n}'),
    # The law's own number is the document's primary reference and the one a
    # learner is most likely to search for. Its gloss is the French form
    # rather than a transliteration: a Moroccan professional says
    # "Loi N° 27.06", never a phonetic rendering of القانون.
    (re.compile(r'القانون\s+رقم\s+(\d+[\-.]\d+)'),
     'القانون رقم {n}', 'Loi N° {n}'),
]

LATIN_REFERENCES = [
    (re.compile(r'Loi\s+n[°o]?\s*([\d\-\.]+)', re.I), 'Loi N° {n}', None),
    (re.compile(r'ISO\s+(\d{4,5})'), 'ISO {n}', None),
    (re.compile(r'\bArticle\s+(\d+)', re.I), 'Article {n}', None),
    # Same parity fix as ARABIC_REFERENCES above, for the French/English
    # structural shapes generate_training_data.py's fabrication gate already
    # covers.
    (re.compile(r'\b(?:Section)\s+(\d+)', re.I), 'Section {n}', None),
    (re.compile(r'\b(?:Chapitre|Chapter)\s+(\d+)', re.I), 'Chapitre {n}', None),
    (re.compile(r'\b(?:Paragraphe|Paragraph)\s+(\d+)', re.I), 'Paragraphe {n}', None),
    (re.compile(r'\b(?:Annexe|Annex)\s+(\d+)', re.I), 'Annexe {n}', None),
    (re.compile(r'\bD[ée]cret\s+n?[°o]?\s*([\d\-\.]+)', re.I), 'Décret n° {n}', None),
    (re.compile(r'\bEN\s*(\d{3,5})'), 'EN {n}', None),
    (re.compile(r'\b[Pp]\.\s*(\d+)\b'), 'P. {n}', None),
]

# How the model actually refers to an Arabic article once it is writing
# Arabizi. Collected from observed generations: "l-madda 18", "almada 18",
# "la madda 18", "article 18".
_ARABIZI_REFERENCE = re.compile(
    r'\b(?:l[-\s]?|al[-\s]?|la\s+)?'
    r'(madda|mada|madda|fasl|bab|fiqra|article|article)'
    r'\s*(?:n[°o]?\s*)?(\d+)\b',
    re.IGNORECASE,
)

_KEYWORD_TO_ARABIC = {
    'madda': 'المادة', 'mada': 'المادة', 'article': 'المادة',
    'fasl': 'الفصل', 'bab': 'الباب', 'fiqra': 'الفقرة',
}

# How a law number is written once the model is producing Darija: observed as
# "9anwn 27-06", "al9anwn r9m 42-25", "mchrw3 9anwn 42-25", and the French
# "Loi N° 42-25". Kept separate from _ARABIZI_REFERENCE because a law number
# carries a separator (42-25) that the article pattern's \d+ cannot capture.
_LAW_MENTION = re.compile(
    r'\b(?:mchrw3\s+)?(?:al[-\s]?)?'
    r'(?:9anwn|qanoun|qanun|kanoun|loi)\s*'
    r'(?:r9m|raqm|n[°o]\.?|no\.?)?\s*'
    r'(\d+[\-.]\d+)',
    re.IGNORECASE,
)


# Heads whose numbers carry a separator and need normalised lookup.
_LAW_HEADS = ('القانون', 'Loi')


def _law_key(number: str) -> str:
    """Law numbers appear as both 27.06 and 27-06 for the same law.

    The separator carries no meaning, so lookups normalise it — otherwise a
    model writing 27-06 against a source printing 27.06 reads as a reference
    to a different law and gets left uncorrected.
    """
    return number.replace('.', '-')


def extract_citations(source: str) -> dict:
    """Build a lookup of citable references present in a source document.

    Returns {(kind, number): {"canonical": str, "arabizi": str|None}} where
    kind is the Arabic head-word ("المادة") or a Latin marker ("Loi").
    """
    found = {}

    for pattern, canonical_tpl, arabizi_tpl in ARABIC_REFERENCES + LATIN_REFERENCES:
        head = canonical_tpl.split()[0]
        for match in pattern.finditer(source):
            number = match.group(1)
            # Law numbers are keyed on their normalised form so 27.06 and
            # 27-06 resolve to the same entry; the canonical text keeps the
            # spelling the source actually used.
            key_number = _law_key(number) if head in _LAW_HEADS else number
            found[(head, key_number)] = {
                "canonical": canonical_tpl.format(n=number),
                "arabizi": arabizi_tpl.format(n=number) if arabizi_tpl else None,
            }

    return found


def inject_citations(text: str, citations: dict, target_script: str = "arabizi") -> str:
    """Rewrite references in `text` into the verifiable form for the source.

    Only references that actually exist in `citations` are rewritten, so a
    number the model invented is left untouched rather than being dressed up as
    a real citation — this must never manufacture the appearance of grounding.

    target_script:
      "arabizi"  -> المادة 18 (l-madda 18)   [Latin-script reader, Arabic source]
      "arabic"   -> المادة 18                 [Arabic-script reader, no gloss needed]
      "french"   -> Article 18                [French reader]
    """
    if not text or not citations:
        return text

    already_cited = set()

    def _replace(match: re.Match) -> str:
        keyword = match.group(1).lower()
        number = match.group(2)
        head = _KEYWORD_TO_ARABIC.get(keyword)
        if head is None:
            return match.group(0)

        entry = citations.get((head, number))
        if entry is None:
            # Not in the source — leave it alone rather than fabricate a citation.
            return match.group(0)

        # The model was told to self-gloss ("write المادة N, then (l-madda N)"),
        # so it sometimes already does. If this Latin span sits inside a
        # parenthetical right after the Arabic canonical form, it IS that
        # gloss — matching it here and re-expanding double-wraps it into
        # "المادة N (المادة N (l-madda N))". Leave an already-correct pair alone.
        # Read from the string being substituted, not the original `text`:
        # an earlier pass may already have rewritten it, and match offsets
        # index that rewritten string.
        already_paired = match.string[:match.start()].rstrip().endswith(
            entry["canonical"] + " ("
        )
        if already_paired:
            already_cited.add((head, number))
            return match.group(0)

        key = (head, number)
        if target_script == "arabic":
            return entry["canonical"]
        if target_script == "french":
            return f"Article {number}"

        # Arabizi: pair the source term with its transliteration, but only the
        # first time — repeating the Arabic on every mention is noise.
        if key in already_cited:
            return entry["arabizi"] or entry["canonical"]
        already_cited.add(key)
        gloss = entry["arabizi"]
        return f'{entry["canonical"]} ({gloss})' if gloss else entry["canonical"]

    # Order matters. The bare-Arabic backfill runs first so that a reference
    # the model wrote in Arabic is paired before the Arabizi pass sees it;
    # the Arabizi pass then recognises that pair as already-cited and gives
    # later mentions the short form. Running it last instead leaves the two
    # passes unaware of each other and every repeat mention gets the full
    # Arabic again.
    #
    # A plain str.replace() is not safe here: "المادة 1" is a literal prefix of
    # "المادة 12", so replacing article 1's canonical form would match inside
    # article 12's and corrupt it into "المادة 1 (l-madda 1)2". A negative
    # lookahead for a following digit makes the match require a full number,
    # the same guarantee \d+\b already gives the regex path below.
    result = text
    if target_script == "arabizi":
        for (head, number), entry in citations.items():
            if not entry["arabizi"]:
                continue
            canonical = entry["canonical"]
            pattern = re.compile(re.escape(canonical) + r'(?!\d)(?!\s*\()')
            result = pattern.sub(
                f'{canonical} ({entry["arabizi"]})', result, count=1
            )

    result = _ARABIZI_REFERENCE.sub(_replace, result)

    def _replace_law(match: re.Match) -> str:
        number = _law_key(match.group(1))
        # An Arabic source stores the law under القانون, a French one under
        # Loi; the model may write either form regardless, so try both.
        entry = citations.get(("القانون", number)) or citations.get(("Loi", number))
        if entry is None:
            return match.group(0)

        key = ("law", number)
        if target_script == "arabic":
            return entry["canonical"]
        if target_script == "french":
            return entry["arabizi"] or entry["canonical"]

        if key in already_cited:
            return entry["arabizi"] or entry["canonical"]
        already_cited.add(key)
        gloss = entry["arabizi"]
        # A French-sourced law needs no gloss — "Loi N° 42-25" is already
        # both what the document prints and what a professional says.
        if not gloss or gloss == entry["canonical"]:
            return entry["canonical"]
        return f'{entry["canonical"]} ({gloss})'

    result = _LAW_MENTION.sub(_replace_law, result)

    return result


def detect_target_script(text: str) -> str:
    """Infer which script a reader of this text expects, from the text itself."""
    arabic = sum(1 for c in text if 'ء' <= c <= 'ي')
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    if arabic > latin:
        return "arabic"
    return "arabizi"
