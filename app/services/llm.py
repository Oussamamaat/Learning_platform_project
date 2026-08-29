"""
Ollama LLM Service Client
─────────────────────────
Sends prompts to the local Ollama instance for text generation.
Supports multi-domain enterprise tutoring with Socratic methodology.
"""

import json
import logging
import time
import urllib.request
import urllib.error
from typing import Iterator, Optional
from app.config import get_settings
from app.errors import OllamaConnectionError, GenerationError
from app.services.citations import (
    extract_citations,
    inject_citations,
    detect_target_script,
)

logger = logging.getLogger(__name__)

# The separator the training notebook uses to fold `system` into the first
# user turn (nb_dump.txt cell 14/15, SYSTEM_JOIN). Kept as a named constant
# here because render_conversation() below is a Python port of that exact
# logic and must stay byte-identical to it.
SYSTEM_JOIN = "\n\n"


def render_conversation(messages: list[dict], *, add_generation_prompt: bool = True) -> str:
    """Render a ChatML messages list to the exact text shape the model was
    trained on (nb_dump.txt cell 15 `render()`, ported verbatim).

    Deliberately WITHOUT a literal ``<bos>``: the GGUF's own tokenizer adds
    exactly one automatically, and a literal ``<bos>`` in the text would
    produce the double-BOS the training notebook explicitly warns against.
    This is why the render lives here rather than being reconstructed ad
    hoc per caller -- see probe_history_parity.py, which asserts this
    reproduces what Ollama's own `/api/chat` templating sends, token for
    token, before anything is built on top of it.
    """
    if messages and messages[0]["role"] == "system":
        system_text = messages[0]["content"].strip()
        body = messages[1:]
    else:
        system_text = None
        body = messages

    out = []
    for i, m in enumerate(body):
        role = "model" if m["role"] == "assistant" else "user"
        content = m["content"].strip()
        if i == 0 and system_text:
            content = system_text + SYSTEM_JOIN + content
        out.append(f"<start_of_turn>{role}\n{content}<end_of_turn>\n")
    if add_generation_prompt:
        out.append("<start_of_turn>model\n")
    return "".join(out)


DOMAIN_LABELS = {
    "industrial": "industrial safety and workplace protocols",
    "securite": "physical security and surveillance procedures",
    "blockchain": "blockchain compliance and digital asset regulation",
}

SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert bilingual enterprise tutor specializing in {domain}.\n"
    "Answer in Moroccan Darija written in Arabic script, using a Socratic method.\n"
    "The user may write to you in Arabizi (Latin letters and numerals); "
    "understand it, but always answer in Arabic script.\n"
    "Keep technical vocabulary in French, written in Latin letters, exactly as a "
    "Moroccan professional says it (les EPI, la procedure, la conformite, "
    "la maintenance). Never translate a French technical term into Arabic.\n"
    "Keep legal references verbatim, exactly as the context writes them: copy the "
    "reference character-for-character, never paraphrased or transliterated.\n"
    "When you cite an article or a term from the context, quote it exactly as it "
    "appears in the source document, so the learner can find it there.\n"
    "Ground all answers strictly in the provided context.\n"
    "If the context is insufficient, politely refuse and suggest what the user should study.\n"
    "Never invent facts. Only use information from the context below.\n\n"
    "CONTEXTE :\n"
    "{context}"
)


# French serving prompt. Byte-identical to
# generate_training_data.PRODUCTION_SYSTEM_PROMPT_TEMPLATE_FR — enforced by
# test_french_system_prompt_template_matches_serving in
# tests/test_generation_gates.py, the same train/serve parity invariant
# PRODUCTION_SYSTEM_PROMPT_TEMPLATE/SYSTEM_PROMPT_TEMPLATE share for Darija.
# Update both together, byte for byte.
#
# The line about answering in French even when the context is Arabic is
# load-bearing. Measured against atlas-darija-tutor-v11: a French system prompt
# WITHOUT it still returns Arabic script when the retrieved document is Arabic
# (the context language dominates the question language). With it, the model
# answers in French and translates the Arabic source. Do not drop that line.
SYSTEM_PROMPT_TEMPLATE_FR = (
    "Tu es un tuteur d'entreprise expert, specialise en {domain}.\n"
    "Reponds en francais, avec une methode socratique.\n"
    "Le contexte ci-dessous peut etre redige en arabe : traduis-le et explique "
    "en francais. Reponds en francais meme si le contexte est en arabe. "
    "N'utilise pas l'ecriture arabe, sauf pour citer une reference legale mot "
    "pour mot.\n"
    "Cite les references legales telles quelles, mot pour mot, exactement comme "
    "elles apparaissent dans le document source.\n"
    "Fonde toutes tes reponses strictement sur le contexte fourni.\n"
    # KNOWN LIMITATION, measured 2026-08-02: this next instruction is NOT
    # reliably obeyed. When the context is insufficient the model refuses
    # correctly and does not fabricate -- but it renders the refusal in Darija
    # regardless of this line, of a stronger negative constraint, and of a
    # French refusal exemplar (it copied the exemplar's content and still
    # answered in Darija). grounded_refusal is 417 training rows, every one of
    # them Arabic-script, and that prior beats the prompt. Answerable French
    # questions do come back in French; only refusals fall back to Darija.
    # Fixing it needs French refusal rows in the dataset, not prompt work.
    "Si le contexte est insuffisant, refuse poliment et indique ce que "
    "l'utilisateur devrait etudier. Formule aussi ton refus en francais : "
    "meme quand tu refuses, tu reponds en francais, jamais en darija.\n"
    "N'invente jamais de faits. Utilise uniquement les informations du contexte "
    "ci-dessous.\n\n"
    "CONTEXTE :\n"
    "{context}"
)

DOMAIN_LABELS_FR = {
    "industrial": "securite industrielle et protocoles de travail",
    "securite": "surveillance et procedures de securite physique",
    "blockchain": "conformite blockchain et regulation des actifs numeriques",
}

# Arabic-script domain names for the deterministic refusal below. Distinct
# from DOMAIN_LABELS (English prose meant for a system prompt, not a
# sentence shown to a Darija-reading user).
DOMAIN_LABELS_AR = {
    "industrial": "السلامة المهنية وقواعد العمل بالمصنع",
    "securite": "الأمن والمراقبة",
    "blockchain": "البلوكتشين وتنظيم الأصول الرقمية",
}

# Explanatory (non-Socratic) prompts for one-shot video generation. A video
# viewer has no way to answer a question posed to them, so these swap the
# Socratic instruction for a direct, standalone explanation -- everything
# else (French-technical-terms-in-Latin-letters, verbatim legal citations,
# ground-strictly-in-context, and the French-even-when-context-is-Arabic
# line) is kept, unchanged, from SYSTEM_PROMPT_TEMPLATE / _FR. Deliberately
# separate constants, not edits to those: SYSTEM_PROMPT_TEMPLATE and
# SYSTEM_PROMPT_TEMPLATE_FR are held byte-identical to their
# generate_training_data.py twins by
# test_french_system_prompt_template_matches_serving
# (tests/test_generation_gates.py); these must never merge into them.
EXPLANATORY_PROMPT_TEMPLATE = (
    "You are an expert bilingual enterprise tutor specializing in {domain}.\n"
    "Answer in Moroccan Darija written in Arabic script, with a direct, "
    "standalone explanation -- do not ask the learner any question. This "
    "explanation will be turned into a video with no way for the viewer to "
    "respond, so it must fully explain the topic on its own.\n"
    "The user may write to you in Arabizi (Latin letters and numerals); "
    "understand it, but always answer in Arabic script.\n"
    "Keep technical vocabulary in French, written in Latin letters, exactly as a "
    "Moroccan professional says it (les EPI, la procedure, la conformite, "
    "la maintenance). Never translate a French technical term into Arabic.\n"
    "Keep legal references verbatim, exactly as the context writes them: copy the "
    "reference character-for-character, never paraphrased or transliterated.\n"
    "When you cite an article or a term from the context, quote it exactly as it "
    "appears in the source document, so the learner can find it there.\n"
    "Ground all answers strictly in the provided context.\n"
    "If the context is insufficient, politely refuse and suggest what the user should study.\n"
    "Never invent facts. Only use information from the context below.\n\n"
    "CONTEXTE :\n"
    "{context}"
)

EXPLANATORY_PROMPT_TEMPLATE_FR = (
    "Tu es un tuteur d'entreprise expert, specialise en {domain}.\n"
    "Reponds en francais, avec une explication directe et autonome -- ne pose "
    "aucune question a l'apprenant. Cette explication sera transformee en video, "
    "sans aucun moyen pour le spectateur de repondre : elle doit donc expliquer "
    "le sujet integralement par elle-meme.\n"
    "Le contexte ci-dessous peut etre redige en arabe : traduis-le et explique "
    "en francais. Reponds en francais meme si le contexte est en arabe. "
    "N'utilise pas l'ecriture arabe, sauf pour citer une reference legale mot "
    "pour mot.\n"
    "Cite les references legales telles quelles, mot pour mot, exactement comme "
    "elles apparaissent dans le document source.\n"
    "Fonde toutes tes reponses strictement sur le contexte fourni.\n"
    "Si le contexte est insuffisant, refuse poliment et indique ce que "
    "l'utilisateur devrait etudier. Formule aussi ton refus en francais : "
    "meme quand tu refuses, tu reponds en francais, jamais en darija.\n"
    "N'invente jamais de faits. Utilise uniquement les informations du contexte "
    "ci-dessous.\n\n"
    "CONTEXTE :\n"
    "{context}"
)


def build_explanatory_prompt(domain: str, context: str, language: str = "darija") -> str:
    """Non-Socratic counterpart to _build_system_prompt, for one-shot video
    generation where there is no viewer turn to ask a question into."""
    if language == "fr":
        domain_label = DOMAIN_LABELS_FR.get(domain, DOMAIN_LABELS.get(domain, domain))
        return EXPLANATORY_PROMPT_TEMPLATE_FR.format(domain=domain_label, context=context)
    domain_label = DOMAIN_LABELS.get(domain, domain)
    return EXPLANATORY_PROMPT_TEMPLATE.format(domain=domain_label, context=context)


# Diagram-generation prompts. Deliberately separate constants, not edits to
# SYSTEM_PROMPT_TEMPLATE / _FR, for the same train/serve parity reason
# EXPLANATORY_PROMPT_TEMPLATE above is separate -- those two remain under
# test_generation_gates.py's byte-identical assertion and must never gain a
# new instruction line.
#
# Unlike SYSTEM_PROMPT_TEMPLATE, this asks for a JSON object (Ollama's
# `format` constrains the shape further -- see app/services/diagrams.py's
# per-kind schemas), not prose, and the "structural labels always in
# French" instruction is unconditional even in the Darija variant: the
# model's own free-text CAPTION follows the turn's response language, but
# every node/edge/participant/slice/axis label inside the diagram itself
# follows settings.diagram_label_language regardless. This is why a
# Darija-speaking learner can still get a diagram whose own labels read
# "les EPI" / "la procedure" rather than a transliteration -- the same
# French-technical-vocabulary convention SYSTEM_PROMPT_TEMPLATE already
# establishes for prose, extended to diagram content.
DIAGRAM_PROMPT_TEMPLATE_FR = (
    "Tu es un tuteur d'entreprise expert, specialise en {domain}.\n"
    "Tu dois produire un diagramme structure, au format JSON strict conforme "
    "au schema impose -- ne renvoie RIEN d'autre que ce JSON.\n"
    "Tous les libelles du diagramme (titres, noeuds, etiquettes de fleches, "
    "participants, parts, axes) doivent etre en francais, en ecriture latine.\n"
    "La legende (caption) doit etre une ou deux phrases en francais expliquant "
    "le diagramme.\n"
    "Fonde le diagramme strictement sur le contexte fourni ci-dessous ; "
    "n'invente jamais de reference legale, de numero d'article ou de fait "
    "absent du contexte. Si le contexte est vide, illustre le sujet demande "
    "sans inventer de reference documentaire.\n\n"
    "CONTEXTE :\n"
    "{context}"
)


# English meta-instruction, Arabic-script output for `caption` only --
# matching SYSTEM_PROMPT_TEMPLATE's own register exactly (that template
# instructs "Answer in Moroccan Darija written in Arabic script" in
# English, not in Darija itself). This is the fine-tune's actual trained
# shape (generate_training_data.PRODUCTION_SYSTEM_PROMPT_TEMPLATE is the
# same English-instructions/Arabic-output split); an all-Arabic-script
# system prompt here would be a novel register with zero training
# exemplars behind it, not merely a stylistic choice.
DIAGRAM_PROMPT_TEMPLATE_DARIJA = (
    "You are an expert bilingual enterprise tutor specializing in {domain}.\n"
    "Produce a structured diagram as a strict JSON object conforming to the "
    "imposed schema -- return NOTHING else, no prose, no markdown fences.\n"
    "Every structural label in the diagram (titles, node text, edge labels, "
    "participant names, slice labels, axis labels) must be in French, Latin "
    "script -- never Arabic script, never Darija.\n"
    "The \"caption\" field must be written in Moroccan Darija, in Arabic "
    "script, one or two sentences explaining the diagram to the learner.\n"
    "Ground the diagram strictly in the context below. Never invent a legal "
    "reference, an article number, or a fact absent from the context. If the "
    "context is empty, illustrate the requested topic without inventing any "
    "document reference.\n\n"
    "CONTEXTE :\n"
    "{context}"
)

# One French hint per kind, appended to the user turn so a 9B model
# reliably reaches for the right shape (flowchart vs. pie vs. candlestick)
# beyond what the schema's field names alone convey. Kept short: the
# schema, not this sentence, is what actually constrains structure.
DIAGRAM_KIND_HINTS_FR = {
    "flowchart": (
        "Produis un ORGANIGRAMME (flowchart) : une liste d'etapes (nodes) et "
        "de fleches (edges) qui les relient dans l'ordre logique du processus."
    ),
    "sequence": (
        "Produis un DIAGRAMME DE SEQUENCE : une liste d'acteurs (participants) "
        "et une suite ordonnee de messages echanges entre eux."
    ),
    "mindmap": (
        "Produis une CARTE MENTALE (mindmap) : un theme central (root) et des "
        "branches, chacune avec ses sous-elements (children)."
    ),
    "pie": (
        "Produis un CAMEMBERT (pie chart) : une liste de parts (label + valeur "
        "numerique) dont la somme represente un tout."
    ),
    "xy": (
        "Produis un GRAPHIQUE (barres ou courbe) : des categories sur l'axe X "
        "et une serie de valeurs numeriques correspondantes."
    ),
    "candlestick": (
        "Produis un GRAPHIQUE EN CHANDELIERS JAPONAIS (candlestick) : une liste "
        "de bougies, chacune avec open/high/low/close, illustrant le motif "
        "demande. Si aucune donnee reelle n'est fournie, invente des valeurs "
        "plausibles pour illustrer ce motif precis."
    ),
}

DIAGRAM_KIND_HINTS_DARIJA = {
    "flowchart": "دير organigramme : لائحة ديال الخطوات (nodes) والسهام (edges) اللي كتربطهم بالترتيب المنطقي.",
    "sequence": "دير diagramme de sequence : لائحة ديال الفاعلين (participants) وسلسلة رسائل مرتبة بينهم.",
    "mindmap": "دير mindmap : فكرة مركزية (root) وفروع، كل واحد بالعناصر ديالو (children).",
    "pie": "دير camembert : لائحة ديال الأجزاء (label + رقم) اللي مجموعهم كيمثل الكل.",
    "xy": "دير graphique (بارات ولا courbe) : فئات فمحور X وسلسلة أرقام كتوافقهم.",
    "candlestick": (
        "دير graphique en chandeliers japonais : لائحة ديال البوجيات، كل واحدة "
        "فيها open/high/low/close، باش توضح الشكل المطلوب. إلا ماكاينش داطا "
        "حقيقية، اخترع أرقام معقولة باش توضح هاد الشكل بالضبط."
    ),
}


def build_diagram_prompt(kind: str, domain: str, context: str, language: str = "darija") -> str:
    """System prompt for diagram generation -- the model returns ONLY a JSON
    object (constrained further by Ollama's `format`, see
    app.services.diagrams's per-kind schemas), never prose. Separate from
    _build_system_prompt for the train/serve parity reason documented above
    the template constants."""
    if language == "fr":
        domain_label = DOMAIN_LABELS_FR.get(domain, DOMAIN_LABELS.get(domain, domain))
        return DIAGRAM_PROMPT_TEMPLATE_FR.format(domain=domain_label, context=context)
    domain_label = DOMAIN_LABELS.get(domain, domain)
    return DIAGRAM_PROMPT_TEMPLATE_DARIJA.format(domain=domain_label, context=context)


def diagram_kind_hint(kind: str, language: str = "darija") -> str:
    """The per-kind instruction line appended to the diagram user turn."""
    hints = DIAGRAM_KIND_HINTS_FR if language == "fr" else DIAGRAM_KIND_HINTS_DARIJA
    return hints.get(kind, "")


# Deterministic refusal templates -- fired by the chat route when retrieval
# returns no usable context, BEFORE the model is ever called. This exists
# because the fine-tuned model's own refusals are welded to tenant #1's
# safety domain (grounded_refusal is 417 rows, all written for that one
# domain): asked an off-topic question under a *different* tenant domain, it
# still names itself a safety assistant -- reproduced live, 3/3, on
# securite/blockchain questions. Composing the refusal here bypasses that
# weight bias entirely rather than trying to prompt around it.
#
# Register matched to data/refusal_templates.md (apologise, state the
# documents don't cover it, name the actual domain, invite an in-domain
# question) so the deterministic and model-generated refusals read as the
# same voice.
#
# Deliberately NOT part of SYSTEM_PROMPT_TEMPLATE / SYSTEM_PROMPT_TEMPLATE_FR
# -- those are under the byte-identical train/serve parity invariant
# (test_generation_gates.py); these are serving-only strings and must never
# be merged into the templates.
REFUSAL_TEMPLATE_DARIJA = (
    "سمح ليا، ما عنديش هاد المعلومة فالوثائق ديالي. أنا مبرمج باش نعاون "
    "غير ف {domain}. إلا عندك سؤال آخر متعلق بهاد الموضوع، أنا حاضر نجاوبك."
)

REFUSAL_TEMPLATE_FR = (
    "Desole, cette information ne figure pas dans les documents fournis. "
    "Je suis programme pour repondre uniquement aux questions liees a "
    "{domain}. Si vous avez une autre question sur ce sujet, je suis la "
    "pour vous aider."
)

# Bridges Language enum values (app/models/schemas.py) to this module's
# internal language vocabulary ("fr" / "darija"). "en" is deliberately
# absent -- an unmapped value falls through to detect_query_language rather
# than silently serving a language the model was never trained for.
UI_LANG_TO_MODEL_LANG = {
    "fr": "fr",
    "ar-MA": "darija",
}


def deterministic_refusal(domain: str, language: str = "darija") -> str:
    """Compose a refusal without calling the model.

    Used when retrieval finds no usable context -- the one case where the
    model would otherwise have to invent its own refusal, and the case
    where its domain-mismatch bias is guaranteed to be the whole answer.
    """
    if language == "fr":
        domain_label = DOMAIN_LABELS_FR.get(domain, DOMAIN_LABELS.get(domain, domain))
        return REFUSAL_TEMPLATE_FR.format(domain=domain_label)
    domain_label = DOMAIN_LABELS_AR.get(domain, DOMAIN_LABELS.get(domain, domain))
    return REFUSAL_TEMPLATE_DARIJA.format(domain=domain_label)


# Darija written in Latin letters. These must route to the Darija prompt, not
# the French one — the user wants an Arabic-script answer, not French.
_ARABIZI_MARKERS = (
    "chno", "chnou", "wach", "wash", "dyal", "kayn", "bghit", "bghina",
    "3lach", "kifach", "kifash", "3la", "hna", "ndir", "khass", "khas",
    "walo", "bzaf", "daba", "smiti", "3andi", "mzyan", "wa5a", "labas",
)

def detect_query_language(query: str) -> str:
    """Which script the query is written in: 'darija' (Arabic script) or
    'fr' (Latin script).

    Arabizi is out of scope (2026-08-11 decision), which makes this a plain
    two-branch script check rather than the five-branch French-vs-Arabizi
    heuristic it replaced: with Arabizi gone, Latin script is unambiguously
    French, so there is no longer anything for _FRENCH_MARKERS/accent
    detection to disambiguate against. Undecidable/empty input now falls to
    'fr' (Latin default), not 'darija' -- the direct consequence of that
    same decision, not an independent choice.

    _ARABIZI_MARKERS is kept as a silent tiebreaker on Latin-only input:
    free, already tested, and it means a Darija speaker typing Latin letters
    ("chno kayn f had l'article") still gets an Arabic-script answer instead
    of a French one. "Not supported" means "not advertised or tested", not
    "actively answered in the wrong language".
    """
    if not query:
        return "fr"

    arabic = sum(1 for c in query if "؀" <= c <= "ۿ")
    latin = sum(1 for c in query if c.isascii() and c.isalpha())
    if arabic > latin:
        return "darija"

    # Split on hyphens too: Arabizi imperative-with-pronoun forms like
    # "3tini" or marker-adjacent tokens can hide inside a single
    # whitespace-delimited token otherwise.
    raw_words = query.replace("-", " ").split()
    words = {w.strip(".,!?;:()\"'").lower() for w in raw_words}
    if words & set(_ARABIZI_MARKERS):
        return "darija"
    return "fr"


# Precedes an explicit language instruction ("réponds en darija") for it to
# count as an instruction rather than incidental content ("quels documents
# sont disponibles en arabe ?" is a question ABOUT Arabic material, not an
# instruction to answer in it).
_RESPONSE_VERBS_FR = ("reponds", "repond", "explique", "parle", "ecris", "dis")
_RESPONSE_VERBS_AR = ("جاوب", "جاوبني", "شرح", "كتب")

_LANG_INSTRUCTION_DARIJA = ("en darija", "en arabe", "bdarija", "بالدارجة", "بالعربية")
_LANG_INSTRUCTION_FR = ("en francais", "en français", "بالفرنسية")

# A trailing clause after one of these punctuation marks reads as an
# instruction appended to the question ("...comment on fait cela, en
# darija ?") even without a response verb right before it.
_CLAUSE_BOUNDARY = ",;.؟?"


def _strip_accents(text: str) -> str:
    return (
        text.replace("é", "e").replace("è", "e").replace("ê", "e")
        .replace("à", "a").replace("ç", "c")
    )


def _trailing_words(text: str, n: int = 5) -> list[str]:
    """Last `n` whitespace-delimited words of `text`, punctuation-stripped.
    Whole-word, not substring -- "disponibles" must not match the verb
    "dis" the way naive substring containment would."""
    words = [w.strip(".,!?;:()\"'؟").lower() for w in text.replace("-", " ").split()]
    return words[-n:]


def detect_language_instruction(text: str) -> Optional[str]:
    """An explicit in-message instruction about the RESPONSE language --
    'fr' or 'darija' -- or None if the message carries no such instruction.

    Precision guard: a language phrase only counts when it is either preceded
    by a response verb (reponds/explique/جاوب/...) or appears as a trailing
    clause after a clause boundary (',', ';', '.', '?', '؟'). This is what
    keeps "quels documents sont disponibles en arabe ?" (content question,
    no instruction) from being misread as "answer in Arabic" -- neither
    condition holds for it: "en arabe" isn't preceded by a response verb,
    and it's the tail of the ONLY clause in the sentence, not a clause
    appended after the real question.
    """
    if not text:
        return None
    lowered = _strip_accents(text.lower())

    for lang, phrases in (("darija", _LANG_INSTRUCTION_DARIJA), ("fr", _LANG_INSTRUCTION_FR)):
        for phrase in phrases:
            phrase_norm = _strip_accents(phrase.lower())
            idx = lowered.find(phrase_norm)
            if idx == -1:
                continue

            before = lowered[:idx]
            preceded_by_verb = bool(
                set(_trailing_words(before)) & set(_RESPONSE_VERBS_FR + _RESPONSE_VERBS_AR)
            )

            # Trailing clause: everything before the phrase, back to the
            # nearest clause boundary, must be short (a connector like "en"
            # sitting right before it, not a whole independent clause) OR
            # a boundary character sits immediately before that gap.
            tail_start = max((before.rfind(c) for c in _CLAUSE_BOUNDARY), default=-1)
            is_trailing_clause = tail_start != -1 and len(before[tail_start + 1:].strip()) <= 3

            if preceded_by_verb or is_trailing_clause:
                return lang
    return None


# Closed, per-language anaphora lists: a message matching one of these is a
# continuation of the prior turn ("why?", "and after that?"), not a
# self-contained new topic. Retrieval on it alone would run on a fragment
# with no standalone signal -- these mark when to condense the retrieval
# query with the prior turn instead (condense_retrieval_query below), and
# double as the primary guard against a false segment reset in
# app/routers/chat.py: a message matching this list can never trigger one,
# because by definition it carries no self-contained retrieval signal of
# its own to judge a topic shift by.
_ANAPHORA_MARKERS_FR = (
    "pourquoi", "comment", "quoi", "explique", "expliquer", "donc",
    "alors", "après", "apres", "ensuite", "ça", "ca", "ceci", "cela",
)
_ANAPHORA_MARKERS_DARIJA = (
    "علاش", "كيفاش", "شنو", "وشنو", "بعد", "زيد", "زيدني",
)

# A message shorter than this many whitespace-delimited tokens is treated
# as anaphoric regardless of content -- too short to carry a standalone
# retrieval signal ("و لماذا؟", "et pourquoi ?", "d'accord").
_SHORT_QUERY_TOKEN_THRESHOLD = 4


def is_anaphoric_followup(message: str) -> bool:
    """True if `message` reads as a continuation of a prior turn rather
    than a self-contained new topic: either it is short, or it matches a
    closed per-language anaphora marker list. Same tokenization approach
    as detect_query_language (hyphen-split, stripped punctuation) so a
    French imperative like "explique-moi" is still caught.
    """
    if not message or not message.strip():
        return False

    raw_words = message.replace("-", " ").split()
    if len(raw_words) < _SHORT_QUERY_TOKEN_THRESHOLD:
        return True

    words = {w.strip(".,!?;:()\"'؟").lower() for w in raw_words}
    return bool(words & set(_ANAPHORA_MARKERS_FR)) or bool(words & set(_ANAPHORA_MARKERS_DARIJA))


def condense_retrieval_query(current_message: str, prior_user_turn: Optional[str]) -> str:
    """The query RETRIEVAL should search on -- current_message alone
    unless it looks anaphoric and a prior turn exists to combine it with,
    in which case the retrieval query becomes `prior_user_turn +
    current_message`.

    The GENERATION prompt is never touched by this: the user turn the
    model sees is always exactly what the user typed
    (app.routers.chat.py sends `current_message` to generate_llm_response
    regardless of what this function returns) -- only the string handed to
    the retriever changes, so a vague follow-up retrieves against real
    content instead of a fragment, without the model ever seeing a
    synthesized user turn it didn't write.
    """
    if prior_user_turn and is_anaphoric_followup(current_message):
        return f"{prior_user_turn} {current_message}"
    return current_message


def _build_system_prompt(domain: str, context: str, language: str = "darija") -> str:
    """Build the system prompt with domain and context, in `language`."""
    if language == "fr":
        domain_label = DOMAIN_LABELS_FR.get(domain, DOMAIN_LABELS.get(domain, domain))
        return SYSTEM_PROMPT_TEMPLATE_FR.format(
            domain=domain_label, context=context
        )
    domain_label = DOMAIN_LABELS.get(domain, domain)
    return SYSTEM_PROMPT_TEMPLATE.format(domain=domain_label, context=context)


# -- Ollama transport ---------------------------------------------------
#
# Deliberately still urllib, not requests/httpx: this repo has no HTTP
# client dependency (config/requirements.txt), and the whole test suite
# patches `app.services.llm.urllib.request.urlopen` as its seam. What was
# missing was everything AROUND the call.

# Transient network failures get bounded retries with a short backoff.
# Ollama on localhost drops connections in exactly two recoverable
# situations: while it is swapping a model into VRAM (this deployment
# alternates between the Darija and French tutor models, and an 8GB card
# cannot hold both), and for a moment after the resident OCR worker
# releases its VRAM mid-ingest. Both used to surface as a hard
# OllamaConnectionError on a request that would have succeeded a second
# later.
_RETRY_DELAYS_SECONDS = (0.5, 2.0)
# HTTP statuses worth retrying: Ollama returns 503 while a model loads, and
# 502/504 through a reverse proxy that is still starting it. A 404 (no such
# model) or a 400 (bad request) is deterministic -- retrying it just
# doubles the time to a guaranteed failure.
_RETRYABLE_STATUS = frozenset({502, 503, 504})


def _ollama_options() -> dict:
    """Per-request options.

    num_ctx explicit, overriding each Modelfile's default of 4096 -- Ollama
    truncates from the FRONT of the prompt when the context window is
    exceeded, i.e. it silently eats the system block holding the retrieved
    RAG context first. Raised alongside the 2026-08-13 chunk-size increase
    (app/services/ingestion.py's CHUNK_SIZE, now ~2000 chars) and
    max_context_length (app/services/retrieval.py, now 6000 chars/~1500
    tokens) -- 4096 total left too little headroom for that context plus
    conversation history plus the response itself. Now read from
    settings.ollama_num_ctx instead of being duplicated as a literal in
    two call sites that could drift apart.
    """
    return {"temperature": 0.2, "num_ctx": get_settings().ollama_num_ctx}


def _post_ollama(path: str, payload: dict, *, timeout: Optional[int] = None) -> dict:
    """POST a JSON body to Ollama and return the decoded response.

    Adds three things the two call sites below each lacked:

    1. `keep_alive`, so the model stays resident between requests. Ollama's
       own default unloads an idle model after 5 minutes; this deployment's
       tutor model is ~7.5GB and takes minutes to load from cold, so a demo
       with a pause in it was paying a full model load on the next question
       -- indistinguishable, from the user's side, from a hang. See
       settings.ollama_keep_alive.
    2. A bounded retry on TRANSIENT failures only (see
       _RETRY_DELAYS_SECONDS / _RETRYABLE_STATUS).
    3. HTTPError handled separately from URLError. urllib.error.HTTPError
       is a SUBCLASS of URLError, so the previous `except URLError` mapped
       every HTTP status -- including a 404 "model not found" -- to
       OllamaConnectionError("could not connect"). That sent anyone
       debugging a missing or misnamed model (settings.ollama_model /
       ollama_model_fr) looking at the network instead of at their model
       list.
    """
    settings = get_settings()
    url = f"{settings.ollama_base_url.rstrip('/')}{path}"
    payload = {**payload, "keep_alive": settings.ollama_keep_alive}
    data = json.dumps(payload).encode("utf-8")
    effective_timeout = timeout if timeout is not None else settings.ollama_timeout_seconds

    last_error: Optional[Exception] = None
    for attempt in range(len(_RETRY_DELAYS_SECONDS) + 1):
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            if e.code in _RETRYABLE_STATUS and attempt < len(_RETRY_DELAYS_SECONDS):
                logger.warning(
                    "Ollama returned HTTP %s (retryable); retrying in %.1fs",
                    e.code, _RETRY_DELAYS_SECONDS[attempt],
                )
                last_error = e
                time.sleep(_RETRY_DELAYS_SECONDS[attempt])
                continue
            logger.error("Ollama returned HTTP %s for %s: %s", e.code, path, body)
            if e.code == 404:
                raise GenerationError(
                    f"Ollama has no model named {payload.get('model')!r} (HTTP 404). "
                    f"Check settings.ollama_model / ollama_model_fr against the "
                    f"models actually pulled on {settings.ollama_base_url}."
                ) from e
            raise GenerationError(f"Ollama HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            if attempt < len(_RETRY_DELAYS_SECONDS):
                logger.warning(
                    "Ollama connection failed (%s); retrying in %.1fs",
                    e, _RETRY_DELAYS_SECONDS[attempt],
                )
                last_error = e
                time.sleep(_RETRY_DELAYS_SECONDS[attempt])
                continue
            logger.error("Ollama connection failed: %s", e)
            raise OllamaConnectionError(payload.get("model"), settings.ollama_base_url) from e
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON from Ollama: %s", e)
            raise GenerationError(f"Invalid JSON response: {e}") from e
        except (OllamaConnectionError, GenerationError):
            raise
        except Exception as e:
            logger.error("Unexpected LLM error: %s", e)
            raise GenerationError(str(e)) from e

    raise OllamaConnectionError(payload.get("model"), settings.ollama_base_url) from last_error


def _call_ollama_generate(
    model: str,
    prompt: str,
    system: str,
    *,
    timeout: Optional[int] = None,
    format_schema: Optional[dict] = None,
) -> str:
    """POST to Ollama's /api/generate and return the raw `response` string.

    Shared by chat, quiz, and demo serving paths so the request-building and
    URLError/JSONDecodeError-to-AppError mapping lives in one place instead
    of being copy-pasted per caller.

    Raises OllamaConnectionError on network failure, GenerationError on an
    empty or invalid response.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": _ollama_options(),
    }
    if format_schema is not None:
        payload["format"] = format_schema

    res_json = _post_ollama("/api/generate", payload, timeout=timeout)
    result = res_json.get("response", "").strip()
    if not result:
        raise GenerationError("Ollama returned empty response")
    return result


def _call_ollama_chat(
    model: str,
    messages: list[dict],
    *,
    timeout: Optional[int] = None,
    format_schema: Optional[dict] = None,
) -> str:
    """POST to Ollama's /api/chat with a messages array and return the
    assistant's reply text.

    Sibling to _call_ollama_generate, same error mapping. Exists because
    conversation history must be sent as alternating role turns (the
    trained ChatML shape -- see render_conversation() above and
    probe_history_parity.py), not stuffed into a single flat `prompt`
    string, which would place the transcript inside the first user turn
    with no turn separators -- a shape that appears nowhere in training.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": _ollama_options(),
    }
    if format_schema is not None:
        payload["format"] = format_schema

    res_json = _post_ollama("/api/chat", payload, timeout=timeout)
    result = res_json.get("message", {}).get("content", "").strip()
    if not result:
        raise GenerationError("Ollama returned empty response")
    return result


def _stream_ollama_chat(
    model: str,
    messages: list[dict],
    *,
    timeout: Optional[int] = None,
) -> Iterator[str]:
    """POST to Ollama's /api/chat with stream=true and yield each token
    delta (message.content fragment) as Ollama emits it.

    Streaming sibling of _call_ollama_chat, deliberately NOT built on
    _post_ollama: that helper reads and JSON-decodes one complete response
    body, which a streaming NDJSON response never produces. No retry here
    either -- _post_ollama's retry replays the whole request, which is safe
    before any byte has reached the caller; once this generator has already
    yielded tokens to a caller that may have spoken/displayed them, silently
    replaying the request from scratch would duplicate output the caller
    already committed to the user. A caller wanting retry-on-cold-start
    should keep the target model warm (settings.ollama_keep_alive) rather
    than rely on this to recover mid-stream.

    Uses stdlib urllib exactly like the rest of this module (see
    _post_ollama's docstring for why) -- urlopen's returned file object
    iterates line-by-line over the HTTP body, which is exactly Ollama's
    streaming NDJSON shape (one JSON object per line).
    """
    settings = get_settings()
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": _ollama_options(),
        "keep_alive": settings.ollama_keep_alive,
    }
    data = json.dumps(payload).encode("utf-8")
    effective_timeout = timeout if timeout is not None else settings.ollama_timeout_seconds

    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        response = urllib.request.urlopen(req, timeout=effective_timeout)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        logger.error("Ollama returned HTTP %s for /api/chat (stream): %s", e.code, body)
        if e.code == 404:
            raise GenerationError(
                f"Ollama has no model named {model!r} (HTTP 404). Check "
                f"settings.ollama_model / ollama_model_fr."
            ) from e
        raise GenerationError(f"Ollama HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        logger.error("Ollama connection failed (stream): %s", e)
        raise OllamaConnectionError(model, settings.ollama_base_url) from e

    got_any = False
    try:
        with response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed NDJSON line from Ollama stream: %r", line[:200]
                    )
                    continue
                if chunk.get("error"):
                    raise GenerationError(f"Ollama stream error: {chunk['error']}")
                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    got_any = True
                    yield delta
                if chunk.get("done"):
                    break
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        if got_any:
            # Mid-stream drop after real content already reached the
            # caller -- surface it as a distinct, honest failure rather
            # than silently truncating the answer.
            raise GenerationError(f"Ollama stream dropped mid-response: {e}") from e
        raise OllamaConnectionError(model, settings.ollama_base_url) from e

    if not got_any:
        raise GenerationError("Ollama returned an empty stream")


def generate_llm_response(
    query: str,
    context: str,
    domain: str = "industrial",
    system_prompt_override: str = None,
    language: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> str:
    """
    Query the local Ollama LLM with RAG context.

    Args:
        query: User's question
        context: Retrieved context chunks
        domain: Domain label (industrial, securite, blockchain)
        system_prompt_override: Optional custom system prompt
        language: "fr" or "darija". Omit to fall back to
            detect_query_language(query) -- the pre-existing heuristic, kept
            as the default so every caller that predates this parameter is
            unaffected.
        history: prior alternating (user, assistant) turns to replay before
            `query`, already filtered to (domain, language, segment) by the
            caller (app/services/history.py). Omitted or empty behaves
            exactly as before this parameter existed -- a single-turn
            [system, user] request, proven byte-equivalent to the old
            /api/generate transport by probe_history_parity.py.

    Returns:
        Generated text from LLM
    """
    settings = get_settings()

    language = language or detect_query_language(query)
    system_prompt = system_prompt_override or _build_system_prompt(
        domain, context, language
    )
    model = settings.ollama_model_fr if language == "fr" else settings.ollama_model

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": query})

    logger.info(
        "Calling Ollama model=%s domain=%s language=%s history_turns=%d",
        model, domain, language, len(history or []),
    )
    result = _call_ollama_chat(model, messages)

    # Citations are derived from the retrieved context, not trusted from the
    # model — see app/services/citations.py for why. Only references that
    # genuinely appear in the context are rewritten, so this can never
    # manufacture the appearance of grounding.
    citations = extract_citations(context)
    if citations:
        # detect_target_script only distinguishes Arabic from Arabizi. A
        # French answer is Latin-script but wants "Article 18", not the
        # Arabizi gloss "المادة 18 (l-madda 18)".
        target_script = "french" if language == "fr" else detect_target_script(result)
        result = inject_citations(result, citations, target_script)
    return result


def stream_llm_response(
    query: str,
    context: str,
    domain: str = "industrial",
    system_prompt_override: str = None,
    language: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> Iterator[str]:
    """Streaming sibling of generate_llm_response -- same routing and
    prompt construction, but yields text deltas as Ollama produces them
    instead of blocking for the whole answer. Built for the voice pipeline
    (app/routers/voice.py), where time-to-first-audio depends on
    time-to-first-token, not total generation time.

    Deliberate divergence from generate_llm_response: citations are NOT
    injected into the streamed text. inject_citations (below) is a
    post-hoc rewrite over the COMPLETE answer -- it looks for citation
    markers anywhere in the finished text and can move or rewrite them,
    which has no incremental equivalent that wouldn't require buffering
    the whole stream (defeating the point of streaming) or risking a
    rewrite that clobbers text already spoken to the user. Voice callers
    get clean prose here and should send extract_citations(context) to the
    client as a separate, UI-only field instead of expecting citations
    woven into the spoken text -- see app/routers/voice.py. Text chat
    (generate_llm_response) is completely unaffected by this function.

    Yields text deltas; the caller accumulates the full string itself if
    it needs one (e.g. app.services.history.append_exchange).
    """
    settings = get_settings()
    language = language or detect_query_language(query)
    system_prompt = system_prompt_override or _build_system_prompt(
        domain, context, language
    )
    model = settings.ollama_model_fr if language == "fr" else settings.ollama_model

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": query})

    logger.info(
        "Streaming Ollama model=%s domain=%s language=%s history_turns=%d",
        model, domain, language, len(history or []),
    )
    yield from _stream_ollama_chat(model, messages)
