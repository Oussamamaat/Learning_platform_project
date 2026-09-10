"""
Ollama LLM Service Client
─────────────────────────
Sends prompts to the local Ollama instance for text generation.
Supports multi-domain enterprise tutoring with Socratic methodology.
"""

import json
import logging
import threading
import time
import urllib.request
import urllib.error
from typing import Iterator, Optional
from app.config import get_settings
from app.errors import OllamaConnectionError, LLMConnectionError, GenerationError
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


# Web-search-fallback prompts -- app.services.web_search's opt-in answer to
# "the tenant corpus has nothing" besides deterministic_refusal(). Separate
# constants, not a variant of SYSTEM_PROMPT_TEMPLATE, on purpose:
#
# 1. SYSTEM_PROMPT_TEMPLATE / _FR are under test_generation_gates.py's
#    byte-identical train/serve parity assertion -- this must never touch
#    them.
# 2. The instructions themselves are different in kind, not just content.
#    SYSTEM_PROMPT_TEMPLATE tells the model to copy legal references
#    character-for-character because they DO appear verbatim in tenant
#    context; a web snippet has no such reference to copy, and instructing
#    the model to invent one would recreate exactly the citation-fabrication
#    failure mode this platform's fine-tune already has a documented history
#    of (see docs/architecture/rectified -- the untouched base model refuses
#    "not in the text" where the adapter fabricates a law number). So this
#    template explicitly forbids article/law-style citation and asks for
#    plain "according to <title>" attribution instead.
# 3. generate_web_fallback_response() below deliberately skips
#    extract_citations/inject_citations -- those pattern-match tenant legal
#    citation shapes (app/services/citations.py) that cannot appear in web
#    content, so running them here would either do nothing or, worse,
#    "inject" a phantom citation into prose that never claimed one.
#
# The disclaimer that this answer is NOT from the tenant's own documents is
# NOT left to the model to remember to say -- generate_web_fallback_response
# prepends it in code, the same "don't trust the model for a hard invariant"
# reasoning deterministic_refusal already applies to refusals.
WEB_FALLBACK_PROMPT_TEMPLATE_FR = (
    "Tu es un assistant qui repond a une question en te basant UNIQUEMENT sur "
    "les extraits de recherche web fournis ci-dessous -- PAS sur les documents "
    "internes du client, auxquels tu n'as pas acces ici. Ce n'est pas une "
    "question de {domain} couverte par la documentation interne.\n"
    "Reponds en francais, de maniere claire et concise, en te basant "
    "strictement sur les extraits fournis. N'invente jamais un fait absent "
    "des extraits.\n"
    "N'utilise JAMAIS de citation au format juridique (pas de \"Article X\", "
    "pas de numero de loi) -- ces extraits ne sont pas des textes "
    "reglementaires. Pour attribuer une information, nomme le VRAI nom du "
    "site ou du document indique entre crochets dans les extraits (par "
    "exemple \"selon Service-Public.fr\" si c'est le nom reel de la source) "
    "-- n'ecris jamais litteralement le mot \"titre\", c'est un exemple.\n"
    "Si les extraits ne repondent pas non plus a la question, dis-le "
    "clairement plutot que d'inventer une reponse.\n\n"
    "EXTRAITS DE RECHERCHE WEB :\n"
    "{context}"
)

WEB_FALLBACK_PROMPT_TEMPLATE_DARIJA = (
    "You are an assistant answering a question based ONLY on the web search "
    "snippets provided below -- NOT on the tenant's internal documents, which "
    "you do not have access to here. This is not a {domain} question covered "
    "by the internal documentation.\n"
    "Answer in Moroccan Darija, written in Arabic script. Keep technical "
    "vocabulary in French, Latin letters, exactly as a Moroccan professional "
    "says it, same as always. Base your answer strictly on the snippets "
    "provided. Never invent a fact absent from them.\n"
    "NEVER use a legal-citation format (no \"Article X\", no law number) -- "
    "these snippets are not regulatory text. To attribute information, name "
    "the REAL site or document name shown in brackets in the snippets (e.g. "
    "\"according to Le Monde\" if that is the actual source name) -- never "
    "write the literal word \"title\", it is only an example.\n"
    "If the snippets also don't answer the question, say so plainly instead "
    "of inventing an answer.\n\n"
    "WEB SEARCH SNIPPETS:\n"
    "{context}"
)

WEB_FALLBACK_DISCLAIMER_FR = (
    "[Reponse basee sur une recherche web -- pas sur vos documents internes.] "
)
WEB_FALLBACK_DISCLAIMER_DARIJA = (
    "[هاد الجواب مبني على بحث فالانترنت، ماشي على الوثائق ديالكم.] "
)


def _format_web_context(results) -> str:
    """`results`: list[app.services.web_search.WebResult]. Numbered so the
    model's "according to <title>" attribution has something concrete to
    reference; the URL is included for the disclaimer's benefit, not because
    the model is expected to reproduce it."""
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[{i}] {r.title} ({r.url})\n{r.snippet}")
    return "\n\n".join(blocks)


def generate_web_fallback_response(
    query: str,
    web_results,
    domain: str = "industrial",
    language: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> str:
    """Answer from live web-search results instead of tenant context --
    app.services.web_search's fallback for chat.py's refusal gate.

    `web_results`: list[app.services.web_search.WebResult], already fetched
    by the caller (this function makes no network call of its own).

    Deliberately NOT a codepath through generate_llm_response: no
    extract_citations/inject_citations (wrong shape for web content, see
    WEB_FALLBACK_PROMPT_TEMPLATE_FR's docstring above), and the disclaimer
    is prepended here in code rather than trusted to the model.
    """
    settings = get_settings()
    language = language or detect_query_language(query)
    web_context = _format_web_context(web_results)

    if language == "fr":
        domain_label = DOMAIN_LABELS_FR.get(domain, DOMAIN_LABELS.get(domain, domain))
        system_prompt = WEB_FALLBACK_PROMPT_TEMPLATE_FR.format(domain=domain_label, context=web_context)
        disclaimer = WEB_FALLBACK_DISCLAIMER_FR
    else:
        domain_label = DOMAIN_LABELS_AR.get(domain, DOMAIN_LABELS.get(domain, domain))
        system_prompt = WEB_FALLBACK_PROMPT_TEMPLATE_DARIJA.format(domain=domain_label, context=web_context)
        disclaimer = WEB_FALLBACK_DISCLAIMER_DARIJA

    # settings.web_search_fallback_model, NOT ollama_model_fr/ollama_model
    # -- see that setting's own comment for the live-reproduced false-
    # refusal defect that rules out the fine-tuned tutors here.
    model = settings.web_search_fallback_model
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": query})

    logger.info(
        "Calling Ollama (web fallback) model=%s domain=%s language=%s results=%d",
        model, domain, language, len(web_results),
    )
    result = _call_ollama_chat(model, messages)
    return disclaimer + result


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

# Bounds concurrent in-flight requests to Ollama -- see settings.
# ollama_max_concurrent's comment for why. threading.Semaphore, not
# asyncio.Semaphore: chat() and generate_quiz() (app/routers/chat.py,
# quiz.py) are plain `def`, dispatched to FastAPI's worker threadpool;
# voice's handler is `async def` but its own Ollama call runs inside
# _answer_worker via asyncio.to_thread (app/routers/voice.py). All four
# call surfaces (chat, quiz, diagrams, voice) end up on worker threads, so
# one shared threading primitive gates all of them uniformly. Built lazily
# (not at import time) so get_settings() -- itself lru_cache'd -- is read
# after any test/deployment override has taken effect, same pattern as
# app.services.ingestion._get_pool.
_ollama_semaphore: Optional[threading.Semaphore] = None
_ollama_semaphore_lock = threading.Lock()


def _get_ollama_semaphore() -> threading.Semaphore:
    global _ollama_semaphore
    if _ollama_semaphore is None:
        with _ollama_semaphore_lock:
            if _ollama_semaphore is None:
                _ollama_semaphore = threading.Semaphore(get_settings().ollama_max_concurrent)
    return _ollama_semaphore


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
    4. Bounded concurrency: acquires app.services.llm's shared
       threading.Semaphore (settings.ollama_max_concurrent) for the
       full duration of this call, including retries -- see that
       semaphore's own module-level comment.
    """
    _get_ollama_semaphore().acquire()
    try:
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
    finally:
        _get_ollama_semaphore().release()


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

    Also acquires app.services.llm's shared threading.Semaphore
    (settings.ollama_max_concurrent) for the WHOLE lifetime of the
    generator -- connect through final token -- released in a
    finally so early abandonment (the caller stops iterating, e.g.
    voice.py's cancel_flag) still frees the slot via GeneratorExit.
    """
    _get_ollama_semaphore().acquire()
    try:
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
    finally:
        _get_ollama_semaphore().release()


# -- vLLM transport -------------------------------------------------------
#
# /v1/completions with a RAW PROMPT (render_conversation()'s output), not
# /v1/chat/completions. This is the plan's single most important transport
# choice: render_conversation() above is already a byte-exact, test-locked
# (tests/test_prompt_format.py) port of the training notebook's template.
# The chat_template.jinja recovered alongside the production adapters
# (docs/architecture/model-artifacts.md) emits a literal {{ bos_token }} --
# sending that through vLLM's own chat templating would double-BOS exactly
# the way the training notebook warns against. Rendering here and sending
# the result as a raw prompt keeps this app owning the one template that
# must stay correct, instead of trusting a second, divergent one server-side.

# The two stop strings Ollama gets for free from each Modelfile's
# `PARAMETER stop` -- vLLM never sees the Modelfile, so these must be sent
# per request. See render_conversation(): every rendered turn is wrapped in
# exactly these two markers.
_VLLM_STOP = ["<end_of_turn>", "<start_of_turn>"]

# Mirrors _ollama_semaphore's module-level comment almost exactly, with one
# difference: settings.llm_max_concurrent is NOT a parallelism ceiling the
# way ollama_max_concurrent is. vLLM does its own continuous-batching
# admission control; this only bounds how many sockets this process opens
# to it at once (backpressure), so it can safely default far higher than
# the Ollama semaphore. A separate primitive from _ollama_semaphore so the
# two backends never contend for the same permits, including in a
# both-backends-configured test or a Step 6 side-by-side benchmark run.
_vllm_semaphore: Optional[threading.Semaphore] = None
_vllm_semaphore_lock = threading.Lock()


def _get_vllm_semaphore() -> threading.Semaphore:
    global _vllm_semaphore
    if _vllm_semaphore is None:
        with _vllm_semaphore_lock:
            if _vllm_semaphore is None:
                _vllm_semaphore = threading.Semaphore(get_settings().llm_max_concurrent)
    return _vllm_semaphore


def _post_vllm(payload: dict, *, timeout: Optional[int] = None) -> dict:
    """POST a JSON body to vLLM's /v1/completions and return the decoded
    response.

    Sibling to _post_ollama, same retry/error-mapping shape (transient
    502/503/504 retried with the same backoff, HTTPError mapped to
    GenerationError, a connection failure mapped to LLMConnectionError) so
    both backends fail the same way from a caller's point of view. Not
    built on _post_ollama itself: that function unconditionally injects
    Ollama's `keep_alive` field (vLLM has no analogue -- a served model is
    always resident) and acquires the Ollama semaphore, neither of which
    applies here.
    """
    _get_vllm_semaphore().acquire()
    try:
        settings = get_settings()
        url = f"{settings.llm_base_url.rstrip('/')}/v1/completions"
        data = json.dumps(payload).encode("utf-8")
        effective_timeout = timeout if timeout is not None else settings.ollama_timeout_seconds

        last_error: Optional[Exception] = None
        for attempt in range(len(_RETRY_DELAYS_SECONDS) + 1):
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}, method="POST",
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
                        "vLLM returned HTTP %s (retryable); retrying in %.1fs",
                        e.code, _RETRY_DELAYS_SECONDS[attempt],
                    )
                    last_error = e
                    time.sleep(_RETRY_DELAYS_SECONDS[attempt])
                    continue
                logger.error("vLLM returned HTTP %s for /v1/completions: %s", e.code, body)
                if e.code == 404:
                    raise GenerationError(
                        f"vLLM has no served model named {payload.get('model')!r} (HTTP 404). "
                        f"Check settings.llm_model_darija / llm_model_fr against "
                        f"--served-model-name at {settings.llm_base_url}."
                    ) from e
                raise GenerationError(f"vLLM HTTP {e.code}: {body}") from e
            except urllib.error.URLError as e:
                if attempt < len(_RETRY_DELAYS_SECONDS):
                    logger.warning(
                        "vLLM connection failed (%s); retrying in %.1fs",
                        e, _RETRY_DELAYS_SECONDS[attempt],
                    )
                    last_error = e
                    time.sleep(_RETRY_DELAYS_SECONDS[attempt])
                    continue
                logger.error("vLLM connection failed: %s", e)
                raise LLMConnectionError("vLLM", payload.get("model"), settings.llm_base_url) from e
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON from vLLM: %s", e)
                raise GenerationError(f"Invalid JSON response: {e}") from e
            except (LLMConnectionError, GenerationError):
                raise
            except Exception as e:
                logger.error("Unexpected vLLM error: %s", e)
                raise GenerationError(str(e)) from e

        raise LLMConnectionError("vLLM", payload.get("model"), settings.llm_base_url) from last_error
    finally:
        _get_vllm_semaphore().release()


def _call_vllm_generate(
    model: str,
    prompt: str,
    system: str,
    *,
    timeout: Optional[int] = None,
    guided_json: Optional[dict] = None,
) -> str:
    """vLLM sibling of _call_ollama_generate -- same (model, prompt, system)
    signature so quiz.py/diagrams.py's call sites are a one-line swap.
    `prompt`/`system` are folded into a [system, user] messages list and
    rendered through render_conversation() into the raw prompt vLLM
    receives; `guided_json` is vLLM's substitution for Ollama's `format`
    (ADR 0003 already anticipated this)."""
    text = render_conversation(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    )
    payload = {
        "model": model,
        "prompt": text,
        "temperature": 0.2,
        "max_tokens": get_settings().llm_max_tokens,
        "stop": _VLLM_STOP,
        "stream": False,
    }
    if guided_json is not None:
        payload["guided_json"] = guided_json

    res_json = _post_vllm(payload, timeout=timeout)
    choices = res_json.get("choices") or []
    result = (choices[0].get("text", "") if choices else "").strip()
    if not result:
        raise GenerationError("vLLM returned empty response")
    return result


def _call_vllm_chat(
    model: str,
    messages: list[dict],
    *,
    timeout: Optional[int] = None,
    guided_json: Optional[dict] = None,
) -> str:
    """vLLM sibling of _call_ollama_chat -- same (model, messages) signature.
    `messages` is rendered through render_conversation() into the raw
    prompt vLLM's /v1/completions receives, rather than sent to a vLLM
    /v1/chat/completions endpoint -- see the module header comment above."""
    text = render_conversation(messages)
    payload = {
        "model": model,
        "prompt": text,
        "temperature": 0.2,
        "max_tokens": get_settings().llm_max_tokens,
        "stop": _VLLM_STOP,
        "stream": False,
    }
    if guided_json is not None:
        payload["guided_json"] = guided_json

    res_json = _post_vllm(payload, timeout=timeout)
    choices = res_json.get("choices") or []
    result = (choices[0].get("text", "") if choices else "").strip()
    if not result:
        raise GenerationError("vLLM returned empty response")
    return result


def _stream_vllm_chat(
    model: str,
    messages: list[dict],
    *,
    timeout: Optional[int] = None,
) -> Iterator[str]:
    """Streaming vLLM sibling of _stream_ollama_chat. vLLM's /v1/completions
    with stream=true emits OpenAI-compatible SSE (`data: {...}\\n\\n`,
    terminated by `data: [DONE]`) -- NOT Ollama's NDJSON -- and the delta
    lives at choices[0].text, not message.content.

    Mirrors _stream_ollama_chat's semantics verbatim on purpose: no retry
    once bytes have reached the caller (replaying would duplicate content
    already spoken/displayed), a distinct GenerationError for a mid-stream
    drop after real content, and the semaphore permit released in `finally`
    so early abandonment (voice.py's cancel_flag) still frees the slot via
    GeneratorExit.
    """
    _get_vllm_semaphore().acquire()
    try:
        settings = get_settings()
        url = f"{settings.llm_base_url.rstrip('/')}/v1/completions"
        text = render_conversation(messages)
        payload = {
            "model": model,
            "prompt": text,
            "temperature": 0.2,
            "max_tokens": settings.llm_max_tokens,
            "stop": _VLLM_STOP,
            "stream": True,
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
            logger.error("vLLM returned HTTP %s for /v1/completions (stream): %s", e.code, body)
            if e.code == 404:
                raise GenerationError(
                    f"vLLM has no served model named {model!r} (HTTP 404). Check "
                    f"settings.llm_model_darija / llm_model_fr."
                ) from e
            raise GenerationError(f"vLLM HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            logger.error("vLLM connection failed (stream): %s", e)
            raise LLMConnectionError("vLLM", model, settings.llm_base_url) from e

        got_any = False
        try:
            with response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Skipping malformed SSE line from vLLM stream: %r", line[:200]
                        )
                        continue
                    choices = chunk.get("choices") or []
                    delta = choices[0].get("text", "") if choices else ""
                    if delta:
                        got_any = True
                        yield delta
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if got_any:
                raise GenerationError(f"vLLM stream dropped mid-response: {e}") from e
            raise LLMConnectionError("vLLM", model, settings.llm_base_url) from e

        if not got_any:
            raise GenerationError("vLLM returned an empty stream")
    finally:
        _get_vllm_semaphore().release()


# -- Backend-neutral dispatchers -------------------------------------------
#
# Everything above this point (Ollama transport + vLLM transport) is
# backend-specific. These three functions are the seam the rest of the app
# calls through: they read settings.llm_backend once and dispatch to the
# matching pair. Kept in this module rather than a new one, deliberately --
# the whole test suite patches app.services.llm.urllib.request.urlopen as
# its seam (14 files), and putting both transports here preserves that for
# free instead of requiring a second patch target.

def resolve_model_name(language: str) -> str:
    """Which served model name to request for `language` ('fr', or anything
    else treated as Darija) under the currently configured backend.

    Single source of truth for a ternary that was previously duplicated,
    identically, at four call sites (quiz.py, diagrams.py, and both of this
    module's own generate_llm_response/stream_llm_response) -- exactly the
    kind of duplication that silently drifts, the way the French quiz path
    once drifted and served every French quiz from the Darija model (see
    quiz.py's own comment on that incident) before it was fixed there.
    """
    settings = get_settings()
    if settings.llm_backend == "vllm":
        return settings.llm_model_fr if language == "fr" else settings.llm_model_darija
    return settings.ollama_model_fr if language == "fr" else settings.ollama_model


def llm_generate(
    model: str,
    prompt: str,
    system: str,
    *,
    timeout: Optional[int] = None,
    format_schema: Optional[dict] = None,
) -> str:
    """Backend-neutral sibling of _call_ollama_generate / _call_vllm_generate.
    quiz.py and diagrams.py call this instead of reaching for either
    backend's function directly."""
    if get_settings().llm_backend == "vllm":
        return _call_vllm_generate(model, prompt, system, timeout=timeout, guided_json=format_schema)
    return _call_ollama_generate(model, prompt, system, timeout=timeout, format_schema=format_schema)


def llm_chat(
    model: str,
    messages: list[dict],
    *,
    timeout: Optional[int] = None,
    format_schema: Optional[dict] = None,
) -> str:
    """Backend-neutral sibling of _call_ollama_chat / _call_vllm_chat."""
    if get_settings().llm_backend == "vllm":
        return _call_vllm_chat(model, messages, timeout=timeout, guided_json=format_schema)
    return _call_ollama_chat(model, messages, timeout=timeout, format_schema=format_schema)


def llm_stream_chat(
    model: str,
    messages: list[dict],
    *,
    timeout: Optional[int] = None,
) -> Iterator[str]:
    """Backend-neutral sibling of _stream_ollama_chat / _stream_vllm_chat."""
    settings = get_settings()
    if settings.llm_backend == "vllm":
        yield from _stream_vllm_chat(model, messages, timeout=timeout)
    else:
        yield from _stream_ollama_chat(model, messages, timeout=timeout)


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
    model = resolve_model_name(language)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": query})

    logger.info(
        "Calling LLM (%s) model=%s domain=%s language=%s history_turns=%d",
        settings.llm_backend, model, domain, language, len(history or []),
    )
    result = llm_chat(model, messages)

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
    model = resolve_model_name(language)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": query})

    logger.info(
        "Streaming LLM (%s) model=%s domain=%s language=%s history_turns=%d",
        settings.llm_backend, model, domain, language, len(history or []),
    )
    yield from llm_stream_chat(model, messages)
