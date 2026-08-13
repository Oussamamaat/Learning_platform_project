"""
Ollama LLM Service Client
─────────────────────────
Sends prompts to the local Ollama instance for text generation.
Supports multi-domain enterprise tutoring with Socratic methodology.
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Optional
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


def _call_ollama_generate(
    model: str,
    prompt: str,
    system: str,
    *,
    timeout: int = 180,
    format_schema: Optional[dict] = None,
) -> str:
    """POST to Ollama's /api/generate and return the raw `response` string.

    Shared by chat, quiz, and demo serving paths so the request-building and
    URLError/JSONDecodeError-to-AppError mapping lives in one place instead
    of being copy-pasted per caller.

    Raises OllamaConnectionError on network failure, GenerationError on an
    empty or invalid response.
    """
    settings = get_settings()
    url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    if format_schema is not None:
        payload["format"] = format_schema

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            result = res_json.get("response", "").strip()
            if not result:
                raise GenerationError("Ollama returned empty response")
            return result
    except urllib.error.URLError as e:
        logger.error("Ollama connection failed: %s", e)
        raise OllamaConnectionError(model, settings.ollama_base_url) from e
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON from Ollama: %s", e)
        raise GenerationError(f"Invalid JSON response: {e}") from e
    except (OllamaConnectionError, GenerationError):
        raise
    except Exception as e:
        logger.error("Unexpected LLM error: %s", e)
        raise GenerationError(str(e)) from e


def _call_ollama_chat(
    model: str,
    messages: list[dict],
    *,
    timeout: int = 180,
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
    settings = get_settings()
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    if format_schema is not None:
        payload["format"] = format_schema

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            result = res_json.get("message", {}).get("content", "").strip()
            if not result:
                raise GenerationError("Ollama returned empty response")
            return result
    except urllib.error.URLError as e:
        logger.error("Ollama connection failed: %s", e)
        raise OllamaConnectionError(model, settings.ollama_base_url) from e
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON from Ollama: %s", e)
        raise GenerationError(f"Invalid JSON response: {e}") from e
    except (OllamaConnectionError, GenerationError):
        raise
    except Exception as e:
        logger.error("Unexpected LLM error: %s", e)
        raise GenerationError(str(e)) from e


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
