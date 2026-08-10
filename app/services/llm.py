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
from app.errors import OllamaConnectionError, OllamaTimeoutError, GenerationError
from app.services.citations import (
    extract_citations,
    inject_citations,
    detect_target_script,
)

logger = logging.getLogger(__name__)

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

# Function words that are French and not shared with English, so an English
# question does not get misrouted here.
_FRENCH_MARKERS = (
    "que", "qui", "quoi", "quel", "quelle", "quelles", "quels", "est",
    "sont", "les", "des", "une", "dans", "pour", "avec", "sur", "comment",
    "pourquoi", "quand", "doit", "peut", "vous", "je", "nous", "bonjour",
    "merci", "s'il", "ce", "cette", "ces", "du", "au", "aux", "et", "ou",
    "un", "le", "la", "de", "en", "par", "plus", "mais", "donc", "aussi",
    "tout", "tous", "toute", "toutes", "faire", "etre", "avoir", "selon",
    "entre", "apres", "avant", "sans", "sous", "leur", "leurs", "notre",
    "votre", "mon", "ma", "mes", "son", "sa", "ses", "il", "elle", "ils",
    "elles", "on", "peuvent", "expliquer", "quelles",
    "moi", "toi", "lui", "leur", "y", "en", "veux", "veuillez", "svp",
)

# Accented characters French uses and Arabizi does not. One is enough on its
# own — nobody hits e-acute by accident writing Darija in Latin letters.
_FRENCH_ACCENTS = "éèêëàâäùûüôöîïçœ"


def detect_query_language(query: str) -> str:
    """Which language the answer should come back in: 'fr' or 'darija'.

    Arabic script is unambiguous. Latin script is not — it is either French or
    Arabizi, and those want opposite answers, so Arabizi markers win over
    French ones. Anything undecidable falls back to 'darija', which is the
    trained behaviour and therefore the safe default.
    """
    if not query:
        return "darija"

    arabic = sum(1 for c in query if "؀" <= c <= "ۿ")
    latin = sum(1 for c in query if c.isascii() and c.isalpha())
    if arabic > latin:
        return "darija"

    # Split on hyphens too: French imperative-with-pronoun forms like
    # "explique-moi" or "dis-moi" hide their pronoun marker inside a single
    # whitespace-delimited token otherwise (found 2026-08-02, the §4.2 live
    # demo script routed "Explique-moi la procedure LOTO." to Darija because
    # "moi" was never split out of "explique-moi" to be checked).
    raw_words = query.replace("-", " ").split()
    words = {w.strip(".,!?;:()\"'").lower() for w in raw_words}
    if words & set(_ARABIZI_MARKERS):
        return "darija"
    if any(c in _FRENCH_ACCENTS for c in query.lower()):
        return "fr"
    if len(words & set(_FRENCH_MARKERS)) >= 2:
        return "fr"
    return "darija"


def _build_system_prompt(domain: str, context: str, language: str = "darija") -> str:
    """Build the system prompt with domain and context, in `language`."""
    if language == "fr":
        domain_label = DOMAIN_LABELS_FR.get(domain, DOMAIN_LABELS.get(domain, domain))
        return SYSTEM_PROMPT_TEMPLATE_FR.format(
            domain=domain_label, context=context
        )
    domain_label = DOMAIN_LABELS.get(domain, domain)
    return SYSTEM_PROMPT_TEMPLATE.format(domain=domain_label, context=context)


def generate_llm_response(
    query: str,
    context: str,
    domain: str = "industrial",
    system_prompt_override: str = None,
    language: Optional[str] = None,
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

    Returns:
        Generated text from LLM
    """
    settings = get_settings()

    language = language or detect_query_language(query)
    system_prompt = system_prompt_override or _build_system_prompt(
        domain, context, language
    )

    model = settings.ollama_model_fr if language == "fr" else settings.ollama_model
    url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": query,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        logger.info("Calling Ollama model=%s domain=%s language=%s", model, domain, language)
        with urllib.request.urlopen(req, timeout=180) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            result = res_json.get("response", "").strip()
            if not result:
                raise GenerationError("Ollama returned empty response")
            # Citations are derived from the retrieved context, not trusted
            # from the model — see app/services/citations.py for why. Only
            # references that genuinely appear in the context are rewritten,
            # so this can never manufacture the appearance of grounding.
            citations = extract_citations(context)
            if citations:
                # detect_target_script only distinguishes Arabic from Arabizi.
                # A French answer is Latin-script but wants "Article 18", not
                # the Arabizi gloss "المادة 18 (l-madda 18)".
                target_script = (
                    "french" if language == "fr" else detect_target_script(result)
                )
                result = inject_citations(result, citations, target_script)
            return result
    except urllib.error.URLError as e:
        logger.error("Ollama connection failed: %s", e)
        raise OllamaConnectionError(settings.ollama_model, settings.ollama_base_url) from e
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON from Ollama: %s", e)
        raise GenerationError(f"Invalid JSON response: {e}") from e
    except (OllamaConnectionError, GenerationError):
        raise
    except Exception as e:
        logger.error("Unexpected LLM error: %s", e)
        raise GenerationError(str(e)) from e
