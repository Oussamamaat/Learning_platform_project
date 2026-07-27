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
from app.config import get_settings
from app.errors import OllamaConnectionError, OllamaTimeoutError, GenerationError

logger = logging.getLogger(__name__)

DOMAIN_LABELS = {
    "industrial": "industrial safety and workplace protocols",
    "securite": "physical security and surveillance procedures",
    "blockchain": "blockchain compliance and digital asset regulation",
}

SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert bilingual enterprise tutor specializing in {domain}.\n"
    "Guide the user in French and Moroccan Darija (Arabizi script) using a Socratic method.\n"
    "Use formal punctuation and capitalization.\n"
    "Ground all answers strictly in the provided context.\n"
    "If the context is insufficient, politely refuse and suggest what the user should study.\n"
    "Never invent facts. Only use information from the context below.\n\n"
    "CONTEXTE :\n"
    "{context}"
)


def _build_system_prompt(domain: str, context: str) -> str:
    """Build the system prompt with domain and context."""
    domain_label = DOMAIN_LABELS.get(domain, domain)
    return SYSTEM_PROMPT_TEMPLATE.format(domain=domain_label, context=context)


def generate_llm_response(
    query: str,
    context: str,
    domain: str = "industrial",
    system_prompt_override: str = None,
) -> str:
    """
    Query the local Ollama LLM with RAG context.

    Args:
        query: User's question
        context: Retrieved context chunks
        domain: Domain label (industrial, securite, blockchain)
        system_prompt_override: Optional custom system prompt

    Returns:
        Generated text from LLM
    """
    settings = get_settings()

    system_prompt = system_prompt_override or _build_system_prompt(domain, context)

    url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
    payload = {
        "model": settings.ollama_model,
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
        logger.info("Calling Ollama model=%s domain=%s", settings.ollama_model, domain)
        with urllib.request.urlopen(req, timeout=60) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            result = res_json.get("response", "").strip()
            if not result:
                raise GenerationError("Ollama returned empty response")
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
