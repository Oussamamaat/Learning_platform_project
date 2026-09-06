"""Optional web-search fallback for chat.py's refusal gate.

Tenant RAG can only answer what's in the tenant's own uploaded/curated
corpus. Today, when retrieval finds nothing (app.services.turn.TurnContext.
is_refusal / chat.py step 3's `domain_source == "no_match" or not
context.strip()` gate), the tutor refuses deterministically -- correct
default, since answering from nowhere would either hallucinate or silently
borrow the base model's untrusted world knowledge. This module lets an
operator opt a tenant INTO a live web search instead of a refusal, with the
result clearly kept separate from tenant-grounded content end to end (see
docs/architecture/rectified/adr/0010-web-search-fallback.md) -- never
merged into `ChatResponse.sources` (which the whole platform treats as "this
came from the tenant's own documents"), never run through
app.services.llm.generate_llm_response's citation-injection path (built for
article/law-number citations that appear verbatim in tenant context; a web
snippet has neither), and never fed to the fine-tuned tutor models (their
refusal register and citation habits are trained on tenant #1's regulatory
corpus specifically -- see llm.py's SYSTEM_PROMPT_TEMPLATE docstring on the
refusal-welding problem; there's no evidence they generalize safely to
arbitrary web content, and this module doesn't gamble on it).

Engine pattern mirrors app/services/{ocr,stt,tts}.py: an ABC, a NullEngine
default (chat behaviour is completely unchanged unless explicitly opted
in), and one real engine behind an API key setting. Uses stdlib
urllib.request, like the rest of the network calls in this codebase (see
app/services/llm.py's _post_ollama docstring for why) -- no new HTTP
dependency for one JSON POST.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str


class WebSearchEngine(ABC):
    name: str

    @abstractmethod
    def search(self, query: str, *, max_results: int) -> list[WebResult]:
        """Return up to `max_results` results, or [] on any failure.

        Fail-OPEN, same contract as app.services.history's DB calls and
        app.services.domain_context's disk fallback: an empty list means
        "no web fallback available this turn", which chat.py's caller then
        treats identically to the engine being "none" -- it falls through
        to the ORIGINAL deterministic refusal rather than raising into a
        500 on top of an already-unanswerable question.
        """
        raise NotImplementedError


class NullWebSearchEngine(WebSearchEngine):
    name = "none"

    def search(self, query: str, *, max_results: int) -> list[WebResult]:
        return []


class TavilyWebSearchEngine(WebSearchEngine):
    """https://tavily.com -- built for feeding LLM/RAG pipelines: results
    come back as short pre-extracted snippets, not raw HTML to scrape, so
    there's no parser to maintain here. A free tier exists (1000 req/mo as
    of this writing). Chosen over Google/Bing Custom Search (needs a
    separate search-engine-ID resource, not just an API key) and over
    scraping DuckDuckGo/Google result pages (fragile, and against their
    ToS -- this platform already treats fail-open/graceful-degradation as
    a hard requirement elsewhere; a scraper that silently breaks on a
    markup change is the opposite of that).
    """

    name = "tavily"
    _ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, api_key: str, *, timeout_seconds: float):
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def search(self, query: str, *, max_results: int) -> list[WebResult]:
        payload = json.dumps(
            {
                "api_key": self._api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self._ENDPOINT,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Most common cause here is an invalid/exhausted API key --
            # surfaced in the log (not raised) so an operator can find it,
            # but the chat turn still degrades to the ordinary refusal
            # rather than a 500.
            logger.warning(
                "Tavily search HTTP %s for query=%r: %s",
                e.code, query[:120], e.read(200) if hasattr(e, "read") else "",
            )
            return []
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            logger.exception("Tavily search failed for query=%r", query[:120])
            return []

        results = []
        for r in (body.get("results") or [])[:max_results]:
            url = r.get("url")
            if not url:
                continue
            results.append(
                WebResult(
                    title=(r.get("title") or url).strip(),
                    url=url,
                    snippet=(r.get("content") or "").strip(),
                )
            )
        return results


_ENGINES = {"none": NullWebSearchEngine, "tavily": TavilyWebSearchEngine}


@lru_cache(maxsize=1)
def get_web_search_engine() -> WebSearchEngine:
    """Cached like app.services.tts.get_tts_engine -- settings are
    @lru_cache'd per-process already, so this stays consistent with them
    for the process lifetime rather than re-branching on every call."""
    settings = get_settings()
    name = settings.web_search_engine
    if name == "tavily":
        if not settings.tavily_api_key:
            logger.warning(
                "web_search_engine='tavily' but tavily_api_key is unset -- "
                "falling back to 'none' (no web fallback this run)"
            )
            return NullWebSearchEngine()
        return TavilyWebSearchEngine(
            settings.tavily_api_key, timeout_seconds=settings.web_search_timeout_seconds
        )
    if name not in _ENGINES:
        raise ValueError(f"unknown web_search_engine {name!r}")
    return _ENGINES[name]()
