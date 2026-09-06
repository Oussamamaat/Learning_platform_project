"""
Live web-search-fallback self-test (ADR 0010's own "Open, for later" item:
"no live Tavily key has been exercised" -- TavilyWebSearchEngine's HTTP
shape is verified against Tavily's documented response format and
unit-tested with mocked urlopen, never against the real API).

Runs the ACTUAL two-step path chat.py's refusal gate uses when
WEB_SEARCH_ENGINE=tavily: a real Tavily search
(app.services.web_search.get_web_search_engine().search(...)), then a real
generation through settings.web_search_fallback_model ("gemma2:9b", NOT
either fine-tuned tutor -- see that setting's own comment in app/config.py
for the live-reproduced false-refusal defect that rules the tutors out
here) via app.services.llm.generate_web_fallback_response.

Also re-checks the second defect ADR 0010 records catching only by hand: a
literal "{title}"/"{titre}" placeholder echoed verbatim into the answer
instead of a real attribution.

Requires (from repo root):
  - .env: TAVILY_API_KEY set to a real key, WEB_SEARCH_ENGINE=tavily
    (only needed for get_web_search_engine() to pick Tavily; this script
    does not require the app's own env var name to already be flipped on
    a lease -- it is a standalone check).
  - Ollama running locally with gemma2:9b pulled (`ollama pull gemma2:9b`,
    ~5.4GB -- already on this laptop per ADR 0010's own live test).
  - No GPU contention requirement: gemma2:9b is a normal Ollama model like
    any tutor; run this with XTTS/whisper NOT resident if VRAM is tight.

Run:
    .gguf_venv/Scripts/python.exe scripts/web_search_selftest.py

Never prints the API key.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# One question per language, deliberately genuinely out-of-corpus (nothing
# to do with industrial/safety regulations) so the refusal gate's own
# no_match/empty-context condition would fire in the real app -- this
# script calls the fallback directly rather than going through chat.py, so
# it does not re-derive that condition, but the questions are chosen to be
# representative of what actually reaches this path in production.
TEST_CASES = [
    {"language": "fr", "query": "Comment faire du pain au levain ?"},
    {"language": "darija", "query": "شحال الجو اليوم فالرباط؟"},
]

_PLACEHOLDER_MARKERS = ("{title}", "{titre}")


def main() -> None:
    from app.config import get_settings
    from app.services.llm import generate_web_fallback_response
    from app.services.web_search import TavilyWebSearchEngine, get_web_search_engine

    settings = get_settings()
    print(f"web_search_engine={settings.web_search_engine!r} "
          f"web_search_fallback_model={settings.web_search_fallback_model!r}\n")

    engine = get_web_search_engine()
    if not isinstance(engine, TavilyWebSearchEngine):
        print(
            f"get_web_search_engine() returned {type(engine).__name__}, not "
            f"TavilyWebSearchEngine -- set WEB_SEARCH_ENGINE=tavily and "
            f"TAVILY_API_KEY in .env before running this."
        )
        sys.exit(1)

    failures = []
    for case in TEST_CASES:
        language, query = case["language"], case["query"]
        print(f"--- {language}: {query}")

        results = engine.search(query, max_results=settings.web_search_max_results)
        if not results:
            print("  Tavily returned NO results -- either a bad/exhausted key, a network "
                  "problem, or a genuinely unanswerable query. Cannot proceed for this case.")
            failures.append((language, "no web results"))
            print()
            continue
        print(f"  Tavily returned {len(results)} result(s):")
        for r in results:
            print(f"    - {r.title!r} <{r.url}>")

        answer = generate_web_fallback_response(query, results, language=language)
        print(f"  answer:\n    {answer}")

        found_placeholder = [m for m in _PLACEHOLDER_MARKERS if m in answer]
        if found_placeholder:
            print(f"  DEFECT: literal placeholder {found_placeholder} echoed into the answer "
                  f"(the exact bug ADR 0010 caught by hand).")
            failures.append((language, f"placeholder echo {found_placeholder}"))
        else:
            print("  OK: no placeholder echo")
        print()

    print("=" * 60)
    if failures:
        print(f"{len(failures)}/{len(TEST_CASES)} case(s) FAILED:")
        for language, reason in failures:
            print(f"  - {language}: {reason}")
        sys.exit(1)
    print(f"All {len(TEST_CASES)} case(s) passed -- live Tavily + gemma2:9b fallback verified.")


if __name__ == "__main__":
    main()
