"""
Live multi-turn tutoring evaluation, company_efg, both documents, both
languages.

Drives the LIVE HTTP chat API (POST /api/v1/chat/), not an in-process
call -- this is what an actual user session goes through, including
routing, retrieval, and history persistence, none of which
probe_language_routing.py's in-process chat_router.chat() call exercises.

WHY this exists (see conversation history / docs/architecture for the full
incident): an 80-page benchmark PDF scored the fine-tuned models as
non-Socratic and hallucinating, but retrieval was silently handing them the
WRONG document (12 of 80 pages had never been ingested, and a separate
routing defect served french_test.pdf's corpus for arabic_test.pdf
questions). That benchmark answered nothing about tutoring quality because
the models were never actually shown the right content. This script re-runs
the same kind of question, AFTER the ingestion pipeline fix (see
app/services/pdf_classify.py, app/services/ingestion.py) and under a fresh
tenant (company_efg) holding ONLY these two documents -- so a wrong-source
answer here is a real defect, not cross-tenant contamination.

Prerequisites (see this run's own printed preflight):
  - Backend running as company_efg: .\\start_backend.ps1 -TenantId company_efg
  - arabic_test.pdf and french_test.pdf uploaded to that tenant and
    status in ('ready', 'partial')

Four tracks -- (document) x (response language) -- because both documents
are genuinely useful for this: arabic_test.pdf is Arabic-script source
content, so asking it in French exercises the trained cross-lingual
translate-and-explain path (SYSTEM_PROMPT_TEMPLATE_FR's "traduis-le et
explique en francais" instruction); french_test.pdf is French source
content, so asking it in Darija exercises the same cross-lingual path in
the other direction. Same-language tracks (arabic_test.pdf/darija,
french_test.pdf/french) are the baseline case with no translation
involved.

Each track is a 3-turn conversation, same session_id throughout (server-
side history keys off it -- app/services/history.py):
  Turn 1: a real content question, answerable from the document.
  Turn 2: the "learner" replies to the tutor's own check-question from
          turn 1 with a DELIBERATELY PARTLY WRONG answer -- this is the
          one thing a single-turn eval structurally cannot test: does the
          tutor correct a wrong answer, or just agree with it.
  Turn 3: pushes for the next related point, to see if a third turn stays
          coherent and still grounded (MAX_WINDOW_MESSAGES=4 means only
          the prior exchange is replayed into the prompt at this point --
          see app/services/history.py).

Scoring is against the contract build_socratic_prompt (used to GENERATE
the training data these models were fine-tuned on --
app/services/generate_training_data.py:604) actually specifies:
  - explains before asking (a turn that is ONLY a question is invalid
    per that function's own rules)
  - ends with exactly one check-understanding question
  - turn 2 reacts to what the learner said (references it, doesn't ignore
    it) and adds ONE NEW point rather than repeating turn 1
  - stays in the requested language throughout
  - grounded: response text overlaps with known real content from the
    source document (a deterministic proxy for "did it actually use the
    right corpus", NOT a claim of full grounding verification)

These are deterministic PROXIES, not a full semantic judge -- flagged
per-check below. The full transcript is saved verbatim specifically so a
human (or a separate LLM judge pass) can read what the heuristics can't
score, e.g. "did it correct the wrong answer, or paper over it."

Writes probe_tutoring_eval_results.json (structured) and
probe_tutoring_eval_transcript.txt (human-readable) next to this script.
Never asserts / exits nonzero -- this is a report generator, not a gate.
"""
import json
import re
import time
import uuid

import requests

API_BASE = "http://127.0.0.1:8000"
CHAT_URL = f"{API_BASE}/api/v1/chat/"

RESULTS_JSON = "probe_tutoring_eval_results.json"
TRANSCRIPT_TXT = "probe_tutoring_eval_transcript.txt"


# ---------------------------------------------------------------------------
# Tracks: (label, response_language, [turn1, turn2_wrong_answer, turn3])
# turn2 is what the SIMULATED LEARNER says in reply to turn1's check-
# question -- written to be plausible but factually off, per document.
# ---------------------------------------------------------------------------

# ORDERED BY LANGUAGE, NOT BY DOCUMENT, deliberately: the French and
# Darija tutors are two separate ~5.7GB GGUF models
# (settings.ollama_model_fr / .ollama_model) and only one fits in this
# laptop's 8GB VRAM at a time, so every language switch costs a full
# model unload+reload from Ollama. Interleaving documents within a
# language (fr, fr, darija, darija) costs ONE swap; alternating by
# document (fr, darija, fr, darija) costs THREE, which is what made the
# first attempt at this eval exceed its runner timeout before finishing a
# single track.
TRACKS = [
    {
        "label": "arabic_test.pdf / French (cross-lingual)",
        "language": "fr",
        "document_hint": "arabic_test.pdf",
        # Ground truth (measured earlier by direct SQL/OCR against this
        # corpus): built-up area <=30%, beach/greenery >=50%, setback from
        # reclamation line >=40m.
        "turns": [
            "Quelle est la surface maximale constructible sur le front de "
            "mer, et quelle distance de retrait faut-il respecter par "
            "rapport a la ligne de remblai ?",
            "Donc si je comprends bien, la limite est de 50% de la "
            "surface totale qui peut etre construite ?",
            "Et concernant la hauteur maximale des batiments dans cette "
            "zone, quelle est la regle ?",
        ],
    },
    {
        "label": "french_test.pdf / French",
        "language": "fr",
        "document_hint": "french_test.pdf",
        # Ground truth (p13): replacing a safety component with one of
        # DIFFERENT performance/type IS a modification. Replacing with an
        # identical one is not what the text describes as a modification.
        "turns": [
            "Selon ce guide, dans quel cas le remplacement d'un composant "
            "de securite est-il considere comme une modification de la "
            "machine ?",
            "Donc remplacer n'importe quelle piece, meme a l'identique, "
            "compte toujours comme une modification ?",
            "Et le simple deplacement d'un composant de securite, sans le "
            "remplacer, est-ce aussi considere comme une modification ?",
        ],
    },
    # --- language switch happens here, exactly once ---
    {
        "label": "arabic_test.pdf / Darija",
        "language": "ar-MA",
        "document_hint": "arabic_test.pdf",
        # Ground truth: primary substation required above 12,000 KVA;
        # telecom tower plot is 10m x 10m.
        "turns": [
            "شحال كيبان الحمل الكهربائي اللي كيوجب توفير محطة رئيسية؟",
            "يعني إلا وصل الحمل ديال البناية ل 10000 كيلوفولط أمبير خاصنا "
            "نديرو محطة رئيسية؟",
            "وشحال كتكون المساحة اللي خاصها برج الاتصالات؟",
        ],
    },
    {
        "label": "french_test.pdf / Darija (cross-lingual)",
        "language": "ar-MA",
        "document_hint": "french_test.pdf",
        "turns": [
            "شنو كيقول هاد الدليل على تبديل مكون السلامة ديال الآلة، "
            "فأي حالة كيتعتبر تعديل؟",
            "يعني كيفما بدلتي شي جزء، حتى إلا كان نفسو بالضبط، كيتعتبر "
            "دايما تعديل؟",
            "وإلا غير بدلتي بلاصة المكون ديال السلامة من غير ما تبدلوه، "
            "واش هادشي كيتعتبر تعديل؟",
        ],
    },
]


# ---------------------------------------------------------------------------
# Deterministic scoring proxies
# ---------------------------------------------------------------------------

_ARABIC_RANGE = ("؀", "ۿ")


def _script_langs(text: str) -> tuple[int, int]:
    arabic = sum(1 for c in text if _ARABIC_RANGE[0] <= c <= _ARABIC_RANGE[1])
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    return arabic, latin


def _question_count(text: str) -> int:
    # Counts '?' and the Arabic question mark '؟'
    return text.count("?") + text.count("؟")


def _explains_before_asking(text: str) -> bool:
    """Proxy for build_socratic_prompt's 'EVERY assistant turn MUST first
    explain ... before asking anything' rule: the response must have
    substantive content (>40 chars) before its first question mark, not
    just open with a question."""
    idx = min(
        (i for i in (text.find("?"), text.find("؟")) if i != -1),
        default=len(text),
    )
    return idx > 40


def _ends_with_question(text: str) -> bool:
    tail = text.strip()[-3:]
    return "?" in tail or "؟" in tail


def score_turn(response_text: str, expected_lang: str) -> dict:
    arabic_chars, latin_chars = _script_langs(response_text)
    if expected_lang == "ar-MA":
        language_ok = arabic_chars > latin_chars
    else:
        language_ok = latin_chars >= arabic_chars

    q_count = _question_count(response_text)
    return {
        "language_ok": language_ok,
        "arabic_chars": arabic_chars,
        "latin_chars": latin_chars,
        "explains_before_asking": _explains_before_asking(response_text),
        "question_count": q_count,
        "ends_with_exactly_one_question": q_count == 1 and _ends_with_question(response_text),
        "char_count": len(response_text),
        "nonempty": bool(response_text.strip()),
    }


def score_turn2_reaction(turn1_response: str, turn2_learner_msg: str, turn2_response: str) -> dict:
    """Weak lexical-overlap proxies only -- NOT a semantic judge. Read the
    transcript for the real answer to 'did it correct the wrong answer.'
    """

    def _content_words(s: str) -> set:
        return {w for w in re.findall(r"[\w؀-ۿ]{4,}", s.lower())}

    t1_words = _content_words(turn1_response)
    t2_words = _content_words(turn2_response)
    new_words = t2_words - t1_words

    return {
        "turn2_word_count": len(t2_words),
        "new_content_word_ratio": round(len(new_words) / len(t2_words), 2) if t2_words else 0.0,
        "note": "Lexical-overlap proxy only. Read the transcript to judge whether "
                "the tutor actually corrected the learner's wrong answer in turn 2, "
                "vs. silently agreeing with it -- no automated check here does that.",
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_track(track: dict) -> dict:
    session_id = f"eval-{uuid.uuid4().hex[:12]}"
    turns_out = []
    prev_response_text = None

    for i, message in enumerate(track["turns"], start=1):
        payload = {"message": message, "session_id": session_id, "language": track["language"]}
        t0 = time.time()
        try:
            resp = requests.post(CHAT_URL, json=payload, timeout=180)
            elapsed = time.time() - t0
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            turns_out.append({"turn": i, "message": message, "error": f"{type(e).__name__}: {e}"})
            break

        response_text = data.get("response", "")
        scores = score_turn(response_text, track["language"])
        if i == 2 and prev_response_text is not None:
            scores["reaction"] = score_turn2_reaction(prev_response_text, message, response_text)

        turns_out.append({
            "turn": i,
            "learner_message": message,
            "response": response_text,
            "sources": data.get("sources", []),
            "domain": data.get("domain"),
            "domain_source": data.get("domain_source"),
            "language_reported": data.get("language"),
            "cross_language": data.get("cross_language"),
            "degraded": data.get("degraded"),
            "elapsed_seconds": round(elapsed, 1),
            "scores": scores,
        })
        prev_response_text = response_text

    return {
        "label": track["label"],
        "document_hint": track["document_hint"],
        "requested_language": track["language"],
        "session_id": session_id,
        "turns": turns_out,
    }


def write_transcript(all_results: list) -> None:
    lines = []
    for track_result in all_results:
        lines.append("=" * 78)
        lines.append(f"TRACK: {track_result['label']}  (session={track_result['session_id']})")
        lines.append("=" * 78)
        for t in track_result["turns"]:
            if "error" in t:
                lines.append(f"\n--- Turn {t['turn']}: ERROR ---")
                lines.append(t["error"])
                continue
            lines.append(f"\n--- Turn {t['turn']} ({t['elapsed_seconds']}s) ---")
            lines.append(f"LEARNER: {t['learner_message']}")
            lines.append(f"TUTOR:   {t['response']}")
            lines.append(
                f"[sources={t['sources']} domain={t['domain']}/{t['domain_source']} "
                f"lang={t['language_reported']} cross_lang={t['cross_language']} "
                f"degraded={t['degraded']}]"
            )
            s = t["scores"]
            lines.append(
                f"[score: lang_ok={s['language_ok']} explains_first={s['explains_before_asking']} "
                f"q_count={s['question_count']} ends_with_1_q={s['ends_with_exactly_one_question']}]"
            )
            if "reaction" in s:
                lines.append(f"[reaction proxy: {s['reaction']}]")
        lines.append("")
    with open(TRANSCRIPT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    print(f"Target: {CHAT_URL}")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=5).json()
        print(f"Backend health: {health}")
    except Exception as e:
        print(f"WARNING: backend not reachable at {API_BASE}: {e}")
        print("This must be run against a backend started as company_efg -- "
              "see this script's module docstring.")
        return

    all_results = []
    for track in TRACKS:
        print(f"\n>>> {track['label']} ...", flush=True)
        result = run_track(track)
        all_results.append(result)
        for t in result["turns"]:
            if "error" in t:
                print(f"    turn {t['turn']}: ERROR {t['error']}", flush=True)
            else:
                s = t["scores"]
                print(
                    f"    turn {t['turn']}: {t['elapsed_seconds']}s  "
                    f"sources={t['sources']}  lang_ok={s['language_ok']}  "
                    f"q_count={s['question_count']}",
                    flush=True,
                )
        # Persist after EVERY track, not once at the end: a 12-turn run
        # against two ~5.7GB models on an 8GB card is slow enough that an
        # outer timeout is a real possibility, and losing three completed
        # tracks because the fourth was still running would mean paying
        # the whole cost again for nothing.
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        write_transcript(all_results)

    print(f"\nWrote {RESULTS_JSON} and {TRANSCRIPT_TXT}")


if __name__ == "__main__":
    main()
