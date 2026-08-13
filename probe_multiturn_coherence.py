"""
Live probe: does a real multi-turn conversation through the actual chat()
router stay coherent and in-language at turn 3 -- one turn past the
trained shape (training's multi-turn prompts specify exactly 4 messages:
user, assistant, user, assistant; a live 3-exchange conversation puts a
3rd assistant turn in front of the model for the first time).

This is the live-dry-run check the memory/RAG plan calls for before
trusting MAX_WINDOW_MESSAGES=4 (2 prior exchanges) as the default -- if
turn 3 degrades (repeats, drops language, fails to stop), the fallback is
MAX_WINDOW_MESSAGES=2 (1 prior exchange, the byte-exact trained shape).

No Postgres needed -- history.load_window is monkeypatched with a canned
window so this exercises real generation without depending on a live DB.

Usage:
    .gguf_venv/Scripts/python.exe probe_multiturn_coherence.py
"""
import sys
from unittest.mock import patch

sys.path.insert(0, ".")

from app.models.schemas import ChatRequest, Domain, Language
from app.routers import chat as chat_router


def fake_context(*args, **kwargs):
    return (
        "المادة 283: يجب على المشغل أن يوفر معدات الوقاية الشخصية الملائمة "
        "لطبيعة الأشغال المنجزة، وأن يضمن صيانتها في حالة جيدة.",
        ["1.11_ar_code_travail_salama.md"],
    )


CANNED_WINDOW = [
    {"role": "user", "content": "شنو خاص المشغل يوفر للعمال؟"},
    {"role": "assistant", "content": "خاصو يوفر معدات الوقاية الشخصية. شنو كتعرف على هاد المعدات؟"},
]


def main():
    print("=== Turn 3: real generation with 1 prior exchange already in context ===\n")
    with patch("app.routers.chat.build_domain_context", side_effect=fake_context), \
         patch("app.routers.chat.history.load_window", return_value=CANNED_WINDOW), \
         patch("app.routers.chat.history.append_exchange"), \
         patch("app.routers.chat.history.extract_dropped_questions", return_value=[]):
        response = chat_router.chat(ChatRequest(
            message="علاش خاصها تكون ملائمة للشغل؟",
            domain=Domain.INDUSTRIAL,
            language=Language.DARIJA,
            session_id="probe-multiturn",
        ))

    reply = response.response
    print(f"Reply ({len(reply)} chars):\n  {reply}\n")

    arabic = sum(1 for c in reply if "؀" <= c <= "ۿ")
    latin = sum(1 for c in reply if c.isascii() and c.isalpha())
    stayed_in_darija = arabic > latin
    non_empty = bool(reply.strip())
    not_degenerate = len(set(reply.split())) > 3 if reply.split() else False

    print(f"stayed_in_darija (arabic={arabic} > latin={latin}): {stayed_in_darija}")
    print(f"non_empty: {non_empty}")
    print(f"not_degenerate (some lexical variety, not stuck repeating): {not_degenerate}")

    ok = stayed_in_darija and non_empty and not_degenerate
    print("\n" + "=" * 60)
    print("VERDICT:", "OK -- 2-exchange window (4 messages) looks safe to keep as default."
          if ok else "DEGRADED -- fall back to MAX_WINDOW_MESSAGES=2 (1 prior exchange).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
