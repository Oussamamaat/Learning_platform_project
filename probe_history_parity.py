"""
Live probe: does Ollama's own `/api/chat` templating reproduce the exact
trained ChatML shape, or do we need to hand-render and send `raw: true`?

This is the Stage 1a blocking gate (see the memory/RAG plan): everything
downstream (server-side conversation history) depends on the answer. For 3
fixed conversations, send the SAME logical prompt two ways and compare
Ollama's own `prompt_eval_count` (the number of tokens it counted as
prompt/prefill):

  (a) POST /api/chat  with a `messages` array -- Ollama applies the
      Modelfile's own TEMPLATE.
  (b) POST /api/generate with "raw": true and a hand-rendered string from
      app.services.llm.render_conversation() (the training notebook's
      render(), ported) -- bypasses Ollama's templating entirely.

If prompt_eval_count matches for every case, (a) is proven equivalent to
the trained format and /api/chat is the transport to build history on. If
it does not match, ship the render_conversation() + raw:true path instead
-- same code, one flag, per the plan's own decision rule.

Usage:
    .gguf_venv/Scripts/python.exe probe_history_parity.py

Requires: Ollama running with IBLOG_TUTOR:latest and iblog-tutor-fr:latest.
"""

import json
import sys
import urllib.request

sys.path.insert(0, ".")

from app.services.llm import render_conversation
from app.config import get_settings

OLLAMA_URL = get_settings().ollama_base_url.rstrip("/")

# Three fixed conversations: single-turn, the trained 4-message shape
# (system, user, assistant, user), and one turn past the trained shape
# (3 assistant turns) -- exactly the case Stage 1e's window-size fallback
# rule depends on. Content is realistic corpus-grounded phrasing, not toy
# sentences (see the project's "test with real phrasing" lesson).
DARIJA_SYSTEM = (
    "You are an expert bilingual enterprise tutor specializing in "
    "industrial safety and workplace protocols.\n"
    "Answer in Moroccan Darija written in Arabic script, using a Socratic method.\n"
    "Ground all answers strictly in the provided context.\n\n"
    "CONTEXTE :\n"
    "المادة 283: يجب على المشغل أن يوفر معدات الوقاية الشخصية الملائمة لطبيعة "
    "الأشغال المنجزة، وأن يضمن صيانتها في حالة جيدة."
)

FRENCH_SYSTEM = (
    "Tu es un tuteur d'entreprise expert, specialise en securite industrielle et "
    "protocoles de travail.\n"
    "Reponds en francais, avec une methode socratique.\n"
    "Fonde toutes tes reponses strictement sur le contexte fourni.\n\n"
    "CONTEXTE :\n"
    "Article 283 : L'employeur doit fournir des equipements de protection "
    "individuelle adaptes a la nature des travaux effectues, et garantir leur "
    "maintenance en bon etat."
)

CONVERSATIONS = [
    {
        "name": "trained-shape 4-message (french)",
        "model": "iblog-tutor-fr:latest",
        "messages": [
            {"role": "system", "content": FRENCH_SYSTEM},
            {"role": "user", "content": "Que doit fournir l'employeur aux travailleurs ?"},
            {"role": "assistant", "content": "Il doit fournir des equipements de protection individuelle. Sais-tu a quoi ils servent ?"},
            {"role": "user", "content": "Pourquoi doivent-ils etre adaptes au travail effectue ?"},
        ],
    },
    {
        "name": "single-turn (darija)",
        "model": "IBLOG_TUTOR:latest",
        "messages": [
            {"role": "system", "content": DARIJA_SYSTEM},
            {"role": "user", "content": "شنو خاص المشغل يوفر للعمال؟"},
        ],
    },
    {
        "name": "trained-shape 4-message (darija)",
        "model": "IBLOG_TUTOR:latest",
        "messages": [
            {"role": "system", "content": DARIJA_SYSTEM},
            {"role": "user", "content": "شنو خاص المشغل يوفر للعمال؟"},
            {"role": "assistant", "content": "خاصو يوفر معدات الوقاية الشخصية. شنو كتعرف على هاد المعدات؟"},
            {"role": "user", "content": "علاش خاصها تكون ملائمة للشغل؟"},
        ],
    },
    {
        "name": "one-past-trained 6-message (darija)",
        "model": "IBLOG_TUTOR:latest",
        "messages": [
            {"role": "system", "content": DARIJA_SYSTEM},
            {"role": "user", "content": "شنو خاص المشغل يوفر للعمال؟"},
            {"role": "assistant", "content": "خاصو يوفر معدات الوقاية الشخصية. شنو كتعرف على هاد المعدات؟"},
            {"role": "user", "content": "علاش خاصها تكون ملائمة للشغل؟"},
            {"role": "assistant", "content": "لأن كل شغل عندو مخاطر مختلفة. واش عندك مثال على شغل فيه خطر؟"},
            {"role": "user", "content": "شكون المسؤول على الصيانة ديال هاد المعدات؟"},
        ],
    },
]


def call_chat(model: str, messages: list) -> dict:
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"temperature": 0.2, "num_predict": 1}}
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_generate_raw(model: str, rendered: str) -> dict:
    payload = {"model": model, "prompt": rendered, "raw": True, "stream": False,
               "options": {"temperature": 0.2, "num_predict": 1}}
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    all_match = True
    for conv in CONVERSATIONS:
        print(f"\n=== {conv['name']} ({conv['model']}) ===")
        try:
            chat_res = call_chat(conv["model"], conv["messages"])
        except Exception as e:
            print(f"  /api/chat FAILED: {type(e).__name__}: {e}")
            all_match = False
            continue

        rendered = render_conversation(conv["messages"], add_generation_prompt=True)
        try:
            gen_res = call_generate_raw(conv["model"], rendered)
        except Exception as e:
            print(f"  /api/generate raw:true FAILED: {type(e).__name__}: {e}")
            all_match = False
            continue

        chat_count = chat_res.get("prompt_eval_count")
        gen_count = gen_res.get("prompt_eval_count")
        match = chat_count == gen_count
        all_match = all_match and match
        print(f"  /api/chat        prompt_eval_count = {chat_count}")
        print(f"  /api/generate raw prompt_eval_count = {gen_count}")
        print(f"  MATCH: {match}")
        if not match:
            print("  --- rendered text sent to /api/generate ---")
            print(rendered[:600].replace("\n", "\\n\n"))

    print("\n" + "=" * 60)
    if all_match:
        print("VERDICT: /api/chat reproduces the trained format. Use it as the transport.")
    else:
        print("VERDICT: MISMATCH — ship render_conversation() + raw:true instead of /api/chat.")
    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
