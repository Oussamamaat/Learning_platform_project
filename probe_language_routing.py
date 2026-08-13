"""
Live probe: does the sticky in-message language override
(app.services.routing.resolve_language) actually behave across a real
multi-turn conversation through the real chat() router -- explicit
instruction sets it, a same-script follow-up keeps it without repeating
the instruction, and a script change clears it?

Real generation, real Postgres (chat_sessions.response_lang_override /
override_query_lang), real pgvector retrieval against the ingested
industrial corpus -- domain is held constant (Domain.INDUSTRIAL, explicit)
throughout so this probe isolates language behaviour from domain routing.

Every assertion is observable through ChatResponse.language alone (no DB
introspection needed): each turn's language is compared against what
SCRIPT-DEFAULT ALONE would produce, so a turn that diverges from its own
script's default is proof the override (not coincidence) drove the result.

    Turn 1 (plain French, no instruction)      -> "fr"      (script default)
    Turn 2 (French + "réponds en darija")      -> "darija"  (explicit instruction)
    Turn 3 (plain French follow-up)            -> "darija"  (STICKY -- the actual
                                                    proof; script-default alone
                                                    would give "fr")
    Turn 4 (Arabic-script question)            -> "darija"  (script default --
                                                    but also clears the override,
                                                    since query_lang changed)
    Turn 5 (plain French follow-up again)      -> "fr"      (proves turn 4 truly
                                                    cleared the override, not
                                                    just coincidentally matching)

Usage:
    .gguf_venv/Scripts/python.exe probe_language_routing.py
"""
import sys

sys.path.insert(0, ".")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.models.schemas import ChatRequest, Domain
from app.routers import chat as chat_router

SESSION_ID = "probe-language-routing"

TURNS = [
    ("Quelle est l'obligation du travailleur en matière de sécurité au travail ?", "fr"),
    ("Comment fonctionne un permis de travail ? Reponds en darija.", "darija"),
    ("Et concrètement, qui doit le signer ?", "darija"),
    ("شنو هوما لخطارات د الكهرباء ف الشغل؟", "darija"),
    ("Et pour les equipements de protection, il y a des regles particulieres ?", "fr"),
]


def main():
    print("=== Sticky language override: 5-turn live conversation ===\n")
    all_ok = True
    for i, (message, expected) in enumerate(TURNS, start=1):
        response = chat_router.chat(ChatRequest(
            message=message, domain=Domain.INDUSTRIAL, session_id=SESSION_ID,
        ))
        ok = response.language == expected
        all_ok &= ok
        status = "OK" if ok else "MISMATCH"
        print(f"Turn {i} [{status}]")
        print(f"  message:  {message}")
        print(f"  expected: {expected}  actual: {response.language}  (domain_source={response.domain_source})")
        print(f"  reply:    {response.response[:160]}")
        print()

    print("=" * 60)
    print(
        "VERDICT:", "OK -- explicit instruction, stickiness, and clear-on-script-change "
        "all behave as designed." if all_ok else "FAILED -- see mismatches above."
    )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
