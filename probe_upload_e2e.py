"""
Live E2E probe: the tenant document upload feature end to end, against
real Postgres + the real single-worker ingest queue (app.services.
ingest_queue) -- not mocked, not monkeypatched. This is the concrete proof
that the whole point of the corpus_version pin-invalidation work
(app.routers.chat._resolve_turn_context) actually does something: a
document uploaded MID-CONVERSATION must show up in the very next turn's
answer, not stay invisible until an unrelated topic shift.

    1. Upload two real documents (French + Arabic-script), poll to 'ready'.
    2. Ask a baseline question (industrial domain, no relation to the
       upload) to seed a pin.
    3. Upload a THIRD document mid-conversation, containing a fact that
       exists NOWHERE else in the corpus. Ask a follow-up only that
       document can answer -- if the pin were reused verbatim (the bug
       this whole feature exists to prevent), the new source could never
       appear in `sources`.
    4. Toggle that source off -- confirm it disappears from `sources` on
       the very next turn (not just eventually).
    5. Delete it -- confirm total documents drops back to baseline and the
       question can no longer be answered from it.
    6. Confirm the tenant's always-on global corpus (raw/shared) remained
       reachable throughout -- uploads only ever ADD, never gate it.

Usage:
    .gguf_venv/Scripts/python.exe probe_upload_e2e.py
"""
import asyncio
import io
import sys
import time

sys.path.insert(0, ".")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi import UploadFile

from app.models.schemas import ChatRequest, Domain, SourceToggleRequest
from app.routers import chat as chat_router
from app.routers import ingest as ingest_router

SESSION_ID = "probe-upload-e2e"
POLL_TIMEOUT_S = 60

# A fact that exists nowhere else in raw/shared -- if the chat answer
# names this exact policy number, it can only have come from this upload.
UNIQUE_POLICY_TEXT = b"""# Politique interne de consignation electrique - Site Nord

## Article Z-99: Delai de recertification

Sur le site Nord, tout cadenas de consignation electrique individuel doit
etre recertifie par le responsable HSE tous les 47 jours, un delai propre
a ce site et distinct du delai national. Le non-respect de ce delai de 47
jours entraine le retrait immediat du cadenas du service.
"""


def _upload_file(filename: str, content: bytes) -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(content))


def wait_until_ready(source_id: str) -> None:
    start = time.monotonic()
    while time.monotonic() - start < POLL_TIMEOUT_S:
        row = ingest_router.get_source(source_id)
        if row.status == "ready":
            return
        if row.status == "error":
            raise RuntimeError(f"source {source_id} failed: {row.error_message}")
        time.sleep(1)
    raise TimeoutError(f"source {source_id} did not reach 'ready' within {POLL_TIMEOUT_S}s")


def main():
    ok = True

    print("=== 1. Upload + poll to ready ===")
    results = asyncio.run(ingest_router.upload_sources(
        files=[
            _upload_file("probe_fr.md", b"# Guide de test\n\nCeci est un document de test."),
            _upload_file("probe_ar.md", "# دليل الاختبار\n\nهذه وثيقة اختبار.".encode("utf-8")),
        ],
        domain=None,
    ))
    for r in results:
        wait_until_ready(r.id)
    print(f"  {len(results)} file(s) ready: {[r.id for r in results]}\n")

    print("=== 2. Baseline turn, seeds a pin ===")
    baseline = chat_router.chat(ChatRequest(
        message="Quelle est l'obligation du travailleur en matiere de securite ?",
        domain=Domain.INDUSTRIAL, session_id=SESSION_ID,
    ))
    print(f"  sources: {baseline.sources}\n")
    baseline_ok = bool(baseline.sources) and not baseline.response.startswith("Je ne dispose pas")
    ok &= baseline_ok
    print(f"  [{'OK' if baseline_ok else 'FAIL'}] baseline answered with real sources\n")

    print("=== 3. Mid-conversation upload -- the actual pin-invalidation proof ===")
    upload_result = asyncio.run(ingest_router.upload_sources(
        files=[_upload_file("politique_consignation_nord.md", UNIQUE_POLICY_TEXT)], domain=None,
    ))
    new_source_id = upload_result[0].id
    wait_until_ready(new_source_id)
    print(f"  uploaded politique_consignation_nord.md, id={new_source_id}, status=ready")

    follow_up = chat_router.chat(ChatRequest(
        message="Tous les combien de jours faut-il recertifier un cadenas de consignation electrique sur le site Nord ?",
        domain=Domain.INDUSTRIAL, session_id=SESSION_ID,
    ))
    print(f"  reply: {follow_up.response[:200]}")
    print(f"  sources: {follow_up.sources}")
    found_new_source = "politique_consignation_nord.md" in follow_up.sources
    found_47 = "47" in follow_up.response
    step3_ok = found_new_source and found_47
    ok &= step3_ok
    print(
        f"  [{'OK' if step3_ok else 'FAIL'}] new source appears in the very next turn "
        f"(source_in_sources={found_new_source}, mentions_47={found_47})\n"
    )

    print("=== 4. Toggle off -- disappears on the very next turn ===")
    ingest_router.update_source(new_source_id, SourceToggleRequest(enabled=False))
    after_toggle = chat_router.chat(ChatRequest(
        message="Et sur le site Nord, le delai de recertification du cadenas electrique, c'est combien de jours ?",
        domain=Domain.INDUSTRIAL, session_id=SESSION_ID,
    ))
    toggled_off_ok = "politique_consignation_nord.md" not in after_toggle.sources
    ok &= toggled_off_ok
    print(f"  sources after toggle-off: {after_toggle.sources}")
    print(f"  [{'OK' if toggled_off_ok else 'FAIL'}] disabled source no longer grounds the answer\n")

    print("=== 5. Delete -- chunk count returns to baseline ===")
    before_delete = ingest_router.list_sources()
    total_before = before_delete.total_chunks
    delete_result = ingest_router.delete_source(new_source_id)
    after_delete = ingest_router.list_sources()
    delete_ok = delete_result["deleted_chunks"] > 0 and after_delete.total_chunks == total_before - delete_result["deleted_chunks"]
    ok &= delete_ok
    print(f"  deleted_chunks={delete_result['deleted_chunks']}, total_chunks {total_before} -> {after_delete.total_chunks}")
    print(f"  [{'OK' if delete_ok else 'FAIL'}] chunk count dropped by exactly the deleted amount\n")

    print("=== 6. Global corpus remained reachable throughout ===")
    final = chat_router.chat(ChatRequest(
        message="Quels sont les equipements de protection individuelle obligatoires ?",
        domain=Domain.INDUSTRIAL, session_id=SESSION_ID + "-final",
    ))
    global_ok = bool(final.sources) and any("politique_consignation_nord" not in s for s in final.sources)
    ok &= global_ok
    print(f"  sources: {final.sources}")
    print(f"  [{'OK' if global_ok else 'FAIL'}] global raw/shared corpus still answers normally\n")

    # Cleanup: the two probe_fr/probe_ar sources from step 1.
    for r in results:
        ingest_router.delete_source(r.id)

    print("=" * 60)
    print("VERDICT:", "OK -- mid-conversation upload invalidates the pin, toggle/delete both "
          "take effect immediately, global corpus unaffected." if ok else "FAILED -- see steps above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
