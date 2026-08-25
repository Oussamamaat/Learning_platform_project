from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models.schemas import AudioResponse, Language

router = APIRouter(prefix="/api/v1/audio", tags=["audio"])


@router.post("/", response_model=AudioResponse)
async def process_audio(
    audio: UploadFile = File(...),
    session_id: str = None,
    tenant_id: str = None,
    language: Language = Language.DARIJA,
):
    """
    Audio processing endpoint -- reserved for the voice-chat feature
    (STT -> RAG pipeline -> TTS). Not implemented: no STT/TTS vendor has
    been selected yet (resurrection.md Q0.2, the program's single largest
    unresolved MVP item, tracked #1 in resurrection.md's "Known-open,
    carried forward" list). This must fail loudly rather than return a
    placeholder transcription -- a mounted endpoint that fakes success is
    worse than one that says so, once a real client is wired to it.
    """
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio type")

    raise HTTPException(
        status_code=501,
        detail="Audio transcription is not implemented yet (no STT/TTS vendor selected).",
    )
