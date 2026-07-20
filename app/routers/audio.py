from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models.schemas import AudioResponse

router = APIRouter(prefix="/api/v1/audio", tags=["audio"])


@router.post("/", response_model=AudioResponse)
async def process_audio(
    audio: UploadFile = File(...),
    session_id: str = None,
    tenant_id: str = None,
    language: str = "ar-MA",
):
    """
    Audio processing endpoint.
    Receives audio blob, transcribes via Whisper, processes through RAG pipeline.
    """
    # TODO: Week 6 - Integrate Whisper STT
    # TODO: Week 6 - Pass transcription to RAG pipeline
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio type")

    return AudioResponse(
        transcription="[Placeholder] Audio transcription will appear here.",
        response="[Placeholder] AI response to transcribed audio.",
        session_id=session_id or "new-session",
        sources=[],
    )
