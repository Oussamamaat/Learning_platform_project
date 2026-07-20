from fastapi import APIRouter
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Text-based conversation endpoint.
    Receives user message, performs RAG retrieval, returns AI response.
    """
    # TODO: Week 2 - Implement RAG pipeline
    # TODO: Week 3 - Connect to LLM
    return ChatResponse(
        response="[Placeholder] This endpoint will connect to the RAG pipeline and LLM.",
        session_id=request.session_id or "new-session",
        sources=[],
        tokens_used=0,
    )
