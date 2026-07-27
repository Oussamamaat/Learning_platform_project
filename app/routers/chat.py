import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.models.schemas import ChatRequest, ChatResponse
from app.services.search import build_rag_context
from app.services.llm import generate_llm_response
from app.errors import AppError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Text-based conversation endpoint.
    Receives user message, performs RAG retrieval, and queries the local LLM.
    """
    tenant = request.tenant_id or "company_abc"
    domain = request.domain.value

    # 1. Retrieve relevant context and sources from pgvector
    context, sources = build_rag_context(
        query=request.message,
        tenant_id=tenant,
        top_k=4,
        similarity_threshold=0.35
    )

    # 2. Call local Ollama model to generate answer based on context
    try:
        ai_reply = generate_llm_response(
            query=request.message,
            context=context,
            domain=domain,
        )
    except AppError as e:
        logger.error("LLM error in chat: %s", e.code)
        return JSONResponse(
            status_code=e.status_code,
            content={"error": {"code": e.code, "message": e.message}},
        )

    return ChatResponse(
        response=ai_reply,
        session_id=request.session_id or "new-session",
        sources=sources,
        tokens_used=0,
    )
