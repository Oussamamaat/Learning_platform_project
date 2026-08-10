import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.models.schemas import ChatRequest, ChatResponse
from app.services.domain_context import build_domain_context
from app.services.llm import (
    generate_llm_response,
    deterministic_refusal,
    detect_query_language,
    UI_LANG_TO_MODEL_LANG,
)
from app.errors import AppError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Text-based conversation endpoint.
    Receives user message, performs RAG retrieval, and queries the local LLM.
    """
    domain = request.domain.value

    # Explicit ui_lang wins; omitted falls back to the heuristic. Resolved
    # once so both the deterministic-refusal path and the model path agree.
    ui_lang = (
        UI_LANG_TO_MODEL_LANG.get(request.language.value)
        if request.language
        else None
    )
    ui_lang = ui_lang or detect_query_language(request.message)

    # 1. Retrieve relevant chunks from the selected domain's own reference
    # documents (raw/shared/<domain>/text/) -- not pgvector. See
    # app/services/domain_context.py for why.
    context, sources = build_domain_context(
        query=request.message,
        domain=domain,
        top_k=4,
    )

    # 2. Empty context: refuse deterministically rather than let the model
    # compose its own refusal. The fine-tuned model's refusal register is
    # welded to tenant #1's safety domain regardless of the actual tenant
    # domain -- this bypasses that bias entirely instead of prompting
    # around it. See app/services/llm.py:deterministic_refusal.
    if not context.strip():
        return ChatResponse(
            response=deterministic_refusal(domain, ui_lang),
            session_id=request.session_id or "new-session",
            sources=[],
            tokens_used=0,
        )

    # 3. Call local Ollama model to generate answer based on context
    try:
        ai_reply = generate_llm_response(
            query=request.message,
            context=context,
            domain=domain,
            language=ui_lang,
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
