import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.models.schemas import QuizRequest, QuizResponse, QuizQuestion
from app.services.search import build_rag_context
from app.services.quiz import generate_quiz_questions
from app.services.grounding import filter_grounded_questions
from app.services.llm import (
    deterministic_refusal,
    detect_query_language,
    UI_LANG_TO_MODEL_LANG,
)
from app.errors import AppError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/quiz", tags=["quiz"])


@router.post("/", response_model=QuizResponse)
def generate_quiz(request: QuizRequest):
    """
    Quiz generation endpoint.
    Retrieves relevant course material via RAG, generates pedagogical quiz
    questions with the fine-tuned model, and filters out any question that
    cites something absent from the retrieved material before returning it.
    """
    tenant = request.tenant_id or "company_abc"
    domain = "industrial"

    ui_lang = (
        UI_LANG_TO_MODEL_LANG.get(request.language.value)
        if request.language
        else None
    )
    ui_lang = ui_lang or detect_query_language(request.topic)

    # 1. Retrieve relevant context and sources from pgvector
    context, sources = build_rag_context(
        query=request.topic,
        tenant_id=tenant,
        top_k=4,
        similarity_threshold=0.35,
    )

    # 2. No usable source material: refuse deterministically rather than let
    # the model invent quiz content with nothing to ground it in.
    if not context.strip():
        return QuizResponse(
            questions=[],
            topic=request.topic,
            total_questions=0,
            message=deterministic_refusal(domain, ui_lang),
            sources=[],
        )

    # 3. Generate, then filter out anything not actually grounded in context.
    try:
        raw_questions = generate_quiz_questions(
            topic=request.topic,
            context=context,
            domain=domain,
            language=ui_lang,
            n=request.num_questions,
        )
    except AppError as e:
        logger.error("LLM error in quiz: %s", e.code)
        return JSONResponse(
            status_code=e.status_code,
            content={"error": {"code": e.code, "message": e.message}},
        )

    kept, dropped = filter_grounded_questions(raw_questions, context)
    if dropped:
        logger.warning(
            "quiz: dropped %d/%d ungrounded/invalid question(s) for topic=%s",
            len(dropped), len(raw_questions), request.topic,
        )

    if not kept:
        return QuizResponse(
            questions=[],
            topic=request.topic,
            total_questions=0,
            message=deterministic_refusal(domain, ui_lang),
            sources=[],
        )

    questions = [
        QuizQuestion(
            question=q["question"],
            options=q["options"],
            correct_index=q["answer"],
            explanation=q.get("explanation", ""),
        )
        for q in kept
    ]

    return QuizResponse(
        questions=questions,
        topic=request.topic,
        total_questions=len(questions),
        sources=sources,
    )
