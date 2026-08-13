import logging
import time
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.config import get_tenant_id
from app.models.schemas import QuizRequest, QuizResponse, QuizQuestion
from app.services.search import build_rag_context
from app.services.quiz import generate_quiz_questions
from app.services.grounding import filter_grounded_questions
from app.services.llm import deterministic_refusal, UI_LANG_TO_MODEL_LANG
from app.services.routing import resolve_domain, resolve_language
from app.errors import AppError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/quiz", tags=["quiz"])

# A single generate_quiz_questions round measured at 91-119s live (10
# questions, industrial/FR). One extra top-up round is worth the wait; a
# second would risk a ~5 minute spinner behind the modal for a diminishing
# return, so it's capped at 1. QUIZ_TOPUP_BUDGET_S is a backstop against an
# unusually slow round rather than the primary limit -- checked before
# starting the extra round, not mid-call.
QUIZ_MAX_EXTRA_ROUNDS = 1
QUIZ_TOPUP_BUDGET_S = 180


@router.post("/", response_model=QuizResponse)
def generate_quiz(request: QuizRequest):
    """
    Quiz generation endpoint.
    Retrieves relevant course material via RAG, generates pedagogical quiz
    questions with the fine-tuned model, and filters out any question that
    cites something absent from the retrieved material before returning it.
    """
    # get_tenant_id ignores request.tenant_id entirely (single-tenant MVP
    # seam, app/config.py) -- a client-supplied tenant_id was a recorded
    # security bug (a request could claim any tenant's data by naming it).
    tenant = get_tenant_id(request.tenant_id)

    # Domain: an explicit request.domain (a course-module page that already
    # knows its own domain) is tier 1 and skips routing entirely. Omitted
    # routes automatically -- app.services.routing.resolve_domain's tier 2
    # (retrieval-as-router: vote over an unfiltered search) then tier 3
    # (tenant default).
    if request.domain is not None:
        domain, domain_source = request.domain.value, "page_context"
    else:
        domain, domain_source = resolve_domain(
            request.topic, tenant_id=tenant, backend="pgvector"
        )

    # Language: same split as chat -- query_lang drives retrieval affinity,
    # response_lang drives the quiz's own language and refusal wording.
    # Quiz has no session, so there is nothing to be sticky across; only
    # the explicit-field and script-detection tiers of resolve_language
    # apply here.
    explicit_lang = (
        UI_LANG_TO_MODEL_LANG.get(request.language.value) if request.language else None
    )
    lang = resolve_language(request.topic, explicit_language=explicit_lang)

    # 1. Retrieve relevant context and sources from pgvector, scoped to
    # `domain` -- previously build_rag_context had no domain filter at
    # all, so a quiz labelled "industrial" could ground in blockchain
    # chunks (nobody noticed because there was no retrieval eval).
    context, sources = build_rag_context(
        query=request.topic,
        tenant_id=tenant,
        top_k=4,
        domain=domain,
        ui_lang=lang.query_lang,
    )

    # 2. No usable source material: refuse deterministically rather than let
    # the model invent quiz content with nothing to ground it in.
    if not context.strip():
        return QuizResponse(
            questions=[],
            topic=request.topic,
            total_questions=0,
            requested_questions=request.num_questions,
            message=deterministic_refusal(domain, lang.response_lang),
            sources=[],
            domain=domain,
            domain_source=domain_source,
            language=lang.response_lang,
        )

    # 3. Generate, then filter out anything not actually grounded in context.
    # Top up on a shortfall: the model doesn't reliably hit `num_questions`
    # exactly (out-of-distribution count + grounding/structure rejections
    # both shave some off), so ask for the deficit again -- pointed away
    # from what's already kept -- rather than silently returning fewer than
    # requested. Bounded at QUIZ_MAX_EXTRA_ROUNDS extra rounds and a wall-clock
    # backstop so a stubborn topic (context too thin to ground more questions,
    # or an unusually slow round) can't turn into an open-ended spinner.
    kept: list = []
    dropped: list = []
    seen_questions: set = set()
    loop_start = time.monotonic()
    try:
        for round_idx in range(QUIZ_MAX_EXTRA_ROUNDS + 1):
            deficit = request.num_questions - len(kept)
            if deficit <= 0:
                break
            if round_idx > 0 and time.monotonic() - loop_start > QUIZ_TOPUP_BUDGET_S:
                logger.warning(
                    "quiz: topup budget (%ds) exhausted for topic=%s, stopping at %d/%d",
                    QUIZ_TOPUP_BUDGET_S, request.topic, len(kept), request.num_questions,
                )
                break
            raw_questions = generate_quiz_questions(
                topic=request.topic,
                context=context,
                domain=domain,
                language=lang.response_lang,
                n=deficit,
                avoid_questions=[q["question"] for q in kept] or None,
            )
            round_kept, round_dropped = filter_grounded_questions(raw_questions, context)
            dropped.extend(round_dropped)
            for q in round_kept:
                key = q["question"].strip().casefold()
                if key in seen_questions:
                    continue
                seen_questions.add(key)
                kept.append(q)
    except AppError as e:
        logger.error("LLM error in quiz: %s", e.code)
        return JSONResponse(
            status_code=e.status_code,
            content={"error": {"code": e.code, "message": e.message}},
        )

    if dropped:
        logger.warning(
            "quiz: dropped %d ungrounded/invalid/duplicate question(s) for topic=%s "
            "(kept %d/%d requested)",
            len(dropped), request.topic, len(kept), request.num_questions,
        )

    if not kept:
        return QuizResponse(
            questions=[],
            topic=request.topic,
            total_questions=0,
            requested_questions=request.num_questions,
            message=deterministic_refusal(domain, lang.response_lang),
            sources=[],
            domain=domain,
            domain_source=domain_source,
            language=lang.response_lang,
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
        requested_questions=request.num_questions,
        sources=sources,
        domain=domain,
        domain_source=domain_source,
        language=lang.response_lang,
    )
