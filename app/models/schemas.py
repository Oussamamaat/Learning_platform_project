from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


class Language(str, Enum):
    FRENCH = "fr"
    ENGLISH = "en"
    DARIJA = "ar-MA"


class Domain(str, Enum):
    INDUSTRIAL = "industrial"
    SECURITE = "securite"
    BLOCKCHAIN = "blockchain"


class DiagramKind(str, Enum):
    FLOWCHART = "flowchart"
    SEQUENCE = "sequence"
    MINDMAP = "mindmap"
    PIE = "pie"
    XY = "xy"
    CANDLESTICK = "candlestick"


class Candle(BaseModel):
    """One OHLC bar. Used both as ChatRequest.candles (caller-supplied real
    data, rendered verbatim) and inside DiagramPayload.spec (model-invented
    or caller-supplied, whichever generate_diagram actually used)."""

    label: str = Field(..., description="X-axis tick, e.g. a date or session label")
    open: float
    high: float
    low: float
    close: float


class DiagramPayload(BaseModel):
    """A generated diagram, attached to ChatResponse when the turn's message
    triggered app.services.diagrams.detect_diagram_intent. See
    docs/architecture/diagram-generation.md."""

    kind: DiagramKind
    kind_source: str = Field(
        ..., description="How `kind` was chosen: 'keyword' (Tier 1, deterministic) or "
        "'semantic' (Tier 2 fallback, not yet implemented)."
    )
    title: str
    caption: str = Field(
        ..., description="One-to-two sentence explanation, also used as ChatResponse.response "
        "for this turn -- follows the turn's response language, unlike the diagram's own "
        "structural labels, which always follow settings.diagram_label_language."
    )
    mermaid: Optional[str] = Field(
        None, description="Mermaid source, populated for every kind except 'candlestick' "
        "(Mermaid has no OHLC diagram type -- see `spec` instead)."
    )
    spec: dict = Field(
        default_factory=dict,
        description="Raw render data for kinds Mermaid can't express. Only populated for "
        "'candlestick': {'candles': [Candle, ...]}.",
    )
    grounded: bool = Field(
        ..., description="True when retrieval found tenant context and every structural label "
        "passed the same ungrounded-reference check quiz questions do. False means the diagram "
        "is illustrative (e.g. a candlestick pattern with no corpus) -- still shown, just "
        "flagged so the UI can say it isn't from the tenant's own documents."
    )
    repairs: list[str] = Field(
        default_factory=list,
        description="Deterministic salvage actions taken on the model's raw output before "
        "rendering (e.g. a dangling edge dropped) -- never a fabrication, only a removal. "
        "Empty when the model's output needed no repair.",
    )
    sources: list[str] = Field(default_factory=list, description="Retrieved document sources, when grounded")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: Optional[str] = Field(None, description="Conversation session ID")
    tenant_id: Optional[str] = Field(None, description="Company/tenant identifier")
    domain: Optional[Domain] = Field(
        None,
        description=(
            "Domain for tutoring context. Explicit value (e.g. a course-module "
            "page that already knows its own domain) is authoritative and skips "
            "auto-routing entirely; omitted routes automatically via "
            "app.services.routing (retrieval-as-router, then the tenant "
            "default) -- see ChatResponse.domain_source for what was chosen."
        ),
    )
    language: Optional[Language] = Field(
        None,
        description=(
            "Response language. Explicit value is authoritative for both "
            "retrieval and the response; omitted routes automatically via "
            "app.services.routing (script detection, an in-message "
            "instruction like 'réponds en darija', or a sticky prior "
            "override) -- see ChatResponse.language."
        ),
    )
    active_source_ids: Optional[list[str]] = Field(
        None,
        description=(
            "Which of the tenant's uploaded sources (app/routers/ingest.py) "
            "may ground this turn's answer. A NARROWING hint only, like "
            "tenant_id -- app.services.sources.active_source_ids intersects "
            "this against server-side (status='ready' AND enabled) state, "
            "never widens it, so a stale client can't resurrect a source the "
            "tenant just disabled. The tenant's always-on global corpus is "
            "never affected by this field. Omitted means every currently-"
            "enabled uploaded source is eligible."
        ),
    )
    candles: Optional[list[Candle]] = Field(
        None,
        description=(
            "Real OHLC data for a candlestick diagram, when the caller already "
            "has it. Rendered verbatim -- app.services.diagrams overrides "
            "whatever the model invented with this list rather than the "
            "reverse, so a real dataset can never be silently replaced by "
            "illustrative model output. Ignored unless the message's detected "
            "diagram intent is 'candlestick' (app.services.diagrams."
            "detect_diagram_intent); omitted means the model invents "
            "illustrative values for the named pattern (e.g. 'marteau haussier')."
        ),
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI assistant response")
    session_id: str = Field(..., description="Conversation session ID")
    sources: list[str] = Field(default_factory=list, description="Retrieved document sources")
    tokens_used: int = Field(0, description="Total tokens consumed")
    domain: str = Field(..., description="The domain this turn was actually answered in")
    domain_source: str = Field(
        ...,
        description=(
            "How `domain` was resolved: 'page_context' (caller supplied it), "
            "'retrieval' (auto-routed by a similarity-weighted vote over "
            "unfiltered top chunks), 'pinned' (reused this session's existing "
            "segment), 'no_match' (the vote ran and NOTHING cleared the "
            "threshold -- the query is out of corpus, and this response is a "
            "deterministic refusal), or 'tenant_default' (routing could form "
            "no opinion at all: disk backend, or the vote's own search "
            "failed)."
        ),
    )
    language: str = Field(
        ..., description="The response language actually used: 'fr' or 'darija'."
    )
    prior_questions: list[str] = Field(
        default_factory=list,
        description=(
            "Socratic questions the tutor already asked in turns that fell out of "
            "the replayed history window (deterministic extraction, not a model "
            "summary) -- for the UI to show 'already asked' state without the "
            "server re-asking them in the prompt."
        ),
    )
    cross_language: bool = Field(
        False,
        description=(
            "True when retrieval could not find enough same-language source material "
            "(ADR 0002 decision 5) and fell back across the language line -- the answer "
            "is still grounded and in the requested language, but the UI should surface "
            "that the underlying sources are in a different script."
        ),
    )
    degraded: bool = Field(
        False,
        description=(
            "True when pgvector retrieval raised and this answer fell back to the "
            "disk backend (app/routers/chat.py's _retrieve_context) -- which knows "
            "nothing about tenant uploads. The answer is still grounded in the "
            "built-in corpus, but any uploaded sources are silently absent from it "
            "until Postgres recovers; the UI should say so rather than let a "
            "confident, cited-looking answer imply otherwise."
        ),
    )
    diagram: Optional[DiagramPayload] = Field(
        None,
        description=(
            "Set when the message's own text triggered "
            "app.services.diagrams.detect_diagram_intent -- there is no separate "
            "diagram endpoint. `response` is the diagram's own caption in that "
            "case. None for every ordinary chat turn."
        ),
    )


class AudioResponse(BaseModel):
    transcription: str = Field(..., description="Transcribed text from audio")
    response: str = Field(..., description="AI assistant response to transcribed query")
    session_id: str = Field(..., description="Conversation session ID")
    sources: list[str] = Field(default_factory=list, description="Retrieved document sources")


class QuizRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500, description="Topic to generate quiz about")
    num_questions: int = Field(5, ge=1, le=20, description="Number of questions to generate")
    tenant_id: Optional[str] = Field(None, description="Company/tenant identifier")
    domain: Optional[Domain] = Field(
        None,
        description=(
            "Domain to ground the quiz in. Explicit value is authoritative and "
            "enforced at retrieval time (was previously hardcoded server-side to "
            "'industrial' with no domain filter at all -- a quiz labelled "
            "industrial could ground in blockchain chunks). Omitted routes "
            "automatically via app.services.routing, same as chat -- see "
            "QuizResponse.domain_source."
        ),
    )
    language: Optional[Language] = Field(
        None,
        description=(
            "Quiz language. Explicit value is authoritative; omitted falls "
            "back to detecting the language from the topic text."
        ),
    )


class QuizQuestion(BaseModel):
    question: str = Field(..., description="Question text")
    options: list[str] = Field(..., min_length=2, max_length=6, description="Answer options")
    correct_index: int = Field(..., ge=0, description="Index of correct answer")
    explanation: str = Field("", description="Explanation of the correct answer")


class QuizResponse(BaseModel):
    questions: list[QuizQuestion] = Field(..., description="Generated quiz questions")
    topic: str = Field(..., description="Quiz topic")
    total_questions: int = Field(..., description="Total number of questions actually returned")
    requested_questions: int = Field(
        ..., description="Number of questions the caller asked for (QuizRequest.num_questions) "
        "-- compare against total_questions to detect a shortfall, e.g. thin source material."
    )
    message: Optional[str] = Field(
        None,
        description="Set when no grounded questions could be returned (e.g. "
        "no matching source material, or every generated question failed "
        "the grounding check).",
    )
    sources: list[str] = Field(
        default_factory=list, description="Retrieved document sources the quiz was grounded in"
    )
    domain: str = Field(..., description="The domain this quiz was actually grounded in")
    domain_source: str = Field(
        ...,
        description=(
            "How `domain` was resolved: 'page_context' | 'retrieval' | "
            "'no_match' (vote cleared nothing -- out of corpus, and this "
            "response is a refusal) | 'tenant_default'"
        ),
    )
    language: str = Field(..., description="The quiz language actually used: 'fr' or 'darija'.")


class SourceStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    PARTIAL = "partial"
    ERROR = "error"


class SourceFileOut(BaseModel):
    """One tenant-uploaded document, mirroring app.models.database.SourceFile
    -- see app/routers/ingest.py."""

    id: str
    filename: str
    status: SourceStatus
    error_message: Optional[str] = None
    enabled: bool
    domain: Optional[str] = None
    language: Optional[str] = None
    chunk_count: int = 0
    page_count: Optional[int] = None
    pages_done: Optional[int] = Field(
        None,
        description="Live progress while status='processing': pages attempted so far, "
        "out of page_count. NULL once ready/partial/error -- read chunk_count/page_count "
        "for the finished result instead.",
    )
    parser: Optional[str] = Field(
        None,
        description="How the file was parsed: pdf_text | pdf_ocr | pdf_mixed | docx | pptx | "
        "xlsx | csv | text | image_ocr -- an audit trail for 'was this text OCR'd'.",
    )
    ocr_engine: Optional[str] = None
    unprocessed_pages: Optional[list[dict]] = Field(
        None,
        description="Set when status='partial': pages skipped rather than failing the "
        "whole document, e.g. [{'page': 4, 'reason': 'ocr_required'}].",
    )
    size_bytes: int = 0
    created_at: datetime
    duplicate_of: Optional[str] = Field(
        None,
        description="Set when this upload's sha256 matched an existing ready source -- "
        "the returned row IS that existing source, not a new one.",
    )


class SourceListResponse(BaseModel):
    sources: list[SourceFileOut]
    ready_count: int
    total_chunks: int


class SourceToggleRequest(BaseModel):
    enabled: bool


class VideoJobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class VideoGenerateRequest(BaseModel):
    """Input contract: what our app sends to request a video."""

    text: str = Field(..., min_length=1, max_length=8000, description="Explanation/content to turn into video")
    title: Optional[str] = Field(
        None,
        max_length=300,
        description="Short topic label for the video's opening frame, in the same language as text",
    )
    language: Language = Field(..., description="fr | en | ar-MA")
    session_id: Optional[str] = Field(None, description="Chat/quiz session this was triggered from, if any")
    tenant_id: Optional[str] = Field(None, description="Company/tenant identifier")


class VideoJobUpdateRequest(BaseModel):
    """Output contract: what the video worker PATCHes back as it progresses."""

    status: VideoJobStatus
    video_url: Optional[str] = Field(None, description="Required when status='ready'")
    error_message: Optional[str] = Field(None, description="Required when status='error'")


class VideoJobOut(BaseModel):
    id: str
    tenant_id: str
    session_id: Optional[str] = None
    input_text: str
    title: Optional[str] = None
    language: str
    status: VideoJobStatus
    video_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
