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
            "unfiltered top chunks), or 'tenant_default' (nothing cleared "
            "the routing threshold)."
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
        ..., description="How `domain` was resolved: 'page_context' | 'retrieval' | 'tenant_default'"
    )
    language: str = Field(..., description="The quiz language actually used: 'fr' or 'darija'.")
