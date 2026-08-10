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
    domain: Domain = Field(Domain.INDUSTRIAL, description="Domain for tutoring context")
    language: Optional[Language] = Field(
        None,
        description=(
            "Response language. Explicit value is authoritative; omitted "
            "falls back to detecting the language from the message text."
        ),
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI assistant response")
    session_id: str = Field(..., description="Conversation session ID")
    sources: list[str] = Field(default_factory=list, description="Retrieved document sources")
    tokens_used: int = Field(0, description="Total tokens consumed")


class AudioResponse(BaseModel):
    transcription: str = Field(..., description="Transcribed text from audio")
    response: str = Field(..., description="AI assistant response to transcribed query")
    session_id: str = Field(..., description="Conversation session ID")
    sources: list[str] = Field(default_factory=list, description="Retrieved document sources")


class QuizRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500, description="Topic to generate quiz about")
    num_questions: int = Field(5, ge=1, le=20, description="Number of questions to generate")
    tenant_id: Optional[str] = Field(None, description="Company/tenant identifier")
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
    total_questions: int = Field(..., description="Total number of questions")
    message: Optional[str] = Field(
        None,
        description="Set when no grounded questions could be returned (e.g. "
        "no matching source material, or every generated question failed "
        "the grounding check).",
    )
    sources: list[str] = Field(
        default_factory=list, description="Retrieved document sources the quiz was grounded in"
    )
