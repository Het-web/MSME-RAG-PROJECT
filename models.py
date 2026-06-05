from typing import Any, Literal

from pydantic import BaseModel, Field


RouteName = Literal["MSME", "General", "Blocked"]


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)


class Source(BaseModel):
    source_file: str | None = None
    document_type: str | None = None
    upload_date: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    content_preview: str | None = None


class ChatResponse(BaseModel):
    route: RouteName
    answer: str
    sources: list[Source] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class GuardrailResult(BaseModel):
    passed: bool
    reason: str | None = None
    detected: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    context: str
    sources: list[Source]
    metrics: dict[str, Any]


class TavilyResult(BaseModel):
    context: str = ""
    sources: list[Source] = Field(default_factory=list)
    used: bool = False
    reason: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
