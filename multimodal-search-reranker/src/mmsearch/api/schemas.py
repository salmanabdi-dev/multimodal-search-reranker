"""Request/response contracts for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Modality = Literal["text", "image"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000,
                       examples=["how do I keep a sourdough starter alive"])
    top_k: int = Field(5, ge=1, le=20, description="results in the merged list")
    rerank: bool = Field(True, description="false = stage 1 only (vector search)")
    modalities: list[Modality] = Field(default_factory=lambda: ["text", "image"],
                                       min_length=1)

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v

    @field_validator("modalities")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(v))


class SearchResult(BaseModel):
    rank: int
    modality: Modality
    document_id: str
    title: str | None = Field(description="document title, or the image caption")
    snippet: str | None = Field(None, description="text results only")
    image_url: str | None = Field(None, description="image results only")
    width: int | None = None
    height: int | None = None
    relevance: float = Field(description="calibrated probability; comparable across modalities")
    vector_score: float = Field(description="stage-1 cosine (MiniLM for text, CLIP for images)")
    vector_rank: int = Field(description="rank within its modality after stage 1")
    rerank_score: float | None = Field(
        None, description="cross-encoder logit (on the passage, or on the image caption)")


class SearchResponse(BaseModel):
    query: str
    reranked: bool
    modalities: list[Modality] = Field(description="modalities actually searched")
    candidates_considered: dict[str, int]
    results: list[SearchResult]
    timings_ms: dict[str, float]


class ImageUploadResponse(BaseModel):
    document_id: str
    created: bool = Field(description="false if identical bytes were already indexed")
    sha256: str
    width: int
    height: int
    image_url: str
