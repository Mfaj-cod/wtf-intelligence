from pydantic import BaseModel, Field


class ProfileSchema(BaseModel):
    primary_goal: str = Field(..., min_length=1)
    horizon_years: int = Field(..., gt=0)
    risk_score: int = Field(..., ge=1, le=10)
    investor_profile: str = Field(..., min_length=1)
    current_holdings: list[str] = Field(..., min_length=1)


class AskRequest(BaseModel):
    profile: ProfileSchema
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    response: str
    sources_used: list[str]
    model: str
    inference_ms: int
    tokens_generated: int
    retrieved_chunks: list[str] | None = None
    retrieval_ms: int | None = None
    total_latency_ms: int | None = None


class HealthResponse(BaseModel):
    status: str
    ollama: dict
    chroma: dict


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    chunks_stored: int
    collection: str
    embedding_model: str
    duration_ms: int


class ModelsResponse(BaseModel):
    models: list[str]