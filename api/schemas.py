from pydantic import BaseModel

class ProfileSchema(BaseModel):
    primary_goal: str
    horizon_years: int
    risk_score: int
    investor_profile: str
    current_holdings: list[str]

class AskModel(BaseModel):
    profile: ProfileSchema
    query: str


class OutputModel(BaseModel):
    response: str
    sources_used: list[str]
    model: str
    inference_ms: int
    tokens_generated: int

class DocumentIngestion(BaseModel):
    pass