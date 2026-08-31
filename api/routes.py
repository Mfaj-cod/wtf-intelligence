from fastapi import APIRouter, Depends, File, HTTPException, UploadFile # type: ignore
from .schemas import AskModel, OutputModel

router = APIRouter(tags=["routes"])


@router.post("/ask")
def generate_response(payload: AskModel):
    query = payload.query

    return OutputModel(response="", sources_used=[], model="llama3.2:8b", inference_ms=1842, tokens_generated=312)

@router.ingest("/ingest")
def ingest_doc():
    pass

@router.get("/models")
def get_models():
    pass

@router.get("/health")
def health_check():
    pass

