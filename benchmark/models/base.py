from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelResult:
    """Structured result from model generation."""
    response: str
    latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None
    model_name: str = ""
    error: Optional[str] = None


class ModelAdapter:
    """Abstract adapter for comparing different models."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str) -> ModelResult:
        """Generate response from the model.
        
        Args:
            system_prompt: The system/instruction prompt
            user_prompt: The user question/request
            
        Returns:
            ModelResult with response and metrics
        """
        raise NotImplementedError

    def warmup(self):
        """Optional warm-up call to initialize model (load weights, etc)."""
        pass

    def validate(self) -> bool:
        """Validate that the model is accessible and working."""
        raise NotImplementedError
