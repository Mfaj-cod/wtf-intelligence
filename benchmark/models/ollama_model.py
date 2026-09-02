import time
from typing import Any

from api.utils import get_ollama_client
from .base import ModelAdapter, ModelResult


class OllamaModelAdapter(ModelAdapter):
    """Adapter for local Ollama LLM."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = get_ollama_client()
        self._actual_model = None

    def validate(self) -> bool:
        """Check if Ollama is accessible and model is available."""
        try:
            response = self.client.list()
            available_models = []

            # Handle both dict and object responses from Ollama
            if isinstance(response, dict):
                for item in response.get("models", []):
                    if isinstance(item, dict):
                        available_models.append(item.get("name"))
                    elif isinstance(item, str):
                        available_models.append(item)
            else:
                for item in getattr(response, "models", []) or []:
                    if hasattr(item, "model"):
                        available_models.append(getattr(item, "model"))
                    elif isinstance(item, dict):
                        available_models.append(item.get("name"))
                    else:
                        available_models.append(str(item))

            if self.model_name in available_models:
                self._actual_model = self.model_name
                return True

            # Fallback to first available
            if available_models:
                self._actual_model = available_models[0]
                return True

            return False
        except Exception as e:
            print(f"Ollama validation error: {e}")
            return False

    def get_actual_model(self) -> str:
        """Return the actual model name being used."""
        if not self._actual_model:
            self.validate()
        return self._actual_model or self.model_name

    def warmup(self):
        """Warm up the model with a simple request."""
        try:
            self.client.generate(
                model=self.get_actual_model(),
                prompt="Test",
                stream=False,
            )
        except Exception as e:
            print(f"Warmup error: {e}")

    def generate(self, system_prompt: str, user_prompt: str) -> ModelResult:
        """Generate response from Ollama model."""
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        start = time.perf_counter()
        try:
            response = self.client.generate(
                model=self.get_actual_model(),
                prompt=combined_prompt,
                stream=False,
                options={
                    "temperature": 0.2,
                    "num_predict": 500,
                },
            )
            end = time.perf_counter()
            latency_ms = (end - start) * 1000

            # Extract response text
            response_text = ""
            if isinstance(response, dict):
                response_text = response.get("response", "").strip()
            elif hasattr(response, "response"):
                response_text = str(getattr(response, "response", "")).strip()

            prompt_tokens = response.get("prompt_eval_count")
            completion_tokens = response.get("eval_count")

            total_tokens = None

            if prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens

            tokens_per_second = None

            eval_duration = response.get("eval_duration")

            if completion_tokens and eval_duration:
                tokens_per_second = (
                    completion_tokens / (eval_duration / 1_000_000_000)
                )

            return ModelResult(
                response=response_text,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                tokens_per_second=tokens_per_second,
                model_name=self.get_actual_model(),
                error=None,
            )
        except Exception as e:
            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            return ModelResult(
                response="",
                latency_ms=latency_ms,
                model_name=self.get_actual_model(),
                error=str(e),
            )
