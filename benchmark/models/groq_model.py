import os
import time
from typing import Optional

from .base import ModelAdapter, ModelResult

try:
    from groq import Groq
except ImportError:
    Groq = None


class GroqModelAdapter(ModelAdapter):
    """Adapter for Groq API with GPT-OSS 120B model."""

    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        super().__init__(model_name)
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = None
        if self.api_key and Groq:
            self.client = Groq(api_key=self.api_key)

    def validate(self) -> bool:
        """Check if Groq API is accessible."""
        if not self.api_key:
            print("GROQ_API_KEY not set")
            return False
        if not Groq:
            print("groq package not installed: pip install groq")
            return False
        if not self.client:
            print("Groq client not initialized")
            return False

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10,
                temperature=0.2,
            )
            return response is not None
        except Exception as e:
            print(f"Groq validation error: {e}")
            return False

    def warmup(self):
        """Warm up the API with a simple request."""
        if not self.client:
            return
        try:
            self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10,
                temperature=0.2,
            )
        except Exception as e:
            print(f"Groq warmup error: {e}")

    def generate(self, system_prompt: str, user_prompt: str) -> ModelResult:
        """Generate response from Groq API."""
        if not self.client:
            return ModelResult(
                response="",
                latency_ms=0,
                model_name=self.model_name,
                error="Groq client not initialized",
            )

        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=500,
                temperature=0.2,
            )
            end = time.perf_counter()
            latency_ms = (end - start) * 1000

            response_text = response.choices[0].message.content.strip()

            # Extract token counts if available
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None
            tokens_per_second = None

            if hasattr(response, "usage") and response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                if completion_tokens and latency_ms > 0:
                    tokens_per_second = (completion_tokens / latency_ms) * 1000

            return ModelResult(
                response=response_text,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                tokens_per_second=tokens_per_second,
                model_name=self.model_name,
                error=None,
            )
        except Exception as e:
            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            return ModelResult(
                response="",
                latency_ms=latency_ms,
                model_name=self.model_name,
                error=str(e),
            )
