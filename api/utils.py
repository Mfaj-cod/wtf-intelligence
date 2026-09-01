from typing import Any
from ollama import Client
from api.config import settings


def get_ollama_client() -> Client:
    return Client(host=settings.OLLAMA_BASE_URL or "http://localhost:11434")


def resolve_model_name() -> str:
    preferred = settings.OLLAMA_MODEL or "llama3.2:latest"
    try:
        client = get_ollama_client()
        response = client.list()
        available: list[str] = []

        def add_name(name: Any) -> None:
            if isinstance(name, str) and name:
                available.append(name)

        if isinstance(response, dict):
            for item in response.get("models", []):
                if isinstance(item, dict):
                    add_name(item.get("name"))
                else:
                    add_name(item)
        else:
            for item in getattr(response, "models", []) or []:
                if hasattr(item, "model"):
                    add_name(getattr(item, "model"))
                elif isinstance(item, dict):
                    add_name(item.get("name"))
                else:
                    add_name(item)

        if preferred in available:
            return preferred
        for candidate in ["llama3.2:latest", "llama3.2", "llama3.2:8b", "llama3.1:latest", "llama3.1"]:
            if candidate in available:
                return candidate
        if available:
            return available[0]
    except Exception:
        pass

    return "llama3.2:latest" if "llama3.2:latest" in [preferred, "llama3.2:latest"] else preferred
