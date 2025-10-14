"""API surface for serving Chatterbox via FastAPI."""

from .openai_tts import app

__all__ = ["app"]
