"""FastAPI app exposing a streaming endpoint compatible with OpenAI's TTS API."""

from __future__ import annotations

import base64
import json
import logging
import threading
from typing import Dict, Iterable, Iterator, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator

from ..mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from ..tts import ChatterboxTTS

LOGGER = logging.getLogger(__name__)


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _ModelRegistry:
    """Thread-safe registry that lazily loads TTS backends."""

    def __init__(self) -> None:
        self._device = _select_device()
        self._english: Optional[ChatterboxTTS] = None
        self._multilingual: Optional[ChatterboxMultilingualTTS] = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        return self._device

    def _load_english(self) -> ChatterboxTTS:
        if self._english is None:
            with self._lock:
                if self._english is None:
                    LOGGER.info("Loading ChatterboxTTS on %s", self._device)
                    self._english = ChatterboxTTS.from_pretrained(device=self._device)
        return self._english

    def _load_multilingual(self) -> ChatterboxMultilingualTTS:
        if self._multilingual is None:
            with self._lock:
                if self._multilingual is None:
                    LOGGER.info("Loading ChatterboxMultilingualTTS on %s", self._device)
                    self._multilingual = ChatterboxMultilingualTTS.from_pretrained(device=self._device)
        return self._multilingual

    def resolve(self, model_name: str, language: Optional[str]) -> tuple[object, Dict[str, int]]:
        preferred_multi = False
        if language and language.lower() != "en":
            preferred_multi = True
        normalized = (model_name or "").lower()
        if normalized in {"chatterbox-multilingual", "chatterbox-multi", "faster_multi_api"}:
            preferred_multi = True
        if preferred_multi:
            model = self._load_multilingual()
        else:
            model = self._load_english()
        return model, {"sample_rate": getattr(model, "sr", 24_000)}


class SpeechRequest(BaseModel):
    """Body for POST /v1/audio/speech."""

    model: str = Field(default="faster_multi_api", description="Model identifier")
    input: str = Field(..., description="Text to synthesise")
    language: Optional[str] = Field(default=None, description="Language code for multilingual synthesis")
    audio_prompt_path: Optional[str] = Field(default=None, description="Path to a reference audio prompt")
    exaggeration: float = Field(default=0.5, ge=0.0, le=1.5)
    cfg_weight: float = Field(default=0.5, ge=0.0, le=3.0)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=1000, ge=1, le=1500)
    max_cache_len: int = Field(default=1500, ge=1)
    repetition_penalty: float = Field(default=1.2, ge=0.5, le=2.0)
    min_p: float = Field(default=0.05, ge=0.0, le=1.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = Field(default=True, description="Stream audio chunks as SSE events")
    response_format: str = Field(default="pcm16", description="Audio payload encoding")
    min_chunk_tokens: int = Field(default=10, ge=1, description="Tokens to buffer before emitting the first chunk")
    t3_params: Dict[str, float] = Field(default_factory=dict, description="Advanced overrides passed to T3.inference")

    @validator("response_format")
    def _check_response_format(cls, value: str) -> str:
        if value != "pcm16":
            raise ValueError("Only pcm16 response_format is currently supported")
        return value


registry = _ModelRegistry()
app = FastAPI(title="Chatterbox OpenAI-Compatible API", version="1.0.0")


def _tensor_to_pcm16(chunk: torch.Tensor) -> bytes:
    array = chunk.detach().cpu().numpy()
    if array.ndim > 1:
        array = np.squeeze(array, axis=0)
    array = np.clip(array, -1.0, 1.0)
    return (array * 32767.0).astype("<i2").tobytes()


def _format_sse(event: str, payload: Dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _stream_to_sse(chunks: Iterable[torch.Tensor], sample_rate: int) -> Iterator[str]:
    yield _format_sse("response.metadata", {"sample_rate": sample_rate, "format": "pcm16"})
    try:
        for index, chunk in enumerate(chunks):
            pcm = _tensor_to_pcm16(chunk)
            encoded = base64.b64encode(pcm).decode("ascii")
            yield _format_sse(
                "response.output_audio.delta",
                {"index": index, "audio": encoded},
            )
    except Exception as exc:  # pragma: no cover - runtime safeguard
        LOGGER.exception("Streaming failure", exc_info=exc)
        yield _format_sse("response.error", {"message": str(exc)})
        return
    yield _format_sse("response.completed", {})


@app.get("/healthz", tags=["system"])
def healthcheck() -> Dict[str, str]:
    return {"status": "ok", "device": registry.device}


@app.post("/v1/audio/speech", tags=["tts"])
def create_speech(request: SpeechRequest):
    try:
        model, metadata = registry.resolve(request.model, request.language)
    except ValueError as exc:  # pragma: no cover - resolution errors should be rare
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    language = request.language.lower() if request.language else None
    if isinstance(model, ChatterboxMultilingualTTS):
        language = language or "en"
        if language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language '{language}'. Supported values: {', '.join(SUPPORTED_LANGUAGES)}",
            )
    else:
        language = language or "en"

    t3_kwargs = dict(request.t3_params)

    if request.stream:
        if isinstance(model, ChatterboxMultilingualTTS):
            chunk_iterator = model.generate_stream(
                request.input,
                language,
                audio_prompt_path=request.audio_prompt_path,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
                temperature=request.temperature,
                max_new_tokens=request.max_new_tokens,
                max_cache_len=request.max_cache_len,
                repetition_penalty=request.repetition_penalty,
                min_p=request.min_p,
                top_p=request.top_p,
                t3_params=t3_kwargs,
                min_chunk_tokens=request.min_chunk_tokens,
            )
        else:
            chunk_iterator = model.generate_stream(
                request.input,
                language,
                audio_prompt_path=request.audio_prompt_path,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
                temperature=request.temperature,
                max_new_tokens=request.max_new_tokens,
                max_cache_len=request.max_cache_len,
                repetition_penalty=request.repetition_penalty,
                min_p=request.min_p,
                top_p=request.top_p,
                t3_params=t3_kwargs,
                min_chunk_tokens=request.min_chunk_tokens,
            )

        return StreamingResponse(
            _stream_to_sse(chunk_iterator, metadata["sample_rate"]),
            media_type="text/event-stream",
        )

    with torch.inference_mode():
        if isinstance(model, ChatterboxMultilingualTTS):
            wav = model.generate(
                request.input,
                language,
                audio_prompt_path=request.audio_prompt_path,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
                temperature=request.temperature,
                max_new_tokens=request.max_new_tokens,
                max_cache_len=request.max_cache_len,
                repetition_penalty=request.repetition_penalty,
                min_p=request.min_p,
                top_p=request.top_p,
                t3_params=t3_kwargs,
            )
        else:
            wav = model.generate(
                request.input,
                language,
                audio_prompt_path=request.audio_prompt_path,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
                temperature=request.temperature,
                max_new_tokens=request.max_new_tokens,
                max_cache_len=request.max_cache_len,
                repetition_penalty=request.repetition_penalty,
                min_p=request.min_p,
                top_p=request.top_p,
                t3_params=t3_kwargs,
            )

    pcm = _tensor_to_pcm16(wav.squeeze(0))
    encoded = base64.b64encode(pcm).decode("ascii")
    return JSONResponse(
        content={
            "audio": encoded,
            "format": request.response_format,
            "sample_rate": metadata["sample_rate"],
        }
    )
