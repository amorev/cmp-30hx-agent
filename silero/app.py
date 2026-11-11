"""Silero TTS v5_6 (русский, диктор xenia) — отдельный HTTP-сервис.

Почему отдельным процессом: torch и onnxruntime-gpu в одном процессе
конфликтуют по CUDA-контекстам и дают segfault (ТЗ, п.15.15). Здесь torch
строго CPU-only и живёт в своём venv (`.venv-silero`).

Протокол повторяет qwen-сервис, чтобы voice-сервис работал с ними одинаково:

  GET  /health                 -> {"status": ..., "ready": bool, "voices": [...]}
  GET  /voices                 -> {"voices": [...]}
  POST /tts                    -> JSON {"text","voice"} -> WAV (base64 не нужен)
  POST /v1/audio/speech        -> OpenAI-совместимо: response_format=pcm
                                  -> s16le mono 16-bit потоком

Стриминга внутри модели нет (Silero синтезирует клаузу целиком), поэтому поток
эмулируется нарезкой по предложениям: первая клауза отдаётся сразу, остальные
по мере синтеза. Частота дискретизации v5_6 — 48 кГц.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import struct
import time
import wave
from typing import Iterable

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

try:                                  # запуск как пакета: uvicorn silero.app:app
    from .normalize import normalize as normalize_text
except ImportError:                   # запуск как скрипта: python silero/app.py
    from normalize import normalize as normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("silero-tts")

MODEL_PATH = os.environ.get("SILERO_MODEL", os.path.expanduser("~/cmp-voice/models/silero/v5_6_ru.pt"))
DEFAULT_VOICE = os.environ.get("SILERO_VOICE", "xenia")
SAMPLE_RATE = int(os.environ.get("SILERO_SR", "48000"))
THREADS = int(os.environ.get("SILERO_THREADS", "3"))
HOST = os.environ.get("SILERO_HOST", "127.0.0.1")
PORT = int(os.environ.get("SILERO_PORT", "8023"))

# Ограничение длины клаузы: длинные куски синтезируются дольше, а звук нужен раньше
MAX_CLAUSE = int(os.environ.get("SILERO_MAX_CLAUSE", "220"))
MIN_CLAUSE = int(os.environ.get("SILERO_MIN_CLAUSE", "12"))

# Нормализация текста: в токенсете модели нет цифр и латиницы, поэтому
# без неё «в 2024 году» читается как «в году». Отключается SILERO_NORMALIZE=0.
NORMALIZE = os.environ.get("SILERO_NORMALIZE", "1").strip().lower() not in (
    "0", "false", "no", "off")
NORMALIZE_LATIN = os.environ.get("SILERO_NORMALIZE_LATIN", "1").strip().lower() not in (
    "0", "false", "no", "off")

app = FastAPI(title="Silero TTS v5_6 ru")

_state: dict = {"model": None, "voices": [], "ready": False, "error": None}
_sem = asyncio.Semaphore(1)          # модель не потокобезопасна
_lock = asyncio.Lock()


def _prepare(text: str) -> str:
    """Нормализация перед синтезом (только для Silero).

    Делается до нарезки на клаузы: числа разворачиваются в слова, и длина
    клаузы должна считаться уже по развёрнутому тексту, иначе клауза
    окажется вдвое длиннее лимита.
    """
    if not NORMALIZE:
        return text
    try:
        return normalize_text(text, latin=NORMALIZE_LATIN)
    except Exception as e:  # noqa: BLE001
        # нормализация не должна ронять синтез: отдаём текст как есть
        log.warning("нормализация не удалась (%s: %s) — синтез без неё",
                    e.__class__.__name__, e)
        return text


def _split(text: str) -> list[str]:
    """Нарезка на клаузы: по границам предложений, длинные — по запятой."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…])\s+", text)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) <= MAX_CLAUSE:
            if len(p) >= MIN_CLAUSE or not out:
                out.append(p)
            else:
                out[-1] = (out[-1] + " " + p).strip()
            continue
        # длинное предложение — режем по запятым, затем по словам
        for chunk in re.split(r"(?<=[,;:—–])\s+", p):
            chunk = chunk.strip()
            while len(chunk) > MAX_CLAUSE:
                cut = chunk.rfind(" ", 0, MAX_CLAUSE)
                if cut < MIN_CLAUSE:
                    cut = MAX_CLAUSE
                out.append(chunk[:cut].strip())
                chunk = chunk[cut:].strip()
            if chunk:
                out.append(chunk)
    return out


def _load() -> None:
    t0 = time.monotonic()
    torch.set_num_threads(THREADS)
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"нет модели Silero: {MODEL_PATH}")
    model = torch.package.PackageImporter(MODEL_PATH).load_pickle("tts_models", "model")
    model.to(torch.device("cpu"))
    voices = list(getattr(model, "speakers", []) or [])
    _state.update({"model": model, "voices": voices, "ready": True, "error": None})
    log.info("silero загружен за %.1f с, голоса: %s", time.monotonic() - t0, voices)


def _synth(text: str, voice: str) -> bytes:
    """Возвращает сырой PCM s16le mono."""
    model = _state["model"]
    if model is None:
        raise RuntimeError("модель не загружена")
    audio = model.apply_tts(text=text, speaker=voice, sample_rate=SAMPLE_RATE)
    # NaN/Inf санитизация (иначе браузер получает мусор)
    audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(audio.abs().max()) if audio.numel() else 0.0
    if peak > 1.0:
        audio = audio / peak
    audio = audio.clamp(-1.0, 1.0)
    pcm = (audio * 32767.0).to(torch.int16).numpy().tobytes()
    return pcm


class SpeechIn(BaseModel):
    input: str
    voice: str | None = None
    response_format: str | None = "pcm"
    language: str | None = None


class TTSIn(BaseModel):
    text: str
    voice: str | None = None


@app.on_event("startup")
async def _startup() -> None:
    try:
        await asyncio.get_running_loop().run_in_executor(None, _load)
    except Exception as e:  # noqa: BLE001
        _state["error"] = f"{e.__class__.__name__}: {e}"
        log.error("silero не загрузился: %s", e)


@app.get("/health")
async def health():
    return {
        "status": "ok" if _state["ready"] else "degraded",
        "engine": "silero",
        "model": os.path.basename(MODEL_PATH),
        "ready": _state["ready"],
        "error": _state["error"],
        "voice": DEFAULT_VOICE,
        "voices": _state["voices"],
        "sample_rate": SAMPLE_RATE,
        "device": "cpu",
        "normalize": NORMALIZE,
        "normalize_latin": NORMALIZE_LATIN,
    }


@app.get("/voices")
async def voices():
    return {"voices": _state["voices"], "default": DEFAULT_VOICE}


def _check_voice(voice: str) -> str | None:
    v = (voice or DEFAULT_VOICE).strip().lower()
    if _state["voices"] and v not in _state["voices"]:
        return None
    return v


@app.post("/v1/audio/speech")
async def speech(body: SpeechIn):
    """OpenAI-совместимый эндпоинт: pcm -> s16le поток, wav -> файл."""
    if not _state["ready"]:
        return JSONResponse({"error": "модель не загружена"}, status_code=503)
    text = (body.input or "").strip()
    if not text:
        return JSONResponse({"error": "пустой input"}, status_code=400)
    voice = _check_voice(body.voice or "")
    if voice is None:
        return JSONResponse({"error": f"неизвестный голос: {body.voice}"}, status_code=400)

    fmt = (body.response_format or "pcm").lower()
    clauses = _split(_prepare(text))
    loop = asyncio.get_running_loop()

    if fmt == "wav":
        async with _sem:
            pcm = b"".join([await loop.run_in_executor(None, _synth, c, voice) for c in clauses])
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)
        return Response(content=buf.getvalue(), media_type="audio/wav")

    async def gen() -> Iterable[bytes]:
        for c in clauses:
            async with _sem:
                pcm = await loop.run_in_executor(None, _synth, c, voice)
            if pcm:
                yield pcm

    return StreamingResponse(
        gen(), media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(SAMPLE_RATE), "X-Engine": "silero"})


@app.post("/tts")
async def tts_raw(body: TTSIn):
    """Прямой замер: сырой PCM целиком."""
    if not _state["ready"]:
        return JSONResponse({"error": "модель не загружена"}, status_code=503)
    text = (body.text or "").strip()
    if not text:
        return JSONResponse({"error": "пустой text"}, status_code=400)
    voice = _check_voice(body.voice or "")
    if voice is None:
        return JSONResponse({"error": f"неизвестный голос: {body.voice}"}, status_code=400)
    loop = asyncio.get_running_loop()
    out = []
    for c in _split(_prepare(text)):
        async with _sem:
            out.append(await loop.run_in_executor(None, _synth, c, voice))
    return Response(content=b"".join(out), media_type="application/octet-stream",
                    headers={"X-Sample-Rate": str(SAMPLE_RATE)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
