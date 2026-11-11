"""TTS: единый интерфейс бэкенда + два движка (silero по HTTP, kokoro-ru ONNX).

qwen-TTS (qwentts.cpp) выведен из поставки: его VRAM отдана под модель LLM.

Общие требования (раздел 8 ТЗ):
  * асинхронный поток байтов PCM16 mono + атрибут частоты дискретизации;
  * метод прогрева и описание состояния для /health;
  * TTS НИКОГДА не блокирует чтение токенов LLM (очередь + отдельный воркер);
  * синтез в 16-битный PCM, NaN/Inf санитизируется;
  * неизвестный голос — 400, выключенный TTS — 503.
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
from pathlib import Path
from urllib.parse import urlparse

from . import config

log = logging.getLogger("voice.tts")


class TTSError(Exception):
    def __init__(self, message: str, status: int = 503):
        super().__init__(message)
        self.status = status


class TTSBackend:
    name = "none"
    sample_rate = 24000

    async def stream(self, text: str, voice: str | None = None):
        raise NotImplementedError
        yield b""

    async def warmup(self) -> None:
        return None

    def describe(self) -> dict:
        return {"backend": self.name}

    def voices(self) -> list[str]:
        return []


# ---------------------------------------------------------------------------
# kokoro-ru: ONNX в процессе
# ---------------------------------------------------------------------------
class KokoroTTS(TTSBackend):
    name = "kokoro"
    sample_rate = 24000

    def __init__(self, model_dir: str, voice: str = "sveta",
                 provider: str = "cuda", speed: float = 1.0):
        self.model_dir = Path(model_dir)
        self.voice = voice
        self.provider = provider
        self.speed = speed
        self._session = None
        self._vocab: dict[str, int] = {}
        self._style_cache: dict[str, object] = {}
        self._lock = asyncio.Lock()
        self.loaded_provider: str | None = None

    def available(self) -> bool:
        return (self.model_dir / "model.onnx").is_file() and \
               (self.model_dir / "config.json").is_file()

    @staticmethod
    def _preload_cuda_libs() -> None:
        """onnxruntime-gpu требует системный cuDNN; из pip-пакета nvidia-* он не
        виден автоматически. Подгружаем .so заранее, иначе CUDA-провайдер
        падает с dlopen failed for libcudnn.so."""
        import ctypes
        import glob
        import os
        import sysconfig

        site = sysconfig.get_paths().get("purelib") or ""
        dirs = sorted(glob.glob(os.path.join(site, "nvidia", "*", "lib")))
        if not dirs:
            return
        os.environ["LD_LIBRARY_PATH"] = ":".join(dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        for d in dirs:
            for so in sorted(glob.glob(os.path.join(d, "*.so*"))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass

    def _load(self) -> None:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        if self.provider.lower() == "cuda":
            self._preload_cuda_libs()

        import onnxruntime as ort

        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

        providers: list[str] = []
        avail = ort.get_available_providers()
        if self.provider.lower() == "cuda" and "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        try:
            self._session = ort.InferenceSession(
                str(self.model_dir / "model.onnx"), sess_options=opts, providers=providers)
        except Exception as e:  # noqa: BLE001
            if providers and providers[0] != "CPUExecutionProvider":
                log.warning("kokoro: %s недоступен (%s) — откат на CPU",
                            providers[0], e.__class__.__name__)
                self._session = ort.InferenceSession(
                    str(self.model_dir / "model.onnx"), sess_options=opts,
                    providers=["CPUExecutionProvider"])
            else:
                raise
        self.loaded_provider = self._session.get_providers()[0]

        cfg = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self._vocab = {k: int(v) for k, v in cfg["vocab"].items()}
        log.info("kokoro: провайдер=%s voice=%s", self.loaded_provider, self.voice)
        self._style(self.voice)

    def _style(self, voice: str):
        import numpy as np

        if voice not in self._style_cache:
            p = self.model_dir / f"{voice}.bin"
            if not p.is_file():
                raise TTSError(f"неизвестный голос: {voice}", 400)
            self._style_cache[voice] = np.fromfile(str(p), dtype=np.float32)
        return self._style_cache[voice]

    def voices(self) -> list[str]:
        return sorted(p.stem for p in self.model_dir.glob("*.bin"))

    def describe(self) -> dict:
        return {"backend": "kokoro", "dir": str(self.model_dir), "voice": self.voice,
                "provider": self.loaded_provider or self.provider,
                "model": "model.onnx", "sample_rate": self.sample_rate,
                "speed": self.speed}

    def _synth_pcm(self, text: str, voice: str) -> bytes:
        import numpy as np
        from phonemizer import phonemize

        ps = phonemize(text, language="ru", backend="espeak", strip=True,
                       preserve_punctuation=True, with_stress=False)
        if isinstance(ps, list):
            ps = " ".join(ps)
        ps = re.sub(r"\s+", " ", ps).strip()
        if not ps:
            return b""

        ids = [self._vocab[c] for c in ps if c in self._vocab]
        if not ids:
            log.warning("kokoro: пустая фонемная последовательность для %r", text[:40])
            return b""

        style = self._style(voice)
        n = len(ids)
        idx = min(n, style.shape[0] - 1)
        ref_s = style[idx * 256:(idx + 1) * 256].reshape(1, -1).astype(np.float32)

        inputs = {
            "input_ids": np.array([ids], dtype=np.int64),
            "style": ref_s,
            "speed": np.array([self.speed], dtype=np.float32),
        }
        wave = self._session.run(None, inputs)[0]
        audio = np.asarray(wave, dtype=np.float32).reshape(-1)

        # NaN/Inf санитизация: иначе браузер получает мусор (баг fp16 на Turing)
        if not np.all(np.isfinite(audio)):
            log.warning("kokoro: NaN/Inf в аудио, санитизирую")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    async def stream(self, text: str, voice: str | None = None):
        if self._session is None:
            async with self._lock:
                if self._session is None:
                    await asyncio.get_running_loop().run_in_executor(None, self._load)
        v = voice or self.voice
        self._style(v)  # 400 до начала потока
        loop = asyncio.get_running_loop()
        pcm = await loop.run_in_executor(None, functools.partial(self._synth_pcm, text, v))
        if pcm:
            yield pcm

    async def warmup(self) -> None:
        try:
            async for _ in self.stream("Прогрев синтеза."):
                break
            log.info("kokoro прогрет (%s)", self.loaded_provider)
        except Exception as e:  # noqa: BLE001
            log.warning("kokoro прогрев не удался: %s", e)


class SileroTTS(TTSBackend):
    """Silero v5_6 ru через отдельный HTTP-сервис (torch CPU-only).

    Отдельный процесс обязателен: torch и onnxruntime-gpu несовместимы
    в одном процессе (ТЗ, грабли 15). Протокол — OpenAI-совместимый
    /v1/audio/speech с response_format=pcm; частота 48 кГц отдаётся заголовком.
    """

    name = "silero"

    def __init__(self, url: str, speaker: str = "xenia", timeout: float = 60.0):
        self.url = urlparse(url) if "://" in url else urlparse("http://" + url)
        self.speaker = speaker
        self.timeout = timeout
        self.sample_rate = 48000
        self._voices: list[str] = []

    def voices(self) -> list[str]:
        return self._voices or ["xenia", "aidar", "baya", "kseniya", "eugene"]

    def describe(self) -> dict:
        return {"backend": "silero", "url": self.url.geturl(), "voice": self.speaker,
                "endpoint": "/v1/audio/speech", "sample_rate": self.sample_rate,
                "device": "cpu", "model": "v5_6_ru"}

    def _host_port(self) -> tuple[str, int]:
        return (self.url.hostname or "127.0.0.1", self.url.port or 80)

    async def _open(self):
        host, port = self._host_port()
        try:
            return await asyncio.wait_for(asyncio.open_connection(host, port), timeout=5.0)
        except (OSError, asyncio.TimeoutError) as e:
            raise TTSError("TTS silero недоступен: " + e.__class__.__name__, 503)

    @staticmethod
    def _request_bytes(path: str, host: str, port: int, body: bytes) -> bytes:
        head = [
            "POST " + path + " HTTP/1.1",
            "Host: " + host + ":" + str(port),
            "Content-Type: application/json",
            "Content-Length: " + str(len(body)),
            "Connection: close",
            "",
            "",
        ]
        return "\r\n".join(head).encode() + body

    async def _read_headers(self, reader):
        status_line = await asyncio.wait_for(reader.readline(), timeout=self.timeout)
        parts = status_line.decode(errors="replace").split()
        status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        headers: dict[str, str] = {}
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=self.timeout)
            if line in (b"\r\n", b"\n", b""):
                break
            k, _, v = line.decode(errors="replace").partition(":")
            headers[k.strip().lower()] = v.strip()
        return status, headers

    async def _http(self, path: str, body: bytes):
        reader, writer = await self._open()
        host, port = self._host_port()
        try:
            writer.write(self._request_bytes(path, host, port, body))
            await writer.drain()

            status, headers = await self._read_headers(reader)
            if status != 200:
                raise TTSError("TTS silero вернул " + str(status),
                               400 if status in (400, 404) else 503)

            sr = headers.get("x-sample-rate") or headers.get("x-audio-sample-rate")
            if sr and sr.isdigit():
                self.sample_rate = int(sr)

            chunked = "chunked" in headers.get("transfer-encoding", "").lower()
            if chunked:
                while True:
                    size_line = await asyncio.wait_for(reader.readline(), timeout=self.timeout)
                    try:
                        size = int(size_line.strip().split(b";")[0], 16)
                    except ValueError:
                        break
                    if size == 0:
                        try:
                            await reader.readline()
                        except Exception:
                            pass
                        break
                    data = await asyncio.wait_for(reader.readexactly(size),
                                                  timeout=self.timeout)
                    try:
                        await reader.readline()
                    except Exception:
                        pass
                    if data:
                        yield data
            else:
                while True:
                    data = await asyncio.wait_for(reader.read(8192), timeout=self.timeout)
                    if not data:
                        break
                    yield data
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def stream(self, text: str, voice: str | None = None):
        payload = {"input": text, "voice": voice or self.speaker,
                   "response_format": "pcm"}
        body = json.dumps(payload, ensure_ascii=False).encode()
        async for chunk in self._http("/v1/audio/speech", body):
            yield chunk

    async def warmup(self) -> None:
        # подтягиваем список голосов и греем путь синтеза
        try:
            reader, writer = await self._open()
            host, port = self._host_port()
            req = ("GET /voices HTTP/1.1\r\nHost: " + host + ":" + str(port) +
                   "\r\nConnection: close\r\n\r\n").encode()
            writer.write(req)
            await writer.drain()
            raw = await asyncio.wait_for(reader.read(), timeout=10)
            writer.close()
            body = raw.split(b"\r\n\r\n", 1)[-1]
            data = json.loads(body)
            if isinstance(data.get("voices"), list) and data["voices"]:
                self._voices = data["voices"]
        except Exception:
            pass
        try:
            async for _ in self.stream("Прогрев."):
                break
            log.info("silero прогрет, голоса: %s", self._voices or "по умолчанию")
        except TTSError as e:
            log.warning("silero прогрев не удался: %s", e)


class NoTTS(TTSBackend):
    name = "none"

    def describe(self) -> dict:
        return {"backend": "none", "reason": "TTS выключен"}


# ---------------------------------------------------------------------------
# Реестр бэкендов
# ---------------------------------------------------------------------------
class TTSRegistry:
    def __init__(self):
        self._backends: dict[str, TTSBackend] = {}
        self._errors: dict[str, str] = {}
        self.active = "none"

    def register(self, b: TTSBackend) -> None:
        self._backends[b.name] = b

    def fail(self, name: str, reason: str) -> None:
        self._errors[name] = reason

    def has(self, name: str) -> bool:
        return name in self._backends

    def get(self, name: str) -> TTSBackend:
        if name not in self._backends:
            if name in self._errors:
                raise TTSError(f"TTS «{name}» недоступен: {self._errors[name]}", 503)
            raise TTSError(f"неизвестный TTS-движок: {name}", 400)
        b = self._backends[name]
        if isinstance(b, KokoroTTS) and not b.available():
            raise TTSError("TTS kokoro недоступен: нет модели", 503)
        return b

    def available(self) -> list[dict]:
        out = []
        for name in ("silero", "kokoro"):
            if name in self._backends:
                b = self._backends[name]
                out.append({"name": name, "available": True, "voices": b.voices(),
                            "description": b.describe()})
            else:
                out.append({"name": name, "available": False,
                            "reason": self._errors.get(name, "не сконфигурирован")})
        return out

    async def warmup_all(self) -> None:
        for name, b in self._backends.items():
            try:
                await b.warmup()
            except Exception as e:  # noqa: BLE001
                log.warning("прогрев %s: %s", name, e)


def build_registry() -> TTSRegistry:
    """Собирает доступные движки из окружения. Ничего не грузит на импорте."""
    reg = TTSRegistry()

    kok = KokoroTTS(config.tts_kokoro_dir(), config.tts_kokoro_voice(),
                    config.tts_kokoro_provider(), config.tts_kokoro_speed())
    if kok.available():
        reg.register(kok)
    else:
        reg.fail("kokoro", f"нет модели в {config.tts_kokoro_dir()}")

    surl = config.tts_silero_url()
    if surl and urlparse(surl).hostname:
        reg.register(SileroTTS(surl, config.tts_silero_voice(), config.tts_timeout()))
    else:
        reg.fail("silero", "TTS_SILERO_URL не задан")

    want = config.tts_backend_default()
    if want in reg._backends:
        reg.active = want
    elif "kokoro" in reg._backends:
        reg.active = "kokoro"
    else:
        reg.active = "none"
    log.info("TTS: доступны %s, активен %s",
             list(reg._backends.keys()) or "нет", reg.active)
    return reg


# ---------------------------------------------------------------------------
# Нарезка текста на клаузы (для раннего старта и параллельности)
# ---------------------------------------------------------------------------
_SENT_RE = re.compile(r"[^.!?…]*[.!?…]+")
_SOFT_RE = re.compile(r"[^,;:—–-]*[,;:—–-]+")
# Число, разрезанное границей буфера, нельзя отдавать в синтез:
# «3,» + «5 кг» прочитается как «три» и «пять килограммов».
_DIGIT_TAIL_RE = re.compile(r"[\d.,]\s*$")
_DIGIT_HEAD_RE = re.compile(r"\s*[\d.,]")


def _splits_number(sent: str, rest: str) -> bool:
    """Проверяет, что граница не проходит внутри числа."""
    if not sent or not rest:
        return False
    if _DIGIT_TAIL_RE.search(sent) and _DIGIT_HEAD_RE.match(rest):
        return True
    # «1 000» — пробел внутри разрядов
    if re.search(r"\d\s*$", sent) and re.match(r"\s*\d{3}(?!\d)", rest):
        return True
    return False


def split_clauses(buf: str, final: bool = False) -> tuple[list[str], str]:
    """Возвращает (готовые клаузы, остаток буфера)."""
    out: list[str] = []
    rest = buf

    while True:
        m = _SENT_RE.match(rest)
        if not (m and m.group(0).strip()):
            break
        if final or len(m.group(0)) >= config.tts_min_clause_chars():
            sent, tail = m.group(0).strip(), rest[m.end():]
            # «Цена 1.» — точка может оказаться десятичным разделителем
            if not final and _splits_number(sent, tail):
                break
            out.append(sent)
            rest = tail
            continue
        break

    # ранний старт: рвём по запятой, если уже накопилось достаточно символов
    if len(rest) >= config.tts_soft_break_chars():
        m = _SOFT_RE.match(rest)
        if m and m.group(0).strip():
            sent, tail = m.group(0).strip(), rest[m.end():]
            if not _splits_number(sent, tail):
                out.append(sent)
                rest = tail

    # принудительная нарезка на длинных кусках
    while len(rest) > config.tts_max_clause_chars():
        cut = rest.rfind(" ", 0, config.tts_max_clause_chars())
        if cut < config.tts_min_clause_chars():
            cut = config.tts_max_clause_chars()
        sent, tail = rest[:cut].strip(), rest[cut:]
        if _splits_number(sent, tail):
            # сдвигаем границу до конца числа
            nxt = re.search(r"\S+", tail)
            if nxt:
                sent = rest[:cut + nxt.end()].strip()
                tail = rest[cut + nxt.end():]
        out.append(sent)
        rest = tail

    if final and rest.strip():
        out.append(rest.strip())
        rest = ""

    return [c for c in out if c], rest.lstrip()
