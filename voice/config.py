"""Конфигурация: свой парсер .env без внешних зависимостей.

Порядок поиска .env: явный путь из VOICE_ENV_FILE -> .env в корне проекта ->
.env рядом с приложением. Уже заданные переменные НЕ перезаписываются
(приоритет у процесса/юнита).

ВАЖНО: значения читаются ЛЕНИВО (через функций), а не на импорте модуля.
Иначе результат зависит от порядка импортов и от того, успел ли загрузиться
.env — известные грабли (11a в ТЗ).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("voice.config")


def parse_env_file(path: Path) -> dict[str, str]:
    """Читает .env: UTF-8 с BOM, export, кавычки, комментарии, CRLF."""
    out: dict[str, str] = {}
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return out
    except OSError as e:
        log.warning("не читается %s: %s", path, e)
        return out

    for line in raw.splitlines():
        # CRLF: \r остаётся в значении и даёт невидимый хвост (401 на токене)
        line = line.replace("\r", "").strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        else:
            for sep in (" #", "\t#"):
                if sep in val:
                    val = val.split(sep, 1)[0].rstrip()
                    break
        out[key] = val
    return out


def candidate_env_files() -> list[Path]:
    files: list[Path] = []
    explicit = os.environ.get("VOICE_ENV_FILE", "").strip()
    if explicit:
        files.append(Path(explicit).expanduser())
    here = Path(__file__).resolve().parent
    files.append(here.parent / ".env")   # корень проекта
    files.append(here / ".env")          # рядом с приложением
    return files


_loaded = False


def load_env(force: bool = False) -> list[Path]:
    """Подмешивает .env в os.environ, не перезаписывая уже заданное."""
    global _loaded
    if _loaded and not force:
        return []
    _loaded = True
    used: list[Path] = []
    for f in candidate_env_files():
        if not f.is_file():
            continue
        vals = parse_env_file(f)
        applied = 0
        for k, v in vals.items():
            if k in os.environ and os.environ[k] != "":
                continue
            os.environ[k] = v
            applied += 1
        used.append(f)
        log.info("env: прочитан %s (%d переменных, применено %d)", f, len(vals), applied)
    if not used:
        log.info("env: файлы .env не найдены (проверено: %s)",
                 ", ".join(str(p) for p in candidate_env_files()))
    return used


# ---------------------------------------------------------------------------
# Ленивые геттеры
# ---------------------------------------------------------------------------
def s(name: str, default: str) -> str:
    load_env()
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def i(name: str, default: int) -> int:
    try:
        return int(s(name, str(default)))
    except ValueError:
        return default


def f(name: str, default: float) -> float:
    try:
        return float(s(name, str(default)))
    except ValueError:
        return default


def b(name: str, default: bool) -> bool:
    v = s(name, "").strip().lower()
    if not v:
        return default
    return v not in ("0", "false", "no", "off", "none")


HOME = Path.home()


# --- LLM-демон ---
def llm_host() -> str: return s("LLM_HOST", "127.0.0.1")
def llm_port() -> int: return i("LLM_PORT", 18521)
def llm_timeout_read() -> float: return f("LLM_TIMEOUT_READ", 120.0)
def llm_timeout_connect() -> float: return f("LLM_TIMEOUT_CONNECT", 5.0)


# --- Voice ---
def voice_host() -> str: return s("VOICE_HOST", "127.0.0.1")
def voice_port() -> int: return i("VOICE_PORT", 8110)
def max_gen_tokens() -> int: return i("VOICE_MAX_GEN_TOKENS", 200)
def max_tool_rounds() -> int: return i("VOICE_MAX_TOOL_ROUNDS", 3)
def tools_enabled() -> bool: return b("VOICE_TOOLS_ENABLED", True)


# --- TTS ---
def tts_backend_default() -> str: return s("TTS_BACKEND", "silero")
def tts_timeout() -> float: return f("TTS_TIMEOUT", 60.0)
def tts_kokoro_dir() -> str: return s("TTS_KOKORO_DIR", str(HOME / "models" / "kokoro-ru"))
def tts_kokoro_voice() -> str: return s("TTS_KOKORO_VOICE", "sveta")
def tts_kokoro_provider() -> str: return s("TTS_KOKORO_PROVIDER", "cuda")
def tts_kokoro_speed() -> float: return f("TTS_KOKORO_SPEED", 1.0)
def tts_silero_url() -> str:
    return s("TTS_SILERO_URL", "http://127.0.0.1:8023")


def tts_silero_voice() -> str:
    return s("TTS_SILERO_VOICE", "xenia")


def tts_min_clause_chars() -> int: return i("TTS_MIN_CLAUSE_CHARS", 12)
def tts_soft_break_chars() -> int: return i("TTS_SOFT_BREAK_CHARS", 60)
def tts_max_clause_chars() -> int: return i("TTS_MAX_CLAUSE_CHARS", 210)


# --- ASR ---
def asr_model_dir() -> str:
    return s("ASR_MODEL_DIR",
             str(HOME / "tone_asr" / "sherpa-onnx-streaming-t-one-russian-2025-09-08"))


def asr_mode() -> str: return s("ASR_MODE", "server")
def asr_num_threads() -> int: return i("ASR_NUM_THREADS", 2)
def asr_silence_final() -> float: return f("ASR_SILENCE_FINAL", 2.4)
def asr_silence_after_text() -> float: return f("ASR_SILENCE_AFTER_TEXT", 1.2)
def asr_max_utterance() -> float: return f("ASR_MAX_UTTERANCE", 20.0)
def asr_lead_silence() -> float: return f("ASR_LEAD_SILENCE", 0.75)
def asr_tail_silence() -> float: return f("ASR_TAIL_SILENCE", 1.0)


# --- MCP / инструменты ---
def worm_mcp_url() -> str: return s("WORM_MCP_URL", "https://ai.wormsoft.ru/api/mcp")
def worm_mcp_timeout() -> float: return f("WORM_MCP_TIMEOUT", 45.0)
def worm_token_file() -> str: return s("WORM_TOKEN_FILE", str(HOME / ".wormsoft-token"))
