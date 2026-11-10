"""Инструменты агента: MCP-поиск wormsoft + локальные (время, калькулятор, таймер).

Требования раздела 6 ТЗ:
  * токен читается ЛЕНИВО (в момент первого вызова), не на импорте;
  * .env ищется: явный путь -> корень проекта -> рядом с приложением;
  * если токена нет — внятный текст с ПЕРЕЧНЕМ проверенных путей, без падения;
  * токен никогда не попадает в логи/ответы/health — только {present, source};
  * ошибки сети/HTTP -> короткий текст результата, без стектрейсов и тела ответа;
  * результаты сжимаются: без markdown-таблиц и длинных URL внутри фразы.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from . import config

log = logging.getLogger("voice.tools")

_token_cache: dict[str, object] = {"loaded": False, "value": "", "source": None}


# ---------------------------------------------------------------------------
# Токен: ленивое чтение
# ---------------------------------------------------------------------------
def _token_paths() -> list[Path]:
    paths: list[Path] = []
    explicit = os.environ.get("WORM_TOKEN_FILE", "").strip()
    if explicit:
        paths.append(Path(explicit).expanduser())
    paths.append(Path(str(config.worm_token_file())).expanduser())
    paths.append(Path.home() / ".wormsoft-token")
    here = Path(__file__).resolve().parent
    paths.append(here.parent / ".wormsoft-token")
    # уникализировать, сохранив порядок
    seen, out = set(), []
    for p in paths:
        s = str(p)
        if s not in seen:
            seen.add(s)
            out.append(p)
    return out


def _read_token() -> tuple[str, str | None]:
    """Возвращает (токен, источник). Источник — 'WORM_TOKEN' или путь к файлу."""
    if _token_cache["loaded"]:
        return str(_token_cache["value"]), _token_cache["source"]  # type: ignore[return-value]

    tok = os.environ.get("WORM_TOKEN", "").strip()
    source: str | None = "WORM_TOKEN" if tok else None
    if not tok:
        for p in _token_paths():
            try:
                if not p.is_file():
                    continue
                # utf-8-sig: BOM; strip: CRLF-хвост из Windows -> 401
                val = p.read_text(encoding="utf-8-sig").strip().replace("\r", "").replace("\n", "")
                if val:
                    tok = val
                    source = str(p)
                    break
            except OSError:
                continue

    _token_cache.update({"loaded": True, "value": tok, "source": source})
    if tok:
        log.info("токен wormsoft найден, источник=%s, длина=%d", source, len(tok))
    else:
        log.warning("токен wormsoft не найден; проверено: %s",
                    ", ".join(str(p) for p in _token_paths()))
    return tok, source


def token_status() -> dict:
    """Для /health: только факт наличия и источник, без значения."""
    tok, source = _read_token()
    return {"present": bool(tok), "source": source}


def reset_token_cache() -> None:
    _token_cache.update({"loaded": False, "value": "", "source": None})


# ---------------------------------------------------------------------------
# MCP JSON-RPC
# ---------------------------------------------------------------------------
class MCPError(RuntimeError):
    pass


def mcp_call(name: str, arguments: dict) -> str:
    """Прямой tools/call, один запрос — один JSON-ответ (без initialize/SSE)."""
    tok, _ = _read_token()
    if not tok:
        checked = "\n".join(f"  - {p}" for p in _token_paths())
        return ("поиск недоступен: не найден токен доступа. Проверены пути:\n" + checked +
                "\nа также переменная окружения WORM_TOKEN."
                "\nОтветить пользователю честно: данные из интернета получить нельзя.")

    body = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }, ensure_ascii=False).encode()

    req = urllib.request.Request(
        config.worm_mcp_url(), data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + tok,
                 "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=config.worm_mcp_timeout()) as r:
            payload = json.loads(r.read())
    except urllib.error.HTTPError as e:
        # тело ответа не показываем — там может быть чувствительное
        hint = {401: "токен не принят (401)", 403: "доступ запрещён (403)",
                429: "слишком много запросов (429)"}.get(e.code, f"HTTP {e.code}")
        return f"поиск недоступен: {hint}"
    except urllib.error.URLError as e:
        return f"поиск недоступен: сеть недоступна ({e.reason.__class__.__name__})"
    except TimeoutError:
        return "поиск недоступен: истёк таймаут запроса"
    except (json.JSONDecodeError, OSError) as e:
        return f"поиск недоступен: {e.__class__.__name__}"

    if "error" in payload:
        return f"поиск недоступен: ошибка протокола ({payload['error'].get('code', '?')})"

    result = payload.get("result") or {}
    if result.get("isError"):
        return "поиск недоступен: сервер вернул ошибку инструмента"

    parts = []
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "\n".join(parts) if parts else "поиск недоступен: пустой ответ"


# ---------------------------------------------------------------------------
# Сжатие результата для промпта
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+")
_WS_RE = re.compile(r"[ \t]+")


def compress(text: str, limit: int = 900, max_fragments: int = 4) -> str:
    """Убрать дубли пробелов, markdown-таблицы, длинные URL; обрезать по лимиту."""
    if not text:
        return ""

    # если это JSON со списком результатов — разобрать и оставить выжимки
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        data = None

    frags: list[str] = []
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        for r in data["results"][:max_fragments]:
            title = str(r.get("title", "")).strip()
            snippet = str(r.get("content", "") or r.get("snippet", "")).strip()
            url = str(r.get("url", "")).strip()
            host = ""
            if url:
                m = re.match(r"https?://([^/]+)", url)
                host = m.group(1) if m else ""
            line = f"— {title}" if title else "—"
            if snippet:
                line += ": " + snippet
            if host:
                line += f" (источник: {host})"
            frags.append(line)
    else:
        frags = [text]

    out: list[str] = []
    for f in frags:
        # markdown-таблицы не годятся для озвучки
        lines = [ln for ln in f.splitlines() if not ln.strip().startswith("|")]
        f = " ".join(lines)
        f = _URL_RE.sub("", f)          # URL внутри фразы не проговариваем
        f = _WS_RE.sub(" ", f)
        f = re.sub(r"\s*\n\s*", " ", f).strip()
        if f:
            out.append(f)

    res = "\n".join(out)
    if len(res) > limit:
        res = res[:limit].rsplit(" ", 1)[0] + "…"
    return res


# ---------------------------------------------------------------------------
# Реализации инструментов
# ---------------------------------------------------------------------------
def web_search(query: str, max_results: int = 5) -> str:
    raw = mcp_call("web_search", {"query": query, "max_results": int(max_results)})
    if raw.startswith("поиск недоступен"):
        return raw
    return compress(raw)


def web_fetch(url: str) -> str:
    raw = mcp_call("web_fetch", {"url": url})
    if raw.startswith("поиск недоступен"):
        return raw
    return compress(raw, limit=1500, max_fragments=1)


def current_time() -> str:
    now = datetime.now().astimezone()
    days = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
    months = ["января", "февраля", "марта", "апреля", "мая", "июня",
              "июля", "августа", "сентября", "октября", "ноября", "декабря"]
    return (f"Сейчас {now.strftime('%H:%M')}, {days[now.weekday()]}, "
            f"{now.day} {months[now.month - 1]} {now.year} года, "
            f"часовой пояс {now.strftime('%Z') or 'локальный'}.")


_SAFE_EXPR = re.compile(r"^[0-9+\-*/().,%\s^]+$")


def calculator(expression: str) -> str:
    expr = expression.strip().replace("^", "**").replace(",", ".")
    if not expr or not _SAFE_EXPR.match(expression.replace("^", "").replace(",", "")):
        return "калькулятор: допустимы только числа и операции + - * / ( ) % ^"
    try:
        val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — вход отфильтрован
    except Exception as e:  # noqa: BLE001
        return f"калькулятор: не удалось вычислить ({e.__class__.__name__})"
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    return f"{expression.strip()} = {val}"


_timers: dict[str, float] = {}


def set_timer(seconds: int, label: str = "") -> str:
    seconds = max(1, min(int(seconds), 3600))
    key = label.strip() or f"таймер-{len(_timers) + 1}"
    _timers[key] = time.monotonic() + seconds
    return f"Таймер «{key}» установлен на {seconds} секунд."


def check_timer(label: str = "") -> str:
    if not _timers:
        return "Активных таймеров нет."
    if label.strip():
        t = _timers.get(label.strip())
        if t is None:
            return f"Таймер «{label}» не найден."
        left = t - time.monotonic()
        return (f"Таймер «{label}» сработал." if left <= 0
                else f"До конца таймера «{label}» {int(left)} секунд.")
    parts = []
    for k, t in _timers.items():
        left = t - time.monotonic()
        parts.append(f"«{k}»: " + ("сработал" if left <= 0 else f"{int(left)} с"))
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Описания в OpenAI-подобном формате (для демона)
# ---------------------------------------------------------------------------
TOOL_SPECS: list[dict] = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Поиск информации в интернете. Используй для свежих данных: "
                       "курсы валют и криптовалют, новости, погода, цены, события.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "поисковый запрос"},
            "max_results": {"type": "integer", "description": "сколько результатов, по умолчанию 5"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "web_fetch",
        "description": "Загрузить текст веб-страницы по адресу и прочитать его.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "полный адрес страницы"},
        }, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "current_time",
        "description": "Текущие дата, время и день недели. Используй, когда спрашивают "
                       "который час, какое сегодня число или день недели.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Вычислить арифметическое выражение.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "например 2*(3+4)"},
        }, "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "set_timer",
        "description": "Поставить таймер на заданное число секунд.",
        "parameters": {"type": "object", "properties": {
            "seconds": {"type": "integer", "description": "длительность в секундах"},
            "label": {"type": "string", "description": "название таймера"},
        }, "required": ["seconds"]}}},
    {"type": "function", "function": {
        "name": "check_timer",
        "description": "Проверить, сколько осталось до таймера или сработал ли он.",
        "parameters": {"type": "object", "properties": {
            "label": {"type": "string", "description": "название таймера"},
        }}}},
]

_REGISTRY = {
    "web_search": lambda a: web_search(a.get("query", ""), a.get("max_results", 5)),
    "web_fetch": lambda a: web_fetch(a.get("url", "")),
    "current_time": lambda a: current_time(),
    "calculator": lambda a: calculator(str(a.get("expression", ""))),
    "set_timer": lambda a: set_timer(a.get("seconds", 0), a.get("label", "")),
    "check_timer": lambda a: check_timer(a.get("label", "")),
}


def specs() -> list[dict]:
    return TOOL_SPECS if config.tools_enabled() else []


def execute(name: str, arguments: dict) -> str:
    fn = _REGISTRY.get(name)
    if fn is None:
        return f"инструмент {name} не поддерживается"
    try:
        return fn(arguments or {})
    except Exception as e:  # noqa: BLE001
        log.warning("tool %s: %s", name, e)
        return f"инструмент {name} завершился ошибкой ({e.__class__.__name__})"
