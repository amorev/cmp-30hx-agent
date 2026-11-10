"""Клиент LLM-демона: свой текстовый протокол поверх TCP.

Обязательные свойства (раздел 4 ТЗ):
  * таймаут на чтение + переподключение при обрыве;
  * идемпотентные команды (prefill/stat/reset/set_tools) можно повторить один раз;
  * generate НЕ повторяется никогда — часть ответа уже ушла в TTS;
  * prefill вызывается по схеме «последний выигрывает».
"""
from __future__ import annotations

import asyncio
import json
import logging

log = logging.getLogger("voice.llm")


class LLMError(RuntimeError):
    pass


class NativeLLM:
    def __init__(self, host: str, port: int, timeout_read: float = 120.0,
                 timeout_connect: float = 5.0):
        self.host = host
        self.port = port
        self.timeout_read = timeout_read
        self.timeout_connect = timeout_connect
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

    # -- соединение -------------------------------------------------------
    async def _connect(self) -> None:
        if self._writer is not None and not self._writer.is_closing():
            return
        log.info("подключаюсь к демону %s:%s", self.host, self.port)
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port), timeout=self.timeout_connect)
        log.info("демон подключён")

    async def _drop(self) -> None:
        w, self._writer = self._writer, None
        self._reader = None
        if w is not None:
            try:
                w.close()
                await asyncio.wait_for(w.wait_closed(), timeout=1.0)
            except Exception:
                pass

    async def close(self) -> None:
        await self._drop()

    async def _request(self, payload: dict, retry: bool = True) -> dict:
        """Одна команда -> одна JSON-строка."""
        last: Exception | None = None
        for attempt in (0, 1):
            if attempt and not retry:
                break
            try:
                async with self._lock:
                    await self._connect()
                    assert self._writer is not None and self._reader is not None
                    self._writer.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
                    await self._writer.drain()
                    line = await asyncio.wait_for(self._reader.readline(),
                                                  timeout=self.timeout_read)
                if not line:
                    raise LLMError("демон закрыл соединение")
                return json.loads(line)
            except asyncio.CancelledError:
                # Отмена посреди обмена оставляет сокет в неопределённом
                # состоянии (команда ушла, ответ не прочитан) — рвём его,
                # иначе следующий ответ достанется чужой команде.
                await self._drop()
                raise
            except (asyncio.TimeoutError, ConnectionError, OSError, LLMError) as e:
                last = e
                log.warning("команда %s не прошла (%s), сбрасываю сокет",
                            payload.get("cmd"), e.__class__.__name__)
                await self._drop()
                if attempt == 0:
                    await asyncio.sleep(0.2)
                    continue
        raise LLMError(f"демон недоступен: {last}")

    # -- команды ----------------------------------------------------------
    async def ping(self) -> bool:
        try:
            r = await self._request({"cmd": "ping"}, retry=True)
            return bool(r.get("pong"))
        except LLMError:
            return False

    async def stat(self) -> dict:
        return await self._request({"cmd": "stat"}, retry=True)

    async def prefill(self, text: str, history: list[dict] | None = None) -> dict:
        return await self._request(
            {"cmd": "prefill", "text": text, "history": history or []}, retry=True)

    async def reset(self) -> dict:
        return await self._request({"cmd": "reset"}, retry=True)

    async def set_tools(self, tools: list[dict]) -> dict:
        return await self._request({"cmd": "set_tools", "tools": tools}, retry=True)

    async def generate(self, text: str, history: list[dict] | None,
                       max_tokens: int, on_token, is_continue: bool = False) -> dict:
        """Стрим токенов. Повтор запрещён: часть ответа уже ушла клиенту."""
        async with self._lock:
            await self._connect()
            assert self._writer is not None and self._reader is not None
            payload = {"cmd": "generate", "text": text,
                       "history": history or [], "max_tokens": max_tokens,
                       "continue": is_continue}
            self._writer.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
            await self._writer.drain()

            while True:
                line = await asyncio.wait_for(self._reader.readline(),
                                              timeout=self.timeout_read)
                if not line:
                    await self._drop()
                    raise LLMError("демон закрыл соединение во время генерации")
                msg = json.loads(line)
                t = msg.get("type")
                if t == "token":
                    await on_token(msg.get("text", ""))
                elif t == "done":
                    return msg
                elif t == "error":
                    raise LLMError(f"демон: {msg.get('error')}")


class LastWinsPrefill:
    """«Последний выигрывает» — без потери прогресса кэша.

    Важная тонкость. Наивная реализация отменяет текущий prefill при приходе
    нового партиала. Но отмена на клиенте НЕ останавливает демон: он
    досчитывает начатое, а новый запрос встаёт в очередь. В результате при
    потоке партиалов каждые 250–300 мс кэш не продвигается вообще —
    ни один prefill не успевает финишировать (проверено: 11 партиалов →
    1 строка KV).

    Поэтому здесь схема слияния: один prefill в работе + не более одного
    отложенного (самый свежий). Когда текущий завершается, сразу запускается
    отложенный. Кэш монотонно догоняет речь, ничего не отменяется.
    """

    def __init__(self, llm: NativeLLM):
        self.llm = llm
        self._task: asyncio.Task | None = None
        self._pending: tuple[str, list[dict], object] | None = None
        self._seq = 0
        self.n_submitted = 0
        self.n_done = 0
        self.n_coalesced = 0

    def submit(self, text: str, history: list[dict], on_done=None) -> None:
        self._seq += 1
        self.n_submitted += 1
        if self._task is not None and not self._task.done():
            # уже считается — запомнить только самый свежий, остальные не нужны
            self._pending = (text, history, on_done)
            self.n_coalesced += 1
            return
        self._task = asyncio.create_task(self._run(text, history, on_done))

    async def _run(self, text, history, on_done):
        try:
            r = await self.llm.prefill(text, history)
            self.n_done += 1
            if on_done is not None:
                await on_done(r)
        except asyncio.CancelledError:
            raise
        except LLMError as e:
            log.warning("prefill: %s", e)
        finally:
            # сразу забрать отложенный — не теряя уже посчитанный прогресс
            if self._pending is not None:
                nxt = self._pending
                self._pending = None
                self._task = asyncio.create_task(self._run(*nxt))

    async def cancel(self) -> None:
        """Полная остановка (сброс/закрытие сеанса)."""
        self._pending = None
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None

    async def wait(self) -> None:
        """Дождаться, пока кэш догонит последний отправленный партиал."""
        for _ in range(400):
            if self._task is None or self._task.done():
                if self._pending is None:
                    return
            await asyncio.sleep(0.02)
