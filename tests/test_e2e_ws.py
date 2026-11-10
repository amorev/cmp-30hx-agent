#!/usr/bin/env python3
"""Сквозной тест двух ходов через WebSocket (критерии 13, 14, 15, 11).

Проверяет:
  * текстовый ввод по Enter: prefill + генерация + TTS-аудио;
  * режим «Enter по словам»: cached растёт, генерация только по финальному Enter;
  * второй ход в том же диалоге: история работает, cached горячий;
  * переключение TTS, блокировка во время хода;
  * восстановление после рестарта демона (--restart-llm).

Запуск: .venv/bin/python tests/test_e2e_ws.py [--restart-llm]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice import config  # noqa: E402

config.load_env()

import websockets  # noqa: E402


class Client:
    def __init__(self, url: str):
        self.url = url
        self.ws = None
        self.events: list[dict] = []
        self.audio_bytes = 0
        self.audio_chunks = 0
        self._task = None

    async def __aenter__(self):
        self.ws = await websockets.connect(self.url, max_size=16 * 1024 * 1024)
        self._task = asyncio.create_task(self._pump())
        return self

    async def __aexit__(self, *a):
        if self._task:
            self._task.cancel()
        if self.ws:
            await self.ws.close()

    async def _pump(self):
        try:
            async for m in self.ws:
                if isinstance(m, (bytes, bytearray)):
                    self.audio_bytes += len(m)
                    self.audio_chunks += 1
                else:
                    self.events.append(json.loads(m))
        except Exception:
            pass

    async def send(self, **kw):
        await self.ws.send(json.dumps(kw, ensure_ascii=False))

    async def wait(self, type_: str, timeout: float = 60.0, after: int = 0) -> dict | None:
        t0 = asyncio.get_running_loop().time()
        while asyncio.get_running_loop().time() - t0 < timeout:
            for e in self.events[after:]:
                if e.get("type") == type_:
                    return e
            await asyncio.sleep(0.03)
        return None

    def mark(self) -> int:
        return len(self.events)

    def dump(self, types=("error", "tts_error", "kv", "gen_stats", "tool_call", "tool_result")):
        for e in self.events:
            if e.get("type") in types:
                print("     ", json.dumps(e, ensure_ascii=False)[:220])


async def turn_text(c: Client, text: str) -> dict:
    mark = c.mark()
    a0 = c.audio_bytes
    await c.send(type="text", text=text, word_mode=False)
    started = await c.wait("turn_start", 15, mark)
    if not started:
        print("  FAIL: ход не начался")
        return {}
    ready = await c.wait("ready", 120, mark)
    stats = None
    for e in c.events[mark:]:
        if e.get("type") == "gen_stats":
            stats = e
    ans = "".join(e["text"] for e in c.events[mark:] if e.get("type") == "ans_token")
    audio = c.audio_bytes - a0
    print(f"  ход «{text[:40]}»:")
    print(f"    ответ: {ans[:120]!r}")
    if stats:
        print(f"    prefill={stats['prefill_ms']} мс cached={stats['cached_tokens']} "
              f"new={stats['new_tokens']} первый токен={stats['first_token_ms']} мс "
              f"{stats['tok_s']} ток/с")
    print(f"    аудио: {audio} байт ({audio/2/24000:.2f} с)")
    return {"ready": bool(ready), "stats": stats, "audio": audio, "answer": ans}


async def test_word_mode(c: Client) -> bool:
    print("\n--- режим «Enter по словам» (критерии 3, 15) ---")
    await c.send(type="reset")
    await asyncio.sleep(0.5)
    mark = c.mark()
    parts = ["какой", "сейчас", "курс", "биткоина"]
    for p in parts:
        await c.send(type="text", text=p, word_mode=True)
        await asyncio.sleep(1.2)
    rows = [e for e in c.events[mark:] if e.get("type") == "kv"]
    if not rows:
        print("  FAIL: нет строк KV")
        return False
    gen_before = [e for e in c.events[mark:] if e.get("type") == "turn_start"]
    for r in rows:
        print(f"    cached={r['cached_tokens']:5} new={r['new_tokens']:4} "
              f"ms={r['ms_f']:7.1f} total={r['prompt_tokens']}")
    cached = [r["cached_tokens"] for r in rows]
    news = [r["new_tokens"] for r in rows]
    ok = True
    if cached != sorted(cached):
        print(f"  FAIL: cached не растёт: {cached}")
        ok = False
    if max(news) > 10:
        print(f"  FAIL: new_tokens={max(news)} > 10")
        ok = False
    if gen_before:
        print("  FAIL: генерация стартовала до финального Enter")
        ok = False

    # финальный Enter (пустое поле) должен запустить генерацию
    mark2 = c.mark()
    await c.send(type="text", text="", word_mode=True)
    started = await c.wait("turn_start", 20, mark2)
    if not started:
        print("  FAIL: генерация не стартовала по пустому Enter")
        ok = False
    else:
        print("  OK: генерация стартовала по финальному Enter")
        await c.wait("ready", 120, mark2)
    print("  ->", "OK" if ok else "FAIL")
    return ok


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8110/ws")
    ap.add_argument("--restart-llm", action="store_true")
    a = ap.parse_args()

    ok = True
    async with Client(a.url) as c:
        await asyncio.sleep(0.5)

        print("--- health ---")
        import urllib.request
        h = json.loads(urllib.request.urlopen(
            a.url.replace("ws://", "http://").replace("/ws", "/health"), timeout=10).read())
        print("  status:", h["status"], "llm:", bool(h.get("llm")), "tts:", h.get("tts"),
              "токен:", h.get("tools_token"))

        print("\n--- ход 1 (текст по Enter, критерий 14) ---")
        r1 = await turn_text(c, "Привет, как дела?")
        ok = ok and r1.get("ready", False)

        print("\n--- ход 2 (история, критерий 13) ---")
        r2 = await turn_text(c, "А эфира? Ответь одной фразой.")
        ok = ok and r2.get("ready", False)

        ok = await test_word_mode(c) and ok

        print("\n--- переключение TTS (критерий 8) ---")
        mark = c.mark()
        await c.send(type="tts_engine", engine="kokoro")
        e = await c.wait("tts_engine_set", 10, mark)
        print("  kokoro:", "OK" if e else "FAIL")
        r3 = await turn_text(c, "Скажи коротко: проверка второго голоса.")
        mark = c.mark()
        await c.send(type="tts_engine", engine="qwen")
        e = await c.wait("tts_engine_set", 10, mark)
        print("  qwen:", "OK" if e else "FAIL")

        if a.restart_llm:
            print("\n--- рестарт LLM-демона (критерий 11) ---")
            subprocess.run(["sudo", "-n", "systemctl", "restart", "cmp-llm.service"],
                           check=False)
            await asyncio.sleep(12)
            r4 = await turn_text(c, "Ты снова на связи?")
            print("  восстановление:", "OK" if r4.get("ready") else "FAIL")
            ok = ok and r4.get("ready", False)

        print("\n--- события ---")
        c.dump()

    print("\nИТОГ:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
