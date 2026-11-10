#!/usr/bin/env python3
"""Репро: ход с вызовом инструмента → ответ → второй ход.

Печатает ВСЁ: токены ответа, ошибки, объём аудио, и то, что сервер
сохранил в истории (порядок ролей).
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


async def turn(ws, events, text, label):
    mark = len(events)
    audio0 = sum(1 for _ in ())
    before = len(events)
    await ws.send(json.dumps({"type": "text", "text": text, "word_mode": False}))
    for _ in range(1500):
        if any(e.get("type") == "ready" for e in events[before:]):
            break
        await asyncio.sleep(0.1)
    await asyncio.sleep(0.8)

    ans = "".join(e["text"] for e in events[before:] if e.get("type") == "ans_token")
    stats = [e for e in events[before:] if e.get("type") == "gen_stats"]
    calls = [e for e in events[before:] if e.get("type") == "tool_call"]
    errs = [e for e in events[before:] if e.get("type") in ("error", "tts_error")]
    print(f"\n=== {label}: «{text}» ===")
    print(f"  вызовов инструментов: {len(calls)} -> {[c.get('name') for c in calls]}")
    print(f"  раундов генерации: {len(stats)}")
    for i, s in enumerate(stats, 1):
        print(f"    раунд {i}: tokens={s.get('tokens')} first={s.get('first_token_ms')}ms "
              f"cached={s.get('cached_tokens')} new={s.get('new_tokens')}")
    print(f"  ОТВЕТ: {ans[:300]!r}  (длина {len(ans)})")
    if errs:
        print(f"  ОШИБКИ: {errs}")
    return ans


async def main() -> int:
    events = []
    audio = {"bytes": 0}
    async with websockets.connect("ws://127.0.0.1:8110/ws", max_size=1 << 24) as ws:

        async def pump():
            try:
                while True:
                    m = await ws.recv()
                    if isinstance(m, (bytes, bytearray)):
                        audio["bytes"] += len(m)
                        continue
                    events.append(json.loads(m))
            except Exception:
                pass

        task = asyncio.create_task(pump())
        await asyncio.sleep(0.8)
        await ws.send(json.dumps({"type": "reset"}))
        await asyncio.sleep(1.0)

        a1 = await turn(ws, events, "Найди информацию про компанию FromSoftware", "ход 1 (с инструментом)")
        a2 = await turn(ws, events, "А кто её владелец?", "ход 2 (после инструмента)")
        a3 = await turn(ws, events, "Спасибо, а сколько сейчас времени?", "ход 3 (обычный)")

        print(f"\nвсего аудио: {audio['bytes']} байт")
        task.cancel()

    ok = bool(a1) and bool(a2) and bool(a3)
    print("\nИТОГ:", "OK" if ok else "FAIL — на каком-то ходе пустой ответ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
