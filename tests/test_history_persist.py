#!/usr/bin/env python3
"""Проверка: сохраняется ли история между переподключениями WebSocket (F5)."""
import asyncio
import json
import sys

import websockets


async def once(text: str) -> str:
    events: list[dict] = []
    async with websockets.connect("ws://127.0.0.1:8110/ws", max_size=1 << 24) as ws:

        async def pump():
            try:
                while True:
                    m = await ws.recv()
                    if not isinstance(m, (bytes, bytearray)):
                        events.append(json.loads(m))
            except Exception:
                pass

        t = asyncio.create_task(pump())
        await asyncio.sleep(0.7)
        before = len(events)
        await ws.send(json.dumps({"type": "text", "text": text, "word_mode": False}))
        for _ in range(900):
            if any(e.get("type") == "ready" for e in events[before:]):
                break
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.7)
        ans = "".join(e["text"] for e in events[before:] if e.get("type") == "ans_token")
        t.cancel()
        return ans


async def main() -> int:
    a1 = await once("Меня зовут Пётр, запомни это. Ответь коротко.")
    print("соединение 1 (знакомство):", repr(a1[:100]))

    a2 = await once("Как меня зовут? Ответь одним словом.")
    print("соединение 2 (как после F5):", repr(a2[:140]))

    print()
    if "пётр" in a2.lower() or "петр" in a2.lower():
        print("РЕЗУЛЬТАТ: история ПЕРЕЖИЛА переподключение (хранится на сервере)")
    elif "не знаю" in a2.lower() or "не помню" in a2.lower() or "не представил" in a2.lower():
        print("РЕЗУЛЬТАТ: история ПОТЕРЯНА при переподключении (новая сессия на сокет)")
    else:
        print("РЕЗУЛЬТАТ: неоднозначно, смотрите ответ выше")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
