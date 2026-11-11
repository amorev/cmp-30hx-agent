#!/usr/bin/env python3
"""Репро: два хода подряд, второй — с поиском. Печатает историю ходов."""
import asyncio
import json
import sys

import websockets


async def main() -> int:
    events: list[dict] = []
    async with websockets.connect("ws://127.0.0.1:8110/ws", max_size=1 << 24) as ws:

        async def pump():
            try:
                while True:
                    m = await ws.recv()
                    if isinstance(m, (bytes, bytearray)):
                        continue
                    events.append(json.loads(m))
            except Exception:
                pass

        t = asyncio.create_task(pump())
        await asyncio.sleep(0.8)
        await ws.send(json.dumps({"type": "reset"}))
        await asyncio.sleep(1.0)

        for label, text in [
            ("ход 1", "Привет, как дела?"),
            ("ход 2", "Найди в интернете кто такой Антон Морев"),
        ]:
            before = len(events)
            await ws.send(json.dumps({"type": "text", "text": text, "word_mode": False}))
            for _ in range(1500):
                if any(e.get("type") == "ready" for e in events[before:]):
                    break
                await asyncio.sleep(0.1)
            await asyncio.sleep(1.0)
            ans = "".join(e["text"] for e in events[before:] if e.get("type") == "ans_token")
            stats = [e for e in events[before:] if e.get("type") == "gen_stats"]
            calls = [e for e in events[before:] if e.get("type") == "tool_call"]
            print(f"\n=== {label}: «{text}» ===")
            print(f"  вызовов: {[c.get('name') for c in calls]}")
            for i, s in enumerate(stats, 1):
                print(f"  раунд {i}: tokens={s.get('tokens')} cached={s.get('cached_tokens')} "
                      f"new={s.get('new_tokens')} first={s.get('first_token_ms')}ms")
            print(f"  ОТВЕТ: {ans[:250]!r}")

        t.cancel()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
