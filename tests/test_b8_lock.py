#!/usr/bin/env python3
"""B8: переключатель TTS должен быть заблокирован во время хода (по WS)."""
import asyncio
import json
import sys

import websockets


async def main() -> int:
    ok = True
    events = []
    async with websockets.connect("ws://127.0.0.1:8110/ws", max_size=16 << 20) as ws:

        async def pump():
            try:
                while True:
                    m = await ws.recv()
                    if isinstance(m, (bytes, bytearray)):
                        continue
                    events.append(json.loads(m))
            except Exception:
                pass

        task = asyncio.create_task(pump())
        await asyncio.sleep(0.6)

        await ws.send(json.dumps({"type": "text",
                                  "text": "Какой сейчас курс биткоина?",
                                  "word_mode": False}))
        for _ in range(400):
            if any(e.get("type") == "turn_start" for e in events):
                break
            await asyncio.sleep(0.05)
        print("turn_start:", any(e.get("type") == "turn_start" for e in events))

        # переключаем сразу после начала хода
        await ws.send(json.dumps({"type": "tts_engine", "engine": "kokoro"}))
        # и ещё раз посреди хода, когда заведомо busy
        await asyncio.sleep(2.0)
        await ws.send(json.dumps({"type": "tts_engine", "engine": "kokoro"}))
        await asyncio.sleep(1.5)

        rejects = [e for e in events if e.get("status") == 409
                   or "во время хода" in str(e.get("error", ""))]
        switched = [e for e in events if e.get("type") == "tts_engine_set"]

        print("ответов 409:", len(rejects))
        if rejects:
            print("   ", json.dumps(rejects[0], ensure_ascii=False))
        print("переключений:", len(switched))

        if not switched and rejects:
            print("B8 OK: во время хода переключение заблокировано")
        elif switched:
            print("B8 FAIL: переключение прошло во время хода")
            ok = False
        else:
            print("B8 FAIL: нет ни отказа, ни переключения")
            ok = False

        for _ in range(1500):
            if any(e.get("type") == "ready" for e in events):
                break
            await asyncio.sleep(0.1)
        task.cancel()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
