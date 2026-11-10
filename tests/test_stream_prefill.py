#!/usr/bin/env python3
"""Проверка: во время речи партиалы должны до-префилливаться в KV.

Подаём контрольный wav чанками по 100 мс (как это делает браузер) и смотрим,
приходят ли события `kv` ДО того, как придёт `final`.

Запуск: .venv/bin/python tests/test_stream_prefill.py [wav]
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import wave
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice import config  # noqa: E402

config.load_env()


def load_16k(path: str) -> bytes:
    """Читает wav, приводит к 16 кГц mono s16 (через ffmpeg если нужно)."""
    import subprocess
    import tempfile

    with wave.open(path, "rb") as w:
        ok = w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
    if not ok:
        out = tempfile.NamedTemporaryFile(suffix="_16k.wav", delete=False).name
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                        "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", out],
                       check=True)
        path = out
    with wave.open(path, "rb") as w:
        return w.readframes(w.getnframes())


async def main() -> int:
    wav = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(config.asr_model_dir()) / "0.wav")
    pcm = load_16k(wav)
    print(f"аудио: {len(pcm)/2/16000:.2f} с ({wav})")

    events: list[tuple[float, dict]] = []
    audio = {"bytes": 0}
    t0 = time.monotonic()

    async with websockets.connect("ws://127.0.0.1:8110/ws", max_size=1 << 24) as ws:

        async def pump():
            try:
                while True:
                    m = await ws.recv()
                    if isinstance(m, (bytes, bytearray)):
                        audio["bytes"] += len(m)
                        continue
                    events.append((time.monotonic() - t0, json.loads(m)))
            except Exception:
                pass

        task = asyncio.create_task(pump())
        await asyncio.sleep(0.8)

        # старт микрофонного тракта
        await ws.send(json.dumps({"type": "start"}))
        for _ in range(200):
            if any(e.get("type") == "started" for _, e in events):
                break
            await asyncio.sleep(0.05)
        print("ASR запущен:", any(e.get("type") == "started" for _, e in events))
        t_speech = time.monotonic() - t0

        # подаём аудио чанками по 100 мс, как браузер
        chunk = 16000 * 2 // 10
        for off in range(0, len(pcm), chunk):
            await ws.send(pcm[off:off + chunk])
            await asyncio.sleep(0.1)

        t_speech_end = time.monotonic() - t0
        # просим добить хвост тишиной (в браузере это делает сервер по endpoint)
        await ws.send(json.dumps({"type": "finish_utterance"}))

        for _ in range(600):
            if any(e.get("type") == "ready" for _, e in events):
                break
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.5)
        task.cancel()

    t_final = None
    partials = []
    kvs = []
    for ts, e in events:
        if e.get("type") == "partial":
            partials.append((ts, e.get("text", "")))
        elif e.get("type") == "kv":
            kvs.append((ts, e))
        elif e.get("type") == "final":
            t_final = ts

    print(f"\nречь: {t_speech:.1f} – {t_speech_end:.1f} с")
    print(f"партиалов ASR: {len(partials)}, строк KV: {len(kvs)}")
    for ts, txt in partials[:12]:
        print(f"  partial @{ts:5.1f} c  {txt!r}")
    print()
    for ts, e in kvs[:12]:
        print(f"  KV      @{ts:5.1f} c  cached={e.get('cached_tokens')} "
              f"new={e.get('new_tokens')} ms={round(e.get('ms_f') or 0,1)}")

    ok = True
    during = [(ts, e) for ts, e in kvs if t_final is None or ts < (t_final or 1e9)]
    if not kvs:
        print("\nFAIL: ни одной строки KV — партиалы не префиллятся")
        ok = False
    elif not during:
        print("\nFAIL: KV заполнялся только после окончания речи")
        ok = False
    else:
        first = during[0][1]
        print(f"\nOK: {len(during)} строк KV во время речи, первая через "
              f"{during[0][0]:.1f} с (cached={first.get('cached_tokens')}, "
              f"new={first.get('new_tokens')})")
        news = [e.get("new_tokens") for _, e in during]
        cached = [e.get("cached_tokens") for _, e in during]
        print(f"    new_tokens: {news}")
        print(f"    cached:     {cached}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
