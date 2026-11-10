#!/usr/bin/env python3
"""Замер TTFB и RTF для каждого TTS-движка (критерии 5, 6, 7).

Запуск: .venv/bin/python tests/bench_tts.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice import config  # noqa: E402

config.load_env()

from voice import tts as tts_mod  # noqa: E402

TEXTS = [
    "Привет, как дела?",
    "Сейчас двенадцать часов дня, курс биткоина вырос.",
    "Это длинная фраза, чтобы проверить, как движок держит темп речи на протяжении "
    "нескольких секунд звучания. Темп должен оставаться ровным и без разрывов.",
]


async def bench(name: str, reg: tts_mod.TTSRegistry) -> None:
    try:
        b = reg.get(name)
    except tts_mod.TTSError as e:
        print(f"{name}: недоступен ({e})")
        return

    print(f"\n=== {name} ===")
    await b.warmup()
    print("  describe:", b.describe())

    for text in TEXTS:
        t0 = time.monotonic()
        ttfb = None
        total = 0
        chunks = 0
        gaps: list[float] = []
        last = None
        async for c in b.stream(text):
            now = time.monotonic()
            if ttfb is None:
                ttfb = now - t0
            if last is not None:
                gaps.append(now - last)
            last = now
            total += len(c)
            chunks += 1
        gen = time.monotonic() - t0
        sr = b.sample_rate
        dur = total / 2 / sr
        rtf = gen / dur if dur else 0.0
        maxgap = max(gaps) * 1000 if gaps else 0.0
        print(f"  «{text[:40]}…»")
        print(f"    TTFB={ttfb*1000:.0f} мс  генерация={gen:.2f} с  звук={dur:.2f} с  "
              f"RTF={rtf:.2f}  чанков={chunks}  макс.разрыв={maxgap:.0f} мс  sr={sr}")


async def main() -> int:
    reg = tts_mod.build_registry()
    print("доступные движки:", list(reg._backends.keys()))
    for name in ("qwen", "kokoro"):
        await bench(name, reg)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
