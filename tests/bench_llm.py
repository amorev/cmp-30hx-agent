#!/usr/bin/env python3
"""Бенчмарк LLM: prefill, первый токен, ток/с. Критерий 4.

Запуск: .venv/bin/python tests/bench_llm.py [--text "..." ...]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice import config  # noqa: E402

config.load_env()

from voice.llm_client import NativeLLM  # noqa: E402


async def one(llm: NativeLLM, text: str, max_tokens: int) -> dict:
    await llm.reset()
    t0 = time.monotonic()
    first = {"ms": None}
    pieces = []

    async def on_token(p: str):
        if first["ms"] is None:
            first["ms"] = (time.monotonic() - t0) * 1000
        pieces.append(p)

    r = await llm.prefill(text, [])
    t_prefill = (time.monotonic() - t0) * 1000
    done = await llm.generate(text, [], max_tokens, on_token)
    total = (time.monotonic() - t0) * 1000
    return {
        "text": text,
        "prefill_ms": t_prefill,
        "first_token_ms": first["ms"],
        "tokens": done.get("tokens"),
        "tok_s": done.get("tok_s"),
        "gen_prefill_ms": done.get("prefill_ms"),
        "cached": done.get("cached_tokens"),
        "new": done.get("new_tokens"),
        "total_ms": total,
        "answer": "".join(pieces)[:200],
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="append", default=None)
    ap.add_argument("--max-tokens", type=int, default=120)
    a = ap.parse_args()

    texts = a.text or ["Привет, как дела?",
                       "Который час?",
                       "Расскажи коротко, что такое нейросеть."]
    llm = NativeLLM(config.llm_host(), config.llm_port())
    print(await llm.stat())

    worst_first = 0.0
    for t in texts:
        r = await one(llm, t, a.max_tokens)
        worst_first = max(worst_first, r["first_token_ms"] or 0)
        print(f"\n«{r['text']}»")
        print(f"  prefill (партиал)={r['prefill_ms']:.0f} мс  cached={r['cached']} new={r['new']}")
        print(f"  prefill в generate={r['gen_prefill_ms']} мс")
        print(f"  первый токен={r['first_token_ms']:.0f} мс  токенов={r['tokens']}  {r['tok_s']:.1f} ток/с")
        print(f"  ответ: {r['answer']!r}")

    print(f"\nкритерий 4: худший первый токен = {worst_first:.0f} мс "
          f"({'OK' if worst_first <= 2500 else 'FAIL'} при пороге 2500)")
    await llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
