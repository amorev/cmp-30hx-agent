#!/usr/bin/env python3
"""Тест инкрементального prefill по серии партиалов (критерии 2 и 3).

Печатает по каждому партиалу cached/new/ms и проверяет пороги:
  cached растёт, new_tokens <= 10, медиана prefill <= 100 мс.

Запуск:  .venv/bin/python tests/test_prefill.py
"""
from __future__ import annotations

import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice import config  # noqa: E402

config.load_env()

from voice.llm_client import NativeLLM  # noqa: E402

PARTIALS = ["какой", "какой сейчас", "какой сейчас курс", "какой сейчас курс биткоина"]


async def main() -> int:
    llm = NativeLLM(config.llm_host(), config.llm_port())
    st = await llm.stat()
    print(f"демон: kv={st.get('kv_tokens')} base={st.get('base_tokens')} "
          f"ctx={st.get('ctx')} kv_type={st.get('kv_type')} tools={st.get('tools')}")

    await llm.reset()
    st = await llm.stat()
    assert st["kv_tokens"] == st["base_tokens"], "после reset в KV должна остаться база"
    print(f"после reset: kv={st['kv_tokens']} (база)\n")

    rows = []
    for i, text in enumerate(PARTIALS, 1):
        r = await llm.prefill(text, [])
        assert r.get("ok"), r
        rows.append(r)
        print(f"{i}. {text!r:38} cached={r['cached_tokens']:5} new={r['new_tokens']:4} "
              f"total={r['prompt_tokens']:5} ms={r['ms_f']:8.1f} fallbacks={r['fallbacks']}")

    cached = [r["cached_tokens"] for r in rows]
    news = [r["new_tokens"] for r in rows]
    ms = [r["ms_f"] for r in rows]
    ok = True

    if rows[0]["cached_tokens"] < 500:
        print(f"\nFAIL критерий 2: первый cached={rows[0]['cached_tokens']} < 500")
        ok = False
    else:
        print(f"\nOK  критерий 2: база из кэша на первом партиале = {rows[0]['cached_tokens']}")

    if cached != sorted(cached):
        print(f"FAIL: cached не растёт: {cached}")
        ok = False
    else:
        print(f"OK  cached растёт: {cached}")

    if max(news) > 10:
        print(f"FAIL критерий 3: max new_tokens={max(news)} > 10")
        ok = False
    else:
        print(f"OK  критерий 3: new_tokens на партиалах = {news} (<=10)")

    med = statistics.median(ms)
    if med > 100:
        print(f"FAIL критерий 3: медиана prefill={med:.1f} мс > 100")
        ok = False
    else:
        print(f"OK  критерий 3: медиана prefill = {med:.1f} мс")

    fb = rows[-1]["fallbacks"]
    print(f"    аварийных сбросов (снапшот-фолбэк): {fb}")

    await llm.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
