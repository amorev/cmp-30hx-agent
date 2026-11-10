#!/usr/bin/env python3
"""Тест инструментов: MCP-поиск напрямую и через демон (критерии 10, 10a).

Запуск:
  .venv/bin/python tests/test_tools.py            # прямой MCP + токен
  .venv/bin/python tests/test_tools.py --llm      # ещё и вызов моделью
  WORM_TOKEN_FILE=/nonexistent .venv/bin/python tests/test_tools.py   # критерий 10a
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice import config  # noqa: E402

config.load_env()

from voice import tools  # noqa: E402
from voice.llm_client import NativeLLM  # noqa: E402


async def llm_case(llm: NativeLLM, question: str) -> None:
    print(f"\n--- через демон: «{question}» ---")
    await llm.set_tools(tools.specs())
    await llm.reset()

    hist: list[dict] = []
    for rnd in range(1, 4):
        seen: list[str] = []

        async def on_token(p: str):
            seen.append(p)

        done = await llm.generate(question if rnd == 1 else "", hist, 200, on_token)
        raw_text = "".join(seen)
        text = done.get("text", "")
        calls = done.get("tool_calls") or []
        print(f"раунд {rnd}: finish={done.get('finish')} токенов={done.get('tokens')} "
              f"вызовов={len(calls)}")
        print(f"  текст: {text[:200]!r}")
        if "<tool_call>" in raw_text:
            print("  FAIL: разметка вызова утекла в текст!")
        if not calls:
            break
        hist.append({"role": "assistant", "content": "\n\n\n\n" + text})
        for c in calls:
            res = tools.execute(c["name"], c.get("arguments") or {})
            print(f"  {c['name']}({c.get('arguments')}) -> {res[:300]!r}")
            hist.append({"role": "tool", "content": f"{c['name']}: {res}"})
        question = ""


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--query", default="курс биткоина сегодня")
    a = ap.parse_args()

    print("токен:", tools.token_status())
    print(f"MCP URL: {config.worm_mcp_url()}")

    print("\n--- прямой MCP web_search ---")
    res = tools.web_search(a.query, 3)
    print(res[:800])
    if res.startswith("поиск недоступен"):
        print("\n(деградация: поиск недоступен — это ожидаемо, если токена нет)")
        if not tools.token_status()["present"]:
            print("OK критерий 10a: текст ошибки получен, сервис не упал")
    else:
        print("\nOK критерий 10: MCP вернул данные")

    print("\n--- локальные инструменты ---")
    print("время:      ", tools.current_time())
    print("калькулятор:", tools.calculator("2*(3+4)"))
    print("таймер:     ", tools.set_timer(5, "тест"))
    print("проверка:   ", tools.check_timer("тест"))

    if a.llm:
        llm = NativeLLM(config.llm_host(), config.llm_port())
        await llm_case(llm, "Какой сейчас курс биткоина? Ответь одной фразой.")
        await llm_case(llm, "Который час?")
        await llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
