#!/usr/bin/env python3
"""Приёмочный тест нормализатора Silero (silero/normalize.py).

Проверяет то, что заявлено в ТЗ по голосу: цифры и латиница не должны
пропадать из речи. Модель для теста не нужна — только чистые функции,
поэтому запускается в любом venv и без torch/GPU.

    .venv/bin/python tests/test_normalize.py
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "silero"))

from normalize import normalize, to_words, plural  # noqa: E402

FAIL: list[str] = []
PASS = 0


def check(name: str, got: str, must: list[str], must_not: list[str] | None = None) -> None:
    """must — подстроки, которые обязаны быть; must_not — которых быть не должно."""
    global PASS
    low = got.lower()
    missing = [m for m in must if m.lower() not in low]
    extra = [m for m in (must_not or []) if m.lower() in low]
    if missing or extra:
        FAIL.append(f"{name}: got={got!r} missing={missing} extra={extra}")
    else:
        PASS += 1
        print(f"  ok  {name}")


def check_digits(name: str, src: str) -> None:
    """В результате не должно остаться цифр вообще."""
    global PASS
    got = normalize(src)
    if any(c.isdigit() for c in got):
        FAIL.append(f"{name}: остались цифры: {got!r}")
    else:
        PASS += 1
        print(f"  ok  {name}")


print("== числительные ==")
check("1000", to_words(1000), ["тысяча"], ["одна тысяча"])
check("2000", to_words(2000), ["две тысячи"])
check("1000000", to_words(1000000), ["один миллион"])
check("21", to_words(21), ["двадцать один"])
check("согласование", plural(5, "рубль", "рубля", "рублей"), ["рублей"])
check("11 рублей", plural(11, "рубль", "рубля", "рублей"), ["рублей"])
check("21 рубль", plural(21, "рубль", "рубля", "рублей"), ["рубль"])

print("== цифры исчезают ==")
for src in ("У меня 5 яблок и 10 груш.",
            "Код 1234, версия 3.",
            "В 2024 году 365 дней.",
            "Скидка 15% до 31.12.2024.",
            "Стоимость 1 500 000 рублей.",
            "Температура -12 градусов.",
            "Позвони на 8-800-555-35-35.",
            "Файл весит 2,5 ГБ.",
            "В 1990-х годах."):
    check_digits(src, src)

print("== год и падеж ==")
check("в 2024 году", normalize("в 2024 году"),
      ["две тысячи двадцать четвёртом", "году"])
check("с 2020 по 2024", normalize("с 2020 по 2024"),
      ["две тысячи двадцатого", "две тысячи двадцать четвёртый"])
check("1990-х годах", normalize("в 1990-х годах"),
      ["тысяча девятьсот девяностых", "годах"], ["одна тысяча"])

print("== дата и время ==")
check("дата", normalize("12.05.2024"), ["двенадцатое мая", "года"])
check("время", normalize("в 14:30"), ["четырнадцать часов", "тридцать минут"])

print("== деньги и единицы ==")
check("рубли", normalize("1000 рублей"), ["тысяча рублей"])
check("доллары", normalize("$5"), ["пять долларов"])
check("дробные кг", normalize("2,5 кг"), ["две целых пять десятых", "килограмма"])
check("проценты", normalize("15%"), ["пятнадцать процентов"])
check("1 процент", normalize("1%"), ["один процент"], ["процентов"])
check("рубли не двоятся", normalize("1000 рублей"), ["тысяча рублей"],
      ["рублейлей"])
check("знак рубля", normalize("1 000 ₽"), ["тысяча рублей"])
check("доллары с разрядами", normalize("$1 500"), ["тысяча пятьсот долларов"])
check("тыс. руб.", normalize("3 тыс. руб."), ["три тысячи рублей"], ["руб."])
check("копейки", normalize("50 коп."), ["пятьдесят копеек"])
check("евро", normalize("€2 000 000"), ["два миллиона евро"])

print("== латиница ==")
check("словарь", normalize("Python и GitHub"), ["Питон", "гитхаб"])
check("аббревиатура", normalize("API"), ["а-пи-ай"])
check("транслит", normalize("Docker"), ["докер"])
check("латиница выключена", normalize("Python", latin=False), ["Питон"])

print("== согласование рода ==")
check("22 книги", normalize("22 книги"), ["двадцать две книги"])
check("2 стола", normalize("2 стола"), ["два стола"])
check("21 книга", normalize("21 книга"), ["двадцать одна книга"])
check("2 дня", normalize("2 дня"), ["два дня"])

print("== токенсет ==")
_ALLOWED = set("!,-.:? абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
               "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ+")
src = ("Тест: «Python», 5-10 шт., файл test.py, 100% готово! "
       "Цена — 1 000 ₽; адрес https://example.com/a?b=1&c=2 (скобки) [квадрат].")
out = normalize(src)
bad = sorted({c for c in out if c not in _ALLOWED})
if bad:
    FAIL.append(f"токенсет: запрещённые символы {bad} в {out!r}")
else:
    PASS += 1
    print(f"  ok  токенсет чист ({out!r})")

print("== устойчивость ==")
for src in ("", "   ", "привет", "!!!", "123456789012345678901234567890"):
    try:
        normalize(src)
        PASS += 1
    except Exception as e:  # noqa: BLE001
        FAIL.append(f"падение на {src!r}: {e.__class__.__name__}: {e}")

print()
if FAIL:
    print(f"ПРОВАЛЕНО {len(FAIL)} (успешно {PASS}):")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
print(f"ВСЁ ХОРОШО: {PASS} проверок")
