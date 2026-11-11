"""Нормализация русского текста для Silero TTS.

Зачем это нужно
---------------
В токенсете моделей Silero нет цифр, а из латиницы есть лишь единичные
буквы (опубликованный токенсет v4_cyrillic:
`!,-.:?iµöабвгдежзийклмнопрстуфхцчшщъыьэюяё...`). Поэтому «в 2024 году»
превращается в «в году», а «Python» просто выпадает. Лечится только
предобработкой текста до синтеза — здесь.

Что делает
----------
1. числа, в т.ч. с разделителем разрядов `1 000 000`, -> прописью;
2. годы с падежом по следующему слову: «в 2024 году» ->
   «в две тысячи двадцать четвёртом году»;
3. даты `12.05.2024` и `12/05` -> «двенадцатое мая две тысячи двадцать
   четвёртого года»;
4. время `14:30` -> «четырнадцать часов тридцать минут»;
5. проценты `15%` -> «пятнадцать процентов» (согласование числа);
6. деньги `1000 ₽`, `$5`, `300 EUR` -> «тысяча рублей», «пять долларов»;
7. дробные `3,5` / `3.5` -> «три целых пять десятых»;
8. диапазоны `5-10` -> «пять — десять»;
9. порядковые с суффиксом `5-й`, `12-го`, `3-е` -> «пятый», «двенадцатого»;
10. латиница: словарь частых терминов, аббревиатуры по буквам, остальное
    транслитерацией;
11. финальная чистка до символов, которые модель принимает.

Соглашения и ограничения (важно, чтобы не выглядело магией)
-----------------------------------------------------------
* Род для «один/одна» и «два/две» угадывается по следующему слову
  (окончание -а/-я -> женский, -о/-е -> средний, иначе мужской). Это
  эвристика, а не морфология.
* Запятая между цифрами без пробела трактуется как десятичная: `1,2` ->
  «одна целая две десятых». Перечисление писать как `1, 2`.
* «с 2020 по 2024» не распознаётся как пара и читается количественными
  числительными — приемлемо, но не идеально.
* Числа, для которых нужен сложный падеж внутри фразы («о 5 домах»),
  читаются в именительном.
* Стресс-маркер `+` сохраняется (модели ru его понимают).

Модуль намеренно без внешних зависимостей: живёт в venv Silero, где
кроме torch/fastapi ничего не гарантировано.
"""
from __future__ import annotations

import re

__all__ = ["normalize", "to_words", "ALLOWED_EXTRA"]


# --------------------------------------------------------------------------
# Числительные: количественные
# --------------------------------------------------------------------------

_UNITS = {
    "m": ("ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь",
          "восемь", "девять"),
    "f": ("ноль", "одна", "две", "три", "четыре", "пять", "шесть", "семь",
          "восемь", "девять"),
    "n": ("ноль", "одно", "два", "три", "четыре", "пять", "шесть", "семь",
          "восемь", "девять"),
}
_TEENS = ("десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
          "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать",
          "девятнадцать")
_TENS = ("", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят",
         "семьдесят", "восемьдесят", "девяносто")
_HUNDREDS = ("", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот",
             "семьсот", "восемьсот", "девятьсот")

# склоняемые «тысяча», «миллион», ...
_SCALE = (
    None,
    ("тысяча", "тысячи", "тысяч", "f"),
    ("миллион", "миллиона", "миллионов", "m"),
    ("миллиард", "миллиарда", "миллиардов", "m"),
    ("триллион", "триллиона", "триллионов", "m"),
)


def plural(n: int, one: str, few: str, many: str) -> str:
    """Русское согласование существительного с числом."""
    n = abs(int(n))
    if n % 100 in (11, 12, 13, 14):
        return many
    last = n % 10
    if last == 1:
        return one
    if last in (2, 3, 4):
        return few
    return many


def _under_1000(n: int, gender: str = "m") -> str:
    parts: list[str] = []
    h, rest = divmod(n, 100)
    if h:
        parts.append(_HUNDREDS[h])
    if rest >= 20:
        parts.append(_TENS[rest // 10])
        rest %= 10
    elif 10 <= rest < 20:
        parts.append(_TEENS[rest - 10])
        rest = 0
    if rest:
        parts.append(_UNITS[gender][rest])
    return " ".join(parts)


def to_words(n: int, gender: str = "m") -> str:
    """Целое число прописью. Поддерживает до триллионов."""
    n = int(n)
    if n < 0:
        return "минус " + to_words(-n, gender)
    if n == 0:
        return "ноль"
    if n < 1000:
        return _under_1000(n, gender)

    groups: list[int] = []
    while n > 0:
        groups.append(n % 1000)
        n //= 1000
    if len(groups) > len(_SCALE) - 1:
        # выше триллионов не забираемся — отдаём как есть по разрядам
        return " ".join(str(g) for g in reversed(groups))

    out: list[str] = []
    for idx in range(len(groups) - 1, -1, -1):
        g = groups[idx]
        if not g:
            continue
        if idx == 0:
            out.append(_under_1000(g, gender))
            continue
        one, few, many, g_gender = _SCALE[idx]
        # «одна тысяча» по-русски не говорят — только «тысяча»
        if not (idx == 1 and g == 1):
            out.append(_under_1000(g, g_gender))
        out.append(plural(g, one, few, many))
    return " ".join(p for p in out if p)


# --------------------------------------------------------------------------
# Числительные: порядковые
# --------------------------------------------------------------------------

# значение -> (основа, тип): h — твёрдый (четвёрт-ый), st — ударный
# (втор-ой), soft — мягкий (трет-ий)
_ORD_SIMPLE = {
    0: ("нулев", "st"), 1: ("перв", "h"), 2: ("втор", "st"),
    3: ("трет", "soft"), 4: ("четвёрт", "h"), 5: ("пят", "h"),
    6: ("шест", "st"), 7: ("седьм", "st"), 8: ("восьм", "st"),
    9: ("девят", "h"), 10: ("десят", "h"), 11: ("одиннадцат", "h"),
    12: ("двенадцат", "h"), 13: ("тринадцат", "h"), 14: ("четырнадцат", "h"),
    15: ("пятнадцат", "h"), 16: ("шестнадцат", "h"), 17: ("семнадцат", "h"),
    18: ("восемнадцат", "h"), 19: ("девятнадцат", "h"),
}
_ORD_TENS = {2: ("двадцат", "h"), 3: ("тридцат", "h"), 4: ("сороков", "st"),
             5: ("пятидесят", "h"), 6: ("шестидесят", "h"),
             7: ("семидесят", "h"), 8: ("восьмидесят", "h"),
             9: ("девяност", "h")}
_ORD_HUNDREDS = {1: ("сот", "h"), 2: ("двухсот", "h"), 3: ("трёхсот", "h"),
                 4: ("четырёхсот", "h"), 5: ("пятисот", "h"),
                 6: ("шестисот", "h"), 7: ("семисот", "h"),
                 8: ("восьмисот", "h"), 9: ("девятисот", "h")}

_ORD_END = {
    "h": {"nom_m": "ый", "gen_m": "ого", "dat_m": "ому", "prep_m": "ом",
          "nom_n": "ое", "nom_f": "ая", "gen_f": "ой", "acc_f": "ую",
          "pl": "ых"},
    "st": {"nom_m": "ой", "gen_m": "ого", "dat_m": "ому", "prep_m": "ом",
           "nom_n": "ое", "nom_f": "ая", "gen_f": "ой", "acc_f": "ую",
           "pl": "ых"},
    "soft": {"nom_m": "ий", "gen_m": "ьего", "dat_m": "ьему", "prep_m": "ьем",
             "nom_n": "ье", "nom_f": "ья", "gen_f": "ьей", "acc_f": "ью",
             "pl": "ьих"},
}

# соединительные формы для «двухтысячный» (только когда младшие разряды нули)
_COMBINE = {
    2: "двух", 3: "трёх", 4: "четырёх", 5: "пяти", 6: "шести", 7: "семи",
    8: "восьми", 9: "девяти", 10: "десяти", 11: "одиннадцати",
    12: "двенадцати", 13: "тринадцати", 14: "четырнадцати",
    15: "пятнадцати", 16: "шестнадцати", 17: "семнадцати",
    18: "восемнадцати", 19: "девятнадцати", 20: "двадцати", 30: "тридцати",
    40: "сорока", 50: "пятидесяти", 60: "шестидесяти", 70: "семидесяти",
    80: "восьмидесяти", 90: "девяноста",
}


def _ord_form(stem: str, kind: str, form: str) -> str:
    ends = _ORD_END[kind]
    return stem + ends.get(form, ends["nom_m"])


def _ordinal(n: int, form: str = "nom_m") -> str:
    """Порядковое числительное. Меняется только последнее слово."""
    if n < 20:
        stem, kind = _ORD_SIMPLE[n]
        return _ord_form(stem, kind, form)

    if n < 100:
        t, rest = divmod(n, 10)
        if rest == 0:
            stem, kind = _ORD_TENS[t]
            return _ord_form(stem, kind, form)
        return _TENS[t] + " " + _ordinal(rest, form)

    if n < 1000:
        h, rest = divmod(n, 100)
        if rest == 0:
            stem, kind = _ORD_HUNDREDS[h]
            return _ord_form(stem, kind, form)
        return _HUNDREDS[h] + " " + _ordinal(rest, form)

    # от тысяч: всё, кроме младшей части, идёт количественным
    for idx in range(len(_SCALE) - 1, 0, -1):
        unit = 1000 ** idx
        if n >= unit:
            high, rest = divmod(n, unit)
            one, few, many, g = _SCALE[idx]
            if rest == 0:
                # 2000 -> двухтысячный; 1000 -> тысячный
                if high == 1:
                    stem = one[:-1] + "н"       # тысяча -> тысячн
                    return _ord_form(stem, "h", form)
                if high in _COMBINE:
                    stem = _COMBINE[high] + one[:-1] + "н"
                    return _ord_form(stem, "h", form)
                return to_words(high, "f") + " " + plural(high, one, few, many)
            # «тысяча девятьсот девяностых» — без «одна» и «одного»
            head = (one if high == 1
                    else to_words(high, g) + " " + plural(high, one, few, many))
            return head + " " + _ordinal(rest, form)

    return to_words(n)


# --------------------------------------------------------------------------
# Склоняемые слова
# --------------------------------------------------------------------------

_MONTH_GEN = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля",
              "августа", "сентября", "октября", "ноября", "декабря")


# --------------------------------------------------------------------------
# Латиница
# --------------------------------------------------------------------------

# Частые термины: читаются по-русски, а не транслитом. Список легко
# расширять — он проверяется раньше общего правила.
_LATIN_DICT = {
    "python": "Питон", "питон": "Питон",
    "javascript": "джаваскрипт", "typescript": "тайпскрипт",
    "gpt": "гэ-пэ-тэ", "llm": "эл-эл-эм", "api": "а-пи-ай",
    "http": "эйч-ти-ти-пи", "https": "эйч-ти-ти-пи-эс",
    "url": "у-эр-эл", "json": "джейсон", "yaml": "ямл",
    "git": "гит", "github": "гитхаб", "gitlab": "гитлаб",
    "docker": "докер", "linux": "линукс", "ubuntu": "убунту",
    "windows": "виндовс", "macos": "макос", "nvidia": "энвидиа",
    "cuda": "куда", "gpu": "гэ-пэ-ю", "cpu": "цэ-пэ-ю",
    "ram": "озу", "ssd": "эс-эс-ди", "usb": "ю-эс-би",
    "wifi": "вай-фай", "wi-fi": "вай-фай", "email": "имейл",
    "telegram": "телеграм", "google": "гугл", "youtube": "ютуб",
    "c++": "си плюс плюс", "c#": "си шарп", "sql": "эс-кью-эл",
    "html": "эйч-ти-эм-эл", "css": "цэ-эс-эс", "exe": "экзе",
    "pdf": "пэ-дэ-эф", "id": "ай-ди", "ok": "окей", "upd": "апдейт",
}

_LETTER_NAMES = {
    "a": "а", "b": "бэ", "c": "цэ", "d": "дэ", "e": "е", "f": "эф",
    "g": "гэ", "h": "аш", "i": "и", "j": "жи", "k": "ка", "l": "эль",
    "m": "эм", "n": "эн", "o": "о", "p": "пэ", "q": "ку", "r": "эр",
    "s": "эс", "t": "тэ", "u": "у", "v": "вэ", "w": "дабл-ю", "x": "икс",
    "y": "игрек", "z": "зэд",
}

# Порядок важен: длинные сочетания должны срабатывать раньше коротких.
_TRANSLIT = (
    ("shch", "щ"), ("sch", "щ"), ("sh", "ш"), ("ch", "ч"), ("zh", "ж"),
    ("th", "т"), ("ph", "ф"), ("kh", "х"), ("ck", "к"), ("qu", "кв"),
    ("oo", "у"), ("ee", "и"), ("ea", "и"), ("ou", "оу"), ("ya", "я"),
    ("yu", "ю"), ("yo", "ё"), ("ye", "е"), ("ay", "эй"), ("ey", "ей"),
    ("oy", "ой"), ("ai", "ай"), ("au", "ау"), ("ju", "ю"), ("ja", "я"),
)
_TRANSLIT_1 = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "х", "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "в", "x": "кс", "y": "и", "z": "з",
}


def _translit(word: str) -> str:
    w = word.lower()
    out: list[str] = []
    i = 0
    while i < len(w):
        for src, dst in _TRANSLIT:
            if w.startswith(src, i):
                out.append(dst)
                i += len(src)
                break
        else:
            ch = w[i]
            if ch == "c" and i + 1 < len(w) and w[i + 1] in "eiy":
                out.append("с")
            elif ch.isalpha():
                out.append(_TRANSLIT_1.get(ch, ""))
            i += 1
    res = "".join(out)
    # дефисы внутри слова не нужны, но пробелы-разделители оставляем
    return res.replace("--", "-")


def _latin_word(m: re.Match) -> str:
    word = m.group(0)
    low = word.lower()

    if low in _LATIN_DICT:
        return _LATIN_DICT[low]

    core = low.strip("-_")
    if core in _LATIN_DICT:
        return _LATIN_DICT[core]

    # аббревиатуры читаем по буквам: API, GPU, NASA
    letters = [c for c in low if c.isalpha()]
    if 2 <= len(letters) <= 5 and low.isupper() and len(letters) == len(low):
        return "-".join(_LETTER_NAMES[c] for c in letters)

    return _translit(core) or " "


# --------------------------------------------------------------------------
# Вспомогательные преобразования
# --------------------------------------------------------------------------

_ALLOWED_EXTRA = set("!,-.:? ")


def _next_noun(text: str, pos: int) -> str:
    m = re.match(r"\s+([а-яё]+)", text[pos:])
    return m.group(1) if m else ""


def _gender_for_count(text: str, pos: int, last_digit: int) -> str:
    """Род для «один/одна», «два/две» по следующему слову.

    Это эвристика по окончанию, а не морфология:
      * после 1 слово стоит в именительном падеже — «-а/-я» женский,
        «-о/-е» средний, иначе мужской;
      * после 2-4 слово стоит в родительном: единственное число на
        «-а/-я» — мужской («два стола», «два дня»), множественное на
        «-и/-ы» — женский («две книги», «две девочки»).
    """
    w = _next_noun(text, pos)
    if not w:
        return "m"
    if last_digit == 1:
        if w.endswith(("а", "я")):
            return "f"
        if w.endswith(("о", "е")):
            return "n"
        return "m"
    if w.endswith(("а", "я")):
        return "m"
    if w.endswith(("о", "е")):
        return "n"
    if w.endswith(("и", "ы")):
        return "f"
    return "m"


def _num_words(n: int, text: str, pos: int) -> str:
    """Количественное с родом только там, где он различается."""
    n_abs = abs(n)
    if n_abs % 100 in (11, 12, 13, 14):
        return to_words(n)
    last = n_abs % 10
    if last in (1, 2):
        return to_words(n, _gender_for_count(text, pos, last))
    return to_words(n)


def _currency(n, unit: str) -> str:
    if isinstance(n, str):
        raw = re.sub(r"[\s\u00a0\u202f]", "", n)
        if "," in raw or "." in raw:
            whole, frac = re.split(r"[.,]", raw, maxsplit=1)
            if unit in ("$", "usd", "доллар", "доллара", "долларов"):
                return _decimal(whole, frac) + " доллара"
            if unit in ("€", "eur", "евро"):
                return _decimal(whole, frac) + " евро"
            return _decimal(whole, frac) + " рубля"
        n = int(re.sub(r"\D", "", raw) or 0)
    if unit in ("₽", "руб", "руб.", "рубль", "рубля", "рублей", "р", "р."):
        return to_words(n) + " " + plural(n, "рубль", "рубля", "рублей")
    if unit in ("$", "usd", "доллар", "доллара", "долларов"):
        return to_words(n) + " " + plural(n, "доллар", "доллара", "долларов")
    if unit in ("€", "eur", "евро"):
        return to_words(n) + " евро"
    return to_words(n) + " " + unit

_DENOM = {
    1: ("десятая", "десятых"),
    2: ("сотая", "сотых"),
    3: ("тысячная", "тысячных"),
    4: ("десятитысячная", "десятитысячных"),
    5: ("стотысячная", "стотысячных"),
    6: ("миллионная", "миллионных"),
}


def _decimal(whole: str, frac: str) -> str:
    w = int(whole)
    f = int(frac)
    one, many = _DENOM.get(len(frac), ("десятая", "десятых"))
    head = to_words(w, "f") + " " + (
        "целая" if w % 10 == 1 and w % 100 != 11 else "целых")
    if f % 10 == 1 and f % 100 != 11:
        tail = to_words(f, "f") + " " + one
    else:
        tail = to_words(f) + " " + many
    return head + " " + tail


# --------------------------------------------------------------------------
# Основной проход
# --------------------------------------------------------------------------

_RE_DICT = re.compile(
    r"(?<![\w-])(" + "|".join(sorted((re.escape(k) for k in _LATIN_DICT),
                                    key=len, reverse=True)) + r")(?![\w-])",
    re.IGNORECASE)
_RE_DATE = re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?(?!\d)")
_RE_YEAR = re.compile(
    r"(?<!\d)(1\d{3}|20\d{2})\s*-?\s*(год|году|года|годах|годов|гг|г)\b\s*\.?",
    re.IGNORECASE)
_RE_TIME = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)")
_RE_PERCENT = re.compile(r"(?<!\d)(\d+(?:[.,]\d+)?)\s*%")
_RE_CUR_PRE = re.compile(
    r"([$\u20ac])\s*(\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+(?:[.,]\d+)?)(?!\d)")
_RE_CUR_POST = re.compile(
    r"(?<![\d.,])(\d[\d \u00a0\u202f]*)"
    r"\s*(₽|рублей|рубля|рубль|руб\.|руб|USD|EUR|евро|долларов|доллара"
    r"|доллар|копеек|коп\.|коп)",
    re.IGNORECASE)
_RE_ORD_SUF = re.compile(
    r"(?<!\d)(\d+)\s*-\s*(й|го|му|м|я|е|ой|ом|х|ый|ая|ое)\b", re.IGNORECASE)
_RE_RANGE = re.compile(r"(?<!\d)(\d+)\s*[-–—]\s*(\d+)(?!\d)")
_RE_DECIMAL = re.compile(r"(?<!\d)(\d+)[.,](\d{1,6})(?!\d)")
_RE_THOUSANDS = re.compile(r"(?<![\d.])(\d{1,3}(?:[ \u00a0\u202f]\d{3})+)(?![\d.])")
_RE_NUM = re.compile(r"(?<![\d.])(\d+)(?![\d.,])")
_RE_NUMERO = re.compile(r"№\s*(\d+)")
_RE_LATIN = re.compile(r"[A-Za-z][A-Za-z'’\-]*|[A-Za-z]")

_ORD_FORMS = {"й": "nom_m", "ый": "nom_m", "го": "gen_m", "му": "dat_m",
              "м": "prep_m", "ом": "prep_m", "я": "nom_f", "ая": "nom_f",
              "е": "nom_n", "ое": "nom_n", "ой": "gen_f", "х": "pl"}


def _fix_date(m: re.Match) -> str:
    d, mo = int(m.group(1)), int(m.group(2))
    y = m.group(3)
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return m.group(0)
    out = _ordinal(d, "nom_n") + " " + _MONTH_GEN[mo - 1]
    if y:
        yy = int(y)
        if yy < 100:
            yy += 1900 if yy > 50 else 2000
        out += " " + _ordinal(yy, "gen_m") + " года"
    return out


def _fix_year(m: re.Match) -> str:
    y = int(m.group(1))
    form = {"год": "nom_m", "г": "nom_m", "году": "prep_m", "года": "gen_m",
            "годах": "prep_m", "годов": "gen_m", "гг": "gen_m"}
    word = {"год": "год", "г": "году", "году": "году", "года": "года",
            "годах": "годах", "годов": "годов", "гг": "годов"}
    key = m.group(2).lower().rstrip(".")
    # хвостовой пробел обязателен: шаблон съедает его вместе с сокращением
    return _ordinal(y, form.get(key, "nom_m")) + " " + word.get(key, "году") + " "


def _fix_phone(m: re.Match) -> str:
    """Телефоны/длинные номера читаем по цифрам, как человек."""
    raw = m.group(0)
    groups = [g for g in re.split(r"[^\d]+", raw) if g]
    if len(groups) > 1:
        return " , ".join(to_words(int(g)) if len(g) <= 3 else
                          " ".join(_UNITS["m"][int(c)] for c in g)
                          for g in groups)
    digits = re.sub(r"\D", "", raw)
    return " ".join(_UNITS["m"][int(c)] for c in digits)


def _fix_url(m: re.Match) -> str:
    url = m.group(0)
    body = re.sub(r"^[a-zA-Z]+://", "", url).rstrip("/")
    parts = [p for p in re.split(r"[/.]", body) if p]
    out: list[str] = []
    for i, p in enumerate(parts):
        out.append(_latin_word(re.match(r"[A-Za-z0-9\-]+", p))
                   if re.match(r"[A-Za-z]", p) else
                   " ".join(_UNITS["m"][int(c)] for c in p if c.isdigit()))
        if i < len(parts) - 1:
            out.append(" точка ")
    return " ".join(x for x in out if x).replace(" точка  точка ", " точка ")


def _fix_year_range(m: re.Match) -> str:
    a, b = int(m.group(1)), int(m.group(2))
    out = ("с " + _ordinal(a, "gen_m") + " по " + _ordinal(b, "nom_m")
           + " год")
    return out[0].upper() + out[1:] if m.group(0)[:1].isupper() else out


def _fix_unit(m: re.Match) -> str:
    raw = m.group(1).replace("\u00a0", " ").replace("\u202f", " ").strip()
    key = m.group(2).lower()
    one, few, many = _UNITS_MAP.get(key, (key, key, key))
    if "," in raw or "." in raw:
        whole, frac = re.split(r"[.,]", raw, maxsplit=1)
        # при дробном числе нужен родительный падеж единственного числа,
        # он совпадает с формой для 2-4
        return _decimal(whole, frac) + " " + few
    n = int(re.sub(r"\D", "", raw))
    return to_words(n) + " " + plural(n, one, few, many)


def _fix_time(m: re.Match) -> str:
    h, mi = int(m.group(1)), int(m.group(2))
    out = to_words(h) + " " + plural(h, "час", "часа", "часов")
    if mi:
        out += " " + to_words(mi) + " " + plural(mi, "минута", "минуты", "минут")
    return out


def _fix_percent(m: re.Match) -> str:
    raw = m.group(1).replace(",", ".")
    if "." in raw:
        whole, frac = raw.split(".", 1)
        val = _decimal(whole, frac)
        return val + " процента"
    n = int(raw)
    return to_words(n) + " " + plural(n, "процент", "процента", "процентов")


def _sanitize(text: str, keep_stress: bool) -> str:
    allowed = _ALLOWED_EXTRA | set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
                                   "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
    if keep_stress:
        allowed.add("+")
    out = []
    for ch in text:
        if ch in allowed:
            out.append(ch)
        elif ch in "—–":
            out.append("-")
        elif ch == ";":
            out.append(",")
        elif ch.isdigit():
            # цифра осталась = что-то не разобрали; читаем по одной
            out.append(" " + _UNITS["m"][int(ch)] + " ")
        else:
            out.append(" ")
    res = "".join(out)
    res = re.sub(r"\s+", " ", res)
    res = re.sub(r"\s+([,.!?:])", r"\1", res)
    res = re.sub(r"-{2,}", "-", res)
    return res.strip()


# Сокращения: модель их не знает и просто выбрасывает.
_ABBR = {
    "шт": "штук", "тк": "так как", "те": "то есть", "тд": "так далее",
    "тп": "тому подобное", "др": "другие", "пр": "прочее",
    "напр": "например", "рис": "рисунок", "стр": "страница",
    "тыс": "тысяч", "млн": "миллионов", "млрд": "миллиардов",
    "коп": "копеек", "ед": "единиц", "чел": "человек",
    "сек": "секунд", "мин": "минут", "гр": "граждан",
    "им": "имени",
}
_SCALE_ABBR = {"тыс": ("тысяча", "тысячи", "тысяч"),
               "млн": ("миллион", "миллиона", "миллионов"),
               "млрд": ("миллиард", "миллиарда", "миллиардов")}
_RE_SCALE_ABBR = re.compile(r"(?<![\d.,])(\d{1,3}(?:\s\d{3})+|\d+(?:[.,]\d+)?)"
                            r"\s*(тыс|млн|млрд)\.?", re.IGNORECASE)


def _fix_scale_abbr(m: re.Match) -> str:
    raw = re.sub(r"\s", "", m.group(1))
    one, few, many = _SCALE_ABBR[m.group(2).lower()]
    if "," in raw or "." in raw:
        whole, frac = re.split(r"[.,]", raw, maxsplit=1)
        return _decimal(whole, frac) + " " + few
    n = int(raw)
    return to_words(n) + " " + plural(n, one, few, many)


_RE_ABBR = re.compile(
    r"(?<![а-яё\w])(шт|т\.?\s?к|т\.?\s?е|т\.?\s?д|т\.?\s?п|др|пр|напр"
    r"|рис|стр|тыс|млн|млрд|коп|ед|чел|сек|мин|гр|им)\.",
    re.IGNORECASE)

_RE_URL = re.compile(r"(?:https?|ftp)://[^\s]+")
# Телефон = либо международный «+», либо минимум две группы через -()
# (без этого шаблон съедал разряды «1 000 000»)
_RE_PHONE = re.compile(
    r"(?<![\d.,])(?:\+\d[\d \-()]{5,}\d"
    r"|\d{1,4}(?:[-()]\d{1,5}){2,})(?!\d)")
_RE_MINUS = re.compile(r"(?<![\w\d])[-\u2212](?=\s?\d)")
_RE_YEAR_RANGE = re.compile(
    r"\bс\s+(1\d{3}|20\d{2})\s+по\s+(1\d{3}|20\d{2})\s*(?:гг\.?|года?|г\.?)?",
    re.IGNORECASE)

# Единицы измерения: число перед ними должно согласовываться
_UNITS_MAP = {
    "кг": ("килограмм", "килограмма", "килограммов"),
    "г": ("грамм", "грамма", "граммов"),
    "км": ("километр", "километра", "километров"),
    "м": ("метр", "метра", "метров"),
    "см": ("сантиметр", "сантиметра", "сантиметров"),
    "мм": ("миллиметр", "миллиметра", "миллиметров"),
    "л": ("литр", "литра", "литров"),
    "мл": ("миллилитр", "миллилитра", "миллилитров"),
    "ч": ("час", "часа", "часов"),
    "мин": ("минута", "минуты", "минут"),
    "сек": ("секунда", "секунды", "секунд"),
    "мс": ("миллисекунда", "миллисекунды", "миллисекунд"),
    "гб": ("гигабайт", "гигабайта", "гигабайт"),
    "мб": ("мегабайт", "мегабайта", "мегабайт"),
    "кб": ("килобайт", "килобайта", "килобайт"),
    "тб": ("терабайт", "терабайта", "терабайт"),
    "вт": ("ватт", "ватта", "ватт"),
    "гц": ("герц", "герца", "герц"),
    "кгц": ("килогерц", "килогерца", "килогерц"),
}
_RE_UNIT = re.compile(
    r"(?<![\d.,])(\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+(?:[.,]\d+)?)"
    r"\s*(гб|мб|кб|тб|кгц|кг|км|см|мм|мл|мс|мин|сек|ч|г|л|вт|гц)"
    r"(?![а-яё\w])",
    re.IGNORECASE)


def normalize(text: str, *, latin: bool = True, keep_stress: bool = True) -> str:
    """Приводит текст к виду, который Silero реально произносит.

    latin=False отключает трогание латиницы (полезно, если модель другая),
    keep_stress=False выбрасывает `+` вместе с прочими служебными знаками.
    """
    if not text:
        return ""

    # 1. типографика -> обычные символы
    t = (text.replace("«", '"').replace("»", '"').replace("“", '"')
             .replace("”", '"').replace("„", '"').replace("’", "'")
             .replace("…", "...").replace("\u00a0", " ").replace("\u202f", " "))

    # 2. «3 тыс.» -> «три тысячи» (до общих сокращений и цифр)
    t = _RE_SCALE_ABBR.sub(_fix_scale_abbr, t)
    # «3 тысячи руб.» — после масштаба валюта должна стоять в родительном
    # множественного: «три тысячи рублей», а не «три тысячи руб.»
    t = re.sub(
        r"(тысяч[аи]?|тысяч|миллион[аов]*|миллиард[аов]*)\s+"
        r"(руб\.|руб|рублей|рубля|рубль|долларов|доллара|доллар|коп\.|коп|копеек)",
        lambda m: m.group(1) + " " + {
            "руб": "рублей", "руб.": "рублей", "рублей": "рублей",
            "рубля": "рублей", "рубль": "рублей",
            "доллар": "долларов", "доллара": "долларов",
            "долларов": "долларов", "коп": "копеек",
            "коп.": "копеек", "копеек": "копеек",
        }.get(m.group(2).lower(), m.group(2)), t)

    # сокращения: «5 шт.» -> «5 штук» -> «пять штук»
    t = _RE_ABBR.sub(lambda m: " " + _ABBR.get(
        re.sub(r"[\s.]", "", m.group(1)).lower(), m.group(1)) + " ", t)

    # 3. URL и телефоны — раньше всего цифрового, иначе разваливаются
    t = _RE_URL.sub(_fix_url, t)
    t = _RE_PHONE.sub(_fix_phone, t)

    # 3. словарь терминов (до транслита и до цифр)
    t = _RE_DICT.sub(lambda m: _LATIN_DICT[m.group(0).lower()], t)

    # 4. структурированные числа: сначала самые специфичные
    t = _RE_YEAR_RANGE.sub(_fix_year_range, t)
    t = _RE_DATE.sub(_fix_date, t)
    t = _RE_YEAR.sub(_fix_year, t)
    t = _RE_TIME.sub(_fix_time, t)
    t = _RE_MINUS.sub(" минус ", t)
    t = _RE_NUMERO.sub(lambda m: "номер " + to_words(int(m.group(1))), t)
    t = _RE_PERCENT.sub(_fix_percent, t)
    # валюта до разрядов: шаблон сам забирает «1 000 000», иначе от суммы
    # останется только хвост, а знак единицы потеряется
    t = _RE_CUR_PRE.sub(lambda m: _currency(m.group(2), m.group(1)), t)
    t = _RE_CUR_POST.sub(
        lambda m: _currency(int(re.sub(r"\D", "", m.group(1))), m.group(2)), t)
    # единицы — до разрядов и дробей, иначе «2,5 кг» распадётся
    t = _RE_UNIT.sub(_fix_unit, t)
    t = _RE_THOUSANDS.sub(
        lambda m: to_words(int(re.sub(r"\D", "", m.group(1)))), t)
    t = _RE_ORD_SUF.sub(
        lambda m: _ordinal(int(m.group(1)),
                           _ORD_FORMS.get(m.group(2).lower(), "nom_m")), t)
    t = _RE_RANGE.sub(
        lambda m: to_words(int(m.group(1))) + " - " + to_words(int(m.group(2))), t)
    t = _RE_DECIMAL.sub(lambda m: _decimal(m.group(1), m.group(2)), t)
    t = _RE_NUM.sub(lambda m: _num_words(int(m.group(1)), t, m.end()), t)

    # 5. латиница
    if latin:
        t = _RE_LATIN.sub(_latin_word, t)

    # 6. чистка под токенсет
    return _sanitize(t, keep_stress)
