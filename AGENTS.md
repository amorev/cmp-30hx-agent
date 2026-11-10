# AGENTS.md — CMP Voice Agent

Инструкции для агента-исполнителя: как развернуть, проверить и отладить
голосовой LLM-сервис на одной NVIDIA CMP 30HX (6 ГиБ).

> Читать целиком перед любыми действиями на сервере. Раздел 8 — список
> грабель, каждая уже стоила времени; не наступать повторно.

---

## 1. Что это за проект

Голосовой агент: микрофон браузера → стриминговый ASR → **инкрементальный
prefill** в KV запущенной LLM → стриминг ответа токенами → потоковый TTS.
Плюс текстовый ввод (Enter / «Enter по словам»), инструменты с поиском в
интернете через MCP и панель KV-кэша в интерфейсе.

Ключевое отличие от обычного `llama-server`: промпт партиала обрывается сразу
после текста пользователя, поэтому каждый следующий партиал продолжает
предыдущий токен-в-токен, и из кэша берётся весь префикс. Это главное
требование ТЗ — если prefill считается с нуля, задача провалена.

**ТЗ:** `TZ_single_cmp30hx.md` (в корне, читать обязательно).
**Отчёт с замерами:** `REPORT.md`.
**Пользовательская документация:** `README.md`.

---

## 2. Состав репозитория

```
daemon/          нативный LLM-демон на C++ (llama.cpp C API, ручной KV)
  bonsai_daemon.cpp
voice/           Python/ASGI voice-сервис
  app.py         оркестрация: ASR → prefill → генерация → TTS → WebSocket
  asr.py         sherpa-onnx T-one CTC, один recognizer на сеанс
  llm_client.py  TCP-клиент демона + LastWinsPrefill (слияние партиалов)
  tts.py         реестр движков: silero (HTTP), kokoro-ru (ONNX)
  tools.py       MCP web_search/web_fetch + время/калькулятор/таймер
  config.py      свой парсер .env, ЛЕНИВЫЕ геттеры
  static/index.html  интерфейс (панель KV, переключатели, текстовый ввод)
silero/          Silero TTS v5_6 ru — отдельный процесс (torch CPU-only)
units/           4 юнита systemd
scripts/         сборка, режимы VRAM, запуск сервисов
tests/           приёмочные тесты (см. раздел 7)
.env.example     все переменные с дефолтами целевой машины
```

---

## 3. Стек и версии (проверено на целевой машине)

| Компонент | Версия / источник |
|---|---|
| llama.cpp | форк `github.com/PrismML-Eng/llama.cpp`, ветка **`prism`**, HEAD `6447906` |
| Модель LLM | `Ternary-Bonsai-2-9B` PQ2_0, arch `qwen35`, 2.42 ГБ |
| Qwen3-TTS | **выведен из поставки** (VRAM отдана модели) |
| Silero | v5_6 ru, `v5_6_ru.pt` 147 МБ, диктор `xenia`, torch 2.15 **CPU-only** |
| Kokoro | `zaakirio/kokoro-ru`, `onnx/model.onnx` **fp32** 325 МБ |
| ASR | `sherpa-onnx-streaming-t-one-russian-2025-09-08`, sherpa_onnx 2.0.1 |
| GPU | CMP 30HX (TU116), sm_75, драйвер 580.x |
| Сборка | cmake 3.28, nvcc 12.0, gcc 13.3, `CMAKE_CUDA_ARCHITECTURES=75` |

**Почему форк обязателен:** stock llama.cpp знает архитектуру `qwen35`, но не
знает тип `PQ2_0` (`GGML_TYPE_PQ2_0 = 142`) и не применяет runtime-преобразование
Адамара (`prism.hadamard.*`, `fwht.cu`). Без форка модель не грузится, а при
подмене на F16 — генерирует бессвязный текст.

---

## 4. Архитектура процессов

| Процесс | Порт | Устройство | Роль |
|---|---|---|---|
| `cmp-llm` | 18521 | GPU0 | демон: `llama_context`, ручной KV, инкрементальный prefill, снапшот базы |
| `cmp-voice` | 8110 | CPU | ASR, оркестрация, WebSocket, статика |
| `cmp-silero-tts` | 8023 | **CPU** | Silero v5_6, PCM s16le 48 кГц |
| `cmp-qwen08` | 8081 | GPU0 | dev: Qwen3.5-0.8B (`llama-server`, OpenAI API + Bearer), по умолчанию выключен |

Kokoro-ru живёт **внутри** `cmp-voice` (ONNX-сессия), отдельного юнита нет.

**Все процессы пиннятся на GPU0** через `CUDA_VISIBLE_DEVICES=0`. Машина может
быть двухкарточной — вторую не трогать.

---

## 5. Быстрый старт (полный цикл)

```bash
# 0. предпосылки: git, cmake, nvcc, gcc, python3.12, ffmpeg, ninja
#    диск >= 40 ГБ свободно, RAM >= 11 ГБ

# 1. токен MCP (права 600, одна строка, без кавычек!)
printf '%s' '<ТОКЕН>' > ~/.wormsoft-token
chmod 600 ~/.wormsoft-token

# 2. код и окружение
git clone <repo> ~/cmp-voice && cd ~/cmp-voice
cp .env.example .env && chmod 600 .env
python3 -m venv .venv
.venv/bin/pip install fastapi uvicorn numpy sherpa-onnx onnxruntime-gpu \
                        phonemizer espeakng-loader websockets
.venv/bin/pip install nvidia-cudnn-cu12      # нужен CUDA-провайдеру ONNX

# 3. сборка форка (30–60 мин на 4 ядрах, это самый долгий шаг)
./scripts/build_llama.sh
./scripts/build_daemon.sh

# 4. модели (если их нет на машине)
#    ~/models/bonsai9/PQ2_0.gguf                 (2.42 ГБ, PQ2_0)
#    ~/cmp-voice/models/kokoro-ru-fp32/          (model.onnx fp32 + *.bin + config.json)
#    ~/cmp-voice/models/silero/v5_6_ru.pt        (147 МБ)
#    ~/tone_asr/sherpa-onnx-streaming-t-one-russian-2025-09-08/

# 5. venv для silero (torch строго CPU!)
python3 -m venv .venv-silero
.venv-silero/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-silero/bin/pip install fastapi uvicorn numpy pydantic

# 6. юниты
sudo cp units/*.service /etc/systemd/system/
sudo cp .env /etc/cmp-voice.env && sudo chmod 600 /etc/cmp-voice.env
sudo systemctl daemon-reload
sudo systemctl enable --now cmp-llm cmp-silero-tts
sleep 20
sudo systemctl enable --now cmp-voice

# 7. проверка
curl -s localhost:8110/health | python3 -m json.tool
```

Интерфейс: `http://<host>:8110/` (или через SSH-туннель, см. ниже).

---

## 6. Доступ к интерфейсу и микрофон

Браузер разрешает микрофон **только в защищённом контексте** (`localhost`
или HTTPS). По IP без TLS — молча блокирует.

```bash
ssh -N -L 8110:127.0.0.1:8110 ubuntu@<host>
# открыть http://localhost:8110/
```

Если туннель запускается из скрипта или агентом — обязательно полное
отсоединение, иначе процесс наследует пайп и команда «висит» навсегда:

```bash
nohup ssh -f -N -L 8110:127.0.0.1:8110 host < /dev/null >/tmp/tun.log 2>&1 &
```

---

## 7. Тесты и что они доказывают

```bash
.venv/bin/python tests/test_prefill.py        # критерии 2,3: cached растёт, new<=10
.venv/bin/python tests/bench_llm.py           # критерий 4: первый токен, ток/с
.venv/bin/python tests/bench_tts.py           # критерии 5,6,7: TTFB, RTF
.venv/bin/python tests/test_tools.py --llm    # критерии 10,10a: MCP, деградация
.venv/bin/python tests/test_asr.py            # критерий 9: контрольный wav
.venv/bin/python tests/test_e2e_ws.py         # критерии 8,11,13,14,15
.venv/bin/python tests/test_stream_prefill.py # ГЛАВНОЕ: prefill во время речи
.venv/bin/python tests/test_b8_lock.py        # B8: блокировка переключателей
.venv/bin/python tests/repro_tools.py         # ходы с инструментами
```

**`test_stream_prefill.py` — самый важный.** Он подаёт wav чанками по 100 мс
(как браузер) и проверяет, что события `kv` приходят **до** `final`.
Ожидаемо: ~11 партиалов → ~11 строк KV, `cached` растёт во время речи.
Если строк KV 1–2 — инкремент сломан, см. грабли 1 и 2.

---

## 8. Грабли (каждая уже стоила времени)

### 8.1. Префилл не успевает за речью → «последний выигрывает» через отмену

**Симптом:** во время речи KV не растёт, все партиалы «теряются», cached=706
на всех строках, генерация стартует с полным пересчётом.

**Причина:** наивная реализация отменяет текущий prefill при новом партиале.
Отмена на клиенте **не останавливает демон** — он досчитывает начатое, а новый
запрос встаёт в очередь. При партиалах каждые 250–300 мс ни один не успевает
финишировать. Замер до правки: 11 партиалов → 1 строка KV.

**Решение:** в `LastWinsPrefill` — слияние, не отмена: один prefill в работе +
один отложенный (самый свежий). После: 11 партиалов → 11 строк KV.

### 8.2. `set_tools` стирает KV

`set_tools` пересчитывает снапшот базы (~4 с) и **обнуляет накопленный KV**.
Если дёргать его из `/health` на каждый запрос — кэш обнуляется каждые
4 секунды и вся работа партиалов пропадает.

**Решение:** троттлинг ≥15 с и запрет во время речи/хода (`_tools_check`).

### 8.3. Литерал ` 248069` токенизатор выбрасывает

Подаёшь закрывающий маркер think-блока текстом — получаешь **0 токенов**.
Поэтому он вставляется **токеном по id**: найти служебный токен, чей текст
равен `` (в Qwen3 это 248069), и вставлять `llama_token` напрямую.
`THINK_OPEN` при этом идёт обычным текстом.

### 8.4. `llama_sampler_sample` требует индекс `-1`

При `n_outputs_max=1` индекс должен быть **`-1`** (последний выход батча),
не `0`. С `0` — `GGML_ASSERT(logits != nullptr)` и падение.

### 8.5. Логиты нужны только на последнем куске батча

`decode_range` режет промпт на куски по `n_batch`. Флаг
`batch.logits[i] = (want_logits && last_chunk && i == n-1) ? 1 : 0`.
Иначе `llama_get_logits_ith` возвращает NULL.

### 8.6. Prefill уже оставил KV на конце промпта

Если `common >= toks.size()`, декодировать нечего, но логиты нужны. Нельзя
декодировать последний токен повторно в занятую позицию — сначала
`rollback(common-1)`, потом `decode_range(common-1, size, want_logits=true)`.

### 8.7. Разметка вызова не в истории → модель «не понимает», что звала функцию

В историю надо писать **сырой текст генерации** (`done.raw`) вместе с
`<tool_call>`, а не очищенный ответ. Иначе на результате поиска модель отвечает
«откуда вы хотели узнать?» или выдумывает факты.

### 8.8. Пустая реплика user при продолжении после инструмента

После результата инструмента промпт уже заканчивается блоком `<tool_response>`,
новая реплика user **не добавляется**. В протоколе — флаг `continue: true`,
в `tokenize_segments` — `add_user=false`.

### 8.9. `reset()` в повторе пустого ответа обнуляет кэш

При немедленном EOG повторять надо **на том же состоянии** (демон уже откатил
KV на длину промпта). `reset()` теряет весь префикс: cached падал с 1137 до 776.

### 8.10. `parse_special` обязателен

Без него ChatML-разметка рассыпается на обычный текст и модель не видит границ
реплик. `llama_tokenize(..., parse_special=true)`.

### 8.11. `n_outputs_max = 1`, батч 256

Без `n_outputs_max=1` llama.cpp резервирует logits-буфер на весь батч (сотни
МиБ при словаре ~250k) и падает с OOM. Батч 256 (не 512) — иначе
`failed to allocate compute pp buffers`.

### 8.12. История ассистента рендерится ровно как в KV

Демон добавляет `assistant\n think\n\n` + токен `` + `\n\n` перед `content`.
Значит `content` в истории — **только текст ответа**, без префиксов. Лишние
`\n\n\n\n` ломают совпадение промпта с KV → полный пересчёт каждый ход.

### 8.13. torch и onnxruntime-gpu в одном процессе → segfault

Поэтому Silero — **отдельный процесс** с `CUDA_VISIBLE_DEVICES=""` и своим
venv (`.venv-silero`, torch CPU-only). Бонус: RTF 0.07 и ноль VRAM.

### 8.14. CUDA-провайдеру ONNX нужен системный cuDNN

`pip install nvidia-cudnn-cu12` + предзагрузка `.so` из
`site-packages/nvidia/*/lib` через `ctypes.CDLL(..., RTLD_GLOBAL)`.
Без этого `dlopen failed for libcudnn.so`. При нехватке VRAM — штатный откат
на CPU (`KokoroTTS._load`).

### 8.15. Конфиг читается на импорте → .env не успевает загрузиться

Все значения — **ленивые геттеры** в `voice/config.py` (`config.llm_port()` и
т.п.), никаких модульных констант. Иначе результат зависит от порядка импортов.

### 8.16. `EnvironmentFile` перебивает `Environment` в systemd

Значение из `/etc/cmp-voice.env` имеет приоритет над `Environment=` в юните.
Если правишь контекст — правь **env-файл**, иначе изменения не применятся.

### 8.17. Долгий ход блокирует чтение WebSocket

Если обработчик команды ждёт завершения хода, переключатели не обрабатываются
до конца ответа. Ход выносится в фоновую задачу (`Session.spawn_turn`).

### 8.18. `pkill -f <подстрока>` убивает собственную сессию

Только по PID: `ps -eo pid,args | awk "/[b]onsai_daemon/ {print \$1}"`.

### 8.19. heredoc через ssh ломает кавычки и `\n`

Многострочные скрипты с python-кодом через ssh-обёртки портятся на кавычках и
экранировании. **Писать скрипт в файл и запускать файл.** Особенно осторожно с
`\n` внутри строковых литералов — они превращаются в реальные переводы строк.

### 8.20. Порядок запуска: TTS-сервер раньше voice

Если TTS-серверу нужна VRAM, он должен стартовать **первым** (`After/Wants`),
иначе не получает непрерывный блок памяти.

### 8.21. Модель 9B нестабильна в решении «звать ли инструмент»

Qwen3.5-Bonsai-9B:
- «кто владелец FromSoftware» → **не вызовет** поиск;
- «найди в интернете кто владелец FromSoftware» → вызовет;
- «Вызови поиск» → вызовет.

Это свойство модели, не дефект сервиса. Для проверки инструментов формулировать
запрос явно.

---

## 9. Диагностика

```bash
# состояние
curl -s localhost:8110/health | python3 -m json.tool
curl -s localhost:8110/config | python3 -m json.tool

# что реально в KV
printf '{"cmd":"stat"}\n' | nc 127.0.0.1 18521

# логи
sudo journalctl -u cmp-llm -n 50 --no-pager
sudo journalctl -u cmp-voice -n 50 --no-pager   # грепать без scatter_nd/Memcpy
sudo journalctl -u cmp-silero-tts -n 30 --no-pager

# VRAM
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

**Что должно быть в `/health`:** `status: ok`, `llm.base_tokens ≈ 776`,
`llm.tools: 6`, `llm.kv_type: q8_0`, `llm.flash_attn: true`,
`tools_token: {present: true, source: <путь>}` (значение не отдаётся никогда).

---

## 10. VRAM-бюджет (замерено)

| Режим | Состав | Занято | Свободно |
|---|---|---|---|
| **voice** (текущий) | LLM 9B ctx 24576 + Silero (CPU) + Kokoro (CPU) | 2751 | 3393 |
| voice + Kokoro CUDA | LLM 9B + Kokoro на CUDA | ~3400 | ~2700 |
| dev | Qwen3.5-0.8B ctx 32K | ~3000 | ~3100 |

- Демон в одиночку держит **ctx до 131072** (4541 МиБ).
- KV q8_0 ≈ **17.4 КиБ/токен**.
- Контекст переключается через `BONSAI_CTX` в `/etc/cmp-voice.env` (не в юните!).
- Скрипт `scripts/mode.sh voice|voice+kokoro|dev|all-in|status`.

---

## 11. Ожидаемые показатели (проверка «всё ли работает»)

| Метрика | Ожидаемо |
|---|---|
| База префикса в KV | 706–776 токенов |
| Prefill партиала | медиана 20–40 мс, `new` 1–10 |
| Первый токен | 200–350 мс |
| Скорость генерации | 19–26 ток/с |
| Silero TTFB / RTF | 88–239 мс / 0.07 |
| Kokoro CUDA RTF | 0.12–0.28 |
| Свободно VRAM | ≥ 100 МиБ (факт 3393) |

Если prefill партиала показывает `cached=0` или `new` десятки на каждом
партиале — инкремент сломан, искать в разделе 8.1/8.12.

---

## 12. Правила работы

1. **Не использовать готовые исходники на сервере** — только этот репозиторий.
   Модели/веса использовать можно, повторно не качать.
2. **Секреты:** токен только в `~/.wormsoft-token` (права 600) или в
   `WORM_TOKEN`. В логи, `/health` и ответы инструментов — только
   `{present, source}`. Значение не печатать никогда.
3. **Правки — локальным редактором**, затем синхронизация на сервер. Не
   генерировать код через `ssh 'here-doc'` (см. 8.19).
4. **После каждой правки — проверка:** собрать, перезапустить, прогнать
   релевантный тест из раздела 7.
5. **Числа без замера в отчёт не попадают.** Все выводы — из вывода тестов.
