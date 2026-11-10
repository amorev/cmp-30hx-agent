#!/bin/bash
# Silero TTS v5_6 ru — отдельный процесс (torch CPU-only).
# Отдельный процесс обязателен: torch и onnxruntime-gpu в одном процессе
# конфликтуют по CUDA-контекстам и дают segfault (ТЗ, грабли 15).
set -e
ROOT=${ROOT:-$HOME/cmp-voice}
PY=${SILERO_PYTHON:-$ROOT/.venv-silero/bin/python}
MODEL=${SILERO_MODEL:-$ROOT/models/silero/v5_6_ru.pt}

[ -x "$PY" ] || { echo "нет venv silero: $PY" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "нет модели: $MODEL" >&2; exit 1; }

# torch строго на CPU: CUDA не должна быть видна, иначе конфликт с onnxruntime
export CUDA_VISIBLE_DEVICES=""
export SILERO_MODEL="$MODEL"
export SILERO_VOICE="${SILERO_VOICE:-xenia}"
export SILERO_PORT="${SILERO_PORT:-8023}"
export SILERO_THREADS="${SILERO_THREADS:-3}"
export PYTHONUNBUFFERED=1

cd "$ROOT"
echo "[silero] model=$MODEL voice=$SILERO_VOICE port=$SILERO_PORT threads=$SILERO_THREADS"
exec "$PY" -m uvicorn silero.app:app --host 127.0.0.1 --port "$SILERO_PORT" --log-level info
