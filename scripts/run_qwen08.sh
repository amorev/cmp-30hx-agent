#!/bin/bash
# Dev-сервис: Qwen3.5-0.8B через llama-server (OpenAI-совместимый API + Bearer).
set -e
ROOT=${ROOT:-$HOME/cmp-voice}
BIN=$ROOT/src/llama-prism/build/bin/llama-server
MODEL=${QWEN08_MODEL:-$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf}
PORT=${QWEN08_PORT:-8081}
CTX=${QWEN08_CTX:-32768}
KEYFILE=${QWEN08_API_KEY_FILE:-$ROOT/run/qwen08-apikey}

if [ ! -x "$BIN" ]; then echo "нет llama-server: $BIN" >&2; exit 1; fi
if [ ! -f "$MODEL" ]; then echo "нет модели: $MODEL" >&2; exit 1; fi

# ключ в файле, чтобы не светился в ps
mkdir -p "$(dirname "$KEYFILE")"
if [ ! -s "$KEYFILE" ]; then
  head -c 24 /dev/urandom | base64 | tr -d '\n=+/' > "$KEYFILE"
  chmod 600 "$KEYFILE"
fi

export LD_LIBRARY_PATH=$ROOT/src/llama-prism/build/bin
echo "[qwen08] model=$MODEL ctx=$CTX port=$PORT keyfile=$KEYFILE"
exec "$BIN" \
  -m "$MODEL" \
  -ngl 999 -fa on -t 4 \
  -c "$CTX" -b 256 -ub 256 --parallel 1 \
  -ctk q8_0 -ctv q8_0 \
  --reasoning off --jinja \
  --host 127.0.0.1 --port "$PORT" \
  --api-key-file "$KEYFILE"
