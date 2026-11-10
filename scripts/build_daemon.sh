#!/bin/bash
# Сборка нативного LLM-демона против собранного форка llama.cpp.
set -e
ROOT=${ROOT:-$HOME/cmp-voice}
LLAMA=${LLAMA:-$ROOT/src/llama-prism}
OUT=${OUT:-$ROOT/daemon/bonsai_daemon}

if [ ! -f "$LLAMA/build/bin/libllama.so" ]; then
  echo "сначала соберите llama.cpp: $ROOT/scripts/build_llama.sh" >&2
  exit 1
fi

g++ -O2 -std=c++17 -pthread \
  -I"$LLAMA/include" \
  -I"$LLAMA/ggml/include" \
  -I"$LLAMA/vendor" \
  "$ROOT/daemon/bonsai_daemon.cpp" \
  -o "$OUT" \
  -L"$LLAMA/build/bin" -lllama -lggml -lggml-base \
  -Wl,-rpath,"$LLAMA/build/bin"

echo "OK: $OUT"
ls -la "$OUT"
