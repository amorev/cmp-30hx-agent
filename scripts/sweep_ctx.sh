#!/bin/bash
# Подбор контекста: матрица ctx -> веса/KV/compute/свободно + работает ли decode.
# Критерий 12 и таблица VRAM в отчёте (раздел 10 ТЗ).
#
# Запуск: scripts/sweep_ctx.sh "16384 24576 32768 49152 65536"
set -uo pipefail
ROOT=${ROOT:-$HOME/cmp-voice}
CTXS=${1:-"16384 24576 32768 49152 65536"}
MODEL=${BONSAI_MODEL:-$HOME/models/bonsai9/PQ2_0.gguf}
OUT=$ROOT/reports/ctx_sweep.txt
mkdir -p "$ROOT/reports"
: > "$OUT"

echo "ctx,weights_mib,kv_mib,total_mib,free_mib,decode_ok,prefill_ms,base_tokens" | tee -a "$OUT"

for CTX in $CTXS; do
  echo "--- ctx=$CTX ---" >&2
  sudo -n systemctl stop cmp-voice.service >/dev/null 2>&1
  sudo -n systemctl stop cmp-qwen-tts.service >/dev/null 2>&1
  sudo -n systemctl stop cmp-qwen08.service >/dev/null 2>&1
  sudo -n systemctl stop cmp-llm.service >/dev/null 2>&1
  sleep 3

  export CUDA_VISIBLE_DEVICES=0
  export LD_LIBRARY_PATH=$ROOT/src/llama-prism/build/bin
  export BONSAI_MODEL=$MODEL
  export BONSAI_CTX=$CTX
  export BONSAI_BATCH=256
  export BONSAI_KV_Q8=1
  export BONSAI_WARMUP=0
  export BONSAI_PORT=18599

  nohup "$ROOT/daemon/bonsai_daemon" > "$ROOT/logs/sweep_$CTX.log" 2>&1 &
  PID=$!

  DECODE_OK=0; PREFILL=""; BASE=""
  for i in $(seq 1 120); do
    if grep -q "\[ready\]" "$ROOT/logs/sweep_$CTX.log" 2>/dev/null; then DECODE_OK=1; break; fi
    if ! kill -0 $PID 2>/dev/null; then break; fi
    sleep 2
  done

  TOTAL=0
  if [ $DECODE_OK -eq 1 ]; then
    R=$("$ROOT/.venv/bin/python" - <<PY
import socket, json
def cmd(o):
    s=socket.create_connection(("127.0.0.1",18599),timeout=180)
    s.sendall((json.dumps(o)+"\n").encode())
    f=s.makefile("r",encoding="utf-8")
    line=f.readline(); s.close(); return json.loads(line)
st=cmd({"cmd":"stat"})
pr=cmd({"cmd":"prefill","text":"Проверка контекста","history":[]})
print(pr.get("ms_f",0), st.get("base_tokens",0), st.get("kv_tokens",0))
PY
)
    PREFILL=$(echo "$R" | awk '{print $1}')
    BASE=$(echo "$R" | awk '{print $2}')
    DECODE_OK=$([ -n "$PREFILL" ] && echo 1 || echo 0)
    TOTAL=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  fi

  KVTYPE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  FREE=$((6144 - ${TOTAL:-0}))
  # веса ≈ 2355 МиБ (файл 2.42 ГБ), KV оценивается из остатка
  WEIGHTS=2355
  KV=$(( ${TOTAL:-0} - WEIGHTS ))

  kill $PID 2>/dev/null
  wait $PID 2>/dev/null
  sleep 2

  if grep -qi "out of memory\|cudaMalloc failed" "$ROOT/logs/sweep_$CTX.log" 2>/dev/null; then
    DECODE_OK=0
  fi

  echo "$CTX,$WEIGHTS,$KV,${TOTAL:-0},$FREE,$DECODE_OK,$PREFILL,$BASE" | tee -a "$OUT"
done

sudo -n systemctl start cmp-llm.service >/dev/null 2>&1
echo "готово: $OUT"
