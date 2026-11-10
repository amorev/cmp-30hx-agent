#!/bin/bash
# Переключение режимов VRAM на одной карте (раздел 10 ТЗ).
#
#   voice        боевой: LLM-демон 9B + TTS qwen(0.6B). Kokoro на CPU. qwen08 выкл.
#   voice+kokoro LLM-демон 9B + Kokoro на CUDA. TTS qwen-сервер выключен.
#   dev          qwen08 (ctx 32K) + TTS qwen. LLM-демон 9B выключен.
#   all-in       всё сразу, только если замеры позволят.
#
# Использование: scripts/mode.sh voice|voice+kokoro|dev|all-in|status
set -uo pipefail
ROOT=${ROOT:-$HOME/cmp-voice}
MODE=${1:-status}

sst() { sudo -n systemctl "$@" 2>/dev/null; }
act() { systemctl is-active "$1" 2>/dev/null; }

show() {
  echo "=== режим: ${1:-?} ==="
  for u in cmp-llm cmp-silero-tts cmp-qwen08 cmp-voice; do
    printf "  %-14s %s\n" "$u" "$(act "$u.service")"
  done
  echo "--- VRAM ---"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
  echo "--- процессы на GPU ---"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
}

stop_all() {
  for u in cmp-voice cmp-silero-tts cmp-qwen08 cmp-llm; do
    sst stop "$u.service" >/dev/null 2>&1
  done
  # дождаться освобождения VRAM
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "${used:-0}" -lt 200 ] && break
    sleep 1
  done
}

case "$MODE" in
  voice)
    stop_all
    sst start cmp-silero-tts.service
    sst start cmp-silero-tts.service
    sleep 2
    sst start cmp-llm.service
    sleep 3
    sst start cmp-voice.service
    sleep 2
    show voice
    ;;
  voice+kokoro)
    stop_all
    # Kokoro живёт внутри voice-сервиса; TTS qwen-сервер не нужен
    sst start cmp-llm.service
    sleep 3
    TTS_BACKEND=kokoro sst start cmp-voice.service
    sleep 2
    show voice+kokoro
    echo "  (Kokoro на CUDA внутри cmp-voice; TTS-сервер qwen выключен)"
    ;;
  dev)
    stop_all
    sst start cmp-silero-tts.service
    sst start cmp-qwen08.service
    sleep 2
    show dev
    ;;
  all-in)
    stop_all
    sst start cmp-qwen08.service
    sst start cmp-silero-tts.service
    sst start cmp-silero-tts.service
    sleep 2
    sst start cmp-llm.service
    sleep 3
    sst start cmp-voice.service
    sleep 2
    show all-in
    echo "  ⚠ all-in работает только при малых контекстах — проверьте /health и VRAM"
    ;;
  status)
    show status
    ;;
  *)
    echo "usage: $0 voice|voice+kokoro|dev|all-in|status" >&2
    exit 2
    ;;
esac

curl -s -m 5 http://127.0.0.1:8110/health | head -c 600
echo
