#!/usr/bin/env python3
"""Распознавание контрольного wav (критерий 9).

Запуск: .venv/bin/python tests/test_asr.py [файл.wav]
По умолчанию берётся 0.wav из каталога модели.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice import config  # noqa: E402

config.load_env()

from voice.asr import WavASR  # noqa: E402


def _ensure_16k(path: str) -> str:
    """Контрольный wav может быть 8 кГц — приводим к 16 кГц mono s16."""
    import subprocess
    import tempfile
    import wave

    with wave.open(path, "rb") as w:
        if w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2:
            return path
    out = tempfile.NamedTemporaryFile(suffix="_16k.wav", delete=False).name
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                    "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", out],
                   check=True)
    print(f"(ресемплинг {path} -> {out} 16 кГц)")
    return out


def main() -> int:
    wav = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(config.asr_model_dir()) / "0.wav")
    if not Path(wav).is_file():
        print(f"нет файла {wav}")
        return 2
    wav = _ensure_16k(wav)
    a = WavASR()
    t0 = time.monotonic()
    text = a.transcribe(wav)
    ms = (time.monotonic() - t0) * 1000
    print(f"файл: {wav}")
    print(f"время: {ms:.0f} мс")
    print(f"текст: {text!r}")
    print("OK" if text.strip() else "FAIL: пустой результат")
    return 0 if text.strip() else 1


if __name__ == "__main__":
    raise SystemExit(main())
