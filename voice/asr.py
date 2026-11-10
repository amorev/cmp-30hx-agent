"""ASR: стриминговый sherpa-onnx (T-one CTC, русский) на CPU.

Свойства (раздел 7 ТЗ):
  * recognizer создаётся ОДИН раз за сеанс, на фразу — новый поток;
  * каждый поток прогревается лидирующей тишиной (~0.75 с);
  * endpoint-правила: 2.4 с тишины / 1.2 с после непустого текста / 20 с макс;
  * при остановке хвост добивается ~1 с тишины + input_finished();
  * распознавание идёт в отдельном рабочем потоке, не в event loop;
  * события partial/final отдаются наружу одинаково (позже можно вынести в wasm).
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger("voice.asr")

SAMPLE_RATE = 16000


class SessionASR:
    """Один recognizer на сеанс + рабочий поток-воркер."""

    def __init__(self, model_dir: str | None = None, num_threads: int | None = None):
        self.model_dir = Path(model_dir or config.asr_model_dir())
        self.num_threads = num_threads or config.asr_num_threads()
        self._recognizer = None
        self._stream = None
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._on_event = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_text = ""
        self._utterance_ms = 0.0
        self._t_speech_start: float | None = None
        self._silence_since: float | None = None

    # -- жизненный цикл ---------------------------------------------------
    def _build(self):
        import sherpa_onnx as so

        model = self.model_dir / "model.onnx"
        tokens = self.model_dir / "tokens.txt"
        if not model.is_file() or not tokens.is_file():
            raise FileNotFoundError(f"ASR-модель не найдена в {self.model_dir}")
        t0 = time.monotonic()
        rec = so.OnlineRecognizer.from_t_one_ctc(
            tokens=str(tokens),
            model=str(model),
            num_threads=self.num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=config.asr_silence_final(),
            rule2_min_trailing_silence=config.asr_silence_after_text(),
            rule3_min_utterance_length=config.asr_max_utterance(),
            decoding_method="greedy_search",
            provider="cpu",
        )
        log.info("ASR recognizer собран за %.0f мс (%s)", (time.monotonic() - t0) * 1000, self.model_dir)
        return rec

    def start(self, on_event) -> None:
        """on_event(dict) вызывается из рабочего потока; публикуем в loop."""
        self._on_event = on_event
        self._loop = asyncio.get_running_loop()
        if self._recognizer is None:
            self._recognizer = self._build()
        self._new_stream()
        self._running = True
        self._thread = threading.Thread(target=self._worker, name="asr-worker", daemon=True)
        self._thread.start()
        log.info("ASR-воркер запущен")

    def _new_stream(self) -> None:
        assert self._recognizer is not None
        self._stream = self._recognizer.create_stream()
        # прогрев лидирующей тишиной: иначе в начале фразы фантомный токен
        lead = int(config.asr_lead_silence() * SAMPLE_RATE)
        self._stream.accept_waveform(SAMPLE_RATE, np.zeros(lead, dtype=np.float32))
        self._last_text = ""
        self._utterance_ms = 0.0
        self._t_speech_start = None
        self._silence_since = None

    def reset_stream(self) -> None:
        """Сброс потока между фразами. Сам recognizer НЕ пересоздаётся."""
        self._new_stream()

    # -- приём аудио ------------------------------------------------------
    def feed(self, pcm16: bytes) -> None:
        if not pcm16:
            return
        self._queue.put(pcm16)

    def _emit(self, ev: dict) -> None:
        if self._on_event is None or self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._on_event, ev)

    def _worker(self) -> None:
        assert self._stream is not None
        while self._running:
            try:
                chunk = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                self._stream.accept_waveform(SAMPLE_RATE, samples)
                self._utterance_ms += len(samples) / SAMPLE_RATE * 1000.0

                while self._recognizer.is_ready(self._stream):
                    self._recognizer.decode_stream(self._stream)

                text = self._recognizer.get_result(self._stream).strip()
                if text and text != self._last_text:
                    self._last_text = text
                    self._emit({"type": "partial", "text": text,
                                "utterance_ms": self._utterance_ms})

                if self._recognizer.is_endpoint(self._stream):
                    if self._last_text:
                        self._emit({"type": "final", "text": self._last_text,
                                    "utterance_ms": self._utterance_ms})
                    self._recognizer.reset(self._stream)
                    self._last_text = ""
                    self._utterance_ms = 0.0
            except Exception as e:  # noqa: BLE001
                log.warning("ASR worker: %s", e)

    # -- завершение фразы -------------------------------------------------
    def finish_utterance(self) -> None:
        """Добить хвост тишиной и форсированно закрыть вход.

        Без этого модель теряет последние ~1 с речи — свойство модели.
        """
        if self._stream is None:
            return
        tail = int(config.asr_tail_silence() * SAMPLE_RATE)
        try:
            self._stream.accept_waveform(SAMPLE_RATE, np.zeros(tail, dtype=np.float32))
            self._stream.input_finished()
            assert self._recognizer is not None
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)
            text = self._recognizer.get_result(self._stream).strip()
            if text:
                self._emit({"type": "final", "text": text,
                            "utterance_ms": self._utterance_ms + config.asr_tail_silence() * 1000})
            self.reset_stream()
        except Exception as e:  # noqa: BLE001
            log.warning("ASR finish: %s", e)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._queue.put(b"")
            self._thread.join(timeout=3)
            self._thread = None
        log.info("ASR-воркер остановлен")


class WavASR:
    """Утилита для тестов: распознать контрольный wav целиком."""

    def __init__(self, model_dir: str | None = None):
        self.model_dir = Path(model_dir or config.asr_model_dir())

    def transcribe(self, wav_path: str) -> str:
        import wave

        import sherpa_onnx as so

        rec = so.OnlineRecognizer.from_t_one_ctc(
            tokens=str(self.model_dir / "tokens.txt"),
            model=str(self.model_dir / "model.onnx"),
            num_threads=config.asr_num_threads(),
            sample_rate=SAMPLE_RATE,
            enable_endpoint_detection=False,
            decoding_method="greedy_search",
            provider="cpu",
        )
        with wave.open(wav_path, "rb") as w:
            assert w.getframerate() == SAMPLE_RATE, f"нужно {SAMPLE_RATE} Гц"
            raw = w.readframes(w.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        s = rec.create_stream()
        s.accept_waveform(SAMPLE_RATE, np.zeros(int(0.75 * SAMPLE_RATE), dtype=np.float32))
        s.accept_waveform(SAMPLE_RATE, samples)
        s.accept_waveform(SAMPLE_RATE, np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32))
        s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
        return rec.get_result(s).strip()
