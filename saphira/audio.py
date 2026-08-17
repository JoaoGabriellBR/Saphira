"""Captura de microfone, playback PCM e hotkey global (push-to-talk)."""
import logging
import threading
import time

import sounddevice as sd
from pynput import keyboard

MIC_RATE = 16000  # Scribe espera s16le 16kHz mono
TTS_RATE = 24000  # corresponde a output_format="pcm_24000" do TTS
VAD_THRESHOLD = 500  # ponytail: limiar fixo; calibrar por microfone se houver falsos disparos

log = logging.getLogger("saphira.audio")


class Hotkey:
    """Hotkey global; chama callback() a cada pressionada (thread própria do pynput)."""

    def __init__(self, combo: str, callback) -> None:
        self._listener = keyboard.GlobalHotKeys({combo: callback})

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()


def record_until(stop: threading.Event, max_seconds: int = 90) -> bytes:
    """Grava até `stop` ser setado (ou timeout de segurança). Retorna PCM s16le 16kHz mono."""
    chunks: list[bytes] = []
    done = threading.Event()

    def callback(indata, frames, time_info, status) -> None:
        chunks.append(bytes(indata))
        if stop.is_set():
            done.set()
            raise sd.CallbackStop()

    with sd.RawInputStream(samplerate=MIC_RATE, channels=1, dtype="int16",
                           blocksize=4096, callback=callback):
        done.wait(timeout=min(max_seconds, 90))
    return b"".join(chunks)


def play_pcm(chunks, interrupt: threading.Event, on_first_chunk=None) -> None:
    """Reproduz chunks PCM e sinaliza o primeiro enviado ao sink."""
    try:
        with sd.RawOutputStream(samplerate=TTS_RATE, channels=1, dtype="int16") as out:
            first_chunk = True
            for chunk in chunks:
                if interrupt.is_set():
                    log.info("playback interrompido (barge-in)")
                    break
                if isinstance(chunk, bytes):
                    out.write(chunk)
                    if first_chunk and on_first_chunk:
                        first_chunk = False
                        on_first_chunk()
    finally:
        if interrupt.is_set():
            close = getattr(chunks, "close", None)
            if close:
                close()


class VAD:
    """Borda de sinais de fala com debounce, sem depender de um SDK de VAD."""

    def __init__(self, callback, debounce_ms: int = 200) -> None:
        if not 150 <= debounce_ms <= 300:
            raise ValueError("debounce_ms deve estar entre 150 e 300")
        self._callback = callback
        self._debounce = debounce_ms / 1000
        self._started_at: float | None = None
        self._emitted = False

    def signal(self, speaking: bool, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if not speaking:
            self._started_at = None
            self._emitted = False
            return
        if self._emitted:
            return
        if self._started_at is None:
            self._started_at = now
        elif now - self._started_at >= self._debounce:
            self._callback()
            self._emitted = True


def monitor_vad(vad: VAD, stop: threading.Event, threshold: int = VAD_THRESHOLD) -> None:
    """Alimenta o VAD com amplitude PCM enquanto a resposta está sendo falada."""
    def callback(indata, frames, time_info, status) -> None:
        samples = memoryview(indata).cast("h")
        vad.signal(any(abs(sample) >= threshold for sample in samples))

    try:
        with sd.RawInputStream(samplerate=MIC_RATE, channels=1, dtype="int16",
                               blocksize=4096, callback=callback):
            while not stop.wait(0.05):
                pass
    except Exception:
        log.exception("monitoramento VAD indisponível")
