"""Captura de microfone, playback PCM e hotkey global (push-to-talk)."""
import logging
import threading

import sounddevice as sd
from pynput import keyboard

MIC_RATE = 16000  # Scribe espera s16le 16kHz mono
TTS_RATE = 24000  # corresponde a output_format="pcm_24000" do TTS

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
        done.wait(timeout=max_seconds)
    return b"".join(chunks)


def play_pcm(chunks, interrupt: threading.Event) -> None:
    """Reproduz iterável de chunks PCM; aborta cedo se `interrupt` for setado (barge-in)."""
    with sd.RawOutputStream(samplerate=TTS_RATE, channels=1, dtype="int16") as out:
        for chunk in chunks:
            if interrupt.is_set():
                log.info("playback interrompido (barge-in)")
                break
            if isinstance(chunk, bytes):
                out.write(chunk)
