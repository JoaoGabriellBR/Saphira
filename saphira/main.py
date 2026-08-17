"""Saphira v1 — loop principal.

Hotkey (1ª vez) → grava → hotkey (2ª vez) → STT → LLM(+tools) → TTS por frase → playback.
Hotkey durante a fala da Saphira = barge-in (interrompe e já volta a escutar).

Rodar:  python -m saphira.main
"""
from collections import deque
import logging
import queue
import threading
import time

from . import audio, config
from .llm import ClaudeBrain
from .speech import Speech, local_fallback_chunks
from .tools import LocalCommandTool, ToolRegistry, WebSearchTool

log = logging.getLogger("saphira")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


class LatencyMetrics:
    def __init__(self, max_interactions: int = 20, clock=time.perf_counter) -> None:
        if max_interactions < 1:
            raise ValueError("max_interactions deve ser positivo")
        self._clock = clock
        self._ttfa: deque[float] = deque(maxlen=max_interactions)
        self._capture_finished: float | None = None
        self._lock = threading.Lock()

    def _timestamp(self, at: float | None) -> float:
        return self._clock() if at is None else at

    def capture_started(self, at: float | None = None) -> float:
        timestamp = self._timestamp(at)
        log.info("captura iniciada")
        return timestamp

    def capture_finished(self, at: float | None = None) -> float:
        timestamp = self._timestamp(at)
        with self._lock:
            self._capture_finished = timestamp
        log.info("captura finalizada")
        return timestamp

    def first_audio(self, at: float | None = None) -> dict[str, float | int]:
        with self._lock:
            capture_finished = self._capture_finished
        if capture_finished is None:
            raise RuntimeError("primeiro áudio sem fim de captura")
        timestamp = self._timestamp(at)
        log.info("primeiro chunk tocando")
        return self.record(capture_finished, timestamp)

    def record(self, capture_finished: float, first_audio: float) -> dict[str, float | int]:
        ttfa_ms = (first_audio - capture_finished) * 1000
        with self._lock:
            self._ttfa.append(ttfa_ms)
            values = list(self._ttfa)
        snapshot: dict[str, float | int] = {
            "ttfa_ms": ttfa_ms,
            "p50_ms": _percentile(values, 0.50),
            "p95_ms": _percentile(values, 0.95),
            "window_size": len(values),
        }
        log.info("ttfa=%.0fms p50=%.0fms p95=%.0fms janela=%d",
                 snapshot["ttfa_ms"], snapshot["p50_ms"], snapshot["p95_ms"],
                 snapshot["window_size"])
        return snapshot


def timed(label: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    log.info("%s: %.0fms", label, (time.perf_counter() - t0) * 1000)
    return out


def say(speech: Speech, text: str) -> None:
    """Mensagem de voz com fallback em texto se o TTS estiver fora do ar."""
    try:
        audio.play_pcm(speech.synthesize_stream(text), threading.Event())
    except Exception:
        log.warning("TTS indisponível; mensagem só em texto: %s", text)


def speaker_worker(q: queue.Queue, speech: Speech, interrupt: threading.Event,
                   on_first_audio=None) -> None:
    """Consome frases → sintetiza → toca; sentinela None encerra."""
    first_audio = False

    def mark_first_audio() -> None:
        nonlocal first_audio
        if first_audio:
            return
        first_audio = True
        if on_first_audio:
            on_first_audio()

    def play(chunks) -> None:
        if on_first_audio:
            audio.play_pcm(chunks, interrupt, on_first_chunk=mark_first_audio)
        else:
            audio.play_pcm(chunks, interrupt)

    while True:
        sentence = q.get()
        if sentence is None:
            return
        if interrupt.is_set():
            continue  # drena a fala abortada sem tocar
        try:
            chunks = speech.synthesize_stream(sentence)
            if interrupt.is_set():
                continue
            play(chunks)
        except Exception:
            if interrupt.is_set():
                continue
            log.exception("TTS falhou; usando fallback local: %r", sentence)
            try:
                play(local_fallback_chunks())
            except Exception:
                log.exception("fallback de áudio local falhou")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    speech = Speech(config.ELEVENLABS_API_KEY, config.ELEVENLABS_VOICE_ID,
                    config.STT_MODEL, config.TTS_MODEL, config.LANGUAGE)
    registry = ToolRegistry()
    registry.register(WebSearchTool(config.TAVILY_API_KEY))
    commands = LocalCommandTool(config.COMMANDS_FILE)
    registry.register(commands)
    brain = ClaudeBrain(config.ANTHROPIC_API_KEY, config.CLAUDE_MODEL, registry,
                        commands.catalog(), max_history=config.MAX_HISTORY_MESSAGES)
    latency = LatencyMetrics()

    hotkey_event = threading.Event()
    hotkey = audio.Hotkey(config.HOTKEY, hotkey_event.set)
    hotkey.start()
    log.info("Saphira pronta. Hotkey %s: 1ª vez = falar, 2ª = enviar; durante a fala = interromper",
             config.HOTKEY)

    go_listen = False  # após barge-in, volta direto a escutar
    try:
        while True:
            hotkey_event.clear()
            if not go_listen:
                hotkey_event.wait()  # dorme até a hotkey
                hotkey_event.clear()
            go_listen = False

            latency.capture_started()
            pcm = timed("gravacao", audio.record_until, hotkey_event)
            latency.capture_finished()
            if len(pcm) < 3200:  # < 100ms de áudio = toque acidental
                continue

            try:
                text = timed("stt", speech.transcribe, pcm)
            except Exception:
                log.exception("STT falhou")
                say(speech, "Não consegui te ouvir agora. Tenta de novo?")
                continue
            if not text:
                say(speech, "Não entendi. Pode repetir?")
                continue
            log.info("ouvido: %r", text)

            q: queue.Queue = queue.Queue()
            interrupt = hotkey_event  # hotkey durante a fala = barge-in
            speaker = threading.Thread(target=speaker_worker,
                                       args=(q, speech, interrupt),
                                       kwargs={"on_first_audio": latency.first_audio},
                                       daemon=True)
            speaker.start()
            try:
                brain.respond(text, on_sentence=lambda s: (log.info("saphira: %s", s), q.put(s)))
            except Exception:
                log.exception("LLM falhou")
                q.put("Tive um problema para processar agora. Tenta de novo em instantes.")
            finally:
                q.put(None)
            speaker.join()
            if interrupt.is_set():
                go_listen = True
    except KeyboardInterrupt:
        pass
    finally:
        hotkey.stop()
        log.info("Saphira desligada.")


if __name__ == "__main__":
    main()
