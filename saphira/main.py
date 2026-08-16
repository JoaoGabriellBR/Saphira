"""Saphira v1 — loop principal.

Hotkey (1ª vez) → grava → hotkey (2ª vez) → STT → LLM(+tools) → TTS por frase → playback.
Hotkey durante a fala da Saphira = barge-in (interrompe e já volta a escutar).

Rodar:  python -m saphira.main
"""
import logging
import queue
import threading
import time

from . import audio, config
from .llm import ClaudeBrain
from .speech import Speech
from .tools import LocalCommandTool, ToolRegistry, WebSearchTool

log = logging.getLogger("saphira")


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


def speaker_worker(q: queue.Queue, speech: Speech, interrupt: threading.Event) -> None:
    """Consome frases → sintetiza → toca. Sentinela None encerra."""
    while True:
        sentence = q.get()
        if sentence is None:
            return
        if interrupt.is_set():
            continue  # drena a fala abortada sem tocar
        try:
            chunks = speech.synthesize_stream(sentence)
        except Exception:
            log.exception("TTS falhou; frase perdida: %r", sentence)
            continue
        audio.play_pcm(chunks, interrupt)


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

            pcm = timed("gravacao", audio.record_until, hotkey_event)
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
                                       args=(q, speech, interrupt), daemon=True)
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
