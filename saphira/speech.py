"""STT (Scribe) e TTS (streaming) via ElevenLabs."""
import io
import logging
import wave
from pathlib import Path

from elevenlabs import ElevenLabs

log = logging.getLogger("saphira.speech")
TTS_TIMEOUT_SECONDS = 30

LOCAL_FALLBACK_MESSAGE = "não consegui te ouvir, tenta de novo"
LOCAL_FALLBACK_WAV = Path(__file__).with_name("fallback_audio.wav")


def _pcm_to_wav(pcm: bytes, rate: int) -> io.BytesIO:
    """Envelope WAV (stdlib) — evita depender do file_format='pcm_s16le_16' da API."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(rate)
        w.writeframes(pcm)
    buf.seek(0)
    return buf


def local_fallback_chunks(chunk_size: int = 4096):
    """Lê o fallback local como chunks PCM 24 kHz, sem criar cliente ElevenLabs."""
    with wave.open(str(LOCAL_FALLBACK_WAV), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 24000):
            raise ValueError("fallback_audio.wav deve ser PCM mono 16-bit a 24 kHz")
        frames = max(1, chunk_size // 2)
        while chunk := source.readframes(frames):
            yield chunk


class Speech:
    def __init__(self, api_key: str, voice_id: str, stt_model: str,
                 tts_model: str, language: str) -> None:
        self._el = ElevenLabs(api_key=api_key, timeout=TTS_TIMEOUT_SECONDS)
        self._voice_id = voice_id
        self._stt_model = stt_model
        self._tts_model = tts_model
        self._language = language

    def transcribe(self, pcm: bytes, rate: int = 16000) -> str:
        resp = self._el.speech_to_text.convert(
            file=_pcm_to_wav(pcm, rate),
            model_id=self._stt_model,
            language_code=self._language,
        )
        return (resp.text or "").strip()

    def synthesize_stream(self, text: str):
        """Iterador de chunks PCM (pcm_24000). Se o SDK rejeitar output_format aqui,
        veja https://elevenlabs.io/docs/api-reference/text-to-speech/stream"""
        return self._el.text_to_speech.stream(
            text=text,
            voice_id=self._voice_id,
            model_id=self._tts_model,
            output_format="pcm_24000",
        )
