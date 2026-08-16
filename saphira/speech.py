"""STT (Scribe) e TTS (streaming) via ElevenLabs."""
import io
import logging
import wave

from elevenlabs import ElevenLabs

log = logging.getLogger("saphira.speech")


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


class Speech:
    def __init__(self, api_key: str, voice_id: str, stt_model: str,
                 tts_model: str, language: str) -> None:
        self._el = ElevenLabs(api_key=api_key)
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
