"""STT (faster-whisper / ElevenLabs) e TTS streaming via Kokoro."""
import io
import logging
import wave
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    from elevenlabs import ElevenLabs
except ImportError:
    ElevenLabs = None

try:
    from kokoro import KPipeline
except ImportError:
    KPipeline = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

log = logging.getLogger("saphira.speech")
TTS_TIMEOUT_SECONDS = 30
TTS_CHUNK_SIZE = 4096

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


def _convert_audio_to_pcm_chunks(audio, chunk_size: int = TTS_CHUNK_SIZE) -> Iterator[bytes]:
    """Converte saída de áudio (Tensor/ndarray/bytes) para chunks PCM mono s16le 24 kHz."""
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    elif hasattr(audio, "numpy"):
        audio = audio.numpy()

    if isinstance(audio, (list, tuple)):
        audio = np.asarray(audio, dtype=np.float32)

    if isinstance(audio, np.ndarray):
        audio = audio.squeeze()
        if audio.ndim == 0:
            return
        if not np.all(np.isfinite(audio)):
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        audio = np.clip(audio, -1.0, 1.0)
        pcm_int16 = (audio * 32767.0).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()
    elif isinstance(audio, bytes):
        pcm_bytes = audio
    else:
        raise TypeError(f"Formato de áudio inesperado: {type(audio)}")

    for i in range(0, len(pcm_bytes), chunk_size):
        chunk = pcm_bytes[i:i + chunk_size]
        if chunk:
            yield chunk


class Speech:
    def __init__(
        self,
        api_key: str = "",
        voice: str = "pf_dora",
        stt_provider: str = "faster-whisper",
        stt_model: str = "large-v3-turbo",
        speed: float = 1.0,
        language: str = "por",
        kokoro_pipeline=None,
        whisper_model=None,
        whisper_device: str = "auto",
        whisper_compute_type: str = "auto",
    ) -> None:
        self._stt_provider = stt_provider or "faster-whisper"
        self._api_key = api_key
        self._el = None
        if self._stt_provider == "elevenlabs" or (api_key and ElevenLabs is not None):
            try:
                self._el = ElevenLabs(api_key=api_key, timeout=TTS_TIMEOUT_SECONDS) if ElevenLabs else None
            except Exception:
                log.debug("Cliente ElevenLabs não inicializado.")
        self._voice = voice or "pf_dora"
        self._stt_model = stt_model
        try:
            self._speed = float(speed)
        except (ValueError, TypeError):
            self._speed = 1.0
        self._language = language
        self._kokoro = kokoro_pipeline
        self._whisper = whisper_model
        self._whisper_device = whisper_device
        self._whisper_compute_type = whisper_compute_type

    def _get_pipeline(self):
        if self._kokoro is None:
            if KPipeline is None:
                raise RuntimeError("Pacote kokoro não está instalado.")
            self._kokoro = KPipeline(lang_code="p")
        return self._kokoro

    def _get_whisper_model(self):
        if self._whisper is None:
            if WhisperModel is None:
                raise RuntimeError("Pacote faster-whisper não está instalado. Instale com: pip install faster-whisper")
            device = self._whisper_device
            compute_type = self._whisper_compute_type

            if device == "auto":
                try:
                    import torch
                    has_cuda = torch.cuda.is_available()
                except Exception:
                    has_cuda = False
                device = "cuda" if has_cuda else "cpu"

            if compute_type == "auto":
                compute_type = "float16" if device == "cuda" else "int8"

            try:
                log.info("Carregando modelo faster-whisper '%s' (device=%s, compute_type=%s)...",
                         self._stt_model, device, compute_type)
                self._whisper = WhisperModel(self._stt_model, device=device, compute_type=compute_type)
                if device == "cuda":
                    dummy = np.zeros(8000, dtype=np.float32)
                    list(self._whisper.transcribe(dummy, language="pt", beam_size=1)[0])
            except Exception as e:
                if device == "cuda":
                    log.warning("Falha ao inicializar faster-whisper em CUDA (%s). Tentando fallback para CPU (int8)...", e)
                    self._whisper = WhisperModel(self._stt_model, device="cpu", compute_type="int8")
                else:
                    raise
        return self._whisper

    def transcribe(self, pcm: bytes, rate: int = 16000) -> str:
        """Transcreve áudio PCM mono 16 kHz (faster-whisper ou ElevenLabs)."""
        if self._stt_provider == "faster-whisper":
            model = self._get_whisper_model()
            if not isinstance(pcm, np.ndarray):
                audio_data = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_data = pcm

            lang = "pt" if self._language in ("por", "pt", "portuguese") else self._language
            segments, _ = model.transcribe(
                audio_data,
                language=lang,
                beam_size=1,
                temperature=0.0,
                vad_filter=False,
            )
            return "".join(segment.text for segment in segments).strip()
        elif self._stt_provider == "elevenlabs":
            if self._el is None:
                if not self._api_key:
                    raise RuntimeError("ELEVENLABS_API_KEY necessária para STT ElevenLabs.")
                if ElevenLabs is None:
                    raise RuntimeError("Pacote elevenlabs não está instalado.")
                self._el = ElevenLabs(api_key=self._api_key, timeout=TTS_TIMEOUT_SECONDS)
            resp = self._el.speech_to_text.convert(
                file=_pcm_to_wav(pcm, rate),
                model_id=self._stt_model,
                language_code=self._language,
            )
            return (resp.text or "").strip()
        else:
            raise ValueError(f"Provedor de STT desconhecido: {self._stt_provider}")

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """Sintetiza texto em streaming de chunks PCM mono 16-bit 24 kHz via Kokoro."""
        pipeline = self._get_pipeline()
        generator = pipeline(text, voice=self._voice, speed=self._speed)
        for result in generator:
            audio = getattr(result, "audio", None)
            if audio is None:
                if isinstance(result, (tuple, list)) and len(result) >= 3:
                    audio = result[2]
                else:
                    audio = result
            yield from _convert_audio_to_pcm_chunks(audio, chunk_size=TTS_CHUNK_SIZE)

