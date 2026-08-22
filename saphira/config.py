"""Configuração via .env — nenhuma chave hardcoded no código."""
import os

from dotenv import load_dotenv

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
load_dotenv()


def _opt(name: str) -> str:
    return os.getenv(name, "").split("#", 1)[0].strip()  # descarta comentário inline do .env


def _req(name: str) -> str:
    value = _opt(name)
    if not value:
        raise RuntimeError(f"Variável {name} ausente — defina no .env (veja .env.example).")
    return value


# Provedor de STT e parâmetros Whisper
STT_PROVIDER = _opt("STT_PROVIDER") or "faster-whisper"
WHISPER_MODEL = _opt("WHISPER_MODEL") or "large-v3-turbo"
WHISPER_DEVICE = _opt("WHISPER_DEVICE") or "auto"
WHISPER_COMPUTE_TYPE = _opt("WHISPER_COMPUTE_TYPE") or "auto"

# Obrigatórias (ELEVENLABS_API_KEY só é obrigatória se STT_PROVIDER for elevenlabs)
if STT_PROVIDER == "elevenlabs":
    ELEVENLABS_API_KEY = _req("ELEVENLABS_API_KEY")
    STT_MODEL = _opt("STT_MODEL") or "scribe_v2"
else:
    ELEVENLABS_API_KEY = _opt("ELEVENLABS_API_KEY")
    STT_MODEL = _opt("STT_MODEL") or WHISPER_MODEL

TAVILY_API_KEY = _req("TAVILY_API_KEY")
GEMINI_API_KEY = _req("GEMINI_API_KEY")  # fallback LLM — aistudio.google.com → API key

# Opcionais / defaults
ANTHROPIC_API_KEY = _opt("ANTHROPIC_API_KEY")  # sem ela o Gemini vira o provider principal
# Confirme o id do modelo em https://docs.anthropic.com/en/docs/about-claude/models antes de rodar.
CLAUDE_MODEL = _opt("CLAUDE_MODEL") or "claude-sonnet-5"
GEMINI_MODEL = "gemini-3.6-flash"
LANGUAGE = _opt("LANGUAGE") or "por"

# TTS Kokoro
KOKORO_VOICE = _opt("KOKORO_VOICE") or "pf_dora"
_speed_raw = _opt("KOKORO_SPEED")
try:
    KOKORO_SPEED = float(_speed_raw) if _speed_raw else 1.0
except ValueError:
    KOKORO_SPEED = 1.0

# Legado / rollback opcional para ElevenLabs TTS
ELEVENLABS_VOICE_ID = _opt("ELEVENLABS_VOICE_ID")
TTS_MODEL = "eleven_flash_v2_5"

HOTKEY = _opt("SAPHIRA_HOTKEY") or "<f9>"  # formato pynput, ex.: "<ctrl>+<shift>+space"
MAX_HISTORY_MESSAGES = 20  # janela deslizante de memória (mensagens, não turnos)
COMMANDS_FILE = _opt("SAPHIRA_COMMANDS") or "commands.yaml"

