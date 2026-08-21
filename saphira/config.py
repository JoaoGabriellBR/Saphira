"""Configuração via .env — nenhuma chave hardcoded no código."""
import os

from dotenv import load_dotenv

load_dotenv()


def _opt(name: str) -> str:
    return os.getenv(name, "").split("#", 1)[0].strip()  # descarta comentário inline do .env


def _req(name: str) -> str:
    value = _opt(name)
    if not value:
        raise RuntimeError(f"Variável {name} ausente — defina no .env (veja .env.example).")
    return value


# Obrigatórias
ELEVENLABS_API_KEY = _req("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = _req("ELEVENLABS_VOICE_ID")  # elevenlabs.io → Voice Library → copiar Voice ID
TAVILY_API_KEY = _req("TAVILY_API_KEY")
GEMINI_API_KEY = _req("GEMINI_API_KEY")  # fallback LLM — aistudio.google.com → API key

# Opcionais / defaults
ANTHROPIC_API_KEY = _opt("ANTHROPIC_API_KEY")  # sem ela o Gemini vira o provider principal
# Confirme o id do modelo em https://docs.anthropic.com/en/docs/about-claude/models antes de rodar.
CLAUDE_MODEL = _opt("CLAUDE_MODEL") or "claude-sonnet-5"
GEMINI_MODEL = "gemini-3.6-flash"
STT_MODEL = "scribe_v2"
TTS_MODEL = "eleven_flash_v2_5"  # troque por "eleven_multilingual_v2" p/ mais expressão (mais latência)
LANGUAGE = "por"
HOTKEY = _opt("SAPHIRA_HOTKEY") or "<f9>"  # formato pynput, ex.: "<ctrl>+<shift>+space"
MAX_HISTORY_MESSAGES = 20  # janela deslizante de memória (mensagens, não turnos)
COMMANDS_FILE = _opt("SAPHIRA_COMMANDS") or "commands.yaml"
