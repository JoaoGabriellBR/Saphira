"""Configuração via .env — nenhuma chave hardcoded no código."""
import os

from dotenv import load_dotenv

load_dotenv()


def _req(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Variável {name} ausente — defina no .env (veja .env.example).")
    return value


# Obrigatórias
ANTHROPIC_API_KEY = _req("ANTHROPIC_API_KEY")
ELEVENLABS_API_KEY = _req("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = _req("ELEVENLABS_VOICE_ID")  # elevenlabs.io → Voice Library → copiar Voice ID
TAVILY_API_KEY = _req("TAVILY_API_KEY")
GEMINI_API_KEY = _req("GEMINI_API_KEY")  # fallback LLM — aistudio.google.com → API key

# Opcionais / defaults
# Confirme o id do modelo em https://docs.anthropic.com/en/docs/about-claude/models antes de rodar.
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
STT_MODEL = "scribe_v2"
TTS_MODEL = "eleven_flash_v2_5"  # troque por "eleven_multilingual_v2" p/ mais expressão (mais latência)
LANGUAGE = "por"
HOTKEY = os.getenv("SAPHIRA_HOTKEY", "<f9>")  # formato pynput, ex.: "<ctrl>+<shift>+space"
MAX_HISTORY_MESSAGES = 20  # janela deslizante de memória (mensagens, não turnos)
COMMANDS_FILE = os.getenv("SAPHIRA_COMMANDS", "commands.yaml")
