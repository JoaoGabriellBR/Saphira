"""Checks mínimos das partes puras da Saphira (sem frameworks).
Rodar: python tests/test_core.py
"""
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- config: valores do v1 e fail-fast sem chaves reais ---
ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["ANTHROPIC_API_KEY", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID",
            "TAVILY_API_KEY", "GEMINI_API_KEY"]
for k in REQUIRED:
    os.environ[k] = f"test-{k.lower()}"  # setado antes do import: .env não sobrescreve
for opt in ("CLAUDE_MODEL", "SAPHIRA_HOTKEY"):
    os.environ.pop(opt, None)

from saphira import config

assert config.CLAUDE_MODEL == "claude-sonnet-5", config.CLAUDE_MODEL
assert config.GEMINI_API_KEY == "test-gemini_api_key"
assert config.STT_MODEL == "scribe_v2"
assert config.TTS_MODEL == "eleven_flash_v2_5"
assert config.LANGUAGE == "por"
assert config.HOTKEY == "<f9>"
assert config.MAX_HISTORY_MESSAGES == 20

# qualquer obrigatória ausente -> RuntimeError nomeando a variável e .env.example
for missing in REQUIRED:
    env = {**os.environ, missing: "", "PYTHONPATH": str(ROOT)}  # "" impede .env de preencher
    r = subprocess.run([sys.executable, "-c", "import saphira.config"],
                       env=env, cwd=tempfile.gettempdir(), capture_output=True, text=True)
    assert r.returncode != 0 and missing in r.stderr and ".env.example" in r.stderr, (missing, r.stderr)

# dependências declaradas e .env.example sem valores preenchidos
assert "google-genai" in (ROOT / "requirements.txt").read_text(encoding="utf-8")
for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        assert line.split("=", 1)[1].split("#", 1)[0].strip() == "", f"valor no .env.example: {line}"

from saphira.llm import _pop_sentences
from saphira.tools import LocalCommandTool, ToolRegistry

# --- sentence splitter ---
s, rest = _pop_sentences("Oi, tudo bem. Como vai")
assert s == ["Oi, tudo bem."] and rest == "Como vai", (s, rest)

s, rest = _pop_sentences("Um.\nDois.\n")
assert s == ["Um.", "Dois."] and rest == "", (s, rest)

s, rest = _pop_sentences("sem pontuacao ainda")
assert s == [] and rest == "sem pontuacao ainda"

# --- LocalCommandTool ---
plat = "win32" if sys.platform == "win32" else ("darwin" if sys.platform == "darwin" else "linux")
py = "python" if plat == "win32" else "python3"
yaml_text = f"""
apps:
  editor:
    open: {{ {plat}: ["cmd", "/c", "echo", "abriu"] }}
scripts:
  ola:
    {plat}: ["{py}", "-c", "print('oi saphira')"]
"""
with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
    f.write(yaml_text)
    path = f.name

tool = LocalCommandTool(path)
assert "editor" in tool.catalog() and "ola" in tool.catalog()
assert tool.run("run_script", "ola").strip() == "oi saphira"
assert "não cadastrado" in tool.run("open_app", "inexistente")
assert "não cadastrado" in tool.run("close_app", "editor")  # só 'open' foi definido

# --- registry: sucesso, schemas, erro protegido, catálogo e log ---
class Ok:
    name = "ok"
    description = "tool fake de sucesso"
    input_schema = {"type": "object", "properties": {"x": {"type": "integer"}}}

    def run(self, x: int) -> str:
        return f"recebido {x}"

class Quebrada:
    name = "quebrada"
    description = "x"
    input_schema = {"type": "object", "properties": {}}

    def run(self):
        raise RuntimeError("boom")

reg = ToolRegistry()
reg.register(Ok())
reg.register(Quebrada())

# tool registrada expõe name/description/input_schema e execute retorna str
schema = next(s for s in reg.anthropic_schemas() if s["name"] == "ok")
assert schema["description"] == "tool fake de sucesso"
assert schema["input_schema"]["properties"]["x"]["type"] == "integer"
assert reg.execute("ok", {"x": 7}) == "recebido 7"

# exceção vira erro com detalhe sem escapar; fantasma recebe o catálogo completo
assert reg.execute("quebrada", {}) == "Erro ao executar quebrada: boom"
r = reg.execute("fantasma", {})
assert "não registrada" in r and "ok" in r and "quebrada" in r, r

# execução bem-sucedida registra tool, args e duração em milissegundos
class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())

tool_log = logging.getLogger("saphira.tools")
tool_log.setLevel(logging.INFO)
handler = ListHandler()
tool_log.addHandler(handler)
reg.execute("ok", {"x": 7})
assert any("tool=ok" in m and "'x': 7" in m and "ms" in m for m in handler.messages), handler.messages

print("todos os checks passaram")
