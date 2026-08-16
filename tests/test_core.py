"""Checks mínimos das partes puras da Saphira (sem frameworks).
Rodar: python tests/test_core.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

# --- registry: erro de tool nunca derruba a conversa ---
class Quebrada:
    name = "quebrada"
    description = "x"
    input_schema = {"type": "object", "properties": {}}

    def run(self):
        raise RuntimeError("boom")

reg = ToolRegistry()
reg.register(Quebrada())
assert "Erro ao executar quebrada" in reg.execute("quebrada", {})
assert "não registrada" in reg.execute("fantasma", {})

print("todos os checks passaram")
