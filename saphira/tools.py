"""Tools plugáveis: interface única + registry.

Adicionar uma tool nova (ex.: visão no v2) = criar uma classe com
name/description/input_schema/run e chamar registry.register(). O orquestrador não muda.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Protocol

import yaml
from tavily import TavilyClient

log = logging.getLogger("saphira.tools")


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict  # JSON Schema no formato Anthropic tool use

    def run(self, **kwargs) -> str: ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def anthropic_schemas(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in self._tools.values()]

    def execute(self, name: str, inputs: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Erro: tool '{name}' não registrada. Disponíveis: {sorted(self._tools)}"
        start = time.perf_counter()
        try:
            result = tool.run(**inputs)
        except Exception as exc:  # tool quebrada nunca derruba a conversa
            log.exception("tool %s falhou", name)
            return f"Erro ao executar {name}: {exc}"
        log.info("tool=%s args=%s %.0fms", name, inputs, (time.perf_counter() - start) * 1000)
        return result


class WebSearchTool:
    name = "web_search"
    description = ("Busca na web informações atuais (notícias, preços, fatos recentes). "
                   "Use quando a resposta depender de dados que mudam com o tempo.")
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "busca em linguagem natural"}},
        "required": ["query"],
    }

    def __init__(self, api_key: str) -> None:
        self._client = TavilyClient(api_key=api_key)

    def run(self, query: str) -> str:
        r = self._client.search(query, search_depth="basic", max_results=4, include_answer=True)
        parts = [r.get("answer") or ""]
        parts += [f"- {x.get('title', '')}: {x.get('content', '')[:300]}"
                  for x in r.get("results", [])]
        return "\n".join(p for p in parts if p).strip()[:3000] or "Nenhum resultado encontrado."


def _platform() -> str:
    if sys.platform == "win32":
        return "win32"
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform.startswith("linux"):
        return "linux"
    return sys.platform


def _truncate_stream(value: str | None) -> str:
    value = value or ""
    return "[...]truncado[...]" + value[-1500:] if len(value) > 1500 else value


class LocalCommandTool:
    """Executa apenas o que estiver pré-cadastrado no commands.yaml (allowlist, sem shell)."""

    name = "local_command"
    description = ("Executa comandos locais pré-cadastrados: abrir/fechar aplicativos "
                   "e rodar scripts. Use somente os nomes do catálogo disponível.")
    input_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["open_app", "close_app", "run_script"]},
            "target": {"type": "string", "description": "nome cadastrado no commands.yaml"},
        },
        "required": ["action", "target"],
    }

    def __init__(self, commands_file: str) -> None:
        data = yaml.safe_load(Path(commands_file).read_text(encoding="utf-8")) or {}
        self._apps = data.get("apps", {})
        self._scripts = data.get("scripts", {})

    def catalog(self) -> str:
        """Vai no system prompt para o LLM saber o que existe."""
        lines = [f"- app '{n}' (ações: open_app/close_app)" for n in self._apps]
        lines += [f"- script '{n}' (ação: run_script)" for n in self._scripts]
        return "\n".join(lines) or "(nenhum comando cadastrado)"

    def _resolve(self, action: str, target: str):
        if action == "run_script":
            entry = self._scripts.get(target)
        elif action in ("open_app", "close_app"):
            verb = "open" if action == "open_app" else "close"
            entry = (self._apps.get(target) or {}).get(verb)
        else:
            entry = None
        return entry.get(_platform()) if entry else None

    def run(self, action: str, target: str) -> str:
        if action == "close_app" and target == "navegador" and _platform() == "win32":
            return (f"Comando '{action} {target}' não suportado no Windows. "
                    f"Disponíveis:\n{self.catalog()}")

        argv = self._resolve(action, target)
        if argv is None:
            # Erro com o catálogo permite o LLM se autocorrigir na próxima tentativa
            return (f"Comando '{action} {target}' não cadastrado para este SO. "
                    f"Disponíveis:\n{self.catalog()}")
        if action == "run_script":
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=120, shell=False)
            stdout = _truncate_stream(proc.stdout)
            stderr = _truncate_stream(proc.stderr)
            if stdout and stderr:
                return f"stdout:\n{stdout}\nstderr:\n{stderr}"
            return stdout or stderr or f"script '{target}' finalizado (exit {proc.returncode})."
        subprocess.Popen(argv, shell=False)  # fire-and-forget: abrir/fechar app não bloqueia a conversa
        return f"ok: {action} {target}"
