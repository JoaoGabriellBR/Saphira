"""Cérebro: Anthropic Claude com tool use, streaming e chunking por frase p/ TTS."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
import threading
from typing import Any, Iterable, Protocol

import anthropic

try:
    from google import genai
except ImportError:  # google-genai é carregado só quando o fallback é usado
    genai = None

from .tools import ToolRegistry

log = logging.getLogger("saphira.llm")


@dataclass(frozen=True)
class ProviderEvent:
    """Evento independente do SDK emitido por qualquer provider de LLM."""

    type: str
    text: str = ""
    tool_use_id: str | None = None
    name: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    stop_reason: str | None = None

    @property
    def kind(self) -> str:
        """Alias legível para consumidores que tratam eventos por categoria."""
        return self.type

    @classmethod
    def text_delta(cls, text: str) -> "ProviderEvent":
        return cls("text_delta", text=text)

    @classmethod
    def tool_use(cls, tool_use_id: str, name: str, inputs: dict[str, Any]) -> "ProviderEvent":
        return cls("tool_use", tool_use_id=tool_use_id, name=name, inputs=dict(inputs))

    @classmethod
    def end(cls, stop_reason: str | None = None) -> "ProviderEvent":
        return cls("end", stop_reason=stop_reason)

    @classmethod
    def cancelled(cls) -> "ProviderEvent":
        return cls("cancelled")

    @classmethod
    def tool_result(cls, tool_use_id: str, content: str) -> "ProviderEvent":
        return cls("tool_result", tool_use_id=tool_use_id, content=content)


class LLMProvider(Protocol):
    """Contrato comum para Claude, Gemini e fakes de teste."""

    def stream(self, *, model: str, system: str, messages: list[dict],
               tools: list[dict], max_tokens: int,
               cancel_event: threading.Event) -> Iterable[ProviderEvent]: ...


class AnthropicProvider:
    """Adapta o streaming do SDK Anthropic ao contrato interno."""

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)

    def stream(self, *, model: str, system: str, messages: list[dict],
               tools: list[dict], max_tokens: int,
               cancel_event: threading.Event) -> Iterable[ProviderEvent]:
        if cancel_event.is_set():
            yield ProviderEvent.cancelled()
            return
        with self._client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        ) as stream:
            for delta in stream.text_stream:
                if cancel_event.is_set():
                    yield ProviderEvent.cancelled()
                    return
                yield ProviderEvent.text_delta(delta)
            final = stream.get_final_message()
        if cancel_event.is_set():
            yield ProviderEvent.cancelled()
            return
        for block in final.content:
            if block.type == "tool_use":
                yield ProviderEvent.tool_use(block.id, block.name, block.input)
        yield ProviderEvent.end(final.stop_reason)


def _field(value: Any, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    try:
        return getattr(value, name)
    except (AttributeError, ValueError):
        return default


def _gemini_contents(messages: list[dict]) -> list[dict]:
    contents = []
    for message in messages:
        raw_content = message.get("content", "")
        blocks = raw_content if isinstance(raw_content, list) else [{"type": "text", "text": raw_content}]
        role = "model" if message["role"] == "assistant" else "user"
        if any(isinstance(block, dict) and block.get("type") == "tool_result" for block in blocks):
            role = "tool"
        parts = []
        for block in blocks:
            if isinstance(block, str):
                parts.append({"text": block})
                continue
            block_type = block.get("type")
            if block_type == "text":
                parts.append({"text": block.get("text", "")})
            elif block_type == "tool_use":
                parts.append({"function_call": {
                    "name": block.get("name", ""),
                    "args": block.get("input", {}),
                }})
            elif block_type == "tool_result":
                parts.append({"function_response": {
                    "name": block.get("name", ""),
                    "response": {"result": block.get("content", "")},
                }})
        if parts:
            contents.append({"role": role, "parts": parts})
    return contents


def _gemini_tools(tools: list[dict]) -> list[dict]:
    if not tools:
        return []
    return [{"function_declarations": [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters_json_schema": tool.get("input_schema", {"type": "object"}),
        }
        for tool in tools
    ]}]


class GeminiProvider:
    """Adapta o streaming do google-genai ao contrato interno."""

    def __init__(self, api_key: str, model: str = "gemini-3.6-flash", client=None) -> None:
        self._model = model
        if client is not None:
            self._client = client
        elif genai is not None:
            self._client = genai.Client(api_key=api_key)
        else:
            raise RuntimeError("google-genai não instalado; instale requirements.txt")

    def stream(self, *, model: str, system: str, messages: list[dict],
               tools: list[dict], max_tokens: int,
               cancel_event: threading.Event) -> Iterable[ProviderEvent]:
        if cancel_event.is_set():
            yield ProviderEvent.cancelled()
            return
        config = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
            "tools": _gemini_tools(tools),
        }
        chunks = self._client.models.generate_content_stream(
            model=self._model,
            contents=_gemini_contents(messages),
            config=config,
        )
        tool_index = 0
        finish_reason = None
        for chunk in chunks:
            if cancel_event.is_set():
                yield ProviderEvent.cancelled()
                return
            text = _field(chunk, "text", "") or ""
            if text:
                yield ProviderEvent.text_delta(text)
            for call in _gemini_function_calls(chunk):
                tool_id = str(_field(call, "id", None) or f"gemini-tool-{tool_index}")
                tool_index += 1
                yield ProviderEvent.tool_use(
                    tool_id,
                    _field(call, "name", ""),
                    dict(_field(call, "args", {}) or {}),
                )
            for result in _gemini_function_responses(chunk):
                tool_id = str(_field(result, "id", None) or "gemini-tool-result")
                response = _field(result, "response", "")
                if isinstance(response, dict):
                    response = response.get("result", response.get("content", response))
                yield ProviderEvent.tool_result(tool_id, str(response))
            for candidate in _field(chunk, "candidates", []) or []:
                finish_reason = _field(candidate, "finish_reason", finish_reason) or finish_reason
        if cancel_event.is_set():
            yield ProviderEvent.cancelled()
            return
        yield ProviderEvent.end(str(finish_reason) if finish_reason is not None else None)


def _gemini_function_calls(chunk: Any) -> list[Any]:
    calls = _field(chunk, "function_calls", None)
    if calls:
        return list(calls)
    found = []
    for candidate in _field(chunk, "candidates", []) or []:
        content = _field(candidate, "content", None)
        for part in _field(content, "parts", []) or []:
            call = _field(part, "function_call", None)
            if call is not None:
                found.append(call)
    return found


def _gemini_function_responses(chunk: Any) -> list[Any]:
    responses = _field(chunk, "function_responses", None)
    if responses:
        return list(responses)
    found = []
    for candidate in _field(chunk, "candidates", []) or []:
        content = _field(candidate, "content", None)
        for part in _field(content, "parts", []) or []:
            response = _field(part, "function_response", None)
            if response is not None:
                found.append(response)
    return found


SYSTEM_TEMPLATE = """Você é Saphira, assistente pessoal por voz em português do Brasil.
Regras: respostas curtas e naturais para serem FALADAS (sem markdown, sem listas, sem emoji).
Quando a pergunta depender de dados atuais, use a tool web_search.
Para comandos no computador, use a tool local_command com os nomes do catálogo abaixo.
Se uma tool falhar, avise em uma frase e sugira uma alternativa.

Comandos locais disponíveis:
{commands_catalog}"""

# Corta em pontuação final seguida de espaço/quebra. Heurística simples —
# ponytail: abreviações ("Dr.", "Sr.") podem cortar cedo; se incomodar, trocar por
# sentencizer de NLP. Na prática o TTS pausa natural e ninguém nota.
_SENTENCE_END = re.compile(r".+?[.!?…](?:\s|$)|.+?\n+", re.DOTALL)


class _TurnCancelled(Exception):
    pass


def _pop_sentences(buf: str, flush: bool = False) -> tuple[list[str], str]:
    """Separa frases completas do buffer; `flush` emite o resto sem pontuação."""
    sentences, last = [], 0
    for m in _SENTENCE_END.finditer(buf):
        s = m.group(0).strip()
        if s:
            sentences.append(s)
        last = m.end()
    rest = buf[last:]
    if flush and rest.strip():
        sentences.append(rest.strip())
        rest = ""
    return sentences, rest


def _tool_result_blocks(message: dict) -> list[dict]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content
            if isinstance(block, dict) and block.get("type") == "tool_result"]


def _tool_use_blocks(message: dict) -> list[dict]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use"]


def _message_units(message: dict) -> int:
    return max(1, len(_tool_result_blocks(message)))


def _history_units(history: list[dict]) -> int:
    return sum(_message_units(message) for message in history)


def _is_turn_start(message: dict) -> bool:
    return message.get("role") == "user" and not _tool_result_blocks(message)


def _turn_start(history: list[dict], index: int) -> int:
    for position in range(index, -1, -1):
        if _is_turn_start(history[position]):
            return position
    return 0


def _safe_history_start(history: list[dict], start: int) -> int:
    start = _turn_start(history, start)
    while True:
        extended = False
        retained_uses = {
            block.get("id")
            for message in history[start:]
            for block in _tool_use_blocks(message)
        }
        for message in history[start:]:
            for result in _tool_result_blocks(message):
                tool_use_id = result.get("tool_use_id")
                if not tool_use_id or tool_use_id in retained_uses:
                    continue
                for position in range(start - 1, -1, -1):
                    if any(block.get("id") == tool_use_id
                           for block in _tool_use_blocks(history[position])):
                        new_start = _turn_start(history, position)
                        if new_start < start:
                            start = new_start
                            extended = True
                        break
        if not extended:
            return start


class ClaudeBrain:
    def __init__(self, api_key: str, model: str, tools: ToolRegistry,
                 commands_catalog: str, max_history: int = 20,
                 provider: LLMProvider | None = None,
                 fallback_provider: LLMProvider | None = None) -> None:
        self._provider = provider if provider is not None else AnthropicProvider(api_key)
        self._fallback_provider = fallback_provider
        self._model = model
        self._tools = tools
        self._system = SYSTEM_TEMPLATE.format(commands_catalog=commands_catalog)
        self._max_history = max_history
        self._history: list[dict] = []

    def respond(self, user_text: str, on_sentence, on_event=None,
                cancel_event: threading.Event | None = None) -> None:
        """Loop de tool use. on_sentence(frase) é chamado p/ cada frase pronta p/ TTS."""
        self._respond_with_provider(user_text, on_sentence, on_event, cancel_event)

    def _respond_with_provider(self, user_text: str, on_sentence, on_event,
                               cancel_event: threading.Event | None) -> None:
        cancel_event = cancel_event or threading.Event()
        history_start = len(self._history)
        self._history.append({"role": "user", "content": user_text})

        providers = [self._provider]
        if self._fallback_provider is not None:
            providers.append(self._fallback_provider)
        try:
            for index, provider in enumerate(providers):
                try:
                    while True:
                        assistant_content, tool_calls = self._consume_provider(
                            provider, on_sentence, on_event, cancel_event)
                        if assistant_content:
                            self._history.append({"role": "assistant", "content": assistant_content})
                        if not tool_calls:
                            self._trim_history()
                            return

                        results = []
                        for call in tool_calls:
                            if cancel_event.is_set():
                                raise _TurnCancelled
                            output = self._tools.execute(call.name, call.inputs)
                            if cancel_event.is_set():
                                raise _TurnCancelled
                            result = ProviderEvent.tool_result(call.tool_use_id, str(output))
                            if on_event:
                                on_event(result)
                            if cancel_event.is_set():
                                raise _TurnCancelled
                            results.append({
                                "type": "tool_result",
                                "tool_use_id": call.tool_use_id,
                                "content": str(output),
                            })
                        if cancel_event.is_set():
                            raise _TurnCancelled
                        self._history.append({"role": "user", "content": results})
                        self._trim_history()
                except _TurnCancelled:
                    raise
                except Exception:
                    if index == len(providers) - 1:
                        raise
                    if cancel_event.is_set():
                        raise _TurnCancelled
                    log.warning("provider %s falhou; tentando fallback", type(provider).__name__,
                                exc_info=True)
        except _TurnCancelled:
            del self._history[history_start:]
        except Exception:
            del self._history[history_start:]
            raise

    def _consume_provider(self, provider: LLMProvider, on_sentence, on_event,
                          cancel_event: threading.Event) -> tuple[list[dict], list[ProviderEvent]]:
        buffer = ""
        assistant_content = []
        tool_calls = []

        def emit_sentence(sentence: str) -> None:
            if cancel_event.is_set():
                raise _TurnCancelled
            on_sentence(sentence)
            if cancel_event.is_set():
                raise _TurnCancelled

        def flush_buffer() -> None:
            nonlocal buffer
            sentences, buffer = _pop_sentences(buffer, flush=True)
            for sentence in sentences:
                emit_sentence(sentence)

        try:
            events = provider.stream(
                model=self._model,
                system=self._system,
                messages=list(self._history),
                tools=self._tools.anthropic_schemas(),
                max_tokens=600,
                cancel_event=cancel_event,
            )
            for event in events:
                if cancel_event.is_set():
                    raise _TurnCancelled
                if event.type == "text_delta":
                    if on_event:
                        on_event(event)
                    if cancel_event.is_set():
                        raise _TurnCancelled
                    buffer += event.text
                    sentences, buffer = _pop_sentences(buffer)
                    for sentence in sentences:
                        emit_sentence(sentence)
                    assistant_content.append({"type": "text", "text": event.text})
                elif event.type == "tool_use":
                    flush_buffer()  # preâmbulo sempre vem antes da tool
                    if on_event:
                        on_event(event)
                    if cancel_event.is_set():
                        raise _TurnCancelled
                    assistant_content.append({
                        "type": "tool_use",
                        "id": event.tool_use_id,
                        "name": event.name,
                        "input": dict(event.inputs),
                    })
                    tool_calls.append(event)
                elif event.type == "end":
                    flush_buffer()
                    if on_event:
                        on_event(event)
                    break
                elif event.type == "cancelled":
                    if on_event:
                        on_event(event)
                    raise _TurnCancelled
                elif event.type == "tool_result":
                    if on_event:
                        on_event(event)
                else:
                    raise ValueError(f"evento de provider desconhecido: {event.type}")
            if cancel_event.is_set():
                raise _TurnCancelled
            flush_buffer()
            return assistant_content, tool_calls
        except _TurnCancelled:
            raise

    def _trim_history(self) -> None:
        if _history_units(self._history) <= self._max_history:
            return
        kept_units = 0
        start = len(self._history)
        for position in range(len(self._history) - 1, -1, -1):
            units = _message_units(self._history[position])
            if kept_units and kept_units + units > self._max_history:
                break
            kept_units += units
            start = position
        start = _safe_history_start(self._history, start)
        del self._history[:start]
