"""Cérebro: Anthropic Claude com tool use, streaming e chunking por frase p/ TTS."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
import threading
from typing import Any, Iterable, Protocol

import anthropic

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


class LLMProvider(Protocol):
    """Contrato comum para Claude, Gemini e fakes de teste."""

    def stream(self, *, model: str, system: str, messages: list[dict],
               tools: list[dict], max_tokens: int,
               cancel_event: threading.Event) -> Iterable[ProviderEvent]: ...

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


class ClaudeBrain:
    def __init__(self, api_key: str, model: str, tools: ToolRegistry,
                 commands_catalog: str, max_history: int = 20,
                 provider: LLMProvider | None = None) -> None:
        self._provider = provider
        self._client = None if provider is not None else anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._tools = tools
        self._system = SYSTEM_TEMPLATE.format(commands_catalog=commands_catalog)
        self._max_history = max_history
        self._history: list[dict] = []

    def respond(self, user_text: str, on_sentence, on_event=None,
                cancel_event: threading.Event | None = None) -> None:
        """Loop de tool use. on_sentence(frase) é chamado p/ cada frase pronta p/ TTS."""
        if self._provider is not None:
            self._respond_with_provider(user_text, on_sentence, on_event, cancel_event)
            return

        self._respond_with_anthropic(user_text, on_sentence, cancel_event)

    def _respond_with_provider(self, user_text: str, on_sentence, on_event,
                               cancel_event: threading.Event | None) -> None:
        cancel_event = cancel_event or threading.Event()
        history_start = len(self._history)
        self._history.append({"role": "user", "content": user_text})
        buffer = ""
        assistant_content = []

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
            events = self._provider.stream(
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
                elif event.type == "end":
                    flush_buffer()
                    if on_event:
                        on_event(event)
                    break
                elif event.type == "cancelled":
                    if on_event:
                        on_event(event)
                    raise _TurnCancelled
                else:
                    raise ValueError(f"evento de provider desconhecido: {event.type}")
            if cancel_event.is_set():
                raise _TurnCancelled
            flush_buffer()
            if assistant_content:
                self._history.append({"role": "assistant", "content": assistant_content})
        except _TurnCancelled:
            del self._history[history_start:]

    def _respond_with_anthropic(self, user_text: str, on_sentence,
                                cancel_event: threading.Event | None) -> None:
        cancel_event = cancel_event or threading.Event()
        history_start = len(self._history)
        self._history.append({"role": "user", "content": user_text})
        try:
            while True:
                if cancel_event.is_set():
                    raise _TurnCancelled
                with self._client.messages.stream(
                    model=self._model,
                    max_tokens=600,  # respostas faladas são curtas
                    system=self._system,
                    tools=self._tools.anthropic_schemas(),
                    messages=self._history,
                ) as stream:
                    buf = ""
                    for delta in stream.text_stream:
                        if cancel_event.is_set():
                            raise _TurnCancelled
                        buf += delta
                        sentences, buf = _pop_sentences(buf)
                        for s in sentences:
                            on_sentence(s)
                    final = stream.get_final_message()
                if cancel_event.is_set():
                    raise _TurnCancelled
                if buf.strip():
                    # Resto do buffer: resposta final OU preâmbulo antes de tool call
                    # ("deixa eu ver...") — falar os dois é o comportamento desejado.
                    on_sentence(buf.strip())
                self._history.append({"role": "assistant", "content": final.content})
                if final.stop_reason != "tool_use":
                    break
                results = []
                for block in final.content:
                    if block.type == "tool_use":
                        log.info("tool_call: %s %s", block.name, block.input)
                        output = self._tools.execute(block.name, block.input)
                        results.append({"type": "tool_result",
                                        "tool_use_id": block.id,
                                        "content": output})
                self._history.append({"role": "user", "content": results})
        except _TurnCancelled:
            del self._history[history_start:]

        # Memória: janela deslizante. Garante que o corte não deixa tool_result órfão
        # nem começa no meio de uma troca (API exige 1ª mensagem role=user).
        del self._history[:-self._max_history]
        while self._history and (self._history[0]["role"] != "user"
                                 or isinstance(self._history[0]["content"], list)):
            self._history.pop(0)
