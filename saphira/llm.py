"""Cérebro: Anthropic Claude com tool use, streaming e chunking por frase p/ TTS."""
import logging
import re

import anthropic

from .tools import ToolRegistry

log = logging.getLogger("saphira.llm")

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


def _pop_sentences(buf: str) -> tuple[list[str], str]:
    """Separa frases completas do buffer; retorna (frases, resto)."""
    sentences, last = [], 0
    for m in _SENTENCE_END.finditer(buf):
        s = m.group(0).strip()
        if s:
            sentences.append(s)
        last = m.end()
    return sentences, buf[last:]


class ClaudeBrain:
    def __init__(self, api_key: str, model: str, tools: ToolRegistry,
                 commands_catalog: str, max_history: int = 20) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._tools = tools
        self._system = SYSTEM_TEMPLATE.format(commands_catalog=commands_catalog)
        self._max_history = max_history
        self._history: list[dict] = []

    def respond(self, user_text: str, on_sentence) -> None:
        """Loop de tool use. on_sentence(frase) é chamado p/ cada frase pronta p/ TTS."""
        self._history.append({"role": "user", "content": user_text})
        while True:
            with self._client.messages.stream(
                model=self._model,
                max_tokens=600,  # respostas faladas são curtas
                system=self._system,
                tools=self._tools.anthropic_schemas(),
                messages=self._history,
            ) as stream:
                buf = ""
                for delta in stream.text_stream:
                    buf += delta
                    sentences, buf = _pop_sentences(buf)
                    for s in sentences:
                        on_sentence(s)
                final = stream.get_final_message()
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

        # Memória: janela deslizante. Garante que o corte não deixa tool_result órfão
        # nem começa no meio de uma troca (API exige 1ª mensagem role=user).
        del self._history[:-self._max_history]
        while self._history and (self._history[0]["role"] != "user"
                                 or isinstance(self._history[0]["content"], list)):
            self._history.pop(0)
