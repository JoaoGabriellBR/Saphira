"""Checks mínimos das partes puras da Saphira (sem frameworks).
Rodar: python tests/test_core.py
"""
import io
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- config: valores do v1 e fail-fast sem chaves reais ---
ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID", "TAVILY_API_KEY", "GEMINI_API_KEY"]
for k in REQUIRED + ["ANTHROPIC_API_KEY"]:
    os.environ[k] = f"test-{k.lower()}"  # setado antes do import: .env não sobrescreve
for opt in ("CLAUDE_MODEL", "SAPHIRA_HOTKEY"):
    os.environ.pop(opt, None)

from saphira import config

assert config.CLAUDE_MODEL == "claude-sonnet-5", config.CLAUDE_MODEL
assert config.GEMINI_API_KEY == "test-gemini_api_key"
assert config.GEMINI_MODEL == "gemini-3.6-flash"
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

# valor que é só o comentário inline do template (chave não preenchida) -> RuntimeError como ausente
env = {**os.environ, "GEMINI_API_KEY": "# aistudio.google.com → API key (fallback LLM)",
       "PYTHONPATH": str(ROOT)}
r = subprocess.run([sys.executable, "-c", "import saphira.config"],
                   env=env, cwd=tempfile.gettempdir(), capture_output=True, text=True)
assert r.returncode != 0 and "GEMINI_API_KEY" in r.stderr, r.stderr

# ANTHROPIC_API_KEY é opcional: só comentário no .env vira "" e o boot segue
env = {**os.environ, "ANTHROPIC_API_KEY": "# console.anthropic.com → API Keys",
       "PYTHONPATH": str(ROOT)}
r = subprocess.run([sys.executable, "-c",
                    "import saphira.config; print(repr(saphira.config.ANTHROPIC_API_KEY))"],
                   env=env, cwd=tempfile.gettempdir(), capture_output=True, text=True)
assert r.returncode == 0 and r.stdout.strip() == "''", (r.returncode, r.stderr)

# dependências declaradas e .env.example sem valores preenchidos
assert "google-genai" in (ROOT / "requirements.txt").read_text(encoding="utf-8")
for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        assert line.split("=", 1)[1].split("#", 1)[0].strip() == "", f"valor no .env.example: {line}"

from saphira import audio as audio_module
from saphira import llm as llm_module
from saphira import main as main_module
from saphira import speech as speech_module
from saphira import tools as tools_module
from saphira.llm import ClaudeBrain, GeminiProvider, ProviderEvent, _history_units, _pop_sentences
from saphira.tools import LocalCommandTool, ToolRegistry

# --- sentence splitter ---
s, rest = _pop_sentences("Oi, tudo bem. Como vai")
assert s == ["Oi, tudo bem."] and rest == "Como vai", (s, rest)

s, rest = _pop_sentences("Um.\nDois.\n")
assert s == ["Um.", "Dois."] and rest == "", (s, rest)

s, rest = _pop_sentences("sem pontuacao ainda")
assert s == [] and rest == "sem pontuacao ainda"
s, rest = _pop_sentences("sem pontuacao ainda", flush=True)
assert s == ["sem pontuacao ainda"] and rest == ""


# --- contrato comum de provider e cancelamento ---
class FakeProvider:
    def __init__(self, events, follow_up=None):
        self.events = events
        self.follow_up = follow_up or [ProviderEvent.end()]
        self.request = None
        self.calls = 0

    def stream(self, **request):
        self.request = request
        self.calls += 1
        yield from (self.events if self.calls == 1 else self.follow_up)


provider = FakeProvider([
    ProviderEvent.text_delta("Olá, tudo bem? "),
    ProviderEvent.text_delta("Deixa eu verificar"),
    ProviderEvent.tool_use("call-1", "web_search", {"query": "tempo"}),
    ProviderEvent.end("tool_use"),
])
brain = ClaudeBrain("unused", "test-model", ToolRegistry(), "catálogo",
                    provider=provider)
trace = []
brain.respond("como está?", on_sentence=lambda s: trace.append(("sentence", s)),
              on_event=lambda event: trace.append((event.type, event)))
assert [kind for kind, _ in trace] == [
    "text_delta", "sentence", "text_delta", "sentence", "tool_use", "end",
    "tool_result", "end",
], trace
assert trace[1][1] == "Olá, tudo bem?"
assert trace[3][1] == "Deixa eu verificar"
assert provider.request["model"] == "test-model"
assert provider.request["max_tokens"] == 600
assert provider.request["system"].endswith("catálogo")


cancel = threading.Event()
cancel_provider = FakeProvider([
    ProviderEvent.text_delta("resposta parcial"),
    ProviderEvent.cancelled(),
    ProviderEvent.text_delta("não deve ser emitida."),
    ProviderEvent.end(),
])


def cancel_after_first(event):
    if event.type == "text_delta":
        cancel.set()


cancel_brain = ClaudeBrain("unused", "test-model", ToolRegistry(), "catálogo",
                           provider=cancel_provider)
cancelled_sentences = []
cancelled_events = []
cancel_brain.respond("cancele", cancelled_sentences.append,
                     on_event=lambda event: (cancelled_events.append(event),
                                             cancel_after_first(event)),
                     cancel_event=cancel)
assert cancelled_sentences == []
assert [event.type for event in cancelled_events] == ["text_delta"]
assert cancel_brain._history == []


# --- adaptador Anthropic: streaming, prompt, tool request e flush final ---
class FakeClaudeBlock:
    type = "tool_use"
    id = "claude-call-1"
    name = "fake_tool"
    input = {"value": "teste"}


class FakeClaudeFinal:
    def __init__(self, tool_use):
        self.stop_reason = "tool_use" if tool_use else "end_turn"
        self.content = [FakeClaudeBlock()] if tool_use else []


class FakeClaudeStream:
    def __init__(self, tool_use):
        self.text_stream = (["Vou verificar. ", "resto sem pontuação"]
                            if tool_use else ["resposta final."])
        self._tool_use = tool_use

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get_final_message(self):
        return FakeClaudeFinal(self._tool_use)


class FakeClaudeMessages:
    def __init__(self):
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return FakeClaudeStream(len(self.calls) == 1)


class FakeAnthropic:
    instances = []

    def __init__(self, api_key):
        self.api_key = api_key
        self.messages = FakeClaudeMessages()
        self.instances.append(self)


class UnicodeFailingClaudeStream:
    def __enter__(self):
        raise UnicodeEncodeError("ascii", "header →", 7, 8, "ordinal not in range(128)")

    def __exit__(self, *args):
        return False


class UnicodeFailingClaudeMessages:
    def stream(self, **kwargs):
        return UnicodeFailingClaudeStream()


class UnicodeFailingAnthropic:
    def __init__(self, api_key):
        self.api_key = api_key
        self.messages = UnicodeFailingClaudeMessages()


class FakeClaudeTool:
    name = "fake_tool"
    description = "tool de teste"
    input_schema = {"type": "object", "properties": {"value": {"type": "string"}}}

    def run(self, value):
        return f"ok {value}"


old_anthropic = llm_module.anthropic.Anthropic
llm_module.anthropic.Anthropic = FakeAnthropic
try:
    claude_tools = ToolRegistry()
    claude_tools.register(FakeClaudeTool())
    claude = ClaudeBrain("fake-key", "claude-sonnet-5", claude_tools,
                         "- app 'spotify' (ações: open_app/close_app)")
    claude_trace = []

    def observe_claude(event):
        claude_trace.append((event.type, event))
        if event.type == "tool_use":
            claude_trace.append(("tool_seen", event.name))

    claude.respond("verifique", lambda sentence: claude_trace.append(("sentence", sentence)),
                   on_event=observe_claude)
    claude_request = FakeAnthropic.instances[-1].messages.calls[0]
finally:
    llm_module.anthropic.Anthropic = old_anthropic

assert [kind for kind, _ in claude_trace] == [
    "text_delta", "sentence", "text_delta", "sentence",
    "tool_use", "tool_seen", "end", "tool_result",
    "text_delta", "sentence", "end",
], claude_trace
assert claude_trace[1][1] == "Vou verificar."
assert claude_trace[3][1] == "resto sem pontuação"
assert claude_request["model"] == "claude-sonnet-5"
assert claude_request["max_tokens"] == 600
assert "português do Brasil" in claude_request["system"]
assert "spotify" in claude_request["system"]
assert claude_request["tools"] == claude_tools.anthropic_schemas()


# --- provider Gemini: parity de eventos, modelo próprio e cancelamento ---
class FakeGeminiFunctionCall:
    id = "gemini-call-1"
    name = "fake_tool"
    args = {"value": "gemini"}


class FakeGeminiFunctionResponse:
    id = "gemini-call-1"
    name = "fake_tool"
    response = {"result": "ok gemini"}


class FakeGeminiChunk:
    def __init__(self, text=None, function_calls=None, function_responses=None):
        self.text = text
        self.function_calls = function_calls or []
        self.function_responses = function_responses or []


class FakeGeminiModels:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def generate_content_stream(self, **kwargs):
        self.calls.append(kwargs)
        yield from self.chunks


class FakeGeminiClient:
    def __init__(self, chunks):
        self.models = FakeGeminiModels(chunks)


gemini_client = FakeGeminiClient([
    FakeGeminiChunk(text="Resposta Gemini. "),
    FakeGeminiChunk(function_calls=[FakeGeminiFunctionCall()]),
    FakeGeminiChunk(function_responses=[FakeGeminiFunctionResponse()]),
])
gemini = GeminiProvider("fake-gemini-key", client=gemini_client)
gemini_events = list(gemini.stream(
    model="ignored-by-gemini",
    system="sistema",
    messages=[{"role": "user", "content": "oi"}],
    tools=[{"name": "fake_tool", "description": "x", "input_schema": {"type": "object"}}],
    max_tokens=600,
    cancel_event=threading.Event(),
))
assert [event.type for event in gemini_events] == [
    "text_delta", "tool_use", "tool_result", "end"
], gemini_events
assert gemini_events[1].name == "fake_tool"
assert gemini_events[2].content == "ok gemini"
gemini_request = gemini_client.models.calls[0]
assert gemini_request["model"] == "gemini-3.6-flash"
assert gemini_request["config"]["max_output_tokens"] == 600
assert gemini_request["config"]["system_instruction"] == "sistema"
assert gemini_request["config"]["tools"][0]["function_declarations"][0]["name"] == "fake_tool"
assert "parameters_json_schema" in gemini_request["config"]["tools"][0]["function_declarations"][0]

cancel_gemini = threading.Event()
cancel_gemini.set()
assert [event.type for event in gemini.stream(
    model="ignored",
    system="sistema",
    messages=[],
    tools=[],
    max_tokens=600,
    cancel_event=cancel_gemini,
)] == ["cancelled"]


# --- timeout de rede do fallback Gemini ---
class CapturingGenaiClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)


old_genai_client = llm_module.genai.Client
llm_module.genai.Client = CapturingGenaiClient
try:
    GeminiProvider("fake-gemini-key")
finally:
    llm_module.genai.Client = old_genai_client

assert CapturingGenaiClient.instances[-1].kwargs["http_options"]["timeout"] == 60_000


# --- failover Claude -> Gemini sem duplicar o user_text ---
class FailingProvider:
    def __init__(self, detail):
        self.detail = detail
        self.calls = []

    def stream(self, **request):
        self.calls.append(request)
        raise RuntimeError(self.detail)


class SuccessfulProvider:
    def __init__(self):
        self.calls = []

    def stream(self, **request):
        self.calls.append(request)
        yield ProviderEvent.text_delta("resposta do fallback")
        yield ProviderEvent.end()


old_anthropic = llm_module.anthropic.Anthropic
llm_module.anthropic.Anthropic = UnicodeFailingAnthropic
try:
    unicode_primary = llm_module.AnthropicProvider("fake-key")
    unicode_fallback = SuccessfulProvider()
    unicode_brain = ClaudeBrain("unused", "claude-sonnet-5", ToolRegistry(), "catálogo",
                                provider=unicode_primary, fallback_provider=unicode_fallback)
    unicode_sentences = []
    unicode_brain.respond("uma pergunta", unicode_sentences.append)
finally:
    llm_module.anthropic.Anthropic = old_anthropic

assert unicode_sentences == ["resposta do fallback"]
assert len(unicode_fallback.calls) == 1
assert unicode_fallback.calls[0]["messages"] == [{"role": "user", "content": "uma pergunta"}]


primary = FailingProvider("Claude indisponível")
fallback = SuccessfulProvider()
fallback_brain = ClaudeBrain("unused", "claude-sonnet-5", ToolRegistry(), "catálogo",
                             provider=primary, fallback_provider=fallback)
fallback_sentences = []
fallback_brain.respond("uma pergunta", fallback_sentences.append)
assert fallback_sentences == ["resposta do fallback"]
assert len(primary.calls) == len(fallback.calls) == 1
assert [m["role"] for m in primary.calls[0]["messages"]] == ["user"]
assert [m["role"] for m in fallback.calls[0]["messages"]] == ["user"]
assert fallback.calls[0]["messages"][0]["content"] == "uma pergunta"

first_failure = FailingProvider("Claude indisponível")
second_failure = FailingProvider("Gemini indisponível")
both_failed_brain = ClaudeBrain("unused", "claude-sonnet-5", ToolRegistry(), "catálogo",
                                provider=first_failure, fallback_provider=second_failure)
try:
    both_failed_brain.respond("falhe", lambda _: None)
except RuntimeError as exc:
    assert str(exc) == "Gemini indisponível"
else:
    assert False, "falha dos dois providers deveria chegar ao orquestrador"
assert len(first_failure.calls) == len(second_failure.calls) == 1
assert both_failed_brain._history == []


# --- janela de memória: contagem, corte seguro e cancelamento ---
window_brain = ClaudeBrain("unused", "test-model", ToolRegistry(), "catálogo",
                           max_history=20, provider=SuccessfulProvider())
window_brain._history = [
    {"role": role, "content": f"mensagem {i}"}
    for i in range(11)
    for role in ("user", "assistant")
]
assert _history_units(window_brain._history) == 22
window_brain._trim_history()
assert len(window_brain._history) == 20
assert window_brain._history[0]["content"] == "mensagem 1"
assert _history_units([{"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "a", "content": "ok"},
    {"type": "tool_result", "tool_use_id": "b", "content": "ok"},
]}]) == 2

tool_window = ClaudeBrain("unused", "test-model", ToolRegistry(), "catálogo",
                          max_history=4, provider=SuccessfulProvider())
tool_window._history = [
    {"role": "user", "content": "pergunta antiga"},
    {"role": "assistant", "content": [{"type": "text", "text": "resposta antiga"}]},
    {"role": "user", "content": "pergunta com tool"},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "call-1", "name": "fake_tool", "input": {}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "ok"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "resultado final"}]},
    {"role": "user", "content": "pergunta nova"},
    {"role": "assistant", "content": [{"type": "text", "text": "resposta nova"}]},
]
tool_window._trim_history()
assert tool_window._history[0]["content"] == "pergunta com tool"
assert any(block.get("type") == "tool_use" for block in tool_window._history[1]["content"])
assert any(block.get("type") == "tool_result" for block in tool_window._history[2]["content"])
assert _history_units(tool_window._history) == 6  # extensão segura pode passar de 4

new_brain = ClaudeBrain("unused", "test-model", ToolRegistry(), "catálogo",
                        provider=SuccessfulProvider())
assert new_brain._history == []

cancel_history = [{"role": "user", "content": "histórico preservado"}]
cancel_memory_brain = ClaudeBrain("unused", "test-model", ToolRegistry(), "catálogo",
                                  provider=FakeProvider([ProviderEvent.text_delta("parcial")]))
cancel_memory_brain._history = list(cancel_history)
cancel_memory_event = threading.Event()
cancel_memory_brain.respond(
    "turno cancelado", lambda _: None,
    on_event=lambda event: cancel_memory_event.set(),
    cancel_event=cancel_memory_event,
)
assert cancel_memory_brain._history == cancel_history


# --- loop de tools: execução, reinjeção, erro e cancelamento ---
class LoopTool:
    name = "loop_tool"
    description = "tool do loop"
    input_schema = {"type": "object", "properties": {"query": {"type": "string"}}}

    def __init__(self, order, result="resultado da tool", cancel=None, error=None):
        self.order = order
        self.result = result
        self.cancel = cancel
        self.error = error

    def run(self, query):
        self.order.append(("tool", query))
        if self.cancel is not None:
            self.cancel.set()
        if self.error:
            raise RuntimeError(self.error)
        return self.result


class ToolLoopProvider:
    def __init__(self, final_text="resultado tratado"):
        self.requests = []
        self.final_text = final_text

    def stream(self, **request):
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ProviderEvent.text_delta("Vou consultar. ")
            yield ProviderEvent.tool_use("loop-call-1", "loop_tool", {"query": "agora"})
            yield ProviderEvent.end("tool_use")
            return
        tool_results = [
            block
            for message in request["messages"]
            if isinstance(message.get("content"), list)
            for block in message["content"]
            if block.get("type") == "tool_result"
        ]
        assert tool_results, request["messages"]
        yield ProviderEvent.text_delta(self.final_text)
        yield ProviderEvent.end()


tool_order = []
tool_registry = ToolRegistry()
tool_registry.register(LoopTool(tool_order, result="42"))
tool_provider = ToolLoopProvider(final_text="O resultado foi 42.")
tool_brain = ClaudeBrain("unused", "test-model", tool_registry, "catálogo",
                         provider=tool_provider)
spoken = []
provider_events = []
tool_brain.respond(
    "qual o resultado?",
    lambda sentence: (spoken.append(sentence), tool_order.append(("sentence", sentence))),
    on_event=provider_events.append,
)
assert spoken == ["Vou consultar.", "O resultado foi 42."]
assert tool_order == [
    ("sentence", "Vou consultar."),
    ("tool", "agora"),
    ("sentence", "O resultado foi 42."),
]
assert len(tool_provider.requests) == 2
assert [event.type for event in provider_events] == [
    "text_delta", "tool_use", "end", "tool_result", "text_delta", "end"
]
result_messages = [
    message for message in tool_provider.requests[1]["messages"]
    if isinstance(message.get("content"), list)
]
assert result_messages[-1]["content"] == [{
    "type": "tool_result",
    "tool_use_id": "loop-call-1",
    "content": "42",
}]

error_order = []
error_registry = ToolRegistry()
error_registry.register(LoopTool(error_order, error="falha controlada"))
error_provider = ToolLoopProvider(final_text="A tool falhou, mas a conversa continua.")
error_brain = ClaudeBrain("unused", "test-model", error_registry, "catálogo",
                          provider=error_provider)
error_spoken = []
error_brain.respond("execute", error_spoken.append)
error_results = [
    block
    for message in error_provider.requests[1]["messages"]
    if isinstance(message.get("content"), list)
    for block in message["content"]
    if block.get("type") == "tool_result"
]
assert error_spoken == ["Vou consultar.", "A tool falhou, mas a conversa continua."]
assert error_results[0]["content"] == "Erro ao executar loop_tool: falha controlada"

cancel_tool_event = threading.Event()
cancel_order = []
cancel_registry = ToolRegistry()
cancel_registry.register(LoopTool(cancel_order, cancel=cancel_tool_event))
cancel_provider = ToolLoopProvider()
cancel_loop_brain = ClaudeBrain("unused", "test-model", cancel_registry, "catálogo",
                                provider=cancel_provider)
cancel_loop_brain.respond("cancele", lambda _: None, cancel_event=cancel_tool_event)
assert len(cancel_provider.requests) == 1
assert cancel_loop_brain._history == []
assert cancel_order == [("tool", "agora")]

# --- WebSearchTool ---
class FakeTavily:
    response = {
        "answer": "resposta atual",
        "results": [{"title": f"Título {i}", "content": "c" * 400} for i in range(6)],
    }
    instances = []

    def __init__(self, api_key):
        self.api_key = api_key
        self.calls = []
        self.instances.append(self)

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.response


old_tavily = tools_module.TavilyClient
tools_module.TavilyClient = FakeTavily
try:
    web = tools_module.WebSearchTool("fake-key")
    result = web.run("notícias de hoje")
    assert web._client.calls == [(
        "notícias de hoje",
        {"search_depth": "basic", "max_results": 4, "include_answer": True},
    )]
    assert "resposta atual" in result
    assert sum(line.startswith("- ") for line in result.splitlines()) <= 4
    assert "- Título 0: " in result and "- Título 3: " in result
    assert "- Título 4: " not in result
    assert len(result) <= 3000

    FakeTavily.response = {"answer": "", "results": []}
    assert tools_module.WebSearchTool("fake-key").run("sem resultados") == "Nenhum resultado encontrado."

    class TimeoutTavily(FakeTavily):
        def search(self, query, **kwargs):
            raise TimeoutError("timeout Tavily")

    tools_module.TavilyClient = TimeoutTavily
    registry = ToolRegistry()
    registry.register(tools_module.WebSearchTool("fake-key"))
    assert registry.execute("web_search", {"query": "falha"}) == \
        "Erro ao executar web_search: timeout Tavily"
finally:
    tools_module.TavilyClient = old_tavily

# --- áudio: captura, playback, hotkey e VAD ---
stop = threading.Event()
input_streams = []


class FakeInputStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        input_streams.append(self)

    def __enter__(self):
        stop.set()
        try:
            self.kwargs["callback"](b"\x01\x02", 1, None, None)
        except audio_module.sd.CallbackStop:
            pass
        return self

    def __exit__(self, *args):
        return False


old_input_stream = audio_module.sd.RawInputStream
audio_module.sd.RawInputStream = FakeInputStream
try:
    captured = audio_module.record_until(stop)
finally:
    audio_module.sd.RawInputStream = old_input_stream

assert captured == b"\x01\x02"
assert input_streams[0].kwargs["samplerate"] == 16000
assert input_streams[0].kwargs["channels"] == 1
assert input_streams[0].kwargs["dtype"] == "int16"

timeouts = []


class FakeDone:
    def set(self):
        pass

    def wait(self, timeout):
        timeouts.append(timeout)
        return False


class IdleInputStream:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


old_event = audio_module.threading.Event
old_input_stream = audio_module.sd.RawInputStream
audio_module.threading.Event = lambda: FakeDone()
audio_module.sd.RawInputStream = IdleInputStream
try:
    assert audio_module.record_until(type("NeverStop", (), {"is_set": lambda self: False})(), 999) == b""
finally:
    audio_module.threading.Event = old_event
    audio_module.sd.RawInputStream = old_input_stream
assert timeouts == [90]

writes = []
first_chunk_events = []
interrupt = threading.Event()
output_streams = []


class FakeOutputStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        output_streams.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def write(self, chunk):
        writes.append(chunk)
        interrupt.set()


old_output_stream = audio_module.sd.RawOutputStream
audio_module.sd.RawOutputStream = FakeOutputStream
try:
    audio_module.play_pcm([b"primeiro", b"descartado"], interrupt,
                          lambda: first_chunk_events.append(len(writes)))
finally:
    audio_module.sd.RawOutputStream = old_output_stream

assert writes == [b"primeiro"]
assert first_chunk_events == [1]
assert output_streams[0].kwargs == {"samplerate": 24000, "channels": 1, "dtype": "int16"}

hotkey_calls = []
listeners = []


class FakeGlobalHotKeys:
    def __init__(self, mapping):
        self.mapping = mapping
        self.started = False
        self.stopped = False
        listeners.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def press(self, combo):
        self.mapping[combo]()


old_global_hotkeys = audio_module.keyboard.GlobalHotKeys
audio_module.keyboard.GlobalHotKeys = FakeGlobalHotKeys
try:
    hotkey = audio_module.Hotkey("<f9>", lambda: hotkey_calls.append("pressed"))
    assert hotkey_calls == []
    hotkey.start()
    listeners[0].press("<f9>")
    hotkey.stop()
finally:
    audio_module.keyboard.GlobalHotKeys = old_global_hotkeys
assert hotkey_calls == ["pressed"]
assert listeners[0].started and listeners[0].stopped

vad_events = []
vad = audio_module.VAD(lambda: vad_events.append("speech"), debounce_ms=200)
vad.signal(True, now=0.0)
vad.signal(True, now=0.199)
assert vad_events == []
vad.signal(True, now=0.201)
vad.signal(True, now=0.4)
assert vad_events == ["speech"]
vad.signal(False, now=0.5)
vad.signal(True, now=1.0)
vad.signal(True, now=1.201)
assert vad_events == ["speech", "speech"]
for invalid_debounce in (149, 301):
    try:
        audio_module.VAD(lambda: None, debounce_ms=invalid_debounce)
    except ValueError:
        pass
    else:
        assert False, invalid_debounce

monitor_signals = []
monitor_stop = threading.Event()


class FakeMonitorVAD:
    def signal(self, speaking):
        monitor_signals.append(speaking)


class FakeMonitorInputStream:
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]

    def __enter__(self):
        self.callback(b"\x00\x00" * 4, 4, None, None)
        self.callback(b"\xc8\x00" * 4, 4, None, None)
        monitor_stop.set()
        return self

    def __exit__(self, *args):
        return False


old_monitor_input = audio_module.sd.RawInputStream
audio_module.sd.RawInputStream = FakeMonitorInputStream
try:
    audio_module.monitor_vad(FakeMonitorVAD(), monitor_stop, threshold=100)
finally:
    audio_module.sd.RawInputStream = old_monitor_input
assert monitor_signals == [False, True]

# --- Speech: ElevenLabs fake e fallback local ---
class FakeSpeechToText:
    def convert(self, **kwargs):
        self.kwargs = kwargs
        self.wav = kwargs["file"].read()
        return type("Response", (), {"text": " texto transcrito "})()


class FakeTextToSpeech:
    def stream(self, **kwargs):
        self.kwargs = kwargs
        return iter([b"\x00\x01", b"\x02\x03"])


class FakeElevenLabs:
    instances = []

    def __init__(self, api_key, **kwargs):
        self.api_key = api_key
        self.timeout = kwargs.get("timeout")
        self.speech_to_text = FakeSpeechToText()
        self.text_to_speech = FakeTextToSpeech()
        self.instances.append(self)


old_elevenlabs = speech_module.ElevenLabs
speech_module.ElevenLabs = FakeElevenLabs
try:
    speech = speech_module.Speech("fake-key", "voice", "scribe_v2", "eleven_flash_v2_5", "por")
    pcm = b"\x01\x02" * 20
    assert speech.transcribe(pcm) == "texto transcrito"
    stt_call = FakeElevenLabs.instances[-1].speech_to_text
    assert FakeElevenLabs.instances[-1].timeout == 30
    with wave.open(io.BytesIO(stt_call.wav), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.readframes(wav.getnframes()) == pcm
    assert stt_call.kwargs["model_id"] == "scribe_v2"
    assert stt_call.kwargs["language_code"] == "por"

    assert list(speech.synthesize_stream("olá")) == [b"\x00\x01", b"\x02\x03"]
    assert FakeElevenLabs.instances[-1].text_to_speech.kwargs == {
        "text": "olá",
        "voice_id": "voice",
        "model_id": "eleven_flash_v2_5",
        "output_format": "pcm_24000",
    }
finally:
    speech_module.ElevenLabs = old_elevenlabs

fallback_before = len(FakeElevenLabs.instances)
fallback_writes = []


class FallbackOutputStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def write(self, chunk):
        fallback_writes.append(chunk)


assert speech_module.LOCAL_FALLBACK_MESSAGE == "não consegui te ouvir, tenta de novo"
old_elevenlabs = speech_module.ElevenLabs


def no_external_client(*args, **kwargs):
    raise AssertionError("fallback local não deve criar cliente ElevenLabs")


speech_module.ElevenLabs = no_external_client
old_output_stream = audio_module.sd.RawOutputStream
audio_module.sd.RawOutputStream = FallbackOutputStream
try:
    audio_module.play_pcm(speech_module.local_fallback_chunks(), threading.Event())
finally:
    speech_module.ElevenLabs = old_elevenlabs
    audio_module.sd.RawOutputStream = old_output_stream
assert fallback_writes and all(isinstance(chunk, bytes) for chunk in fallback_writes)
assert len(FakeElevenLabs.instances) == fallback_before

# --- worker de fala frase a frase ---
class FakeWorkerSpeech:
    def __init__(self):
        self.calls = []

    def synthesize_stream(self, text):
        self.calls.append(text)
        if text == "falha":
            raise RuntimeError("TTS indisponível")
        return [f"{text}-1".encode(), f"{text}-2".encode()]


worker_speech = FakeWorkerSpeech()
worker_played = []
worker_first_audio = []


def fake_worker_play(chunks, interrupt, on_first_chunk=None):
    for chunk in chunks:
        if interrupt.is_set():
            return
        worker_played.append(chunk)
        if on_first_chunk:
            on_first_chunk()


old_worker_play = main_module.audio.play_pcm
old_worker_fallback = main_module.local_fallback_chunks
main_module.audio.play_pcm = fake_worker_play
main_module.local_fallback_chunks = lambda: iter([b"fallback"])
try:
    worker_queue = queue.Queue()
    for sentence in ("falha", "ok"):
        worker_queue.put(sentence)
    worker_queue.put(None)
    main_module.speaker_worker(
        worker_queue,
        worker_speech,
        threading.Event(),
        on_first_audio=lambda: worker_first_audio.append("audio"),
    )

    discarded_queue = queue.Queue()
    discarded_queue.put("descartada")
    discarded_queue.put(None)
    interrupt = threading.Event()
    interrupt.set()
    main_module.speaker_worker(discarded_queue, worker_speech, interrupt)
finally:
    main_module.audio.play_pcm = old_worker_play
    main_module.local_fallback_chunks = old_worker_fallback

assert worker_speech.calls == ["falha", "ok"]
assert worker_played == [b"fallback", b"ok-1", b"ok-2"]
assert worker_first_audio == ["audio"]

# --- degradação de voz ---
class FailingTtsSpeech:
    def __init__(self):
        self.calls = []

    def synthesize_stream(self, text):
        self.calls.append(text)
        raise RuntimeError("ElevenLabs indisponível")


fallback_log_messages = []


class FallbackLogHandler(logging.Handler):
    def __init__(self, messages):
        super().__init__()
        self.messages = messages

    def emit(self, record):
        self.messages.append(record.getMessage())


fallback_handler = FallbackLogHandler(fallback_log_messages)
fallback_log = logging.getLogger("saphira")
fallback_log.setLevel(logging.INFO)
fallback_log.addHandler(fallback_handler)
local_audio = []
old_fallback_play = main_module.audio.play_pcm
old_fallback_chunks = main_module.local_fallback_chunks


def fake_local_play(chunks, interrupt, on_first_chunk=None):
    local_audio.extend(chunks)


main_module.audio.play_pcm = fake_local_play
main_module.local_fallback_chunks = lambda: iter([b"audio local"])
try:
    failing_tts = FailingTtsSpeech()
    main_module.say(failing_tts, "mensagem fixa")
finally:
    main_module.audio.play_pcm = old_fallback_play
    main_module.local_fallback_chunks = old_fallback_chunks
    fallback_log.removeHandler(fallback_handler)

assert failing_tts.calls == ["mensagem fixa"]
assert local_audio == [b"audio local"]

class FailingSttSpeech(FailingTtsSpeech):
    def transcribe(self, pcm):
        raise RuntimeError("Scribe indisponível")


stt_local_audio = []
old_fallback_play = main_module.audio.play_pcm
old_fallback_chunks = main_module.local_fallback_chunks


def fake_stt_fallback_play(chunks, interrupt, on_first_chunk=None):
    stt_local_audio.extend(chunks)


main_module.audio.play_pcm = fake_stt_fallback_play
main_module.local_fallback_chunks = lambda: iter([b"audio local STT"])
stt_latency = main_module.LatencyMetrics(clock=lambda: 2.0)
stt_latency.capture_finished(at=1.0)
try:
    assert not main_module.run_turn(
        b"x" * 3200,
        FailingSttSpeech(),
        None,
        stt_latency,
    )
finally:
    main_module.audio.play_pcm = old_fallback_play
    main_module.local_fallback_chunks = old_fallback_chunks
assert stt_local_audio == [b"audio local STT"]

failed_fallback_logs = []
failed_fallback_handler = FallbackLogHandler(failed_fallback_logs)
fallback_log.addHandler(failed_fallback_handler)
old_fallback_play = main_module.audio.play_pcm
old_fallback_chunks = main_module.local_fallback_chunks


def unavailable_audio(*args, **kwargs):
    raise RuntimeError("sink indisponível")


main_module.audio.play_pcm = unavailable_audio
main_module.local_fallback_chunks = lambda: iter([b"sem saida"])
try:
    main_module.say(FailingTtsSpeech(), "sem voz")
finally:
    main_module.audio.play_pcm = old_fallback_play
    main_module.local_fallback_chunks = old_fallback_chunks
    fallback_log.removeHandler(failed_fallback_handler)
assert any("fallback local" in message and speech_module.LOCAL_FALLBACK_MESSAGE in message
           for message in failed_fallback_logs)


class MidResponseSpeech:
    def __init__(self):
        self.calls = []

    def synthesize_stream(self, text):
        self.calls.append(text)
        return [text.encode()]


mid_speech = MidResponseSpeech()
mid_play_calls = 0
mid_played = []


def fail_first_play(chunks, interrupt, on_first_chunk=None):
    global mid_play_calls
    mid_play_calls += 1
    if mid_play_calls == 1:
        raise RuntimeError("TTS caiu durante playback")
    mid_played.extend(chunks)


old_mid_play = main_module.audio.play_pcm
old_mid_fallback = main_module.local_fallback_chunks
main_module.audio.play_pcm = fail_first_play
main_module.local_fallback_chunks = lambda: iter([b"fallback local"])
try:
    mid_queue = queue.Queue()
    mid_queue.put("primeira")
    mid_queue.put("segunda")
    mid_queue.put(None)
    main_module.speaker_worker(mid_queue, mid_speech, threading.Event())
finally:
    main_module.audio.play_pcm = old_mid_play
    main_module.local_fallback_chunks = old_mid_fallback
assert mid_speech.calls == ["primeira", "segunda"]
assert mid_played == [b"fallback local", b"segunda"]

# --- eventos e métricas de latência ---
latency_messages = []


class LatencyHandler(logging.Handler):
    def emit(self, record):
        latency_messages.append(record.getMessage())


latency_handler = LatencyHandler()
main_log = logging.getLogger("saphira")
main_log.setLevel(logging.INFO)
main_log.addHandler(latency_handler)
try:
    latency = main_module.LatencyMetrics(max_interactions=20)
    latency.capture_started(at=10.0)
    latency.capture_finished(at=11.0)
    first_snapshot = latency.first_audio(at=11.25)
finally:
    main_log.removeHandler(latency_handler)

assert first_snapshot == {
    "ttfa_ms": 250.0,
    "p50_ms": 250.0,
    "p95_ms": 250.0,
    "window_size": 1,
}
assert any("captura iniciada" in message for message in latency_messages)
assert any("captura finalizada" in message for message in latency_messages)
assert any("primeiro chunk tocando" in message for message in latency_messages)

for i in range(1, 21):
    finished = 100.0 + i
    snapshot = latency.record(finished, finished + i / 1000)
assert snapshot["window_size"] == 20
assert round(snapshot["ttfa_ms"], 2) == 20.0
assert round(snapshot["p50_ms"], 2) == 10.5
assert round(snapshot["p95_ms"], 2) == 19.05

# --- loop PTT para turno normal ---
class FakeTurnSpeech:
    def __init__(self, text="resposta transcrita", error=None):
        self.text = text
        self.error = error
        self.transcribe_calls = []
        self.synthesize_calls = []

    def transcribe(self, pcm):
        self.transcribe_calls.append(pcm)
        if self.error:
            raise self.error
        return self.text

    def synthesize_stream(self, text):
        self.synthesize_calls.append(text)
        return [b"pcm da resposta"]


class FakeTurnBrain:
    def __init__(self):
        self.calls = []

    def respond(self, text, on_sentence, cancel_event=None):
        self.calls.append(text)
        on_sentence("Resposta factual.")


turn_playback = []


def fake_turn_play(chunks, interrupt, on_first_chunk=None):
    for chunk in chunks:
        if interrupt.is_set():
            return
        turn_playback.append(chunk)
        if on_first_chunk:
            on_first_chunk()


turn_speech = FakeTurnSpeech()
turn_brain = FakeTurnBrain()
turn_latency = main_module.LatencyMetrics(clock=lambda: 2.0)
turn_latency.capture_finished(at=1.0)
old_turn_play = main_module.audio.play_pcm
main_module.audio.play_pcm = fake_turn_play
try:
    assert main_module.run_turn(b"pcm" * 1100, turn_speech, turn_brain, turn_latency)
finally:
    main_module.audio.play_pcm = old_turn_play

assert turn_speech.transcribe_calls == [b"pcm" * 1100]
assert turn_brain.calls == ["resposta transcrita"]
assert turn_speech.synthesize_calls == ["Resposta factual."]
assert turn_playback == [b"pcm da resposta"]


fallback_turn_provider = SuccessfulProvider()
fallback_turn_speech = FakeTurnSpeech()
fallback_turn_playback = []
old_fallback_turn_play = main_module.audio.play_pcm
old_anthropic = llm_module.anthropic.Anthropic
llm_module.anthropic.Anthropic = UnicodeFailingAnthropic
main_module.audio.play_pcm = lambda chunks, interrupt, on_first_chunk=None: fallback_turn_playback.extend(chunks)
try:
    fallback_turn_brain = ClaudeBrain("unused", "claude-sonnet-5", ToolRegistry(), "catálogo",
                                      fallback_provider=fallback_turn_provider)
    assert main_module.run_turn(
        b"pcm" * 1100,
        fallback_turn_speech,
        fallback_turn_brain,
        turn_latency,
    )
finally:
    main_module.audio.play_pcm = old_fallback_turn_play
    llm_module.anthropic.Anthropic = old_anthropic

assert len(fallback_turn_provider.calls) == 1
assert fallback_turn_speech.synthesize_calls == ["resposta do fallback"]
assert fallback_turn_playback == [b"pcm da resposta"]


short_speech = FakeTurnSpeech()
assert not main_module.run_turn(b"x" * 3199, short_speech, FakeTurnBrain(), turn_latency)
assert short_speech.transcribe_calls == []

recoveries = []
old_say = main_module.say
main_module.say = lambda speech, text: recoveries.append(text)
try:
    empty_speech = FakeTurnSpeech(text="")
    failed_speech = FakeTurnSpeech(error=RuntimeError("STT fora do ar"))
    silence_brain = FakeTurnBrain()
    assert not main_module.run_turn(b"x" * 3200, empty_speech, FakeTurnBrain(), turn_latency)
    assert not main_module.run_turn(b"x" * 3200, failed_speech, FakeTurnBrain(), turn_latency)
    assert not main_module.run_turn(
        b"x" * 3200, FakeTurnSpeech(text="[silêncio]"), silence_brain, turn_latency
    )
finally:
    main_module.say = old_say
assert recoveries == [
    "Não entendi. Pode repetir?",
    "Não consegui te ouvir agora. Tenta de novo?",
    "Não entendi. Pode repetir?",
]
assert silence_brain.calls == []


class StopPttLoop(Exception):
    pass


class FakePttEvent:
    def __init__(self):
        self.wait_calls = 0
        self.clear_calls = 0
        self._set = False

    def clear(self):
        self.clear_calls += 1
        self._set = False

    def is_set(self):
        return self._set

    def set(self):
        self._set = True

    def wait(self):
        self.wait_calls += 1
        if self.wait_calls == 1:
            self.set()
            return True
        raise StopPttLoop


ptt_event = FakePttEvent()
ptt_speech = FakeTurnSpeech()
ptt_brain = FakeTurnBrain()
record_stops = []
old_record_until = main_module.audio.record_until
old_turn_play = main_module.audio.play_pcm


def fake_record_until(stop):
    record_stops.append(stop)
    stop.set()  # segunda ativação encerra a captura
    return b"x" * 3200


main_module.audio.record_until = fake_record_until
main_module.audio.play_pcm = fake_turn_play
try:
    try:
        main_module.run_ptt_loop(
            ptt_speech,
            ptt_brain,
            ptt_event,
            main_module.LatencyMetrics(clock=lambda: 2.0),
        )
    except StopPttLoop:
        pass
finally:
    main_module.audio.record_until = old_record_until
    main_module.audio.play_pcm = old_turn_play

assert ptt_event.wait_calls == 2
assert len(record_stops) == 1 and record_stops[0] is ptt_event
assert not ptt_event.is_set()
assert ptt_speech.transcribe_calls == [b"x" * 3200]
assert ptt_brain.calls == ["resposta transcrita"]

# --- barge-in: VAD, cancelamento coordenado e retomada ---
class BlockingBargeBrain:
    def __init__(self):
        self.cancel_event = None
        self.cancelled = threading.Event()

    def respond(self, text, on_sentence, cancel_event=None):
        self.cancel_event = cancel_event
        on_sentence("Frase antiga.")
        while not cancel_event.is_set():
            cancel_event.wait(0.01)
        self.cancelled.set()
        on_sentence("Frase que não deve tocar.")


class BargeSpeech:
    def __init__(self):
        self.synthesize_calls = []

    def transcribe(self, pcm):
        return "interrompa"

    def synthesize_stream(self, text):
        self.synthesize_calls.append(text)
        return [b"audio antigo"]


barge_play_started = threading.Event()
barge_play_stopped = threading.Event()
barge_writes = []


def blocking_barge_play(chunks, interrupt, on_first_chunk=None):
    barge_play_started.set()
    for chunk in chunks:
        if interrupt.is_set():
            break
        barge_writes.append(chunk)
        if on_first_chunk:
            on_first_chunk()
        while not interrupt.wait(0.001):
            pass
    barge_play_stopped.set()


barge_speech = BargeSpeech()
barge_brain = BlockingBargeBrain()
barge_interrupt = threading.Event()
barge_detected = threading.Event()
barge_vad_ready = threading.Event()
barge_vad = []
barge_latency = main_module.LatencyMetrics(clock=lambda: 2.0)
barge_latency.capture_finished(at=1.0)
old_barge_play = main_module.audio.play_pcm
main_module.audio.play_pcm = blocking_barge_play


def save_barge_vad(vad):
    barge_vad.append(vad)
    barge_vad_ready.set()


barge_thread = threading.Thread(
    target=main_module.run_turn,
    args=(b"x" * 3200, barge_speech, barge_brain, barge_latency),
    kwargs={
        "interrupt": barge_interrupt,
        "barge_in_event": barge_detected,
        "on_vad_ready": save_barge_vad,
    },
)
try:
    barge_thread.start()
    assert barge_vad_ready.wait(1), "VAD não foi disponibilizado"
    assert barge_play_started.wait(1), "playback não começou"
    detected_at = time.perf_counter()
    barge_vad[0].signal(True, now=0.0)
    barge_vad[0].signal(True, now=0.2)
    assert barge_play_stopped.wait(1), "playback não parou"
    stopped_at = time.perf_counter()
    barge_thread.join(1)
finally:
    main_module.audio.play_pcm = old_barge_play

assert not barge_thread.is_alive()
assert stopped_at - detected_at < 0.15
assert barge_detected.is_set() and barge_interrupt.is_set()
assert barge_brain.cancel_event is barge_interrupt
assert barge_brain.cancelled.is_set()
assert barge_speech.synthesize_calls == ["Frase antiga."]
assert barge_writes == [b"audio antigo"]


class ResumePttEvent:
    def __init__(self):
        self.wait_calls = 0
        self._set = False

    def clear(self):
        self._set = False

    def set(self):
        self._set = True

    def is_set(self):
        return self._set

    def wait(self):
        self.wait_calls += 1
        if self.wait_calls == 1:
            self.set()
            return True
        raise StopPttLoop


resume_event = ResumePttEvent()
resume_turns = []
resume_records = []
old_run_turn = main_module.run_turn
old_record_until = main_module.audio.record_until


def fake_resume_record(stop):
    stop.set()
    pcm = b"primeiro" if not resume_records else b"segundo!"
    resume_records.append(pcm)
    return pcm + b"x" * (3200 - len(pcm))


def fake_resume_turn(pcm, speech, brain, latency, interrupt=None, barge_in_event=None,
                     on_vad_ready=None, vad_monitor=None):
    resume_turns.append(pcm[:8])
    if len(resume_turns) == 1:
        barge_in_event.set()
    return True


main_module.audio.record_until = fake_resume_record
main_module.run_turn = fake_resume_turn
try:
    try:
        main_module.run_ptt_loop(
            object(), object(), resume_event, main_module.LatencyMetrics(clock=lambda: 2.0)
        )
    except StopPttLoop:
        pass
finally:
    main_module.audio.record_until = old_record_until
    main_module.run_turn = old_run_turn

assert resume_event.wait_calls == 2, resume_event.wait_calls
assert resume_turns == [b"primeiro", b"segundo!"]
assert len(resume_records) == 2

# --- degradação de raciocínio e tools ---
class FakeConfiguredGemini:
    instances = []

    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
        self.instances.append(self)


class FakeConfiguredBrain:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.instances.append(self)


class FakeCommands:
    def catalog(self):
        return "catálogo"


old_main_gemini = main_module.GeminiProvider
old_main_brain = main_module.ClaudeBrain
main_module.GeminiProvider = FakeConfiguredGemini
main_module.ClaudeBrain = FakeConfiguredBrain
try:
    configured_brain = main_module.build_brain(ToolRegistry(), FakeCommands())
finally:
    main_module.GeminiProvider = old_main_gemini
    main_module.ClaudeBrain = old_main_brain

assert configured_brain.kwargs["fallback_provider"] is FakeConfiguredGemini.instances[-1]
assert configured_brain.kwargs["fallback_provider"].api_key == config.GEMINI_API_KEY
assert configured_brain.kwargs["fallback_provider"].model == config.GEMINI_MODEL

# sem chave Anthropic: Gemini vira provider principal, sem fallback
old_anthropic_key = config.ANTHROPIC_API_KEY
config.ANTHROPIC_API_KEY = ""
main_module.GeminiProvider = FakeConfiguredGemini
main_module.ClaudeBrain = FakeConfiguredBrain
try:
    gemini_only_brain = main_module.build_brain(ToolRegistry(), FakeCommands())
finally:
    config.ANTHROPIC_API_KEY = old_anthropic_key
    main_module.GeminiProvider = old_main_gemini
    main_module.ClaudeBrain = old_main_brain

assert gemini_only_brain.kwargs["provider"] is FakeConfiguredGemini.instances[-1]
assert "fallback_provider" not in gemini_only_brain.kwargs
assert gemini_only_brain.args[1] == config.GEMINI_MODEL


class FailingBrain:
    def __init__(self):
        self.cancel_event = None

    def respond(self, text, on_sentence, cancel_event=None):
        self.cancel_event = cancel_event
        raise RuntimeError("Claude e Gemini indisponíveis")


recovery_audio = []
recovery_speech = FakeTurnSpeech()
recovery_brain = FailingBrain()
recovery_latency = main_module.LatencyMetrics(clock=lambda: 2.0)
recovery_latency.capture_finished(at=1.0)
old_recovery_play = main_module.audio.play_pcm
main_module.audio.play_pcm = lambda chunks, interrupt, on_first_chunk=None: recovery_audio.extend(chunks)
try:
    assert main_module.run_turn(
        b"x" * 3200,
        recovery_speech,
        recovery_brain,
        recovery_latency,
    )
finally:
    main_module.audio.play_pcm = old_recovery_play

assert recovery_brain.cancel_event is not None
assert recovery_speech.synthesize_calls == [
    "Tive um problema para processar agora. Tenta de novo em instantes."
]
assert recovery_audio == [b"pcm da resposta"]

# --- LocalCommandTool ---
plat = "win32" if sys.platform == "win32" else ("darwin" if sys.platform == "darwin" else "linux")
py = "python" if plat == "win32" else "python3"
yaml_text = f"""
apps:
  editor:
    open: {{ {plat}: ["cmd", "/c", "echo", "abriu"] }}
    close: {{ {plat}: ["cmd", "/c", "echo", "fechou"] }}
  navegador:
    open: {{ {plat}: ["fake", "navegador"] }}
scripts:
  ola:
    {plat}: ["{py}", "-c", "print('oi saphira')"]
  remoto:
    plan9: ["fake", "remoto"]
"""
with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
    f.write(yaml_text)
    path = f.name

tool = LocalCommandTool(path)
assert all(name in tool.catalog() for name in ("editor", "navegador", "ola", "remoto"))
assert tool.run("run_script", "ola").strip() == "oi saphira"

# open/close usam somente o argv do YAML e shell=False
popen_calls = []


def fake_popen(argv, **kwargs):
    popen_calls.append((argv, kwargs))
    return object()


old_popen = tools_module.subprocess.Popen
tools_module.subprocess.Popen = fake_popen
try:
    assert tool.run("open_app", "editor") == "ok: open_app editor"
    assert tool.run("close_app", "editor") == "ok: close_app editor"
    assert all(kwargs.get("shell") is False for _, kwargs in popen_calls), popen_calls
    assert popen_calls[0][0] == ["cmd", "/c", "echo", "abriu"]
    assert popen_calls[1][0] == ["cmd", "/c", "echo", "fechou"]

    invalid_action = tool.run("delete_app", "editor")
    invalid_target = tool.run("open_app", "inexistente")
    invalid_so = tool.run("run_script", "remoto")
    old_platform = tools_module.sys.platform
    tools_module.sys.platform = "plan9"
    try:
        invalid_runtime_so = tool.run("open_app", "editor")
    finally:
        tools_module.sys.platform = old_platform
    for result in (invalid_action, invalid_target, invalid_so, invalid_runtime_so):
        assert "não cadastrado" in result
        assert all(name in result for name in ("editor", "navegador", "ola", "remoto")), result

    if plat == "win32":
        before = len(popen_calls)
        unsupported = tool.run("close_app", "navegador")
        assert "não suportado" in unsupported
        assert all(name in unsupported for name in ("editor", "navegador", "ola", "remoto"))
        assert len(popen_calls) == before
finally:
    tools_module.subprocess.Popen = old_popen

# run_script mantém stdout/stderr, limita cada stream aos últimos 1.500 chars e usa timeout
class Completed:
    stdout = "o" * 1600
    stderr = "e" * 1601
    returncode = 7


run_calls = []


def fake_run(argv, **kwargs):
    run_calls.append((argv, kwargs))
    return Completed()


old_run = tools_module.subprocess.run
tools_module.subprocess.run = fake_run
try:
    output = tool.run("run_script", "ola")
finally:
    tools_module.subprocess.run = old_run

assert run_calls == [(
    [py, "-c", "print('oi saphira')"],
    {"capture_output": True, "text": True, "timeout": 120, "shell": False},
)]
assert "stdout:\n[...]truncado[...]" in output
assert "stderr:\n[...]truncado[...]" in output
assert "o" * 1500 in output and "e" * 1500 in output
assert "o" * 1600 not in output and "e" * 1601 not in output

# o catálogo versionado contém os três alvos do SPEC
catalog = LocalCommandTool(str(ROOT / "commands.yaml")).catalog()
assert all(name in catalog for name in ("spotify", "navegador", "ola")), catalog

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
