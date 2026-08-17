"""Checks mínimos das partes puras da Saphira (sem frameworks).
Rodar: python tests/test_core.py
"""
import io
import logging
import os
import subprocess
import sys
import tempfile
import threading
import wave
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

# dependências declaradas e .env.example sem valores preenchidos
assert "google-genai" in (ROOT / "requirements.txt").read_text(encoding="utf-8")
for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        assert line.split("=", 1)[1].split("#", 1)[0].strip() == "", f"valor no .env.example: {line}"

from saphira import audio as audio_module
from saphira import llm as llm_module
from saphira import speech as speech_module
from saphira import tools as tools_module
from saphira.llm import ClaudeBrain, GeminiProvider, ProviderEvent, _pop_sentences
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
    def __init__(self, events):
        self.events = events
        self.request = None

    def stream(self, **request):
        self.request = request
        yield from self.events


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
    "text_delta", "sentence", "text_delta", "sentence", "tool_use", "end"
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
    stop_reason = "tool_use"
    content = [FakeClaudeBlock()]


class FakeClaudeStream:
    text_stream = ["Vou verificar. ", "resto sem pontuação"]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get_final_message(self):
        return FakeClaudeFinal()


class FakeClaudeMessages:
    def __init__(self):
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return FakeClaudeStream()


class FakeAnthropic:
    instances = []

    def __init__(self, api_key):
        self.api_key = api_key
        self.messages = FakeClaudeMessages()
        self.instances.append(self)


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
            claude_trace.append(("tool_execution", claude_tools.execute(event.name, event.inputs)))

    claude.respond("verifique", lambda sentence: claude_trace.append(("sentence", sentence)),
                   on_event=observe_claude)
    claude_request = FakeAnthropic.instances[-1].messages.calls[0]
finally:
    llm_module.anthropic.Anthropic = old_anthropic

assert [kind for kind, _ in claude_trace] == [
    "text_delta", "sentence", "text_delta", "sentence",
    "tool_use", "tool_execution", "end",
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
    audio_module.play_pcm([b"primeiro", b"descartado"], interrupt)
finally:
    audio_module.sd.RawOutputStream = old_output_stream

assert writes == [b"primeiro"]
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

    def __init__(self, api_key):
        self.api_key = api_key
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
