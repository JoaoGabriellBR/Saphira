# SPEC — Saphira v1 (PRD)

- **Data:** 16/ago/2026
- **Fontes de verdade:** `C:\Users\Jão\Documents\Vault\Projeto Saphira\Saphira — Guia Técnico v1.md` + respostas do interview-me (mesma data)
- **Status:** aprovado

## 1. Objetivos e critério de "pronto"

### 1.1 Objetivos

1. Daemon local em **Python 3.12** (runtime validado no Windows), rodando no **Windows** (SO alvo do v1), que converse em **pt-BR** por voz.
2. Latência percebida: **< 2 s** sem tool call e **< 4 s** com busca web — início = fim da captura do turno do usuário (silêncio detectado ou tecla solta); fim = **time-to-first-audio** (TTFA), no primeiro chunk de resposta efetivamente tocando. O orçamento é avaliado em p50 e p95 sobre uma janela deslizante das últimas 20 interações, lidos do log.
3. Ativação por **push-to-talk em modo toggle** (hotkey global, padrão `F9`, configurável via `SAPHIRA_HOTKEY`): 1ª pressionada grava, 2ª envia. Durante a fala da Saphira, o VAD detecta fala nova e inicia **barge-in**: cancela a chamada LLM/tool em andamento e o streaming de TTS, interrompe o playback e permite iniciar a nova gravação imediatamente após o playback parar, com debounce curto de 150–300 ms para evitar falso disparo por eco.
4. Raciocínio via **Anthropic Claude, `model_id="claude-sonnet-5"` como default** (trocável via `CLAUDE_MODEL`), com tool use atrás de um registry plugável. Se o Claude estiver indisponível, o fallback é o Gemini via `google-genai`, usando `GEMINI_API_KEY` e `model_id="gemini-3.6-flash"`, com o mesmo contrato de streaming, tool use e memória.
5. Capacidades implementadas como tools atrás de interface única: `web_search` (Tavily) e `local_command` (allowlist em YAML).
6. Resposta falada via **Kokoro TTS** (voz padrão **`pf_dora`**, configurável via `KOKORO_VOICE` e velocidade via `KOKORO_SPEED`), sintetizada **frase a frase** em pipeline com o streaming do LLM.
7. Nenhuma falha de rede/API (Tavily, ElevenLabs, Anthropic ou Gemini) derruba o loop. STT ou TTS fora do ar usa um áudio local pré-gravado com a mensagem “não consegui te ouvir, tenta de novo”. Falha de Tavily ou `local_command` retorna um `tool_result` de erro ao LLM ativo, que informa o usuário por voz. Falha do Claude aciona o fallback Gemini; se Claude e Gemini estiverem indisponíveis, uma mensagem fixa é emitida pelo TTS e, se o TTS também estiver indisponível, usa-se o áudio local.

### 1.2 Critério de "pronto" (Definition of Done)

- [ ] `python tests/test_core.py` passa.
- [ ] Aceite E2E manual com os 5 cenários da seção 5.3, todos com o comportamento esperado.
- [ ] Latências dos cenários dentro do orçamento (item 1.1.2), avaliadas em p50 e p95 da janela deslizante das últimas 20 interações e lidas do log.
- [ ] Nenhuma chave de API no código (apenas `.env`); daemon falha rápido e com mensagem clara se faltar variável obrigatória.

## 2. Comandos e capacidades (contratos)

### 2.1 Interface comum de tool

```python
class Tool(Protocol):
    name: str
    description: str
    input_schema: dict          # JSON Schema, formato Anthropic tool use
    def run(self, **kwargs) -> str: ...
```

- `ToolRegistry.execute(name, inputs) -> str` **nunca propaga exceção**: falha vira `"Erro ao executar {name}: {detalhe}"` (devolvida como `tool_result` — o LLM ativo, Claude ou Gemini, verbaliza e a conversa segue). Tool inexistente → `"Erro: tool '{name}' não registrada. Disponíveis: [...]"`.
- Toda execução loga `tool`, `args` e duração em ms.

### 2.2 `web_search` (Tavily)

| | |
|---|---|
| Entrada | `query: string` (obrigatório, linguagem natural) |
| Parâmetros fixos | `search_depth="basic"`, `max_results=4`, `include_answer=True` |
| Saída | string ≤ 3.000 chars: `answer` + até 4 resultados no formato `- {título}: {conteúdo[:300]}`; vazio → `"Nenhum resultado encontrado."` |
| Falha (rede/rate limit/timeout) | exceção do SDK → capturada pelo registry → string de erro ao LLM |
| Latência esperada | 400–1.500 ms |

### 2.3 `local_command` (allowlist em `commands.yaml`)

| | |
|---|---|
| Entrada | `action` ∈ `open_app` \| `close_app` \| `run_script`; `target`: nome cadastrado |
| Resolução | `apps[target][open\|close][SO]` ou `scripts[target][SO]`, SO ∈ `win32` \| `darwin` \| `linux` |
| `open_app`/`close_app` | `subprocess.Popen(argv)` fire-and-forget → `"ok: {action} {target}"` |
| `run_script` | `subprocess.run(argv, capture_output=True, timeout=120)` → stdout e stderr mantidos separados; cada stream é truncado nos últimos 1.500 chars e recebe o prefixo `[...]truncado[...]` quando houver truncamento. Sem saída, retorna `"script '{target}' finalizado (exit {code})"` |
| Alvo/ação/SO inexistente | string de erro **com o catálogo completo** (permite autocorreção do LLM); `close_app navegador` em `win32` é não suportado, retorna erro e não executa processo |
| Segurança | `shell=False` sempre; só executa o que está no YAML; o LLM nunca monta argv |

**Conteúdo da allowlist v1:** apps `spotify` (open/close) e `navegador` (open; `close` não suportado em `win32`); script `ola` (demo — win32: `python scripts/ola.py`; darwin/linux: `python3 scripts/ola.py`; assume CWD = raiz do repo). O catálogo é injetado no system prompt.

### 2.4 Contratos internos do pipeline de voz

- **STT:** PCM s16le 16 kHz mono → `speech_to_text.convert(model_id="scribe_v2", language_code="por")` → string. Gravação < 100 ms descartada (toque acidental); máximo 90 s por turno.
- **LLM:** system prompt fixo (pt-BR, respostas curtas para serem faladas, catálogo de comandos); Claude é o provedor primário e Gemini é o fallback com o mesmo contrato; streaming com `max_tokens=600`; chunker de frases por regex de pontuação (limitação conhecida e aceita: abreviações tipo "Dr." podem cortar cedo); preâmbulo antes de tool call é falado. Memória = janela deslizante das últimas 20 mensagens, **só em RAM**: cada entrada do array enviado à API conta como uma mensagem (`user`, `assistant` e cada bloco `tool_result`); o corte só pode ocorrer no início de um turno de usuário e nunca entre um `tool_use` e seu `tool_result` correspondente. Se as 20 mais recentes cortarem um par, estende a janela para trás até um início de turno seguro.
- **TTS:** por frase, `Kokoro TTS (lang_code="p", voice="pf_dora")` → tensor convertido para PCM s16le mono 24 kHz → chunks de 4096 bytes → playback imediato em thread dedicada; ao detectar fala nova via VAD, com debounce de 150–300 ms, o barge-in cancela LLM/tool, streaming de TTS, playback e drena a fila; a nova gravação começa imediatamente após o playback parar.

## 3. Estrutura de pastas e módulos

```text
Saphira/
├── .env.example / .gitignore / requirements.txt / commands.yaml
├── saphira/
│   ├── __init__.py
│   ├── config.py    # env + defaults; falha rápido se faltar variável
│   ├── audio.py     # mic (PCM 16 kHz), playback (PCM 24 kHz), Hotkey global, VAD
│   ├── speech.py    # STT (ElevenLabs Scribe) + TTS streaming (Kokoro)
│   ├── tools.py     # Tool (Protocol), ToolRegistry, WebSearchTool, LocalCommandTool
│   ├── llm.py       # Claude/Gemini: streaming + tool loop + chunker de frases + memória
│   └── main.py      # orquestrador: loop PTT, barge-in, speaker thread, log de latência
├── scripts/ola.py   # script de demo p/ aceite "rodar script"
└── tests/test_core.py
```


Entry point: `python -m saphira.main` a partir da raiz do repo.

## 4. Padrão de código

- Módulos pequenos por responsabilidade; funções onde classes não agregam (`record_until`, `play_pcm`, `timed`).
- SDKs externos isolados nas bordas (`speech.py`, `tools.py`, `llm.py`); o orquestrador (`main.py`) só depende de interfaces.
- Tools retornam `str`, **nunca** lançam exceção para fora do registry.
- Sem frameworks; stdlib primeiro (`wave`, `queue`, `threading`, `subprocess`).
- Simplificações deliberadas com teto conhecido levam comentário `# ponytail:` (ex.: chunker por regex).
- Config somente via `.env`; nada de chave hardcoded, incluindo `GEMINI_API_KEY`; `.env` no `.gitignore`.
- **Logging:** formato `%(asctime)s %(levelname)s %(name)s: %(message)s`; registra texto ouvido, tool chamada + args + duração, cada frase falada, início da captura, fim da captura, primeiro chunk tocando e latência por etapa (`gravacao`, `stt`, `tool=`). Calcula e registra TTFA, p50 e p95 da janela deslizante das últimas 20 interações. Decisão do interview: o log **pode** conter transcrições — tratar como dado pessoal.

## 5. Estratégia de testes

### 5.1 Automatizados (obrigatórios antes do /build)

`tests/test_core.py`, assert-based, **sem framework**, roda sem hardware de áudio e sem chaves:

- `_pop_sentences`: frase completa + resto; múltiplas frases com `\n`; buffer sem pontuação.
- `LocalCommandTool` (YAML temporário): catálogo lista os nomes; `run_script` executa e retorna stdout/stderr separados, com truncamento dos últimos 1.500 chars e prefixo `[...]truncado[...]`; alvo inexistente e ação não definida → `"não cadastrado"`; `close_app navegador` em `win32` → erro de não suportado sem executar processo.
- `ToolRegistry`: exceção dentro de tool → string `"Erro ao executar ..."`; tool fantasma → `"não registrada"`.
- `ClaudeBrain` com clientes externos falsos: fallback Gemini mantém streaming, tool use e memória; corte da janela preserva pares `tool_use`/`tool_result`.
- Pipeline de barge-in com dependências falsas: VAD cancela LLM/tool/TTS, interrompe playback, drena a fila e permite nova captura após debounce.
- Agregação de latência: uma janela de 20 interações produz TTFA, p50 e p95 a partir dos eventos de captura e primeiro áudio.

Comando: `python tests/test_core.py`.

### 5.2 Smoke manual (com chaves reais, antes do aceite)

STT isolado (gravar ~3 s → transcrição correta); TTS isolado (frase → áudio audível); `web_search` isolado; fallback de áudio local para falha de STT/TTS.

### 5.3 Aceite E2E manual (5 cenários)

1. **Busca web básica via Tavily (`web_search`)** — utterance em pt-BR sobre informação atual, por exemplo “Qual é a cotação do dólar hoje?”. Oráculo: o log registra `tool=web_search`, a resposta cita o resultado e o cenário respeita o orçamento.
2. **Execução de comando local (abrir app)** — utterance “Abra o Spotify”. Oráculo: o processo/URI do app é iniciado por uma entrada permitida no catálogo, sem o LLM montar `argv`.
3. **Barge-in no meio de uma resposta** — iniciar uma resposta longa e produzir fala nova durante o playback. Oráculo: o playback para em até 150 ms após a detecção pelo VAD, a nova captura começa após o debounce e não há `tool_result` órfão na memória.
4. **Falha de dependência externa (Tavily simulada)** — executar uma pergunta que exigiria `web_search` com a API indisponível. Oráculo: o erro chega como `tool_result`, uma mensagem tratada é falada pelo LLM ativo, sem crash e com retorno ao loop.
5. **Corte de memória após mais de 20 mensagens** — executar turnos suficientes para exceder a janela. Oráculo: mensagens mais antigas são removidas, o corte ocorre em início de turno seguro e nenhum par `tool_use`/`tool_result` é quebrado. Latências dos cenários aplicáveis são lidas do log e comparadas ao orçamento da seção 1.1.2.

### 5.4 Sem CI no v1

Quando houver, roda apenas os checks puros (runner não tem hardware de áudio).

## 6. Fora de escopo — v1 NÃO faz

- Memória persistente entre sessões (histórico só em RAM, zera ao reiniciar).
- Visão/câmera (v2 — arquitetura pronta via `ToolRegistry`, sem código agora).
- Wake-word e AEC. O barge-in por voz com VAD limitado à detecção de fala nova durante o playback está no escopo do v1, conforme as seções 1.1.3 e 2.4.
- Smart-home, agenda, e-mail e demais integrações.
- STT realtime (Scribe WebSocket) e TTS input-streaming por WebSocket.
- UI gráfica, app mobile, instalador/empacotamento, Raspberry Pi (apenas não bloquear).
- Retry/backoff de APIs, resumo de memória, telemetria além de log, CI.
