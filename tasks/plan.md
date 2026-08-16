# Implementation Plan: Saphira v1

## Overview

Implementar o daemon local de voz em pt-BR definido em `SPEC.md`: captura por PTT, STT/TTS via ElevenLabs, raciocínio com Claude e fallback Gemini, tools `web_search` e `local_command`, streaming frase a frase, memória em RAM, barge-in por VAD, degradação controlada e métricas de TTFA.

Fonte única de escopo: `SPEC.md` atualizado e aprovado. Não entram memória persistente, câmera, wake-word, AEC, integrações adicionais, WebSocket de voz, UI, empacotamento, retries/backoff, resumo de memória, telemetria além de log ou CI.

## Architecture Decisions

- Manter a estrutura de módulos do SPEC; adapters de Claude/Gemini ficam em `saphira/llm.py` e o orquestrador depende do contrato interno, não dos SDKs.
- Usar fakes nos testes assert-based para SDKs, subprocessos e hardware; `python tests/test_core.py` permanece sem chaves e sem dispositivo de áudio.
- Normalizar Claude e Gemini para o mesmo fluxo observável: texto em streaming, eventos de `tool_use`, `tool_result`, encerramento e cancelamento; o `ClaudeBrain` mantém a mesma API para o orquestrador.
- Tratar `GEMINI_API_KEY` como variável obrigatória de configuração, pois o fallback Gemini é parte do caminho de produção do v1 e o SPEC exige falha rápida para variável obrigatória ausente.
- O VAD terá uma borda testável por sinais injetados; algoritmo/limiar de detecção permanece detalhe de implementação, desde que produza o comportamento observável do SPEC.

## Task List

### Phase 1: Foundation

### Task 1: Configuração e dependências do runtime

**Description:** Atualizar dependências, `.env.example` e `config.py` para Claude Sonnet 5, Gemini, chaves obrigatórias, defaults de voz, hotkey, memória e catálogo de comandos.

**Acceptance criteria:**
- [ ] Com ambiente completo, a configuração expõe `CLAUDE_MODEL="claude-sonnet-5"`, `GEMINI_API_KEY`, `STT_MODEL="scribe_v2"`, `TTS_MODEL="eleven_flash_v2_5"`, `LANGUAGE="por"`, hotkey padrão `<f9>` e janela 20.
- [ ] Com qualquer variável obrigatória ausente, a inicialização falha com `RuntimeError` que nomeia a variável e referencia `.env.example`.
- [ ] `requirements.txt` inclui `google-genai`; nenhum segredo aparece no código ou no `.env.example`.

**Verification:**
- [ ] Checks assert-based de configuração passam em `python tests/test_core.py` sem chaves reais.
- [ ] Revisão manual confirma que `.env` permanece ignorado e que o daemon usa somente valores de ambiente.

**Dependencies:** None.

**Vertical slice:** ambiente `.env` -> configuração validada, sem construir clientes externos nem acessar hardware.

**Files likely touched:** `saphira/config.py`, `.env.example`, `requirements.txt`, `tests/test_core.py`.

**Estimated scope:** Medium.

### Task 2: Registry comum de tools

**Description:** Consolidar o `Tool Protocol`, registro, schemas para o LLM, execução protegida e logging de tool, argumentos e duração.

**Acceptance criteria:**
- [ ] Uma tool fake registrada expõe `name`, `description` e `input_schema`, executa e retorna `str`.
- [ ] Exceção da tool vira `Erro ao executar {name}: {detalhe}` sem escapar; tool inexistente retorna `não registrada` e o catálogo completo.
- [ ] Toda execução bem-sucedida registra `tool`, `args` e duração em milissegundos.

**Verification:** `python tests/test_core.py` com tool fake de sucesso, tool quebrada e tool fantasma.

**Dependencies:** Task 1.

**Vertical slice:** chamada do registry fake -> string de resultado ou erro normalizado e evento de log.

**Files likely touched:** `saphira/tools.py`, `tests/test_core.py`.

**Estimated scope:** Small.

## Checkpoint: Foundation

- [ ] Tasks 1 e 2 concluídas.
- [ ] `python tests/test_core.py` passa.
- [ ] Não há chave hardcoded nem dependência ausente na lista do runtime.

### Phase 2: Tools e borda de áudio

### Task 3: Allowlist de comandos locais

**Description:** Implementar `LocalCommandTool` sobre YAML seguro, incluindo catálogo, resolução por SO, execução sem shell, saída separada de scripts e a regra de navegador no Windows.

**Acceptance criteria:**
- [ ] `open_app` e `close_app` só executam `argv` vindo do YAML com `shell=False`; `close_app navegador` em `win32` retorna erro de não suportado sem chamar subprocesso.
- [ ] `run_script` usa timeout de 120 s, preserva stdout e stderr separados, trunca os últimos 1.500 caracteres de cada stream e prefixa streams truncados com `[...]truncado[...]`.
- [ ] Alvo, ação ou SO inválidos retornam erro contendo o catálogo completo; o catálogo contém `spotify`, `navegador` e `ola` conforme o SPEC.

**Verification:** teste com YAML temporário e subprocesso fake em `python tests/test_core.py`; execução isolada do demo `python scripts/ola.py`.

**Dependencies:** Task 2.

**Vertical slice:** YAML temporário -> resolução de ação -> subprocesso fake/real do demo -> string contratada.

**Files likely touched:** `saphira/tools.py`, `commands.yaml`, `tests/test_core.py`, `scripts/ola.py` somente se o teste revelar divergência.

**Estimated scope:** Medium.

### Task 4: Busca web via Tavily

**Description:** Implementar e testar `WebSearchTool` com os parâmetros fixos, normalização do resultado e integração com o erro protegido do registry.

**Acceptance criteria:**
- [ ] O client fake recebe `search_depth="basic"`, `max_results=4` e `include_answer=True` com a query recebida.
- [ ] A saída contém `answer` e no máximo quatro resultados no formato especificado, nunca excede 3.000 caracteres e retorna `Nenhum resultado encontrado.` quando vazia.
- [ ] Exceção de rede/rate limit/timeout chega ao LLM como string de erro via `ToolRegistry`, sem derrubar o processo.

**Verification:** `python tests/test_core.py` com client Tavily fake e cenário de exceção.

**Dependencies:** Task 2.

**Vertical slice:** query fake -> client Tavily fake -> texto normalizado -> `tool_result` observável.

**Files likely touched:** `saphira/tools.py`, `tests/test_core.py`.

**Estimated scope:** Small.

### Task 5: Captura, playback, hotkey e VAD

**Description:** Completar as primitivas de áudio e a borda de detecção de fala: PCM de entrada/saída, limite de gravação, hotkey global, interrupção de playback e evento de fala nova com debounce configurável entre 150 e 300 ms.

**Acceptance criteria:**
- [ ] A captura configura PCM s16le, 16 kHz, mono, encerra por evento de parada ou em no máximo 90 s, e o playback configura PCM 24 kHz e para ao receber interrupção.
- [ ] A hotkey inicia callback por pressão; a borda VAD transforma um sinal de fala nova em um único evento após o debounce configurado, sem exigir API externa.
- [ ] Fakes de stream e VAD permitem verificar os comportamentos sem hardware.

**Verification:** checks assert-based de streams, hotkey, interrupção e debounce em `python tests/test_core.py`.

**Dependencies:** Task 1.

**Vertical slice:** evento de entrada fake -> bytes PCM/sink fake ou evento VAD -> interrupção observável.

**Files likely touched:** `saphira/audio.py`, `tests/test_core.py`.

**Estimated scope:** Medium.

## Checkpoint: Tools e áudio básico

- [ ] Tasks 3, 4 e 5 concluídas.
- [ ] Tools passam nos contratos sem rede real.
- [ ] Áudio e VAD passam com fakes e não exigem hardware no teste automatizado.

### Phase 3: Speech e raciocínio

### Task 6: Adaptadores ElevenLabs e áudio local de fallback

**Description:** Isolar STT/TTS da ElevenLabs, validar os formatos PCM/WAV e disponibilizar o áudio local pré-gravado usado quando STT ou TTS estiver indisponível.

**Acceptance criteria:**
- [ ] STT envia WAV mono 16-bit derivado do PCM 16 kHz com `model_id="scribe_v2"` e `language_code="por"`.
- [ ] TTS usa `model_id="eleven_flash_v2_5"` e `output_format="pcm_24000"`, devolvendo chunks PCM 24 kHz.
- [ ] A mensagem local “não consegui te ouvir, tenta de novo” pode ser reproduzida sem instanciar nem chamar uma API externa.

**Verification:** client ElevenLabs fake e sink de áudio fake em `python tests/test_core.py`; smoke real fica para Task 18.

**Dependencies:** Task 5.

**Vertical slice:** PCM/texto -> chamada SDK fake -> texto/chunks PCM, com caminho alternativo -> áudio local.

**Files likely touched:** `saphira/speech.py`, `saphira/audio.py`, `tests/test_core.py`, novo arquivo local de áudio pré-gravado.

**Estimated scope:** Medium.

### Task 7: Contrato de provider LLM e chunker de frases

**Description:** Definir dentro de `llm.py` o contrato comum usado por Claude e Gemini e manter o chunker por regex, incluindo eventos normalizados de texto, tool use, finalização e cancelamento.

**Acceptance criteria:**
- [ ] `_pop_sentences` passa os quatro casos obrigatórios do SPEC: frase com resto, múltiplas frases com newline, buffer sem pontuação e flush final.
- [ ] Um provider fake consegue emitir deltas, chamadas de tool e encerramento pelo mesmo contrato consumido pelo `ClaudeBrain`.
- [ ] O contrato expõe cancelamento suficiente para descartar eventos posteriores ao barge-in sem emitir resultado parcial.

**Verification:** `python tests/test_core.py` com provider fake e asserts de ordem dos eventos.

**Dependencies:** Task 2.

**Vertical slice:** deltas fake -> frases completas -> eventos normalizados de provider.

**Files likely touched:** `saphira/llm.py`, `tests/test_core.py`.

**Estimated scope:** Medium.

### Task 8: Streaming e tool request do Claude

**Description:** Adaptar Anthropic ao contrato interno, usando o model configurado, system prompt, catálogo de comandos, `max_tokens=600`, streaming e preâmbulo antes de tool call.

**Acceptance criteria:**
- [ ] O client fake recebe `claude-sonnet-5` por default/configuração, system prompt em pt-BR, catálogo e `max_tokens=600`.
- [ ] Deltas são entregues ao chunker durante o streaming; o restante sem pontuação é emitido ao final.
- [ ] Texto produzido antes de `tool_use` é emitido antes da execução da tool.

**Verification:** `python tests/test_core.py` com client Anthropic fake, sem chave real.

**Dependencies:** Tasks 2 e 7.

**Vertical slice:** texto do usuário -> stream Claude fake -> callbacks de frase e evento de tool.

**Files likely touched:** `saphira/llm.py`, `tests/test_core.py`.

**Estimated scope:** Medium.

### Task 9: Fallback Gemini com contrato equivalente

**Description:** Integrar `google-genai` em `llm.py` como provider de fallback, usando `GEMINI_API_KEY` e `gemini-3.6-flash`, sem alterar a interface consumida pelo orquestrador.

**Acceptance criteria:**
- [ ] O provider fake do Gemini recebe `gemini-3.6-flash` e produz texto, tool use, `tool_result`, encerramento e cancelamento no mesmo formato normalizado do Claude.
- [ ] Falha do provider Claude transfere o turno ao Gemini sem duplicar a entrada do usuário nem perder o callback de frases.
- [ ] Falha de ambos os providers é devolvida ao orquestrador como erro controlável, sem exceção não tratada nem loop infinito.

**Verification:** `python tests/test_core.py` com clients Claude/Gemini fakes; nenhuma chave real.

**Dependencies:** Tasks 1, 7 e 8.

**Vertical slice:** usuário -> tentativa Claude fake -> fallback Gemini fake -> mesmas frases/tool events do contrato interno.

**Files likely touched:** `saphira/llm.py`, `saphira/config.py`, `requirements.txt`, `tests/test_core.py`.

**Estimated scope:** Medium.

### Task 10: Janela de memória em RAM

**Description:** Implementar corte da memória do `ClaudeBrain` conforme a contagem de mensagens da API e preservar turnos e pares `tool_use`/`tool_result`.

**Acceptance criteria:**
- [ ] Cada entrada `user`, `assistant` e cada bloco `tool_result` conta conforme o SPEC; uma sequência acima de 20 remove mensagens mais antigas.
- [ ] O corte só ocorre no início de turno de usuário e nunca deixa `tool_result` órfão; se necessário, a janela é estendida para trás até um início seguro.
- [ ] Criar uma nova instância zera o histórico; cancelar um turno não grava resultado parcial órfão.

**Verification:** sequência de histórico fake em `python tests/test_core.py`, incluindo pares completos e turno cancelado.

**Dependencies:** Tasks 7, 8 e 9.

**Vertical slice:** sequência de turnos fake -> histórico final verificável e seguro para a próxima chamada.

**Files likely touched:** `saphira/llm.py`, `tests/test_core.py`.

**Estimated scope:** Medium.

### Task 11: Loop de tools e reinjeção de `tool_result`

**Description:** Fechar o ciclo de raciocínio para Claude/Gemini: executar todas as chamadas emitidas pelo provider ativo, reinjetar resultados e continuar o streaming até resposta final ou cancelamento.

**Acceptance criteria:**
- [ ] Um `tool_use` chama `ToolRegistry.execute` com nome e inputs corretos e reinjeta o resultado como `tool_result` associado ao identificador da chamada.
- [ ] O LLM ativo recebe erro de Tavily/local tool como conteúdo de `tool_result` e pode produzir a frase tratada; o preâmbulo permanece audível antes da tool.
- [ ] Cancelamento interrompe o loop e não adiciona `tool_result` órfão nem inicia nova iteração.

**Verification:** `python tests/test_core.py` com provider e registry fakes, incluindo sucesso, erro e cancelamento.

**Dependencies:** Tasks 2, 8, 9 e 10.

**Vertical slice:** texto do usuário -> tool fake -> `tool_result` -> segundo stream fake -> frases finais.

**Files likely touched:** `saphira/llm.py`, `tests/test_core.py`.

**Estimated scope:** Medium.

## Checkpoint: Speech e providers

- [ ] Tasks 6 a 11 concluídas.
- [ ] `python tests/test_core.py` cobre chunker, parity Claude/Gemini, memória, tools e cancelamento.
- [ ] Nenhum teste usa chave, rede ou hardware real.

### Phase 4: Pipeline e orquestração

### Task 12: Worker de fala frase a frase

**Description:** Conectar fila de frases, síntese individual e playback em thread dedicada, com ordem, sentinela, primeiro áudio e drenagem segura.

**Acceptance criteria:**
- [ ] Cada frase recebida gera uma síntese separada e os chunks são reproduzidos na mesma ordem.
- [ ] O primeiro chunk efetivamente enviado ao sink gera o evento de início de áudio; `None` encerra a worker sem bloquear.
- [ ] Falha na síntese não encerra a worker; usa o fallback local/log definido e não reproduz frases descartadas após interrupção.

**Verification:** worker, Speech fake e sink de playback fake em `python tests/test_core.py`.

**Dependencies:** Tasks 5, 6 e 7.

**Vertical slice:** frase enfileirada -> síntese fake -> chunk PCM no sink ou fallback observável.

**Files likely touched:** `saphira/main.py`, `saphira/audio.py`, `saphira/speech.py`, `tests/test_core.py`.

**Estimated scope:** Medium.

### Task 13: Eventos e métricas de latência

**Description:** Instrumentar captura, STT, tools, primeiro áudio e TTFA; calcular p50/p95 sobre as últimas 20 interações no log especificado.

**Acceptance criteria:**
- [ ] Cada turno registra fim da captura, duração das etapas relevantes e primeiro chunk tocando, permitindo calcular TTFA desde o fim da captura.
- [ ] A janela deslizante mantém as últimas 20 interações e registra p50 e p95; os valores não dependem de média isolada.
- [ ] O formato de logging permanece `%(asctime)s %(levelname)s %(name)s: %(message)s` e inclui texto ouvido, tools e frases faladas.

**Verification:** eventos com timestamps determinísticos em `python tests/test_core.py`; nenhum hardware ou API.

**Dependencies:** Tasks 2, 5 e 12.

**Vertical slice:** eventos fake de captura/áudio -> TTFA e percentis registrados.

**Files likely touched:** `saphira/main.py`, `tests/test_core.py`.

**Estimated scope:** Small.

### Task 14: Loop PTT para turno normal

**Description:** Integrar configuração, hotkey, captura, STT, provider LLM, tools, worker de fala e métricas para o caminho sem barge-in.

**Acceptance criteria:**
- [ ] Primeira ativação inicia captura e segunda encerra/envia; áudio com menos de 100 ms é descartado sem chamar STT.
- [ ] Um turno factual fake percorre captura -> STT -> LLM sem tool -> frases -> TTS -> playback e retorna ao estado ocioso.
- [ ] STT vazio ou falho não derruba o loop e deixa a mensagem de recuperação disponível.

**Verification:** integração com componentes fakes em `python tests/test_core.py`; comando real do daemon fica para Task 18.

**Dependencies:** Tasks 1, 3, 4, 5, 6, 9, 11, 12 e 13.

**Vertical slice:** duas ativações fake -> pergunta transcrita -> resposta falada fake -> próximo ciclo disponível.

**Files likely touched:** `saphira/main.py`, `tests/test_core.py`.

**Estimated scope:** Medium.

### Task 15: Barge-in com cancelamento e retomada

**Description:** Integrar o evento VAD ao cancelamento coordenado de LLM/tool, TTS, playback e fila, iniciando nova captura depois do debounce.

**Acceptance criteria:**
- [ ] Fala nova durante playback para o sink em até 150 ms após a detecção, cancela o provider/tool/TTS em andamento e impede frases antigas de continuarem.
- [ ] A fila é drenada, a nova captura começa após o debounce configurado e o próximo turno pode ser processado sem reiniciar o daemon.
- [ ] O histórico final não contém `tool_result` órfão nem resultado parcial do turno cancelado.

**Verification:** cenário concorrente com VAD, provider, tool, TTS e playback fakes em `python tests/test_core.py`.

**Dependencies:** Tasks 5, 9, 10, 11, 12 e 14.

**Vertical slice:** resposta longa fake -> fala nova -> cancelamento -> nova captura -> memória segura.

**Files likely touched:** `saphira/audio.py`, `saphira/llm.py`, `saphira/main.py`, `tests/test_core.py`.

**Estimated scope:** Medium.

### Task 16: Degradação de voz

**Description:** Integrar os fallbacks de STT/TTS no loop principal, incluindo áudio local pré-gravado e mensagem fixa quando o TTS estiver disponível.

**Acceptance criteria:**
- [ ] Falha simulada de STT ou TTS reproduz o áudio local da mensagem definida sem depender de API externa.
- [ ] Falha de TTS no meio da resposta não encerra o worker nem o loop PTT; frases descartadas não voltam após recuperação.
- [ ] A mensagem local e o caminho de log permanecem observáveis quando não há saída de voz.

**Verification:** falhas injetadas em `python tests/test_core.py`; smoke real de STT/TTS fica para Task 18.

**Dependencies:** Tasks 6, 12 e 14.

**Vertical slice:** falha de voz fake -> áudio local/log -> daemon retorna ao estado utilizável.

**Files likely touched:** `saphira/main.py`, `saphira/speech.py`, `saphira/audio.py`, `tests/test_core.py`.

**Estimated scope:** Medium.

### Task 17: Degradação de raciocínio e tools

**Description:** Integrar as falhas de Tavily, `local_command`, Claude e Gemini no loop, preservando `tool_result`, failover, mensagem por TTS e fallback local.

**Acceptance criteria:**
- [ ] Falha Tavily ou `local_command` vira `tool_result` de erro; o LLM ativo verbaliza a mensagem tratada e o daemon continua.
- [ ] Falha do Claude tenta Gemini uma vez com o mesmo turno; falha de ambos emite mensagem fixa pelo TTS ou áudio local, sem retry/backoff.
- [ ] Nenhuma das falhas deixa thread, tool result, fila ou estado do PTT preso.

**Verification:** matriz de falhas com dependências fakes em `python tests/test_core.py`.

**Dependencies:** Tasks 3, 4, 9, 11, 12, 14 e 16.

**Vertical slice:** falha externa injetada -> fallback correspondente -> resposta/log -> próximo turno.

**Files likely touched:** `saphira/main.py`, `saphira/llm.py`, `saphira/tools.py`, `tests/test_core.py`.

**Estimated scope:** Medium.

## Checkpoint: Orquestração completa

- [ ] Tasks 12 a 17 concluídas.
- [ ] O caminho normal, barge-in e todas as categorias de falha deixam o daemon utilizável.
- [ ] `python tests/test_core.py` passa sem hardware ou chaves.

### Phase 5: Aceite v1

### Task 18: Smoke, E2E e Definition of Done

**Description:** Validar o daemon real contra os smoke tests, os cinco cenários E2E atualizados e a janela de 20 interações para latência.

**Acceptance criteria:**
- [ ] `python tests/test_core.py` passa e os smoke tests reais confirmam STT, TTS, `web_search` e fallback local.
- [ ] `python -m saphira.main` passa pelos cinco cenários: busca `web_search`, abrir Spotify, barge-in, falha Tavily simulada e corte de memória.
- [ ] Após pelo menos 20 interações reais, o log fornece p50/p95 da janela das últimas 20 dentro de `<2 s` sem tool e `<4 s` com busca; nenhuma chave aparece no código.

**Verification:**
- [ ] Executar `python tests/test_core.py`.
- [ ] A partir da raiz, executar `python -m saphira.main` com `.env` real e seguir os cenários da seção 5.3.
- [ ] Registrar evidências de áudio, processo/URI do Spotify, cancelamento em até 150 ms, memória segura e percentis do log.

**Dependencies:** Tasks 1 a 17.

**Vertical slice:** daemon real -> entrada de voz -> provider/tool/fallback -> áudio/log -> aceite do cenário.

**Files likely touched:** nenhum arquivo de implementação; apenas evidências locais não versionadas, se necessárias.

**Estimated scope:** Medium.

## Dependency Order and Parallelization

### Mandatory order

`Task 1 → Task 2 → {Task 3, Task 4, Task 5} → Task 6 → Task 7 → Task 8 → Task 9 → Task 10 → Task 11 → {Task 12, Task 13} → Task 14 → Task 15 → {Task 16, Task 17} → Task 18`

### Safe parallelization

- Tasks 3, 4 e 5 podem ser desenvolvidas em paralelo depois de Tasks 1 e 2, pois usam fakes e contratos separados.
- Task 12 pode começar após Tasks 5 e 6; Task 13 pode começar após o contrato de eventos de Task 12.
- Tasks 16 e 17 podem ser desenvolvidas em paralelo depois de Task 14, desde que ambas consumam os contratos já definidos.

### Must remain sequential

- Task 8 antes de Task 9: Gemini precisa provar paridade com o provider Claude.
- Tasks 9 e 10 antes de Task 11: o loop depende do provider comum e da memória segura.
- Task 14 antes de Task 15: o barge-in precisa interromper um loop PTT integrado.
- Tasks 1 a 17 antes do aceite real da Task 18.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| SDKs Claude/Gemini divergirem no streaming e tool use | Alto | Adapters isolados e teste de paridade com clients fakes antes da integração real. |
| VAD disparar por eco do próprio TTS | Alto | Debounce de 150–300 ms, cancelamento testado por sinal injetado e validação manual no hardware real. |
| Cancelamento deixar tool result ou frase atrasada | Alto | Token/evento de cancelamento compartilhado, drenagem da fila e asserts de histórico sem órfãos. |
| Áudio local de fallback não tocar em dispositivo indisponível | Médio | Testar arquivo e caminho sem SDK; registrar mensagem no log como último fallback operacional. |
| P95 exceder orçamento por rede ou cold start | Alto | Registrar TTFA por turno, usar janela de 20 interações e investigar etapa com maior duração antes do aceite. |
| `GEMINI_API_KEY` ausente impedir o failover | Médio | Validar a variável no boot e falhar com nome claro antes de iniciar o daemon. |
| Spotify ou dispositivo de áudio ausente no ambiente E2E | Médio | Registrar pré-condições antes do cenário; não substituir o aceite real por mock. |

## Open Questions

Nenhuma lacuna bloqueante identificada no SPEC atualizado. O plano trata `GEMINI_API_KEY` como obrigatória por consequência do fallback obrigatório e do fail-fast definido no Definition of Done; alterar essa política exigiria revisar o SPEC antes do build.

## Final Gate

- [ ] Cada tarefa tem aceite, verificação, dependências e fatia vertical.
- [ ] Nenhuma tarefa excede aproximadamente cinco arquivos.
- [ ] Checkpoints existem entre as fases.
- [ ] O plano não inclui itens do fora de escopo.
- [ ] Implementação só começa após aprovação humana deste plano.
