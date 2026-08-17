# Saphira v1 Task List

## Phase 1: Foundation

- [x] Task 1: Configuração e dependências do runtime
- [x] Task 2: Registry comum de tools
- [ ] Checkpoint: Foundation

## Phase 2: Tools e áudio

- [x] Task 3: Allowlist de comandos locais
- [x] Task 4: Busca web via Tavily
- [x] Task 5: Captura, playback, hotkey e VAD
- [ ] Checkpoint: Tools e áudio básico

## Phase 3: Speech e raciocínio

- [x] Task 6: Adaptadores ElevenLabs e áudio local de fallback
- [x] Task 7: Contrato de provider LLM e chunker de frases
- [x] Task 8: Streaming e tool request do Claude
- [x] Task 9: Fallback Gemini com contrato equivalente
- [x] Task 10: Janela de memória em RAM
- [x] Task 11: Loop de tools e reinjeção de `tool_result`
- [ ] Checkpoint: Speech e providers

## Phase 4: Pipeline e orquestração

- [x] Task 12: Worker de fala frase a frase
- [x] Task 13: Eventos e métricas de latência
- [ ] Task 14: Loop PTT para turno normal
- [ ] Task 15: Barge-in com cancelamento e retomada
- [ ] Task 16: Degradação de voz
- [ ] Task 17: Degradação de raciocínio e tools
- [ ] Checkpoint: Orquestração completa

## Phase 5: Aceite v1

- [ ] Task 18: Smoke, E2E e Definition of Done

Consulte `tasks/plan.md` para critérios de aceite, verificações, dependências, fatias verticais, riscos e ordem de execução.
