# Runbook de teste manual — Saphira v1

Este runbook valida o daemon real a partir da raiz do repositório. A aplicação não possui argumentos de CLI, servidor HTTP, UI ou arquivo de log próprio. O comando de execução é `python -m saphira.main`; os logs são emitidos pelo `logging` para o terminal. Os arquivos `.log` usados abaixo são capturas criadas pelo PowerShell, não arquivos gerados pela aplicação.

## Pré-condições

- Windows com Python 3.10 ou superior.
- Prompt PowerShell aberto em `C:\Users\Jão\Documents\Saphira`.
- `.env` preenchido na raiz. As cinco variáveis obrigatórias são `ANTHROPIC_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `TAVILY_API_KEY` e `GEMINI_API_KEY`.
- Dependências instaladas com `python -m pip install -r requirements.txt`.
- Microfone e saída de áudio funcionais; `sounddevice` usa entrada PCM mono 16 kHz e saída PCM mono 24 kHz.
- Para `Abra o Spotify`, Spotify instalado e a associação `spotify:` disponível no Windows.
- O processo deve ser iniciado a partir da raiz. O default `SAPHIRA_COMMANDS=commands.yaml` e o script `scripts/ola.py` dependem desse diretório de trabalho.

Variáveis lidas pelo código:

| Variável | Obrigatória | Default/uso |
|---|---:|---|
| `ANTHROPIC_API_KEY` | sim | Chave do Claude |
| `ELEVENLABS_API_KEY` | sim | Chave de STT/TTS |
| `ELEVENLABS_VOICE_ID` | sim | Voz do TTS |
| `TAVILY_API_KEY` | sim | Chave de `web_search` |
| `GEMINI_API_KEY` | sim | Chave do fallback |
| `CLAUDE_MODEL` | não | `claude-sonnet-5` |
| `SAPHIRA_HOTKEY` | não | `<f9>` |
| `SAPHIRA_COMMANDS` | não | `commands.yaml` |

`GEMINI_MODEL` (`gemini-3.6-flash`), `STT_MODEL` (`scribe_v2`), `TTS_MODEL` (`eleven_flash_v2_5`), `LANGUAGE` (`por`) e `MAX_HISTORY_MESSAGES` (`20`) são constantes em `saphira/config.py`; não são variáveis de ambiente.

## 1. Subir a aplicação

**Objetivo:** confirmar o boot, o fail-fast de configuração e o estado pronto para ouvir.

**Camada:** processo Python, configuração, hotkey global e áudio.

**Ação exata:** na raiz do repositório, execute:

```powershell
python -m saphira.main 2>&1 | Tee-Object -FilePath .\saphira-manual.log
```

O trecho `2>&1 | Tee-Object ...` apenas mostra e captura o terminal. Para não criar captura, o comando exato da aplicação é `python -m saphira.main`.

**Resultado esperado:** depois de inicializar os clientes externos, o terminal mostra uma linha contendo exatamente:

```text
Saphira pronta. Hotkey <f9>: 1ª vez = falar, 2ª = enviar; durante a fala = interromper
```

Essa linha confirma que o daemon instalou a hotkey e entrou no loop. Pressione `Ctrl+C` para encerrar; o terminal deve mostrar `Saphira desligada.`.

**Fail-fast intencional:** em uma nova janela PowerShell, com o `.env` válido preservado, execute:

```powershell
$env:GEMINI_API_KEY = ""
python -m saphira.main
```

O processo deve terminar antes de `Saphira pronta`, com `RuntimeError` contendo `Variável GEMINI_API_KEY ausente` e `.env.example`. Remova apenas a substituição dessa janela antes dos demais testes:

```powershell
Remove-Item Env:GEMINI_API_KEY
```

**Evidência:** `saphira-manual.log` contendo a linha `Saphira pronta`; para o fail-fast, a saída contendo o nome da variável e `.env.example`.

**Falha se:** não aparece a linha de prontidão, o processo termina antes dela, não há entrada/saída de áudio, ou o processo continua mesmo com uma variável obrigatória vazia.

## 2. Smoke test básico (turno sem tool)

**Objetivo:** confirmar o ciclo PTT completo sem tool: captura, STT, Claude/Gemini, TTS e playback.

**Camada:** microfone, ElevenLabs, provider LLM e saída de áudio.

**Pré-condição:** a seção 1 terminou com `Saphira pronta` e o processo continua rodando.

**Ação exata:** pressione e solte `F9` uma vez, diga:

```text
Qual é a capital de Portugal?
```

Pressione e solte `F9` uma segunda vez para enviar o turno. Não mantenha a tecla pressionada: o código usa duas ativações como toggle.

**Evidência esperada no terminal:**

- `captura iniciada`, seguido de `gravacao: ...ms` e `captura finalizada`.
- `stt: ...ms` e `ouvido: 'Qual é a capital de Portugal?'` (a representação pode variar conforme a transcrição).
- Uma ou mais linhas `saphira: ...` e áudio audível com uma resposta curta, normalmente `Lisboa`.
- `primeiro chunk tocando` e uma linha `ttfa=...ms p50=...ms p95=...ms janela=1`.
- Nenhuma linha `tool=web_search` ou `tool=local_command` entre esse `ouvido` e a resposta.

**Resultado final esperado:** após o áudio, o daemon fica disponível para a próxima ativação de `F9`, sem reiniciar.

**Falha se:** STT não produz texto, não há áudio, uma tool é chamada para essa pergunta estática, ou o daemon sai do loop.

## 3. Tool Tavily

**Objetivo:** provar a chamada real de `web_search`, o retorno ao LLM e o caminho de falha da Tavily.

**Camada:** ElevenLabs STT/TTS, Claude ou Gemini, `ToolRegistry` e Tavily.

**Ação exata para sucesso:** pressione `F9`, diga:

```text
Qual é a cotação do dólar hoje?
```

Pressione `F9` novamente para enviar.

**Evidência esperada:**

- `ouvido: ...` com a pergunta transcrita.
- No logger `saphira.tools`, uma linha contendo `tool=web_search args=` e a query, seguida de uma duração em `ms`, por exemplo `tool=web_search args={'query': '...'}  ...ms`.
- O áudio contém uma resposta baseada em dados atuais, e não apenas uma resposta fixa de conhecimento geral.
- `primeiro chunk tocando` e `ttfa=...` aparecem depois do processamento.

O código não loga o evento textual `tool_result`; a evidência implementada da chamada é a linha `tool=web_search` e a resposta audível.

**Falha Tavily intencional, cenário E2E 4 do SPEC:** encerre o processo com `Ctrl+C`, abra uma nova janela PowerShell e execute apenas nesta janela:

```powershell
$env:TAVILY_API_KEY = "invalid-tavily-key-for-manual-test"
python -m saphira.main 2>&1 | Tee-Object -FilePath .\saphira-tavily-failure.log
```

Quando aparecer `Saphira pronta`, diga novamente `Qual é a cotação do dólar hoje?` e envie com `F9`.

**Evidência esperada da falha:** uma exceção do logger `saphira.tools` contendo `tool web_search falhou`, seguida de uma resposta falada tratada pelo LLM. O processo deve continuar aceitando outro turno. O `ToolRegistry` captura a exceção e entrega ao provider o texto `Erro ao executar web_search: ...`; esse texto não é logado como `tool_result` separado.

**Limpeza:** feche essa janela PowerShell ou execute `Remove-Item Env:TAVILY_API_KEY` antes de voltar ao `.env` normal.

**Falha se:** a exceção encerra o processo, não há resposta falada, ou não aparece a chamada `tool=web_search` no cenário de sucesso.

## 4. Comando local via allowlist

**Objetivo:** confirmar que o LLM usa somente os alvos de `commands.yaml`, incluindo `run_script` e a regra de fechamento do navegador no Windows.

**Camada:** LLM, `LocalCommandTool`, `commands.yaml`, `subprocess` e Windows.

**Catálogo real carregado:** `spotify` aceita `open_app`/`close_app`; `navegador` aceita `open_app` e, no Windows, não aceita `close_app`; `ola` aceita `run_script`.

**Teste 1, abrir Spotify:** diga exatamente:

```text
Abra o Spotify.
```

Envie com a segunda ativação de `F9`. Observe:

- No terminal, uma linha contendo `tool=local_command args={'action': 'open_app', 'target': 'spotify'}`.
- O Spotify ou o handler da URI `spotify:` é iniciado pelo comando YAML `cmd /c start spotify:`.
- A resposta é falada e o daemon volta ao estado ocioso.

O código retorna `ok: open_app spotify` após `subprocess.Popen(..., shell=False)`, mas não registra o `argv` nem confirma no log que a janela do Spotify abriu. A abertura efetiva deve ser confirmada visualmente no Windows.

**Teste 2, `run_script` da allowlist:** diga:

```text
Rode o script ola.
```

O resultado permitido é `scripts/ola.py`, cujo stdout real é:

```text
Ola! Script de demonstracao da Saphira executado com sucesso.
```

Observe a linha `tool=local_command args={'action': 'run_script', 'target': 'ola'}` e a resposta falada baseada nesse stdout. O stdout do script não é escrito diretamente no log; ele é devolvido ao LLM como resultado da tool.

**Teste 3, fechamento não suportado no Windows:** diga:

```text
Feche o navegador.
```

O código retorna sem chamar processo:

```text
Comando 'close_app navegador' não suportado no Windows. Disponíveis:
```

Observe a linha de chamada `tool=local_command args={'action': 'close_app', 'target': 'navegador'}`, uma resposta falada informando a indisponibilidade e que um navegador já aberto permanece aberto. A implementação tem uma guarda explícita para Windows; não existe entrada `win32` de `close` para `navegador` em `commands.yaml`.

**Falha se:** um alvo fora do catálogo for executado, `close_app navegador` iniciar `taskkill` ou outro processo, o script `ola` não for encontrado a partir da raiz, ou o daemon cair.

## 5. Barge-in

**Objetivo:** interromper uma resposta em reprodução e confirmar cancelamento do playback e retomada automática da captura.

**Camada:** VAD, playback PCM, cancelamento do turno e loop PTT.

**Pré-condição:** use o processo normal da seção 1, com o microfone livre para que a fala seja capturada durante o playback.

**Ação exata:** inicie um turno com:

```text
Conte uma história em português com pelo menos dez frases, sem usar ferramentas.
```

Envie com `F9`. Assim que a primeira frase começar a tocar, fale continuamente perto do microfone por pelo menos 300 ms:

```text
Pare. Agora responda apenas: interrompido.
```

Não pressione `F9` para iniciar a nova captura. O VAD deve detectar essa fala durante o playback; depois que o áudio antigo parar, aguarde a linha `captura iniciada`, diga novamente `interrompido` se necessário e pressione `F9` para encerrar a nova gravação.

**Evidência esperada:**

- Antes da interrupção: `primeiro chunk tocando` e áudio da primeira resposta.
- Durante a interrupção: `playback interrompido (barge-in)`.
- Depois: `captura iniciada` para o novo turno sem uma nova pressão de `F9` para começar.
- As frases antigas não continuam tocando depois do log de interrupção; a nova resposta é audível.

O debounce real é de 200 ms (`VAD(..., debounce_ms=200)`), dentro do intervalo de 150–300 ms. O código não loga o instante da detecção do VAD nem uma duração de cancelamento; portanto, o terminal permite confirmar a ordem dos eventos e a parada, mas não provar sozinho o limite de 150 ms entre detecção e parada. O teste automatizado em `tests/test_core.py` é o que mede esse limite com dependências falsas.

Também não há log do `_history` nem de `tool_result` no runtime. A ausência de `tool_result` órfão não pode ser inspecionada diretamente nesta execução manual; isso é coberto apenas pelos asserts internos do teste automatizado.

**Falha se:** frases antigas continuam tocando, a nova captura só começa após pressionar `F9`, o daemon fica preso, ou não é possível iniciar outro turno.

## 6. Fallback Claude → Gemini

**Objetivo:** tornar o Claude indisponível sem alterar a chave Gemini e confirmar que o mesmo turno continua pelo fallback.

**Camada:** `AnthropicProvider`, `GeminiProvider`, `ClaudeBrain` e TTS.

O código não lê variável de endpoint Anthropic. A forma disponível de forçar essa falha é substituir a chave Anthropic por uma chave inválida apenas no processo de teste.

**Ação exata:** em uma nova janela PowerShell, execute:

```powershell
$env:ANTHROPIC_API_KEY = "invalid-claude-key-for-fallback-test"
python -m saphira.main 2>&1 | Tee-Object -FilePath .\saphira-claude-fallback.log
```

Não altere `GEMINI_API_KEY`. Quando aparecer `Saphira pronta`, diga:

```text
Qual é a capital de Portugal?
```

Envie com `F9`.

**Evidência esperada:**

- O terminal mostra no logger `saphira.llm` uma linha contendo `provider AnthropicProvider falhou; tentando fallback`, normalmente acompanhada do traceback da falha HTTP/autenticação.
- O turno ainda produz linhas `saphira: ...`, `primeiro chunk tocando` e áudio de resposta.
- Um segundo turno também é processado sem reiniciar o daemon, confirmando que a falha não prendeu o loop.

Não há uma linha explícita `Gemini selecionado`; a resposta falada após o warning e a continuidade do processo são as evidências implementadas. Feche a janela de teste ou remova a substituição com `Remove-Item Env:ANTHROPIC_API_KEY` antes de usar a configuração normal.

**Falha se:** o processo encerra no erro do Claude, o Gemini não responde apesar de sua chave permanecer válida, ou o mesmo texto de usuário aparece duplicado no comportamento observado.

## 7. Falha de STT/TTS

**Objetivo:** exercitar as falhas de voz e identificar exatamente quando o WAV local `saphira/fallback_audio.wav` é usado.

**Camada:** `Speech`, `run_turn`, `speaker_worker`, playback e áudio local.

### 7.1 Falha de TTS isolada

Com as chaves válidas, em uma nova janela PowerShell execute:

```powershell
$env:ELEVENLABS_VOICE_ID = "invalid-voice-id-for-tts-test"
python -m saphira.main 2>&1 | Tee-Object -FilePath .\saphira-tts-failure.log
```

O override só altera o `voice_id` passado ao TTS; o STT continua usando a chave e o modelo válidos. Depois de `Saphira pronta`, faça o smoke test da seção 2.

**Evidência esperada:** o terminal contém `TTS falhou; usando fallback local` e `usando fallback local: não consegui te ouvir, tenta de novo`. O áudio audível deve ser a mensagem gravada no arquivo `saphira/fallback_audio.wav`, não uma nova resposta da ElevenLabs. O daemon deve aceitar outro turno.

### 7.2 Falha de STT isolada

Não existe variável de ambiente para trocar `STT_MODEL`; ele é fixado em `saphira/config.py` como `STT_MODEL = "scribe_v2"`. Para provocar uma rejeição real do request STT sem invalidar o TTS, faça uma edição temporária exata:

1. Abra `saphira/config.py` com `notepad .\saphira\config.py`.
2. Troque `STT_MODEL = "scribe_v2"` por `STT_MODEL = "stt-model-inexistente-para-teste"`.
3. Mantenha `ELEVENLABS_API_KEY` e `ELEVENLABS_VOICE_ID` válidos e execute `python -m saphira.main 2>&1 | Tee-Object -FilePath .\saphira-stt-failure.log`.
4. Faça um turno normal e restaure exatamente `STT_MODEL = "scribe_v2"` antes de qualquer outro cenário.

**Evidência esperada da implementação atual:** `STT falhou` com traceback. Em seguida, como o TTS ainda está disponível, o código fala via TTS `Não consegui te ouvir agora. Tenta de novo?`; ele não chama o WAV local nesse caso.

**Divergência do SPEC:** o SPEC diz que uma falha de STT deve usar o áudio local, mas `run_turn` chama `say(...)` após a exceção de STT, e `say` só usa o WAV quando o TTS também falha. Para ouvir o áudio local após uma falha STT, seria necessário combinar a edição do modelo STT inválido com o override de `ELEVENLABS_VOICE_ID` da seção 7.1; isso testa STT e TTS indisponíveis juntos, não STT isolado.

Remova os overrides de ambiente ao terminar:

```powershell
Remove-Item Env:ELEVENLABS_VOICE_ID
```

**Falha se:** a falha de TTS não produzir os dois logs de fallback e o WAV audível, ou a edição temporária do modelo STT não gerar `STT falhou`. A ausência do WAV no caso de STT isolado é a divergência descrita acima, não um resultado esperado do SPEC.

## 8. Corte da janela de memória

**Objetivo:** exceder a janela de 20 unidades de histórico e confirmar que o daemon continua utilizável.

**Camada:** `ClaudeBrain._history`, providers e loop de tools.

O código conta cada mensagem `user`/`assistant` como uma unidade e cada bloco `tool_result` como uma unidade. Portanto, nove turnos sem tool já geram aproximadamente 18 unidades; um turno com `web_search` leva o total acima de 20.

**Ação exata:** execute o daemon com `.env` normal e faça estes turnos, sempre pressionando `F9` uma vez para começar, falando a frase, e pressionando `F9` novamente para enviar. Aguarde a resposta terminar antes do próximo item:

1. `Diga uma cor.`
2. `Diga um animal.`
3. `Diga uma fruta.`
4. `Diga um número entre um e dez.`
5. `Qual é o oposto de quente?`
6. `Complete: dois mais dois.`
7. `Diga uma saudação curta.`
8. `Diga uma estação do ano.`
9. `Responda apenas: teste nove.`
10. `Qual é a cotação do dólar hoje?`
11. `Responda apenas: teste onze.`

Avance do turno 10 somente depois de observar `tool=web_search` no log, garantindo que a sequência contenha um par real de tool. O turno 10 deve gerar `user`, `assistant` com `tool_use`, `user` com `tool_result` e `assistant` final; somado aos turnos anteriores, ultrapassa 20 unidades.

**Evidência disponível:** todas as 11 respostas são processadas, há `tool=web_search` no turno 10, o processo não quebra e o turno 11 também é respondido. Não existe log do tamanho da janela, do corte ou do conteúdo de `_history`; portanto, não há lugar no runtime atual onde inspecionar diretamente a ausência de `tool_result` órfão. Essa propriedade só é observável nos asserts de `tests/test_core.py`, que verificam `_trim_history()` e os pares `tool_use`/`tool_result`.

**Falha se:** o daemon não aceita o turno seguinte, a busca deixa o loop preso, ou o próximo request falha por histórico inválido. A execução manual, sozinha, não transforma a ausência de erro em prova do par seguro.

## 9. Métricas de latência

**Objetivo:** coletar TTFA e conferir p50/p95 da janela deslizante de até 20 interações.

**Camada:** `LatencyMetrics` em `saphira/main.py` e logging do terminal.

**Eventos realmente logados:**

- `captura iniciada`
- `gravacao: ...ms`
- `captura finalizada`
- `stt: ...ms`
- `primeiro chunk tocando`
- `ttfa=...ms p50=...ms p95=...ms janela=...`
- `tool=... args=... ...ms` para tools que terminam sem exceção

TTFA é calculado pelo código como o tempo entre `captura finalizada` e o primeiro chunk enviado ao playback. Os valores no texto do log são arredondados para milissegundos inteiros; o cálculo interno usa os valores não arredondados.

**Ação exata:** mantenha a captura da seção 1 e execute turnos até que apareça `janela=20`. Para medir sem tool, faça 20 turnos estáticos como o smoke test. Para medir com busca, inicie uma nova execução e faça 20 turnos que produzam `tool=web_search`; iniciar uma execução separada evita misturar categorias, pois o log não associa cada TTFA a um identificador de turno ou ao tipo de tool.

Depois de `Ctrl+C`, calcule os percentis da captura com este bloco PowerShell, executado na raiz:

```powershell
$all = @(
  Get-Content .\saphira-manual.log |
    ForEach-Object {
      if ($_ -match 'ttfa=(?<ms>[0-9]+(?:\.[0-9]+)?)ms') {
        [double]$Matches['ms']
      }
    }
)
if ($all.Count -eq 0) { throw "Nenhuma linha ttfa= foi encontrada" }
$from = [Math]::Max(0, $all.Count - 20)
$window = @($all[$from..($all.Count - 1)] | Sort-Object)
function Percentile([double[]] $values, [double] $fraction) {
  $position = ($values.Count - 1) * $fraction
  $lower = [int][Math]::Floor($position)
  $upper = [int][Math]::Min($lower + 1, $values.Count - 1)
  $weight = $position - $lower
  $values[$lower] + ($values[$upper] - $values[$lower]) * $weight
}
[pscustomobject]@{
  interacoes = $window.Count
  p50_ms = [Math]::Round((Percentile $window 0.50), 2)
  p95_ms = [Math]::Round((Percentile $window 0.95), 2)
}
```

**Resultado esperado:** para uma janela sem tool, `p50_ms` e `p95_ms` devem ser menores que `2000`; para uma janela com busca web, ambos devem ser menores que `4000`. Compare também a saída com a última linha `ttfa=... p50=... p95=... janela=20` do log; diferenças de até aproximadamente 1 ms podem ocorrer porque o arquivo contém os valores arredondados.

**Divergências de observabilidade:** não há duração separada de LLM ou TTS; falhas de tool não registram a linha de duração `tool=...`; e uma janela mista não permite classificar TTFA por tipo de turno somente a partir do log. O aplicativo também não grava um arquivo de log sem o `Tee-Object` externo.

## Checklist final — Definition of Done do SPEC.md

- [ ] DoD 1: `python tests/test_core.py` termina com `todos os checks passaram`.
- [ ] DoD 2.1: E2E Tavily com pergunta atual, chamada `tool=web_search` e resposta falada, conforme a seção 3.
- [ ] DoD 2.2: E2E abrir Spotify pelo alvo `spotify` da allowlist, conforme a seção 4.
- [ ] DoD 2.3: E2E barge-in para playback e inicia nova captura, conforme a seção 5.
- [ ] DoD 2.4: E2E falha Tavily intencional sem crash e com resposta tratada, conforme a seção 3.
- [ ] DoD 2.5: E2E acima de 20 unidades de histórico; a ausência de órfão fica coberta pelo teste automatizado, e a lacuna de log está registrada na seção 8.
- [ ] DoD 3: janelas de 20 interações sem tool e com busca ficam abaixo de `<2 s` e `<4 s` em p50 e p95, conforme a seção 9.
- [ ] DoD 4: nenhuma chave aparece no código; as cinco variáveis obrigatórias estão somente no `.env`; a remoção de uma variável faz o boot falhar com nome e referência a `.env.example`, conforme a seção 1.
