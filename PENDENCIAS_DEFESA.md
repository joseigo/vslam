# Pendências para a defesa

Lista viva: marcar `[x]` quando concluir (com data e onde está o resultado) e acrescentar o que surgir.
Prioridade: **E** = essencial, **I** = importante, **D** = desejável.

## Testes e experimentos

- [ ] **E** Comparação com o estado da arte: ORB-SLAM3 (monocular; mono-inercial se possível) e/ou VINS-Mono nas
      mesmas sequências do KAIST, mesmo protocolo (ATE com evo, RPE do KITTI).
- [ ] **E** Tabela de ablação em todas as sequências: só roda, só VO, fusão sem laço, fusão com laço, com/sem
      realimentação de escala, com/sem âncora pelo plano do chão, BA fix/marg, cada camada de rejeição de laço
      (porta de deriva, PCM, GNC) desligada.
      *Em andamento (2026-11-06): bateria de 55 execuções -> `resultados/ablacao/`; tabela com
      `Codes/ferramentas/tabela_ablacao.py` (CSV, .tex por métrica, figura). Inclui repetição do sistema completo em
      urban27 p/ checar se o pipeline COMPLETO (grafo assíncrono + realimentação) também é determinístico.*
- [ ] **E** Barras de erro: execuções com perturbações controladas (semente do RANSAC, frame inicial deslocado,
      ruído pequeno na calibração) -> mediana e dispersão. Repetições idênticas não servem (pipeline determinístico).
- [ ] **E** Robustez a laços falsos: injetar 10/50/100 laços errados no grafo e medir o ATE com e sem cada camada
      de rejeição.
- [ ] **I** Outro dataset (Inomi / dados próprios); vocabulário treinado em dados de outro dataset.
- [ ] **I** Sensibilidade a erros dos sensores: fator de escala do encoder (±2%), altura da câmera (±5 cm),
      atraso de tempo roda-câmera.
- [ ] **I** Laços de ponta a ponta: quantos laços com pose ruim (aceitos pelo detector) foram barrados pela porta
      de deriva / PCM no grafo.
- [ ] **D** Consistência da incerteza do grafo (NEES).
- [ ] **D** Tempo de processamento em hardware embarcado (se a dissertação afirmar tempo real no veículo).
- [ ] **D** Análise de casos de falha do VO (urban38/39: RPE 37-47%): onde e por quê, com imagens dos trechos.

## Pendências pequenas

- [ ] Teste `test_vo_geometria::test_klt_fluxo_previsto_deslocamento_grande` falhando (suíte: 42 ok, 1 falha).
- [ ] README: corrigir "o VO não é determinístico... várias execuções" (as repetições saem idênticas bit a bit).
- [ ] Commitar o trabalho atual (BA com marginalização, ferramentas de avaliação, figuras TikZ, ajustes de
      rótulos) -- nada foi commitado ainda.

## Concluído

- [x] 2026-11-04 Rótulos técnicos nos gráficos e GIFs ("odometria de rodas", "odometria visual"...).
- [x] 2026-11-04 Figuras TikZ dos grafos: `figuras/grafo_ba_local.tex` e `figuras/grafo_poses_global.tex`.
- [x] 2026-11-05 BA local com marginalização (Schur + FEJ), `--ba-mode marg`, testes em `Codes/tests/test_ba.py`.
- [x] 2026-11-05 Falhas da marginalização corrigidas (landmark colado à câmera; landmark de 1 fator no grafo) -> 0
      falhas nas 6 sequências.
- [x] 2026-11-05 Comparação fix x marg no VO puro: `resultados/comparacao_ba/` (`Codes/ferramentas/comparar_ba.py`).
      Mediana do RPE de translação ponderada 24,3% -> 18,0%; rotação igual; BA ~80% mais lento.
- [x] 2026-11-05 Mediana do RPE gravada em `metricas.csv` e `rpe.csv`.
- [x] 2026-11-06 Figura de processamento: atraso do grafo numa faixa própria (sem 2º eixo y).
- [x] 2026-11-06 Avaliação isolada do detector de laço: `Codes/ferramentas/avaliar_lacos.py`,
      `resultados/lacos/` (recall@N do BoW, curvas PR da verificação, funil, cobertura por sentido, erro da pose).
      Precisão 0,89-1,00 (lugar) / 0,80-1,00 (pose); sentido oposto ~0%.

## Notas para o texto

- Revisitas em sentido oposto: limitação do sensor (câmera frontal); citar Lowry et al. (T-RO 2016) e
  Garg et al. "LoST?" (RSS 2018); trabalhos futuros: semântica, câmera traseira/panorâmica, Scan Context (LiDAR).
- Laços aceitos "a > 10 m" são quase todos do lugar certo vistos de longe (rotação < 2°, translação 1-13 m).
- A diferença fix x marg numa sequência isolada não é confiável (sensibilidade caótica do VO monocular longo).
