# Odometria visual monocular com fusão em grafo de fatores e fechamento de laço

O sistema estima a trajetória de um veículo a partir de **uma única
câmera**, combinada com a **odometria de roda** (e, opcionalmente, uma IMU) num **grafo de fatores** otimizado de
forma incremental, com **fechamento de laço** para corrigir a deriva acumulada quando o veículo revisita um lugar.
Todo o processamento roda mais rápido que o tempo real.

## Visão geral

```
 imagens ──► Odometria visual monocular ──┐
              (KLT + PnP, keyframes,       │     ┌──────────────────────────────┐
               BA local assíncrono)        ├────►│  Grafo de fatores (iSAM2)    │──► trajetória métrica
 roda / IMU ─────────────────────────────┤     │  poses + escala do VO        │    (online e suavizada)
                                           │     └──────────────▲───────────────┘
 imagens ──► Fechamento de laço ───────────┘                    │ escala estimada
              (BoW → consistência → F → PnP;   ◄────────────────┘ realimentada no VO
               porta de deriva + PCM + GNC)
```

- **Odometria visual (VO) monocular**: rastreamento de pontos (KLT) e pose por PnP sobre um mapa local, criação de
  keyframes e *bundle adjustment* local em janela deslizante, executado em outro processo.
- **Escala como estado do grafo**: a câmera monocular não observa a escala, que além disso deriva. Em vez de
  corrigi-la antes, o grafo estima a log-escala do VO em cada nó (passeio aleatório), a partir da roda/IMU, e a
  **realimenta no VO**, levando o mapa para metros. Quando o VO perde o rastreamento e reinicia com outra escala,
  ela é semeada pela razão roda/VO.
- **Fusão em grafo de fatores** (GTSAM, iSAM2): fatores do VO, da roda (modelo diferencial plano), da IMU
  (pré-integração) e dos laços. Online, a cada nó; no fim, uma otimização em lote refina tudo.
- **Fechamento de laço**: vocabulário ORB próprio no estilo DBoW2, consistência temporal, verificação geométrica
  (matriz fundamental e PnP com pontos triangulados) e rejeição robusta de laços errados em três camadas: porta
  de deriva, PCM (consistência par a par, online) e GNC-TLS (no lote final, partindo da solução só com a
  odometria e só sobre laços com algum parceiro consistente no PCM -- um laço isolado não é verificável).
- **Tempo real**: VO, BA, grafo e detector de laço rodam em processos separados. O custo do grafo foi controlado
  agrupando os fatores do VO (o `CustomFactor` do GTSAM em Python copia o grafo inteiro a cada avaliação, o que
  tornava o custo quadrático com o tamanho do mapa).

## Resultados

Os resultados da dissertação saem de uma **bateria única** (`Codes/ferramentas/bateria.py`), retomável e agendada
para a noite, que gera tudo em `resultados/`. O índice com o estado de cada execução fica em `resultados/README.md`:

- `resultados/kaist/<seq>/<config>/` e `resultados/inomi/<seq>/<config>/`: cada execução (figuras, dados, evo; GIFs
  na 1ª repetição do sistema completo);
- `resultados/sintese/`: tabela de ablação e barras de erro (`ablacao/`), detector de laço isolado
  (`detector_lacos/`), robustez a laços falsos (`lacos_falsos/`), Inomi (`inomi/`);
- `resultados/testes_unitarios/`: histórico dos testes isolados.

Sequências: KAIST urban26/27/30/33/38/39 (avaliação), urban28/31/32/37 (**teste cego**: fora de todo ajuste e do
treino do vocabulário) e as duas sequências da Inomi com GT.

**Determinismo**: o VO puro é determinístico (execuções repetidas saem idênticas bit a bit); o sistema completo não
(o grafo roda num processo assíncrono e realimenta a escala no VO), por isso o sistema completo é executado 3 vezes
por sequência e as tabelas usam a mediana (e mostram mín.–máx.).

**Limitações observadas**: revisitas em **sentido oposto** não são reconhecidas (a câmera frontal vê outra cena);
o VO monocular perde o rastreamento em velocidade alta (urban37, ~78 km/h) e em trânsito denso, quando a fusão cai
para a odometria de roda (sem divergir); paradas longas podem gerar "laços" entre o antes e o depois da parada
(geometricamente corretos, mas sem informação nova); a estimativa online deriva entre laços.

## Avaliação

Cada execução gera métricas e figuras contra o GT: ATE e RMSE ancorado, erro ao longo do tempo e da distância,
RPE por comprimento de segmento (protocolo KITTI), precisão dos laços, **funil do detector de laço** (onde cada
consulta para e se o candidato era verdadeiro), **cobertura das revisitas** (mesmo sentido / sentido oposto, pelo
GT), **tempo de processamento** por componente (orçamento de tempo real, crescimento com o mapa, atraso do grafo),
mapas 3D (no referencial do VO e no fundido, em metros) e GIFs para apresentação. As trajetórias também são
avaliadas com o [evo](https://github.com/MichaelGrupp/evo) (APE com alinhamento SE(3)/Sim(3) e RPE em segmentos
de 100 m), cujo APE coincide com o ATE calculado pelo pipeline.

## Dados

- **[KAIST Complex Urban Dataset](https://sites.google.com/view/complex-urban-dataset)**: sequências urbanas de
  vários km (câmera estéreo, da qual se usa só a esquerda; encoders de roda; IMU; GT por SLAM com LiDAR/GNSS).
  O vocabulário do fechamento de laço foi treinado em sequências **separadas** das de teste (urban25/33/34).
  Teste cego: urban28/31/32/37 (nunca usadas para ajustar parâmetros). O encoder fica em `data_sensor/` ou
  `sensor_data/`, conforme a sequência; `Codes/ferramentas/inspecionar_sequencia.py` confere uma sequência baixada.
- **Robô móvel (Inomi)**: trajetos curtos de robô terrestre (~2,6 m), com odometria; só `dataset_16_16` e
  `dataset_16_16_n2` têm GT.

Os datasets não fazem parte deste repositório.

## Organização do repositório

```
Codes/
├── src/            biblioteca
│   ├── VisualOdometry.py, Map.py, KeyFrame.py, BA.py   odometria visual monocular (BA fixo ou marginalizado)
│   ├── GPO.py, Fusion.py                               grafo de fatores e fusão online
│   ├── LoopClosure.py                                  fechamento de laço e treino do vocabulário
│   └── Viewer.py                                       visualização ao vivo
├── scripts/        execução
│   ├── rodar_sistema.py      pipeline completo (VO + fusão + laço + avaliação + figuras)
│   ├── rodar_vo.py    só a odometria visual; configuração de câmera e parâmetros
│   └── fusao_offline.py               fusão offline a partir de uma trajetória do VO salva
├── ferramentas/    bateria e análise
│   ├── bateria.py                bateria final (todas as sequências e configurações) + síntese -> resultados/
│   ├── tabela_ablacao.py         tabela de ablação e barras de erro (CSV, LaTeX, figuras)
│   ├── avaliar_lacos.py          avaliação isolada do detector de laço (recall@N, precisão, funil, cobertura)
│   ├── lacos_falsos.py           robustez a laços falsos (repete só o grafo a partir das mensagens gravadas)
│   ├── avaliar_evo.py            avaliação com o evo (APE/RPE)
│   ├── gerar_gifs.py              GIFs para apresentação a partir de uma execução
│   ├── inspecionar_sequencia.py  confere uma sequência do KAIST baixada (arquivos, percurso, revisitas)
│   ├── comparar_ba.py            (antigo) VO puro, BA fixo x marginalizado; hoje feito pela bateria
│   └── visualizar_mapa.py               visualizador 3D dos mapas salvos
├── tests/          testes isolados com dados sintéticos (rodar_testes.py; histórico em resultados/testes_unitarios)
└── vocabularios/   vocabulários ORB do fechamento de laço
figuras/            figuras da dissertação (TikZ) e figuras de referência
resultados/         saídas (fora do git; ver resultados/README.md)
```

## Dependências

Python 3.10, [GTSAM](https://github.com/borglab/gtsam) 4.2, OpenCV, NumPy, SciPy, Matplotlib, NetworkX, Pillow e,
opcionalmente, [Open3D](https://www.open3d.org) (visualização 3D).

## Referências principais

- C. Forster et al., *On-Manifold Preintegration for Real-Time Visual-Inertial Odometry*, IEEE T-RO, 2017.
- M. Kaess et al., *iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree*, IJRR, 2012.
- D. Gálvez-López e J. D. Tardós, *Bags of Binary Words for Fast Place Recognition in Image Sequences*, IEEE T-RO, 2012.
- T. Qin, P. Li e S. Shen, *VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator*, IEEE T-RO, 2018.
- J. G. Mangelson et al., *Pairwise Consistent Measurement Set Maximization for Robust Multi-Robot Map Merging*, ICRA, 2018.
- H. Yang et al., *Graduated Non-Convexity for Robust Spatial Perception*, IEEE RA-L, 2020.
- A. Rosinol et al., *Kimera: an Open-Source Library for Real-Time Metric-Semantic Localization and Mapping*, ICRA, 2020.
- J. Jeong et al., *Complex Urban Dataset with Multi-Level Sensors from Highly Diverse Urban Environments*, IJRR, 2019.
