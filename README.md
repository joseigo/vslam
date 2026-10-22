# Odometria visual monocular com fusão em grafo de fatores e fechamento de laço

Código da dissertação de mestrado de José Igo. O sistema estima a trajetória de um veículo a partir de **uma única
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
  de deriva, PCM (consistência par a par, online) e GNC-TLS (no lote final).
- **Tempo real**: VO, BA, grafo e detector de laço rodam em processos separados. O custo do grafo foi controlado
  agrupando os fatores do VO (o `CustomFactor` do GTSAM em Python copia o grafo inteiro a cada avaliação, o que
  tornava o custo quadrático com o tamanho do mapa).

## Resultados (exemplo)

KAIST Complex Urban, sequência **urban38** (11,2 km, 36 min, câmera + roda), uma execução:

| Método | ATE [m] | Deriva final [% do percurso] |
|---|---|---|
| Só odometria de roda | 241,6 | 4,4 |
| VO monocular (escala da fusão, alinhado em Sim3) | 202,4 | 1,0 |
| Fusão online (causal) | 47,4 | 0,4 |
| **Fusão suavizada (grafo + laços)** | **9,6** | **0,3** |

- 99% dos laços aceitos corretos (distância medida vs. GT); erro mediano da medida do laço: 0,25 m.
- Processamento: 0,28 × o tempo real (36 FPS para uma câmera de 10 FPS); grafo ~21 ms por nó.
- Como o VO não é determinístico, os números variam entre execuções; a avaliação da dissertação usa várias
  execuções por configuração.

**Limitações observadas**: revisitas em **sentido oposto** não são reconhecidas (a câmera frontal vê outra cena);
paradas longas podem gerar "laços" entre o antes e o depois da parada (geometricamente corretos, mas sem
informação nova); a estimativa online deriva entre laços.

## Avaliação

Cada execução gera métricas e figuras contra o GT: ATE e RMSE ancorado, erro ao longo do tempo e da distância,
RPE por comprimento de segmento (protocolo KITTI), precisão dos laços, **funil do detector de laço** (onde cada
consulta para e se o candidato era verdadeiro), **cobertura das revisitas** (mesmo sentido / sentido oposto, pelo
GT), **tempo de processamento** por componente (orçamento de tempo real, crescimento com o mapa, atraso do grafo),
mapas 3D (no referencial do VO e no fundido, em metros) e GIFs para apresentação.

## Dados

- **[KAIST Complex Urban Dataset](https://sites.google.com/view/complex-urban-dataset)**: sequências urbanas de
  vários km (câmera estéreo, da qual se usa só a esquerda; encoders de roda; IMU; GT por SLAM com LiDAR/GNSS).
  O vocabulário do fechamento de laço foi treinado em sequências **separadas** das de teste (urban25/33/34).
- **Robô móvel (Inomi)**: trajetos curtos de robô terrestre, com odometria e GT.

Os datasets não fazem parte deste repositório.

## Organização do repositório

```
Codes/
├── src/            biblioteca
│   ├── VisualOdometry.py, Map.py, KeyFrame.py, BA.py   odometria visual monocular
│   ├── GPO.py, Fusion.py                               grafo de fatores e fusão online
│   ├── LoopClosure.py                                  fechamento de laço e treino do vocabulário
│   └── Viewer.py                                       visualização ao vivo
├── scripts/        execução
│   ├── test_vo_fusion_rt.py      pipeline completo (VO + fusão + laço + avaliação + figuras)
│   ├── testE_vo_time_satmp.py    só a odometria visual; configuração de câmera e parâmetros
│   └── test_gpo.py               fusão offline a partir de uma trajetória do VO salva
├── ferramentas/    análise
│   ├── make_gifs.py              GIFs para apresentação a partir de uma execução
│   └── view_map.py               visualizador 3D dos mapas salvos
└── vocabularios/   vocabulários ORB do fechamento de laço
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
