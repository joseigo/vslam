# Odometria visual monocular + fusão em grafo (dissertação)

VO monocular com BA local, fusão em grafo de fatores (GTSAM/iSAM2) com roda (e IMU), escala do VO estimada no
grafo e realimentada no VO, e fechamento de laço (BoW + verificação geométrica + PCM/GNC).

## Estrutura

```
Dissert/
├── Codes/
│   ├── src/              biblioteca (importada pelos scripts)
│   │   ├── VisualOdometry.py  VO monocular (KLT + PnP, keyframes, BA local assíncrono, escala)
│   │   ├── Map.py  KeyFrame.py  BA.py
│   │   ├── GPO.py             grafo de poses (fatores VO/roda/IMU/laço, iSAM2, LM, GNC) + sensores
│   │   ├── Fusion.py          fusão online (front-end no processo principal, grafo em outro processo)
│   │   ├── LoopClosure.py     detector de laço (vocabulário ORB, consistência, F, PnP) + treino do vocabulário
│   │   └── Viewer.py          janelas ao vivo (OpenCV, matplotlib, Open3D)
│   ├── scripts/          o que se roda
│   │   ├── test_vo_fusion_rt.py    pipeline completo (VO + fusão + laço + avaliação + figuras)
│   │   ├── testE_vo_time_satmp.py  só o VO; também guarda a CONFIGURAÇÃO (câmera, dataset_path, parâmetros)
│   │   └── test_gpo.py             fusão offline a partir de uma trajetória do VO já salva
│   ├── ferramentas/      análise depois de rodar
│   │   ├── make_gifs.py            GIFs p/ apresentação a partir de resultados/<dataset>
│   │   └── view_map.py             navegador 3D dos mapas salvos (mapa_vo.npz / mapa_fundido.npz)
│   └── vocabularios/     vocabulários ORB do fechamento de laço
│       ├── orb_vocab_k10L5_indep.npz   PADRÃO: treinado em urban25/33/34 (independente das seqs. de teste)
│       └── orb_vocab_k10L5.npz         origem não registrada -- não usar em resultados
└── resultados/           saídas (fora do código; ignorado pelo git)
    ├── <dataset>/        uma pasta por dataset (nome tirado da pasta das imagens)
    │   ├── fusao_*.pdf   figuras (trajetória, erros, RPE, laços, funil, revisitas, processamento, mapas)
    │   ├── dados/        CSVs das figuras, trajetórias TUM, mapas (.npz/.ply), execucao.txt (configuração)
    │   ├── gifs/         make_gifs.py
    │   └── vo_puro/      saídas do testE_vo_time_satmp.py
    └── _avulsos/         arquivos antigos que estavam soltos na raiz
```

Os datasets ficam FORA do projeto (`~/Documentos/TCC/Datasets/urbanXX`, `~/Documentos/TCC/dataset_inomi/...`).

## Como rodar (a partir de `Dissert/`)

```bash
# pipeline completo no dataset de testE_vo_time_satmp.py (última atribuição de dataset_path)
python3 Codes/scripts/test_vo_fusion_rt.py
python3 Codes/scripts/test_vo_fusion_rt.py --no-gui            # sem janelas (mais rápido; use p/ medir tempo)
python3 Codes/scripts/test_vo_fusion_rt.py --images ~/Documentos/TCC/Datasets/urban26/stereo_left
python3 Codes/scripts/test_vo_fusion_rt.py --no-loop           # ablação: sem fechamento de laço
python3 Codes/scripts/test_vo_fusion_rt.py --no-feedback       # ablação: VO sem a escala do grafo
python3 Codes/scripts/test_vo_fusion_rt.py --out resultados/urban38_teste   # outra pasta (não sobrescreve)

# só o VO
python3 Codes/scripts/testE_vo_time_satmp.py

# depois de rodar
python3 Codes/ferramentas/make_gifs.py resultados/urban38
python3 Codes/ferramentas/view_map.py resultados/urban38/dados/mapa_fundido.npz

# treinar um vocabulário (sequências que NÃO serão usadas no teste)
python3 Codes/src/LoopClosure.py train --images <pasta1> <pasta2> --out Codes/vocabularios/novo.npz
```

Janelas ao vivo: `SHOW_WINDOW`, `SHOW_3D`, `SHOW_TRAJ` no topo de `test_vo_fusion_rt.py`.

## Datasets suportados

- **KAIST Complex Urban**: imagens com o tempo (ns) no nome, `data_sensor/encoder.csv` (ticks),
  `global_pose.csv` (GT). Câmera `urban`.
- **Inomi** (robô): imagens `AAAAMMDD_HHMMSS_micro.jpg` em horário local (**UTC-3**, `INOMI_UTC_OFFSET_H` em
  `GPO.py`), odometria como pose (`odometria_dataset.csv` ou `data_sensor/encoder.csv` com x,y,theta), GT
  `poses_gt_unix.csv` (~2.7 Hz, interpolado; eixo "frente" = y, corrigido). Câmera `inomi` (escolhida
  automaticamente); escala inicial estimada com 0.5 m de roda (`--init-dist`).

## Dependências

Python 3.10, `gtsam` 4.2, `opencv-python`, `numpy`, `scipy`, `matplotlib`, `networkx`, `pillow` (GIFs),
`open3d` (opcional: janelas 3D e view_map).
