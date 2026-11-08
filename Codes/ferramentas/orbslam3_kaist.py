"""
Comparação com o estado da arte: ORB-SLAM3 MONOCULAR nas sequências do KAIST, mesmo protocolo de avaliação (evo).

Pré-requisito: ORB-SLAM3 compilado com o executável mono_kaist (~/sota/construir.sh; ver ~/sota/LEIA-ME.md).

Uso:
    python3 orbslam3_kaist.py rodar --images <seq>/stereo_left --out resultados/kaist/<seq>/orbslam3_r1
    python3 orbslam3_kaist.py resumo resultados/kaist --out resultados/sintese/orbslam3

Protocolo:
  - config da câmera gerada de <seq>/calibration/left.yaml (a mesma calibração que o nosso VO usa);
  - ORB-SLAM3 com 2000 ORB/frame (o valor dos exemplos KITTI), sem viewer;
  - ele cria um MAPA NOVO cada vez que se perde; o padrão (SaveTrajectoryEuRoC) é salvar o MAIOR mapa. O ATE é
    calculado só sobre o trecho coberto por esse mapa (Sim(3), como o nosso "só VO"), e a COBERTURA (% do percurso
    do GT) é sempre reportada junto -- ATE sem cobertura engana;
  - para comparar no MESMO trecho, também calcula o ATE do nosso VO puro (sem_realimentacao) e do sistema completo
    (completo_r1) restritos aos instantes cobertos pelo ORB-SLAM3.
"""
import argparse
import copy
import glob
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SOTA = os.environ.get("SOTA_DIR", os.path.expanduser("~/sota"))
ORB = os.path.join(SOTA, "ORB_SLAM3")
EXE = os.path.join(ORB, "Examples", "Monocular", "mono_kaist")
VOC = os.path.join(ORB, "Vocabulary", "ORBvoc.txt")
LIBS = [os.path.join(SOTA, "local", "lib"), os.path.join(ORB, "lib"), os.path.join(ORB, "Thirdparty", "DBoW2", "lib"),
        os.path.join(ORB, "Thirdparty", "g2o", "lib")]

YAML = """%YAML:1.0
File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: {fx:.6f}
Camera1.fy: {fy:.6f}
Camera1.cx: {cx:.6f}
Camera1.cy: {cy:.6f}
Camera1.k1: {k1:.8f}
Camera1.k2: {k2:.8f}
Camera1.p1: {p1:.8f}
Camera1.p2: {p2:.8f}
Camera1.k3: {k3:.8f}
Camera.width: {w}
Camera.height: {h}
Camera.fps: 10
Camera.RGB: 0
ORBextractor.nFeatures: 2000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.6
Viewer.KeyFrameLineWidth: 2.0
Viewer.GraphLineWidth: 1.0
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.7
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -100.0
Viewer.ViewpointZ: -0.1
Viewer.ViewpointF: 2000.0
"""


def disponivel():
    return os.path.isfile(EXE) and os.path.isfile(VOC)


def seq_root(images):
    d = os.path.abspath(images)
    while os.path.basename(d) in ("stereo_left", "image", ""):
        d = os.path.dirname(d)
    return d


def escrever_config(images, path):
    import cv2
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
    import rodar_vo as base                                    # mesma busca de calibração do nosso VO
    calib = base.load_calibration(images)
    if calib is None:
        raise SystemExit(f"sem calibration/left.yaml acima de {images}")
    K, dist, _ = calib
    dist = np.zeros(5) if dist is None else np.r_[dist, np.zeros(5)][:5]
    im = cv2.imread(sorted(glob.glob(os.path.join(images, "*.png")))[0], cv2.IMREAD_GRAYSCALE)
    with open(path, "w") as f:
        f.write(YAML.format(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], k1=dist[0], k2=dist[1], p1=dist[2],
                            p2=dist[3], k3=dist[4], w=im.shape[1], h=im.shape[0]))


def ler_euroc(path):
    """t_ns tx ty tz qx qy qz qw (SaveTrajectoryEuRoC) -> PoseTrajectory3D do evo (t em s)."""
    from evo.core.trajectory import PoseTrajectory3D
    d = np.loadtxt(path, ndmin=2)
    if not len(d):
        return None
    t = d[:, 0] * 1e-9 if d[0, 0] > 1e12 else d[:, 0]
    return PoseTrajectory3D(positions_xyz=d[:, 1:4], orientations_quat_wxyz=d[:, [7, 4, 5, 6]], timestamps=t)


def ate(ref_all, est, sim3, t_janela=None, max_dt=0.02):
    from evo.core import metrics, sync
    from evo.core.metrics import PoseRelation
    est = copy.deepcopy(est)
    if t_janela is not None:
        est.reduce_to_time_range(*t_janela)
    ref, est = sync.associate_trajectories(copy.deepcopy(ref_all), est, max_diff=max_dt)
    est.align(ref, correct_scale=sim3)
    ape = metrics.APE(PoseRelation.translation_part)
    ape.process_data((ref, est))
    ate.ultimo = (ref, est, ape.error)                         # p/ a figura
    return float(ape.get_statistic(metrics.StatisticsType.rmse)), ref


def comprimento(traj):
    return float(np.linalg.norm(np.diff(traj.positions_xyz, axis=0), axis=1).sum()) if traj.num_poses > 1 else 0.0


def avaliar(out, seq_dir):
    """metricas.csv da execução do ORB-SLAM3 + nossos métodos no mesmo trecho."""
    sys.path.insert(0, HERE)
    from avaliar_evo import read_tum
    gt_f = os.path.join(seq_dir, "completo_r1", "evo", "gt.txt")
    if not os.path.isfile(gt_f):
        raise SystemExit(f"falta {gt_f} (rode o completo_r1 da bateria antes)")
    gt = read_tum(gt_f)
    L_total = comprimento(gt)
    est = ler_euroc(os.path.join(out, "orb_frames.txt"))
    tempos = np.loadtxt(os.path.join(out, "orb_tempos.txt"), ndmin=2)
    n_frames = len(tempos)
    linhas, curvas = [], {}
    if est is None or est.num_poses < 10:
        linhas.append(dict(metodo="ORB-SLAM3 mono (maior mapa)", ate_m=np.nan, cobertura_pct=0.0, frames_mapa=0,
                           frames_rastreados_pct=100 * np.mean(tempos[:, 2] == 2) if n_frames else 0.0))
    else:
        e, ref = ate(gt, est, sim3=True)
        curvas["ORB-SLAM3 mono (maior mapa, Sim(3))"] = ate.ultimo
        janela = (est.timestamps[0], est.timestamps[-1])
        linhas.append(dict(metodo="ORB-SLAM3 mono (maior mapa)", ate_m=e, cobertura_pct=100 * comprimento(ref) / L_total,
                           frames_mapa=est.num_poses, frames_rastreados_pct=100 * np.mean(tempos[:, 2] == 2),
                           t_ini=janela[0] - gt.timestamps[0], t_fim=janela[1] - gt.timestamps[0],
                           tempo_ms_mediana=float(np.median(tempos[:, 1]))))
        for nome, arq, sim3 in (("nosso VO puro (BA fixo), mesmo trecho", "sem_realimentacao/evo/vo.txt", True),
                                ("nosso sistema completo, mesmo trecho", "completo_r1/evo/fusao_suavizada.txt", False),
                                ("só roda, mesmo trecho", "completo_r1/evo/so_roda.txt", False)):
            f = os.path.join(seq_dir, arq)
            if os.path.isfile(f):
                e2, ref2 = ate(gt, read_tum(f), sim3, t_janela=janela)
                curvas[nome.replace(", mesmo trecho", "")] = ate.ultimo
                linhas.append(dict(metodo=nome, ate_m=e2, cobertura_pct=100 * comprimento(ref2) / L_total))
    pd.DataFrame(linhas).to_csv(os.path.join(out, "metricas.csv"), index=False)
    print(pd.DataFrame(linhas).to_string(index=False))
    if curvas:
        figura(out, gt, curvas, linhas, os.path.basename(os.path.normpath(seq_dir)))


def figura(out, gt, curvas, linhas, seq):
    """Trajetórias no trecho coberto pelo maior mapa do ORB-SLAM3 (sobre o GT inteiro) + erro ao longo do trecho."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cores = {"ORB-SLAM3": "#d62728", "nosso VO": "#ff7f0e", "nosso sistema": "#1f77b4", "só roda": "#7f7f7f"}
    cor = lambda n: next((c for k, c in cores.items() if n.startswith(k)), "k")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw=dict(width_ratios=[1.1, 1]))
    P0 = gt.positions_xyz[0]
    g = gt.positions_xyz - P0
    a1.plot(g[:, 0], g[:, 1], color="0.8", lw=4, label="GT (percurso inteiro)")
    ref = next(iter(curvas.values()))[0]
    r = ref.positions_xyz - P0
    a1.plot(r[:, 0], r[:, 1], color="k", lw=1.5, label="GT (trecho do maior mapa do ORB-SLAM3)")
    for nome, (ref_n, est_n, err) in curvas.items():
        e = est_n.positions_xyz - P0
        a1.plot(e[:, 0], e[:, 1], color=cor(nome), lw=1.2, label=f"{nome}: ATE {np.sqrt(np.mean(np.square(err))):.1f} m")
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ref_n.positions_xyz, axis=0), axis=1))]
        a2.plot(d, err, color=cor(nome), lw=1.2, label=nome)
    cob = linhas[0].get("cobertura_pct", np.nan)
    a1.set_title(f"{seq}: trajetórias no trecho coberto pelo ORB-SLAM3 ({cob:.0f}% do percurso)")
    a1.set_xlabel("x [m]")
    a1.set_ylabel("y [m]")
    a1.axis("equal")
    a1.legend(fontsize=8, loc="best")
    a1.grid(alpha=0.3)
    a2.set_title("Erro de posição no trecho (após alinhamento)")
    a2.set_xlabel("distância percorrida no trecho [m]")
    a2.set_ylabel("erro [m]")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"comparacao_orbslam3.{ext}"), dpi=150)
    plt.close(fig)


def rodar(a):
    if not disponivel():
        raise SystemExit(f"ORB-SLAM3 não compilado ({EXE}); ver ~/sota/LEIA-ME.md")
    a.out = os.path.abspath(a.out)                             # o executável roda com cwd = a.out
    os.makedirs(a.out, exist_ok=True)
    cfg = os.path.join(a.out, "kaist.yaml")
    escrever_config(a.images, cfg)
    env = dict(os.environ, LD_LIBRARY_PATH=":".join(LIBS + [os.environ.get("LD_LIBRARY_PATH", "")]))
    t0 = time.time()
    with open(os.path.join(a.out, "orbslam3.log"), "w") as log:
        rc = subprocess.call([EXE, VOC, cfg, os.path.abspath(a.images), os.path.join(a.out, "orb"), "0"],
                             stdout=log, stderr=subprocess.STDOUT, cwd=a.out, env=env)
    print(f"ORB-SLAM3 terminou (rc={rc}) em {(time.time() - t0) / 60:.1f} min")
    if rc != 0 and not os.path.isfile(os.path.join(a.out, "orb_frames.txt")):
        raise SystemExit(rc)
    seq_dir = a.seq_dir or os.path.dirname(os.path.abspath(a.out))
    avaliar(a.out, seq_dir)
    open(os.path.join(a.out, ".feito"), "w").close()


def resumo(a):
    rows = []
    for f in sorted(glob.glob(os.path.join(a.raiz, "*", "orbslam3_r*", "metricas.csv"))):
        d = pd.read_csv(f)
        d.insert(0, "execucao", os.path.basename(os.path.dirname(f)))
        d.insert(0, "seq", f.split(os.sep)[-3])
        rows.append(d)
    if not rows:
        print("nenhuma execução do ORB-SLAM3")
        return
    D = pd.concat(rows, ignore_index=True)
    os.makedirs(a.out, exist_ok=True)
    D.to_csv(os.path.join(a.out, "orbslam3.csv"), index=False)
    M = D.groupby(["seq", "metodo"], sort=False).agg(ate_med=("ate_m", "median"), ate_min=("ate_m", "min"),
                                                     ate_max=("ate_m", "max"), cob=("cobertura_pct", "median"),
                                                     n=("ate_m", "size")).reset_index()
    txt = ["ORB-SLAM3 monocular x nosso sistema (ATE [m], mediana [mín-máx] das execuções; cobertura = % do "
           "percurso do GT no maior mapa do ORB-SLAM3; nossos métodos avaliados no MESMO trecho)", ""]
    for seq, g in M.groupby("seq", sort=False):
        txt.append(seq)
        for _, r in g.iterrows():
            faixa = f" [{r.ate_min:.1f}-{r.ate_max:.1f}]" if r.n > 1 else ""
            txt.append(f"   {r.metodo:42s} {r.ate_med:8.1f}{faixa:16s} cobertura {r.cob:5.1f}%")
    s = "\n".join(txt)
    open(os.path.join(a.out, "tabela.txt"), "w").write(s + "\n")
    print(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rodar")
    r.add_argument("--images", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--seq-dir", default=None, help="resultados/kaist/<seq> (padrão: pasta-mãe de --out)")
    r.add_argument("--so-avaliar", action="store_true", help="não roda; só reavalia orb_frames.txt existente")
    s = sub.add_parser("resumo")
    s.add_argument("raiz")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "rodar" and a.so_avaliar:
        avaliar(a.out, a.seq_dir or os.path.dirname(os.path.abspath(a.out)))
    elif a.cmd == "rodar":
        rodar(a)
    else:
        resumo(a)


if __name__ == "__main__":
    main()
