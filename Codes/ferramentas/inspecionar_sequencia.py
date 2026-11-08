"""
Confere uma sequência do KAIST Complex Urban recém-baixada antes de rodar o sistema: acha imagens / encoder / GT /
calibração (mesma busca do rodar_sistema.py), mede percurso, duração, revisitas no GT (mesmo sentido e oposto)
e fração do percurso em curva.

Uso:
    python3 inspecionar_sequencia.py /home/joseigo/Documentos/TCC/Datasets/urban28 [urban31 ...]
    python3 inspecionar_sequencia.py --todas            # todas as pastas urban* em Datasets/
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

DATASETS = "/home/joseigo/Documentos/TCC/Datasets"
RADIUS, MIN_TRAVEL, SAME_DEG, OPP_DEG = 10.0, 200.0, 30.0, 150.0     # mesmos critérios de REVISIT no script principal


def find(root, name, kind="f"):
    for dirpath, dirnames, filenames in os.walk(root):
        if (kind == "f" and name in filenames) or (kind == "d" and name in dirnames):
            return os.path.join(dirpath, name)
    return None


def revisits(gt_path):
    d = np.loadtxt(gt_path, delimiter=",")
    t = d[:, 0] * 1e-9
    P, yaw = d[:, [4, 8]], np.arctan2(d[:, 5], d[:, 1])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    g = np.arange(0, s[-1], 1.0)                                       # reamostra a cada 1 m de percurso
    Q = np.c_[np.interp(g, s, P[:, 0]), np.interp(g, s, P[:, 1])]
    Y = np.interp(g, s, np.unwrap(yaw))
    tree = cKDTree(Q)
    same = opp = 0
    for i, q in enumerate(Q):
        js = [j for j in tree.query_ball_point(q, RADIUS) if g[i] - g[j] > MIN_TRAVEL]
        if js:
            dy = np.degrees(np.abs(np.angle(np.exp(1j * (Y[i] - Y[js])))))
            same += bool((dy < SAME_DEG).any())
            opp += bool(not (dy < SAME_DEG).any() and (dy > OPP_DEG).any())
    curve = float((np.abs(np.degrees(Y[20:] - Y[:-20])) > 20).mean()) if len(Y) > 20 else 0.0
    return dict(km=g[-1] / 1000, min=(t[-1] - t[0]) / 60, same_km=same / 1000, opp_km=opp / 1000, curve=curve)


def inspect(root):
    name = os.path.basename(os.path.normpath(root))
    img = find(root, "stereo_left", "d")
    n_img = len(glob.glob(os.path.join(img, "*.png"))) if img else 0
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
    from rodar_sistema import find_dataset_files                # MESMA busca do script principal
    _, files = find_dataset_files(img) if img else (None, {})
    enc, gt = files.get("encoder"), files.get("gt")
    calib = {k: find(root, k) for k in ("Vehicle2Stereo.txt", "EncoderParameter.txt", "left.yaml")}
    ok = bool(img and n_img and enc and gt and all(calib.values()))
    print(f"\n== {name}: {'OK' if ok else 'INCOMPLETA'}")
    print(f"  imagens (stereo_left): {img or 'NÃO ACHADA'} ({n_img} png)")
    print(f"  encoder: {enc or 'NÃO ACHADO'}\n  GT:      {gt or 'NÃO ACHADO'}")
    for k, v in calib.items():
        print(f"  {k:22s} {'ok' if v else 'NÃO ACHADO'}")
    if gt:
        r = revisits(gt)
        print(f"  percurso {r['km']:.1f} km, {r['min']:.0f} min | revisita no mesmo sentido {r['same_km']:.2f} km, "
              f"oposto {r['opp_km']:.2f} km | {100 * r['curve']:.0f}% em curva")
    if img and n_img:
        print(f"  rodar: python3 Codes/scripts/rodar_sistema.py --images {img} --no-gui --out resultados/<pasta>")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pastas", nargs="*")
    ap.add_argument("--todas", action="store_true")
    a = ap.parse_args()
    pastas = sorted(glob.glob(os.path.join(DATASETS, "urban*"))) if a.todas else a.pastas
    if not pastas:
        ap.error("informe as pastas ou --todas")
    ok = [inspect(p) for p in pastas]
    sys.exit(0 if all(ok) else 1)
