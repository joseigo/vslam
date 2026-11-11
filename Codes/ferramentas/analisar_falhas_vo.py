#!/usr/bin/env python3
"""Análise dos casos de falha do VO monocular puro: onde e por quê.

Divide a trajetória do VO puro (execução `sem_realimentacao`, alinhada por Sim(3) global) em janelas de tempo e mede,
em cada uma, contra o GT:
    escala local   = |deslocamento do VO| / |deslocamento do GT|      (1 = certo; o Sim(3) global fixa a média)
    erro de rumo   = variação do erro de yaw dentro da janela [graus]  (deriva de rotação)
    direção        = ângulo entre o deslocamento do VO e o do GT [graus]
e cruza com o que pode explicar a falha: velocidade e taxa de curva (do GT), landmarks rastreados, frames com o VO
perdido e qualidade da imagem (brilho, contraste, nitidez, pontos ORB, fração de céu/saturada).

Uso:
    python3 Codes/ferramentas/analisar_falhas_vo.py resultados/kaist --out resultados/sintese/falhas_vo
    python3 Codes/ferramentas/analisar_falhas_vo.py resultados/kaist --seqs urban38 urban39 --janela 10
Saída: falhas_vo.csv (uma linha por janela), resumo.txt (por sequência e correlações), piores_<seq>.png (imagens das
piores janelas) e falhas_vo_<seq>.png (mapa colorido pelo erro + séries).
"""
import argparse
import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATASETS = "/home/joseigo/Documentos/TCC/Datasets"
ORB = cv2.ORB_create(nfeatures=2000)


def pasta_imagens(seq):
    for dirpath, _, _ in os.walk(os.path.join(DATASETS, seq)):
        if os.path.basename(dirpath) == "stereo_left" and glob.glob(os.path.join(dirpath, "*.png")):
            return dirpath
    return None


def qualidade(img_path):
    im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return {}
    if im.ndim == 3:                                            # KAIST já vem em cinza
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h = im.shape[0]
    sup = im[: h // 3]
    return dict(brilho=float(im.mean()), contraste=float(im.std()),
                nitidez=float(cv2.Laplacian(im, cv2.CV_64F).var()),
                orb=len(ORB.detect(im, None)),
                saturada_pct=float(100 * (im >= 250).mean()),
                ceu_pct=float(100 * (sup >= 200).mean()))


def imagem_vis(img_path):
    im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB if im.ndim == 2 else cv2.COLOR_BGR2RGB)
    return im


def ang(a):
    return (a + 180) % 360 - 180


def analisar(raiz, seq, janela, amostras_img):
    d = os.path.join(raiz, seq, "sem_realimentacao", "dados")
    vo_csv = glob.glob(os.path.join(d, "traj_vo_monocular_*.csv"))
    if not vo_csv:
        return None
    T = pd.read_csv(vo_csv[0])
    F = pd.read_csv(os.path.join(d, "tempos_frames.csv"))
    perd = pd.read_csv(os.path.join(d, "vo_perdido.csv")).t.values if os.path.isfile(
        os.path.join(d, "vo_perdido.csv")) else np.array([])
    stamps = np.loadtxt(os.path.join(d, "vo_tum.txt"), usecols=0, dtype=np.int64)
    t_img = (stamps - stamps[0]) / 1e9
    imgs = pasta_imagens(seq)

    t = T.t.values
    p = T[["x", "y"]].values
    g = T[["gt_x", "gt_y"]].values
    ey = T.erro_yaw_deg.values
    linhas = []
    for t0 in np.arange(t[0], t[-1] - janela, janela):
        i0, i1 = np.searchsorted(t, [t0, t0 + janela])
        i1 = min(i1, len(t) - 1)
        if i1 - i0 < 5:
            continue
        dg = g[i1] - g[i0]
        dv = p[i1] - p[i0]
        lg = np.linalg.norm(np.diff(g[i0:i1 + 1], axis=0), axis=1).sum()
        lv = np.linalg.norm(np.diff(p[i0:i1 + 1], axis=0), axis=1).sum()
        rumo_gt = np.degrees(np.arctan2(np.diff(g[i0:i1 + 1, 1]), np.diff(g[i0:i1 + 1, 0])))
        mov = np.linalg.norm(np.diff(g[i0:i1 + 1], axis=0), axis=1) > 0.05
        curva = np.abs(ang(np.diff(rumo_gt[mov]))).sum() if mov.sum() > 2 else 0.0
        fr = F[(F.t >= t0) & (F.t < t0 + janela)]
        linhas.append(dict(
            seq=seq, t0=t0, vel_kmh=3.6 * lg / janela, percurso_gt=lg,
            escala_local=lv / lg if lg > 2 else np.nan,
            erro_rumo_deg=abs(ang(ey[i1] - ey[i0])),
            erro_direcao_deg=abs(ang(np.degrees(np.arctan2(dv[1], dv[0]) - np.arctan2(dg[1], dg[0])))) if lg > 2
            else np.nan,
            erro_rel_pct=100 * np.linalg.norm(dv - dg) / lg if lg > 2 else np.nan,
            curva_deg=curva, landmarks=fr.n_landmarks.median() if len(fr) else np.nan,
            perdido_pct=100 * ((perd >= t0) & (perd < t0 + janela)).sum() / max(len(fr), 1),
            x=g[i0, 0], y=g[i0, 1]))
    J = pd.DataFrame(linhas)
    # qualidade da imagem: amostra no meio de cada janela (todas, é barato com 1 imagem por janela)
    if imgs:
        Q = []
        for t0 in J.t0:
            k = int(np.clip(np.searchsorted(t_img, t0 + janela / 2), 0, len(stamps) - 1))
            q = qualidade(os.path.join(imgs, f"{stamps[k]}.png"))
            q["img"] = f"{stamps[k]}.png"
            Q.append(q)
        J = pd.concat([J, pd.DataFrame(Q)], axis=1)
    J["falha"] = (J.erro_rel_pct > 30) | (J.perdido_pct > 20)
    return J, imgs


def figuras(J, seq, imgs, out, n_piores):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    v = J.erro_rel_pct.clip(0, 100)
    sc = ax[0].scatter(J.x, J.y, c=v, cmap="viridis_r", s=14)
    ax[0].set_aspect("equal"); ax[0].set_title(f"{seq}: erro relativo do VO por janela [%] (GT)")
    plt.colorbar(sc, ax=ax[0], fraction=0.04)
    ax[1].plot(J.t0, J.erro_rel_pct.clip(0, 150), lw=1, label="erro relativo [%]")
    ax[1].plot(J.t0, J.vel_kmh, lw=1, label="velocidade [km/h]")
    ax[1].plot(J.t0, J.landmarks / 10, lw=1, label="landmarks / 10")
    if "ceu_pct" in J:
        ax[1].plot(J.t0, J.ceu_pct, lw=1, label="céu/saturado no terço superior [%]")
    ax[1].set_xlabel("t [s]"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"falhas_vo_{seq}.png"), dpi=110); plt.close(fig)
    if not imgs:
        return
    P = J[J.percurso_gt > 20].nlargest(n_piores, "erro_rel_pct")
    B = J[J.percurso_gt > 20].nsmallest(2, "erro_rel_pct")
    sel = list(P.itertuples()) + list(B.itertuples())
    fig, axs = plt.subplots(int(np.ceil(len(sel) / 2)), 2, figsize=(12, 3.3 * np.ceil(len(sel) / 2)))
    for a, r in zip(axs.flat, sel):
        a.imshow(imagem_vis(os.path.join(imgs, r.img)))
        tag = "PIOR" if r in P.itertuples() else "boa"
        a.set_title(f"{tag} t={r.t0:.0f}s erro {r.erro_rel_pct:.0f}% escala {r.escala_local:.2f} "
                    f"{r.vel_kmh:.0f}km/h lm {r.landmarks:.0f} orb {r.orb}", fontsize=8)
        a.axis("off")
    for a in list(axs.flat)[len(sel):]:
        a.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out, f"piores_{seq}.png"), dpi=90); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raiz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--janela", type=float, default=10.0)
    ap.add_argument("--piores", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seqs = a.seqs or sorted(os.path.basename(s) for s in glob.glob(os.path.join(a.raiz, "urban*")))
    todas = []
    for s in seqs:
        r = analisar(a.raiz, s, a.janela, a.piores)
        if r is None:
            continue
        J, imgs = r
        figuras(J, s, imgs, a.out, a.piores)
        todas.append(J)
        print(f"{s}: {len(J)} janelas", flush=True)
    D = pd.concat(todas, ignore_index=True)
    D.to_csv(os.path.join(a.out, "falhas_vo.csv"), index=False)
    M = D[D.percurso_gt > 20]
    txt = [f"Janelas de {a.janela:.0f} s com > 20 m de percurso (falha = erro relativo > 30% ou VO perdido > 20%)", ""]
    res = M.groupby("seq").agg(janelas=("t0", "size"), erro_mediano=("erro_rel_pct", "median"),
                              falha_pct=("falha", lambda x: 100 * x.mean()),
                              escala_p10=("escala_local", lambda x: x.quantile(.1)),
                              escala_p90=("escala_local", lambda x: x.quantile(.9)),
                              rumo_mediano=("erro_rumo_deg", "median"), vel=("vel_kmh", "median"),
                              landmarks=("landmarks", "median"), ceu=("ceu_pct", "median"), orb=("orb", "median"),
                              perdido=("perdido_pct", "mean"))
    txt.append(res.round(2).to_string())
    txt += ["", "Correlação de Spearman com o erro relativo (todas as janelas):"]
    cols = [c for c in ["vel_kmh", "curva_deg", "landmarks", "perdido_pct", "brilho", "contraste", "nitidez", "orb",
                        "saturada_pct", "ceu_pct"] if c in M]
    txt.append(M[cols + ["erro_rel_pct"]].corr(method="spearman")["erro_rel_pct"].drop("erro_rel_pct").round(2)
               .to_string())
    txt += ["", "Janelas boas x com falha (medianas):"]
    txt.append(M.groupby("falha")[cols + ["escala_local", "erro_rumo_deg"]].median().round(2).T.to_string())
    s = "\n".join(txt)
    open(os.path.join(a.out, "resumo.txt"), "w").write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
