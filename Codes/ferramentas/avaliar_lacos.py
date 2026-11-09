"""
Avaliação ISOLADA do detector de fechamento de laço (LoopClosure.LoopDetector) numa sequência, com o GT.

O VO fica de fora: os keyframes são escolhidos pelo GT (um a cada --kf-dist m rodados) e as poses dos KFs (usadas
só p/ triangular os ORB do candidato) são as do GT. Assim os números medem o DETECTOR, não a deriva do VO.

Para cada KF (consulta) mede três coisas:
    1. reconhecimento de lugar (BoW): ranking de todos os KFs elegíveis pelo score do DBoW2 -> recall@N e a curva
       precisão x recall do score do 1º colocado;
    2. verificação geométrica (sonda): o 1º colocado passa por F + triangulação + PnP com limiares relaxados e
       guarda os inliers -> curvas precisão x recall de min_f_inliers e min_pnp_inliers e erro da pose relativa;
    3. pipeline real (LoopConfig padrão: ilhas, consistência temporal, intervalo entre laços, verificação) ->
       funil por etapa, precisão e cobertura das revisitas, separadas por sentido (mesmo / oposto).

Verdade (GT): consulta tem revisita se há KF elegível a < --radius m; o sentido vem do yaw do KF mais próximo
(< 30°: mesmo sentido, > 150°: oposto). Candidato correto: a < --radius m da consulta no GT.

Uso:
    python3 avaliar_lacos.py --images /.../urban39/stereo_left            -> resultados/sintese/detector_lacos/urban39/
    python3 avaliar_lacos.py --resumo resultados/sintese/detector_lacos                   -> tabela e figura com todas
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import cv2
import matplotlib

CODES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CODES, "src"))
sys.path.insert(0, os.path.join(CODES, "scripts"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from LoopClosure import LoopDetector, LoopConfig, OrbVocabulary  # noqa: E402
from GPO import image_stamp_ns  # noqa: E402

VOCAB = os.path.join(CODES, "vocabularios", "orb_vocab_k10L5_indep.npz")   # treinado em urban25/33/34
RESULTS = os.path.join(os.path.dirname(CODES), "resultados", "sintese", "detector_lacos")
SAME_DEG, OPP_DEG = 30.0, 150.0
LABELS = {0: "sem revisita", 1: "mesmo sentido", 2: "sentido oposto", 3: "transversal"}
COL = dict(same="#2a78d6", opp="#eb6834", none="#c3c2b7", ok="#1baf7a", bad="#e34948", miss="#eda100",
           ink="#52514e", grid="#e4e3df", muted="#8a8984")
RECALL_N = (1, 2, 3, 5, 10, 15, 20, 25)
POSE_OK = (2.0, 2.0)
VOCAB_TRAIN = {"urban25", "urban33", "urban34"}   # sequências de treino do vocabulário padrão            # (°, m): laço com pose relativa "correta" (erro vs. GT abaixo disso)


# ====================================================================== dados
def dataset_dirs(images):
    d = os.path.abspath(images)
    while os.path.basename(d).lower() in ("stereo_left", "image", "images", ""):
        d = os.path.dirname(d)
    return d, os.path.basename(d)


def load_calib(root, max_up=4):
    d = os.path.abspath(root)                    # calibration/ pode ficar acima do GT (urban32: urban32/urban32/urban32)
    for _ in range(max_up):
        if os.path.isfile(os.path.join(d, "calibration", "left.yaml")):
            break
        d = os.path.dirname(d)
    root = d
    fs = cv2.FileStorage(os.path.join(root, "calibration", "left.yaml"), cv2.FILE_STORAGE_READ)
    K, dist = fs.getNode("camera_matrix").mat().astype(np.float64), fs.getNode("distortion_coefficients").mat()
    fs.release()
    T_vc = np.eye(4)
    with open(os.path.join(root, "calibration", "Vehicle2Stereo.txt")) as f:
        for line in f:
            if line.startswith("R:"):
                T_vc[:3, :3] = np.array(line.split()[1:10], float).reshape(3, 3)
            elif line.startswith("T:"):
                T_vc[:3, 3] = np.array(line.split()[1:4], float)
    return K, (None if dist is None else dist.reshape(-1)), T_vc


def load_gt(root):
    path = next(p for p in (os.path.join(root, "global_pose.csv"), os.path.join(root, "poses", "global_pose.csv"))
                if os.path.isfile(p))
    d = np.loadtxt(path, delimiter=",", ndmin=2)
    T = np.tile(np.eye(4), (len(d), 1, 1))
    T[:, :3, :4] = d[:, 1:13].reshape(-1, 3, 4)
    return d[:, 0] * 1e-9, T


def yaw(R):
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def ang_diff_deg(a, b):
    return np.degrees(np.abs(np.angle(np.exp(1j * (a - b)))))


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


# ====================================================================== execução numa sequência
def run(images, out, kf_dist, radius, vocab, max_kf):
    root, name = dataset_dirs(images)
    K, dist, T_vc = load_calib(root)
    t_gt, T_gt = load_gt(root)
    paths = sorted(glob.glob(os.path.join(images, "*.png")))
    t_img = np.array([image_stamp_ns(p) * 1e-9 for p in paths])
    ok = (t_img >= t_gt[0]) & (t_img <= t_gt[-1])
    paths, t_img = [p for p, o in zip(paths, ok) if o], t_img[ok]
    i = np.clip(np.searchsorted(t_gt, t_img), 0, len(t_gt) - 1)
    T_img = T_gt[i]                                                      # GT a 100 Hz: vizinho mais próximo basta
    s_img = np.r_[0, np.cumsum(np.linalg.norm(np.diff(T_img[:, :2, 3], axis=0), axis=1))]
    sel = np.unique(np.searchsorted(s_img, np.arange(0, s_img[-1], kf_dist)))   # 1 KF a cada kf_dist m rodados
    if max_kf:
        sel = sel[:max_kf]
    print(f"{name}: {len(paths)} imagens, {s_img[-1]:.0f} m -> {len(sel)} KFs (1 a cada {kf_dist:g} m)")

    voc = OrbVocabulary.load(vocab)
    cfg = LoopConfig()
    det = LoopDetector(voc, K, cfg)                                       # pipeline real (configuração padrão)
    probe = LoopDetector(voc, K, LoopConfig(min_f_inliers=8, min_pnp_inliers=4))   # sonda: sempre verifica o top-1
    probe.kfs = det.kfs                                                   # mesma lista (cache dos pontos 3D)
    maps = None

    kf_frame, kf_t, kf_s, kf_P, kf_yaw, kf_T = [], [], [], [], [], []
    rows = []
    t0 = time.time()
    for n, f in enumerate(sel):
        img = cv2.imread(paths[f], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if dist is not None:                                              # mesma remoção de distorção do VO
            if maps is None:
                maps = cv2.initUndistortRectifyMap(K, dist, None, K, img.shape[::-1], cv2.CV_16SC2)
            img = cv2.remap(img, *maps, interpolation=cv2.INTER_LINEAR)
        T_wc = T_img[f] @ T_vc
        n_kfs, n_log = len(det.kfs), len(det.stats['log'])
        res = det.add_keyframe(int(f), float(t_img[f]), int(f), img)
        if len(det.kfs) == n_kfs:                                         # < 50 ORB: nem entrou no banco
            continue
        det.set_pose(int(f), T_wc)
        q = det.kfs[-1]
        rec = det.stats['log'][-1] if len(det.stats['log']) > n_log else None
        idx_q = len(det.kfs) - 1
        kf_frame.append(f); kf_t.append(t_img[f]); kf_s.append(s_img[f]); kf_P.append(T_img[f][:3, 3])
        kf_yaw.append(yaw(T_img[f][:3, :3])); kf_T.append(T_wc)
        if rec is None:
            continue
        P, Y = np.array(kf_P), np.array(kf_yaw)
        # elegíveis: mesma regra do detector (tempo e nº de KFs)
        n_elig = min(int(np.searchsorted(kf_t[:-1], kf_t[-1] - cfg.min_time_gap, side='right')),
                     idx_q - cfg.min_kf_gap)
        r = dict(t=kf_t[-1] - t_img[0], s=kf_s[-1], x=P[-1, 0], y=P[-1, 1], n_elig=max(n_elig, 0),
                 label=0, d_near=np.nan, stage=rec['stage'], accepted=res is not None,
                 dt_orb=rec.get('dt_orb', np.nan), dt_query=rec.get('dt_query', np.nan), n_db=idx_q)
        if n_elig > 0:
            d = np.linalg.norm(P[:n_elig, :2] - P[-1, :2], axis=1)
            j = int(np.argmin(d))
            r['d_near'] = float(d[j])
            if d[j] < radius:
                near = d < radius
                dy = ang_diff_deg(Y[-1], Y[:n_elig][near])
                r['label'] = 1 if (dy < SAME_DEG).any() else 2 if (dy > OPP_DEG).any() else 3
            # 1. BoW: ranking dos elegíveis
            sc = det.db.query(q['bow'], n_elig)
            order = np.argsort(-sc)
            good = d[order] < radius
            r['rank'] = int(np.argmax(good)) + 1 if good.any() else 0
            c = int(order[0])
            r.update(score1=float(sc[c]), d1=float(d[c]), score_true=float(sc[d < radius].max()) if (
                d < radius).any() else np.nan)
            # 2. sonda geométrica no top-1
            probe._rec = {}
            out_p = probe._verify(q, c)
            pr = probe._rec
            r.update(n_match=pr.get('n_match', 0), n_f=pr.get('n_f', 0), n_3d=pr.get('n_3d', 0),
                     n_pnp=pr.get('n_pnp', 0))
            if out_p is not None:
                T_gt_cq = np.linalg.inv(kf_T[c]) @ T_wc
                E = np.linalg.inv(T_gt_cq) @ out_p['T_cq']
                r.update(rot_err=rot_angle_deg(E[:3, :3]), trans_err=float(np.linalg.norm(E[:3, 3])))
        # 3. pipeline real: candidato escolhido e erro da pose do laço aceito
        if np.isfinite(rec.get('c_t', np.nan)):
            jc = int(np.argmin(np.abs(np.array(kf_t[:-1]) - rec['c_t'])))
            r['d_pipe'] = float(np.linalg.norm(P[jc, :2] - P[-1, :2]))
            if res is not None:
                E = np.linalg.inv(np.linalg.inv(kf_T[jc]) @ T_wc) @ res['T_cq']
                r.update(pipe_rot_err=rot_angle_deg(E[:3, :3]), pipe_trans_err=float(np.linalg.norm(E[:3, 3])),
                         pipe_n_f=res['n_f'], pipe_n_pnp=res['n_pnp'])
        rows.append(r)
        if n % 500 == 0:
            print(f"  {n}/{len(sel)} KFs ({time.time() - t0:.0f} s)")
    print(f"  {len(rows)} consultas em {time.time() - t0:.0f} s")

    os.makedirs(os.path.join(out, "dados"), exist_ok=True)
    keys = ["t", "s", "x", "y", "n_db", "n_elig", "label", "d_near", "rank", "score1", "d1", "score_true", "n_match",
            "n_f", "n_3d", "n_pnp", "rot_err", "trans_err", "stage", "accepted", "d_pipe", "pipe_rot_err",
            "pipe_trans_err", "pipe_n_f", "pipe_n_pnp", "dt_orb", "dt_query"]
    with open(os.path.join(out, "dados", "consultas.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k, "") for k in keys])
    meta = dict(dataset=name, kf_dist=kf_dist, radius=radius, cfg=cfg, n_kf=len(kf_t), length=float(s_img[-1]))
    analyse(os.path.join(out, "dados", "consultas.csv"), out, meta)


# ====================================================================== métricas
def load_rows(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    num = lambda v: float(v) if v not in ("", "nan") else np.nan
    D = {k: np.array([num(r[k]) if k not in ("stage", "accepted") else r[k] for r in rows])
         for k in rows[0].keys()}
    D['accepted'] = D['accepted'] == "True"
    D['label'] = np.nan_to_num(D['label']).astype(int)
    return D


def pr_curve(score, correct, n_pos, thresholds):
    """Precisão e recall ao aceitar o top-1 quando score >= limiar (positivos = consultas com revisita)."""
    P, R = [], []
    for th in thresholds:
        acc = score >= th
        tp = int((acc & correct).sum())
        fp = int((acc & ~correct).sum())
        P.append(tp / (tp + fp) if tp + fp else 1.0)
        R.append(tp / n_pos if n_pos else 0.0)
    return np.array(P), np.array(R)


def metrics(D, radius, cfg):
    has = D['n_elig'] > 0
    pos = has & (D['label'] > 0)
    same, opp = D['label'] == 1, D['label'] == 2
    c1 = D['d1'] < radius                                                # top-1 do BoW correto
    m = dict(n_q=int(has.sum()), n_pos=int(pos.sum()), n_same=int((has & same).sum()), n_opp=int((has & opp).sum()))
    for N in RECALL_N:
        for nm, mk in (("", pos), ("_mesmo", has & same), ("_oposto", has & opp)):
            m[f"recall@{N}{nm}"] = float(np.mean((D['rank'][mk] >= 1) & (D['rank'][mk] <= N))) if mk.any() else np.nan
    # sonda geométrica com os limiares padrão
    acc_g = has & (D['n_f'] >= cfg.min_f_inliers) & (D['n_pnp'] >= cfg.min_pnp_inliers)
    tp = acc_g & c1
    m.update(sonda_precisao=float(tp.sum() / max(acc_g.sum(), 1)), sonda_recall=float((tp & pos).sum() / max(pos.sum(), 1)),
             sonda_recall_mesmo=float((tp & same).sum() / max((has & same).sum(), 1)),
             sonda_recall_oposto=float((tp & opp).sum() / max((has & opp).sum(), 1)))
    # pipeline real
    acc = D['accepted']
    okp = acc & (D['d_pipe'] < radius)
    # pela POSE: o fator que vai ao grafo é a pose relativa, não o "mesmo lugar" -> erro < POSE_OK vs. GT
    okpose = acc & (D['pipe_rot_err'] < POSE_OK[0]) & (D['pipe_trans_err'] < POSE_OK[1])
    m.update(laco_aceitos=int(acc.sum()), laco_corretos=int(okp.sum()),
             laco_precisao=float(okp.sum() / max(acc.sum(), 1)),
             laco_precisao_pose=float(okpose.sum() / max(acc.sum(), 1)),
             laco_rot_err_med=float(np.nanmedian(D['pipe_rot_err'][okp])) if okp.any() else np.nan,
             laco_trans_err_med=float(np.nanmedian(D['pipe_trans_err'][okp])) if okp.any() else np.nan,
             dt_consulta_ms=float(1000 * np.nanmean(D['dt_orb'] + D['dt_query'])))
    return m


def coverage(D, radius, cover=50.0):
    """Fração (em m) de cada tipo de revisita com um laço aceito correto do pipeline a até `cover` m."""
    good = D['s'][D['accepted'] & (D['d_pipe'] < radius)]
    covd = (np.abs(D['s'][:, None] - good[None, :]) <= cover).any(1) if len(good) else np.zeros(len(D['s']), bool)
    w = np.gradient(D['s']) if len(D['s']) > 1 else np.ones(len(D['s']))
    out = {}
    for c in (1, 2):
        mk = D['label'] == c
        out[c] = (float(w[mk].sum()), float(w[mk & covd].sum() / w[mk].sum()) if mk.any() else np.nan)
    return covd, out


# ====================================================================== figuras
def _style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": COL['muted'], "axes.labelcolor": COL['ink'],
        "xtick.color": COL['ink'], "ytick.color": COL['ink'], "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": COL['grid'], "grid.linewidth": 0.7, "axes.axisbelow": True,
        "legend.frameon": False, "pdf.fonttype": 42})


def _save(fig, path):
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=150)
    plt.close(fig)


def analyse(csv_path, out, meta):
    _style()
    D = load_rows(csv_path)
    name, radius, cfg = meta['dataset'], meta['radius'], meta['cfg']
    has = D['n_elig'] > 0
    pos = has & (D['label'] > 0)
    same, opp = has & (D['label'] == 1), has & (D['label'] == 2)
    c1 = D['d1'] < radius
    m = metrics(D, radius, cfg)
    covd, cov = coverage(D, radius)
    m.update(cobertura_mesmo=cov[1][1], cobertura_oposto=cov[2][1], revisita_mesmo_m=cov[1][0],
             revisita_oposto_m=cov[2][0], dataset=name, kfs=meta['n_kf'], percurso_m=meta['length'])
    with open(os.path.join(out, "dados", "metricas.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metrica", "valor"])
        for k, v in m.items():
            w.writerow([k, f"{v:.4f}" if isinstance(v, float) else v])

    print(f"\n{name}: {m['n_q']} consultas c/ KFs elegíveis, {m['n_pos']} com revisita no GT "
          f"({m['n_same']} mesmo sentido, {m['n_opp']} oposto)")
    print(f"  BoW recall@1/5/10 (mesmo sentido): {m['recall@1_mesmo']:.2f} / {m['recall@5_mesmo']:.2f} / "
          f"{m['recall@10_mesmo']:.2f} | sentido oposto: {m['recall@1_oposto']:.2f} / {m['recall@5_oposto']:.2f} / "
          f"{m['recall@10_oposto']:.2f}")
    print(f"  verificação (top-1, F >= {cfg.min_f_inliers}, PnP >= {cfg.min_pnp_inliers}): precisão "
          f"{m['sonda_precisao']:.3f}, recall {m['sonda_recall']:.2f} (mesmo {m['sonda_recall_mesmo']:.2f}, "
          f"oposto {m['sonda_recall_oposto']:.2f})")
    print(f"  pipeline: {m['laco_aceitos']} laços aceitos, {m['laco_corretos']} corretos (precisão "
          f"{m['laco_precisao']:.3f}; pela pose, erro < {POSE_OK[0]:g}° e {POSE_OK[1]:g} m: {m['laco_precisao_pose']:.3f}) | erro da pose: {m['laco_rot_err_med']:.2f}° / {m['laco_trans_err_med']:.2f} m "
          f"(medianas) | cobertura das revisitas: mesmo sentido {100 * cov[1][1]:.0f}% de {cov[1][0]:.0f} m, "
          f"oposto {100 * np.nan_to_num(cov[2][1]):.0f}% de {cov[2][0]:.0f} m")

    # ---- figura 1: mapa (onde há revisita e onde o detector achou)
    fig, (ax, at) = plt.subplots(2, 1, figsize=(10, 9.5), gridspec_kw=dict(height_ratios=[4, 1]))
    x, y = D['x'] - D['x'][0], D['y'] - D['y'][0]
    ax.plot(x, y, color=COL['none'], lw=1.2, label="trajetória (GT)", zorder=1)
    for c, key in ((1, 'same'), (2, 'opp')):
        mk = D['label'] == c
        ax.scatter(x[mk], y[mk], s=6, color=COL[key], lw=0, label=f"revisita: {LABELS[c]}", zorder=2)
    okp = D['accepted'] & (D['d_pipe'] < radius)
    badp = D['accepted'] & ~(D['d_pipe'] < radius)
    ax.scatter(x[okp], y[okp], s=34, facecolor='none', edgecolor=COL['ok'], lw=1.2, label="laço aceito correto", zorder=3)
    ax.scatter(x[badp], y[badp], s=50, marker='x', color=COL['bad'], lw=1.5, label=f"laço aceito a > {radius:g} m no GT", zorder=4)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set(xlabel="x [m]", ylabel="y [m]")
    ax.set_title(f"{name}: revisitas no GT (raio {radius:g} m) e laços aceitos pelo detector")
    ax.legend(loc='best')
    s = D['s'] / 1000
    for c, key in ((1, 'same'), (2, 'opp')):
        mk = D['label'] == c
        at.scatter(s[mk], np.full(mk.sum(), 3 - c), s=10, marker='|', color=COL[key])
        at.scatter(s[mk & covd], np.full((mk & covd).sum(), 3 - c - 0.3), s=10, marker='|', color=COL['ok'])
    at.scatter(s[okp], np.full(okp.sum(), -0.2), s=16, marker='|', color=COL['ok'])
    at.scatter(s[badp], np.full(badp.sum(), -0.2), s=30, marker='x', color=COL['bad'])
    at.set_yticks([2, 1, -0.2], [f"mesmo sentido ({100 * np.nan_to_num(cov[1][1]):.0f}% coberto)",
                                 f"sentido oposto ({100 * np.nan_to_num(cov[2][1]):.0f}% coberto)", "laços aceitos"])
    at.set_ylim(-0.8, 2.5)
    at.grid(axis='y', visible=False)
    at.set(xlabel="distância percorrida [km]")
    at.set_title("Linha do tempo: revisitas (cor) e trecho coberto por laço correto a até 50 m (verde, logo abaixo)")
    fig.tight_layout()
    _save(fig, os.path.join(out, "lacos_mapa"))

    # ---- figura 2: reconhecimento de lugar (BoW)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.5, 4))
    for mk, key, lab in ((same, 'same', LABELS[1]), (opp, 'opp', LABELS[2])):
        if mk.any():
            rec = [np.mean((D['rank'][mk] >= 1) & (D['rank'][mk] <= N)) for N in RECALL_N]
            a1.plot(RECALL_N, rec, color=COL[key], marker='o', ms=4, lw=2, label=f"{lab} ({mk.sum()} consultas)")
    a1.set(xlabel="N (candidatos do ranking)", ylabel="recall@N", ylim=(0, 1.02),
           title="Reconhecimento de lugar: lugar certo entre os N melhores")
    if a1.get_legend_handles_labels()[0]:
        a1.legend(loc='center right')
    sc = D['score1']
    bins = np.linspace(0, np.nanpercentile(sc[has], 99.5), 40)
    a2.hist(sc[has & c1], bins, color=COL['ok'], alpha=0.75, label="top-1 correto")
    a2.hist(sc[has & ~c1], bins, color=COL['bad'], alpha=0.55, label="top-1 errado")
    a2.axvline(cfg.min_score, color=COL['ink'], ls='--', lw=1, label=f"min_score = {cfg.min_score:g}")
    a2.set(xlabel="score BoW do 1º colocado", ylabel="consultas", title="Score do 1º colocado: correto x errado")
    a2.legend()
    th = np.unique(np.r_[np.nanpercentile(sc[has], np.linspace(0, 100, 200)), cfg.min_score])
    P, R = pr_curve(np.nan_to_num(sc[has]), c1[has], pos.sum(), th)
    a3.plot(R, P, color=COL['ink'], lw=2, label="aceitar o top-1 se score ≥ limiar")
    k = int(np.argmin(np.abs(th - cfg.min_score)))
    a3.scatter(R[k], P[k], s=40, color=COL['ink'], edgecolor='white', zorder=3, label=f"min_score = {cfg.min_score:g}")
    a3.set(xlabel="recall (consultas com revisita)", ylabel="precisão", xlim=(0, 1), ylim=(0, 1.02),
           title="Só o BoW, sem verificação geométrica")
    a3.legend(loc='lower left')
    fig.suptitle(f"{name}: reconhecimento de lugar por bag of words (DBoW2), {m['n_q']} consultas", y=1.0)
    fig.tight_layout()
    _save(fig, os.path.join(out, "lacos_bow"))

    # ---- figura 3: verificação geométrica (separabilidade e curvas PR dos limiares)
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4))
    for ax, key, thr, lab, tit in ((axs[0], 'n_f', cfg.min_f_inliers, "inliers da matriz fundamental",
                                    "Matriz fundamental (2D-2D): correto x errado"),
                                   (axs[1], 'n_pnp', cfg.min_pnp_inliers, "inliers do PnP",
                                    "PnP (3D-2D): correto x errado")):
        v = D[key]
        bins = np.r_[0, np.unique(np.round(np.geomspace(1, max(np.nanmax(v[has]), 2) + 1, 40)))]
        ax.hist(v[has & c1], bins, color=COL['ok'], alpha=0.75, label="top-1 correto")
        ax.hist(v[has & ~c1], bins, color=COL['bad'], alpha=0.55, label="top-1 errado")
        ax.axvline(thr, color=COL['ink'], ls='--', lw=1, label=f"limiar atual = {thr}")
        ax.set_xscale('symlog', linthresh=10)
        ax.set_yscale('log')
        ax.set_xlim(0, bins[-1])
        ax.set(xlabel=lab, ylabel="consultas", title=tit)
        ax.legend()
    a = axs[2]
    for key, other, oth, col, lab in (('n_f', 'n_pnp', cfg.min_pnp_inliers, COL['same'], "min_f_inliers"),
                                       ('n_pnp', 'n_f', cfg.min_f_inliers, COL['opp'], "min_pnp_inliers")):
        th = np.arange(0, int(np.nanmax(D[key][has])) + 2)
        base = has & (D[other] >= oth)
        P, R = pr_curve(np.where(base, D[key], -1), c1, pos.sum(), th)
        a.plot(R, P, color=col, lw=2, label=f"variando {lab}")
        cur = {"n_f": cfg.min_f_inliers, "n_pnp": cfg.min_pnp_inliers}[key]
        a.scatter(R[th == cur], P[th == cur], s=40, color=col, edgecolor='white', zorder=3)
    a.set(xlabel="recall (consultas com revisita)", ylabel="precisão", ylim=(min(0.8, a.get_ylim()[0]), 1.005),
          title="Top-1 verificado: limiares (ponto = configuração atual)")
    a.legend(loc='lower left')
    fig.suptitle(f"{name}: verificação geométrica do 1º colocado (F + triangulação + PnP)", y=1.0)
    fig.tight_layout()
    _save(fig, os.path.join(out, "lacos_verificacao"))

    # ---- figura 4: funil do pipeline real e erro da pose dos laços aceitos
    stages = ["sem_elegivel", "pouco_informativa", "sem_candidato", "inconsistente", "intervalo_laco", "falha_f",
              "falha_3d", "falha_pnp", "verificado"]
    stage_names = ["sem KF elegível", "imagem pouco informativa", "sem candidato (score)", "inconsistente no tempo",
                   "intervalo entre laços", "falhou na F", "falhou na triangulação", "falhou no PnP", "laço aceito"]
    groups = [(D['label'] == 0, "sem revisita", COL['none']), (D['label'] == 1, LABELS[1], COL['same']),
              (D['label'] == 2, LABELS[2], COL['opp'])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
    yy = np.arange(len(stages))[::-1]
    h = 0.27
    for k, (mk, lab, col) in enumerate(groups):
        cnt = np.array([(mk & (D['stage'] == st)).sum() for st in stages])
        a1.barh(yy + (1 - k) * h, cnt, height=h * 0.92, color=col, label=f"{lab} ({mk.sum()})")
        for yi, c in zip(yy + (1 - k) * h, cnt):
            if c:
                a1.text(c, yi, f" {c}", va='center', fontsize=7, color=COL['ink'])
    a1.set_yticks(yy, stage_names)
    a1.set_xscale('log')
    a1.set(xlabel="consultas (escala log)", title="Onde cada consulta parou no detector (pipeline real)")
    a1.grid(axis='y', visible=False)
    a1.legend(loc='lower right', title="revisita no GT")
    okp = D['accepted'] & (D['d_pipe'] < radius)
    if okp.any():
        for v, col, lab in ((D['pipe_rot_err'][okp], COL['same'], "rotação [°]"),
                            (D['pipe_trans_err'][okp], COL['opp'], "translação [m]")):
            v = np.sort(v[np.isfinite(v)])
            a2.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=2,
                    label=f"{lab}: mediana {np.median(v):.2f}, p90 {np.percentile(v, 90):.2f}")
        a2.set_xscale('log')
    a2.set(xlabel="erro da pose relativa do laço (vs. GT)", ylabel="fração dos laços aceitos corretos",
           ylim=(0, 1.02), title=f"Erro da pose dos {okp.sum()} laços aceitos corretos (CDF)")
    if a2.get_legend_handles_labels()[0]:
        a2.legend(loc='lower right')
    fig.suptitle(f"{name}: pipeline completo (LoopConfig padrão) | {m['laco_aceitos']} aceitos, precisão "
                 f"{m['laco_precisao']:.3f}", y=1.0)
    fig.tight_layout()
    _save(fig, os.path.join(out, "lacos_funil"))
    print(f"  figuras: {out}/lacos_mapa, lacos_bow, lacos_verificacao, lacos_funil (.pdf/.png)")


# ====================================================================== resumo de várias sequências
def resumo(pasta):
    _style()
    ms = []
    for p in sorted(glob.glob(os.path.join(pasta, "*", "dados", "metricas.csv"))):
        with open(p) as fh:
            d = {r['metrica']: r['valor'] for r in csv.DictReader(fh)}
        ms.append({k: (v if k == 'dataset' else float(v)) for k, v in d.items()})
    if not ms:
        raise SystemExit(f"nada em {pasta}")
    cols = ["dataset", "percurso_m", "kfs", "n_q", "n_same", "n_opp", "recall@1_mesmo", "recall@10_mesmo",
            "recall@1_oposto", "recall@10_oposto", "sonda_precisao", "sonda_recall_mesmo", "laco_aceitos",
            "laco_precisao", "laco_precisao_pose", "cobertura_mesmo", "cobertura_oposto", "laco_rot_err_med", "laco_trans_err_med",
            "dt_consulta_ms"]
    with open(os.path.join(pasta, "resumo_lacos.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for m in ms:
            w.writerow([m.get(c, "") if c == "dataset" else f"{m.get(c, np.nan):.4f}" for c in cols])
    print(f"{'dataset':9s} {'km':>5s} {'consultas':>9s} {'mesmo':>6s} {'oposto':>6s} | {'R@1 m':>5s} {'R@10 m':>6s} "
          f"{'R@1 o':>5s} | {'aceitos':>7s} {'prec.':>5s} {'p.pose':>6s} | {'cob. m':>6s} {'cob. o':>6s} | {'rot°':>5s} {'tr m':>5s}")
    for m in ms:
        print(f"{m['dataset']:9s} {m['percurso_m'] / 1000:5.1f} {m['n_q']:9.0f} {m['n_same']:6.0f} {m['n_opp']:6.0f} | "
              f"{m['recall@1_mesmo']:5.2f} {m['recall@10_mesmo']:6.2f} {np.nan_to_num(m['recall@1_oposto']):5.2f} | "
              f"{m['laco_aceitos']:7.0f} {m['laco_precisao']:5.3f} {m['laco_precisao_pose']:6.3f} | {100 * np.nan_to_num(m['cobertura_mesmo']):5.0f}% "
              f"{100 * np.nan_to_num(m['cobertura_oposto']):5.0f}% | {m['laco_rot_err_med']:5.2f} {m['laco_trans_err_med']:5.2f}")

    names = [m['dataset'] for m in ms]
    y = np.arange(len(ms))[::-1]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 0.55 * len(ms) + 2.4), sharey=True)
    for ax, keys, title, xl in (
            (axs[0], (("recall@1_mesmo", COL['same'], "mesmo sentido"), ("recall@1_oposto", COL['opp'], "sentido oposto")),
             "BoW: recall@1", "recall@1"),
            (axs[1], (("cobertura_mesmo", COL['same'], "mesmo sentido"), ("cobertura_oposto", COL['opp'], "sentido oposto")),
             "Pipeline: revisita coberta por laço correto", "fração do trecho de revisita"),
            (axs[2], (("laco_precisao", COL['ok'], f"lugar (< raio no GT)"),
                      ("laco_precisao_pose", COL['ink'], f"pose (erro < {POSE_OK[0]:g}° e {POSE_OK[1]:g} m)")),
             "Pipeline: precisão dos laços aceitos", "precisão")):
        hh = 0.38
        for k, (key, col, lab) in enumerate(keys):
            v = np.array([m.get(key, np.nan) for m in ms])
            if key.startswith("laco_precisao"):          # sem laço aceito: precisão indefinida
                v = np.where([m['laco_aceitos'] > 0 for m in ms], v, np.nan)
            off = (0.5 - k) * hh
            ax.barh(y + off, np.nan_to_num(v), height=hh * 0.9, color=col, label=lab)
            for yi, vi in zip(y + off, v):
                ax.text(np.nan_to_num(vi), yi, f" {vi:.2f}" if np.isfinite(vi) else " não há",
                        va='center', fontsize=7.5, color=COL['ink'] if np.isfinite(vi) else COL['muted'])
        ax.set(xlim=(0, 1.15), xlabel=xl, title=title)
        ax.grid(axis='y', visible=False)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)
    axs[0].set_yticks(y, [f"{n}*" if n in VOCAB_TRAIN else n for n in names])
    fig.suptitle("Detector de fechamento de laço avaliado isoladamente (KFs e poses do GT)"
                 + (" | * vocabulário treinado nesta sequência" if VOCAB_TRAIN & set(names) else ""), y=1.0)
    fig.tight_layout()
    _save(fig, os.path.join(pasta, "resumo_lacos"))
    print(f"\n{pasta}/resumo_lacos.csv, resumo_lacos.pdf/.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", help="pasta das imagens (stereo_left) de uma sequência do KAIST")
    ap.add_argument("--out", default=None, help="padrão: resultados/sintese/detector_lacos/<dataset>")
    ap.add_argument("--kf-dist", type=float, default=2.0, help="m rodados entre KFs (VO: ~2 m/KF no KAIST)")
    ap.add_argument("--radius", type=float, default=10.0, help="m: mesmo lugar no GT")
    ap.add_argument("--vocab", default=VOCAB)
    ap.add_argument("--max-kf", type=int, default=0)
    ap.add_argument("--resumo", default=None, help="pasta com várias sequências já avaliadas")
    ap.add_argument("--so-figuras", action="store_true", help="refaz só as figuras a partir de dados/consultas.csv")
    args = ap.parse_args()
    if args.resumo:
        return resumo(args.resumo)
    if not args.images:
        ap.error("--images ou --resumo")
    _, name = dataset_dirs(args.images)
    out = args.out or os.path.join(RESULTS, name)
    if args.so_figuras:
        root, _ = dataset_dirs(args.images)
        D = load_rows(os.path.join(out, "dados", "consultas.csv"))
        return analyse(os.path.join(out, "dados", "consultas.csv"), out,
                       dict(dataset=name, kf_dist=args.kf_dist, radius=args.radius, cfg=LoopConfig(),
                            n_kf=len(D['t']), length=float(np.nanmax(D['s']))))
    run(args.images, out, args.kf_dist, args.radius, args.vocab, args.max_kf)


if __name__ == "__main__":
    main()
