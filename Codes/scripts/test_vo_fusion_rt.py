"""
VO + fusão de sensores em TEMPO REAL: a cada frame roda o VisualOdometry e, em paralelo (outro processo), um grafo
de poses (ISAM2) que funde o VO com os sensores selecionados. Mostra ao vivo a trajetória fundida (em metros).

Parâmetros da câmera / VO vêm de testE_vo_time_satmp.py (mesma configuração do teste do VO).

Uso:
    python3 test_vo_fusion_rt.py                                   # usa SENSORS / dataset_path abaixo
    python3 test_vo_fusion_rt.py --sensors vo,wheel,imu --sim-imu  # IMU simulada do GT (o dataset não tem IMU)
    python3 test_vo_fusion_rt.py --max-frames 2000 --no-gui
    python3 test_vo_fusion_rt.py --no-loop                         # sem fechamento de laço (p/ comparar)
    python3 test_vo_fusion_rt.py --no-feedback                     # VO sem a escala do grafo (VO puro, p/ comparar)
    python3 test_vo_fusion_rt.py --vo-view                         # 3D no mundo do VO (como antes), não no fundido

Um só referencial: a escala estimada no grafo volta p/ o VO (mapa em ~metros) e o mapa/trajetória do VO são levados
ao mundo fundido (Sim3 por nó do grafo, acompanha as correções e os laços) -> o 3D mostra o mapa já fundido.

Fechamento de laço (LoopClosure.py + GPO.add_loop): DBoW2 + verificação geométrica (VINS-Mono) -> fator 4-DoF com
escala no grafo; robustez: porta de deriva + PCM incremental (online) e GNC-TLS (lote final).
"""
import argparse
import glob
import os
import time

import testE_vo_time_satmp as base           # (configura BLAS c/ 1 thread antes do numpy)
import numpy as np
from VisualOdometry import VisualOdometry
from Viewer import FrameViewer, TrajViewer, Open3DViewer, plot_snapshot, draw_snapshot
from GPO import GPOConfig, WheelOdometry, PoseOdometry, ImuData, image_stamp_ns
from Fusion import OnlineFusion
from LoopClosure import LoopClosure, LoopConfig
from test_gpo import load_gt, simulate_imu, evaluate, save_tum

# ------------------------------------------------------------------ CONFIGURAÇÃO
dataset_path = base.dataset_path    # imagens (mesmo do testE_vo_time_satmp.py); a raiz do dataset é achada a partir dela

SENSORS = ["vo", "wheel"]           # opções: "vo", "wheel", "imu"  (ex.: ["vo", "imu"], ["vo", "wheel", "imu"])
NODE_DT = 0.5                       # s entre nós do grafo
TIME_OFFSETS = {"wheel": 0.0,       # s: relógio do sensor -> relógio da câmera (t_câmera = t_sensor + offset)
                "imu": 0.0}
IMU_R_VI = np.eye(3)                # rotação IMU -> veículo (KAIST: xsens ~ alinhada ao veículo)
SIM_IMU = False                     # sem xsens_imu.csv: simula a IMU a partir do GT (só p/ validar o código)
BACKEND_PROCESS = True              # grafo em outro processo (não trava o VO)
SCALE_FEEDBACK = True               # escala do grafo (m/unid.) realimenta o VO -> mapa do VO em ~metros
FUSED_VIEW = True                   # 3D / mapa no mundo FUNDIDO (métrico, veículo); False = mundo do VO

LOOP_CLOSURE = True                 # fechamento de laço
VOCAB_FILE = os.path.join(base.VOCAB_DIR, "orb_vocab_k10L5_indep.npz")   # treinado em urban25/33/34
LOOP_GT_TOL = 3.0                   # m: laço correto se |distância medida - distância no GT| < isso (só avaliação)
REVISIT = dict(radius=10.0,         # m: mesmo lugar no GT
               min_travel=200.0,    # m rodados entre as passagens (exclui paradas em semáforo)
               same_deg=30.0,       # |dyaw| < isso: mesmo sentido; > opp_deg: sentido oposto
               opp_deg=150.0,
               cover=50.0)          # m: trecho de revisita "coberto" se há laço aceito correto a até isso
RPE_LENGTHS = (100, 200, 300, 400, 500, 600, 700, 800)   # m: segmentos do RPE (protocolo KITTI)

SHOW_WINDOW = True                  # janela do OpenCV com os tracks (como no testE_vo_time_satmp.py)
SHOW_3D = base.SHOW_3D              # mapa 3D ao vivo no Open3D (1ª e 3ª pessoa), como no testE_vo_time_satmp.py
SHOW_TRAJ = True                    # janela matplotlib das trajetórias (fusão x VO x GT) ao vivo
PLOT_EVERY = 10                     # atualiza o plot das trajetórias a cada N frames
OUT_PREFIX = "fusao"
FIG_FORMAT = "pdf"                  # formato das figuras (pdf = vetorial p/ a dissertação; "png" também funciona)
RESULTS_DIR = base.RESULTS_DIR      # Dissert/resultados/<dataset>/(dados/)

# cores fixas por trajetória (mesma cor em todas as figuras)
VO_PURE, VO_FB = "VO puro (Sim3)", "VO c/ escala da fusão (Sim3)"
COLORS = {"GT": "#0b0b0b", VO_PURE: "#2a78d6", VO_FB: "#2a78d6", "fusão online": "#eb6834",
          "fusão suavizada": "#1baf7a", "só roda": "#eda100"}
LOOP_COLORS = {True: "#008300", False: "#e34948"}      # laço aceito no final / rejeitado
STYLES = {VO_PURE: "--", VO_FB: "--", "só roda": ":"}


def find_dataset_files(images_dir):
    """Procura encoder / GT / IMU subindo a partir da pasta das imagens (layouts do KAIST usados aqui)."""
    d = os.path.abspath(images_dir)
    for _ in range(4):
        cand = {k: None for k in ("encoder", "gt", "imu", "odom")}
        for sub in ("data_sensor", ""):
            for k, name in (("encoder", "encoder.csv"), ("imu", "xsens_imu.csv"),
                            ("odom", "odometria_dataset.csv")):          # Inomi: pose do robô (x, y, theta)
                p = os.path.join(d, sub, name)
                if cand[k] is None and os.path.isfile(p):
                    cand[k] = p
        if cand["encoder"] and "theta" in open(cand["encoder"]).readline():   # 'encoder.csv' com POSE (Inomi)
            cand["odom"], cand["encoder"] = cand["odom"] or cand["encoder"], None
        for p in (os.path.join(d, "poses", "global_pose.csv"), os.path.join(d, "global_pose.csv"),
                  os.path.join(d, "poses_gt_unix.csv"), os.path.join(d, "data_sensor", "poses_gt_unix.csv")):
            if cand["gt"] is None and os.path.isfile(p):
                cand["gt"] = p
        if any(cand.values()):
            return d, cand
        d = os.path.dirname(d)
    return None, dict(encoder=None, gt=None, imu=None, odom=None)


GT_BODY_YAW_DEG = {"inomi": 90.0}     # GT da Inomi: eixo 'frente' do corpo é o y (yaw -90° do movimento) -> gira p/ x


def adapt_gt(t_gt, T_gt, t_img, body_yaw_deg=0.0, max_dt=0.05):
    """GT no formato do resto do código: (1) convenção do corpo com x p/ frente; (2) GT esparso (Inomi ~2.7 Hz)
    interpolado nos instantes das imagens (posição linear, rotação slerp) -- a avaliação casa por tempo (tol. 20 ms)."""
    from scipy.spatial.transform import Rotation, Slerp
    T_gt = T_gt.copy()
    if body_yaw_deg:
        T_gt[:, :3, :3] = T_gt[:, :3, :3] @ Rotation.from_euler('z', body_yaw_deg, degrees=True).as_matrix()
    if len(t_gt) > 1 and np.median(np.diff(t_gt)) > max_dt:
        ti = np.asarray(t_img)
        ti = ti[(ti >= t_gt[0]) & (ti <= t_gt[-1])]
        if len(ti) > len(t_gt):
            T = np.tile(np.eye(4), (len(ti), 1, 1))
            T[:, :3, :3] = Slerp(t_gt, Rotation.from_matrix(T_gt[:, :3, :3]))(ti).as_matrix()
            T[:, :3, 3] = np.column_stack([np.interp(ti, t_gt, T_gt[:, i, 3]) for i in range(3)])
            print(f"  GT esparso ({1 / np.median(np.diff(t_gt)):.1f} Hz) interpolado em {len(ti)} instantes das imagens")
            return ti, T
    return t_gt, T_gt


class GtAligner:
    """GT só p/ VISUALIZAÇÃO ao vivo: alinha o GT à pose fundida no 1º instante em que ambos existem."""

    def __init__(self, t_gt, T_gt):
        self.t, self.T, self.A = t_gt, T_gt, None

    def at(self, t):
        i = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        return self.T[i] if abs(self.t[i] - t) < 0.05 else None

    def line(self, fused_t, fused):
        if self.A is None:
            for t, F in zip(fused_t, fused):
                G = self.at(t)
                if G is not None:
                    self.A = F @ np.linalg.inv(G)
                    break
            if self.A is None:
                return None
        m = (self.t >= fused_t[0]) & (self.t <= fused_t[-1])      # GT a partir do 1º instante fundido: mesmo zero
        P = (self.A[:3, :3] @ self.T[m, :3, 3].T).T + self.A[:3, 3]
        return P[::10, :2]


# ------------------------------------------------------------------ avaliação ancorada no início
def _yaw(R):
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def _match_gt(t, t_gt, tol=0.02):
    i = np.clip(np.searchsorted(t_gt, t), 1, len(t_gt) - 1)
    i = np.where(np.abs(t_gt[i - 1] - t) < np.abs(t_gt[i] - t), i - 1, i)
    return i, np.abs(t_gt[i] - t) < tol


def align_at_start(A, B, with_scale):
    """s R (A - A[0]) + B[0] ~ B com o 1º ponto COINCIDENTE (Umeyama centrado no início, não na média):
    todas as trajetórias saem do mesmo zero e o erro cresce a partir de 0 (mostra a deriva)."""
    A0, B0 = A - A[0], B - B[0]
    U, Sv, Vt = np.linalg.svd(B0.T @ A0)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(Sv) @ D) / max((A0 ** 2).sum(), 1e-12) if with_scale else 1.0
    return R, s


def describe(t, T, t_gt, T_gt, origin, with_scale):
    """Métricas + séries p/ os gráficos, num referencial local com origem em `origin` (GT no instante comum t0)."""
    t, T = np.asarray(t), np.asarray(T)
    i, ok = _match_gt(t, t_gt)
    t, T, G = t[ok], T[ok], T_gt[i[ok]]
    Pg = G[:, :3, 3] - origin
    R, s = align_at_start(T[:, :3, 3], Pg, with_scale)
    P = s * (T[:, :3, 3] - T[0, :3, 3]) @ R.T + Pg[0]
    err = np.linalg.norm(P - Pg, axis=1)
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Pg, axis=0), axis=1))]
    dyaw = np.degrees(np.angle(np.exp(1j * (_yaw(R @ T[:, :3, :3]) - _yaw(G[:, :3, :3])))))
    ume = evaluate(t, T[:, :3, 3], t_gt, T_gt, with_scale)          # ATE padrão (Umeyama na média), p/ referência
    return dict(t=t, P=P, G=Pg, Rw=R @ T[:, :3, :3], Rg=G[:, :3, :3], err=err, dist=dist, dyaw=dyaw, scale=s,
                ate=ume['ate'],
                ate0=float(np.sqrt(np.mean(err ** 2))), final=float(err[-1]),
                drift=100 * float(err[-1]) / max(dist[-1], 1e-9), length=float(dist[-1]))


def plot_results(rows, sensors, t0, scale_t, scale_s, vo_lost_t, out, show, loops=()):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    title = f"Fusão {'+'.join(sensors)}"
    G = max(rows.values(), key=lambda e: len(e['G']))      # GT mais denso (por frame) p/ desenhar a referência
    figs = []

    def style(name):
        return dict(color=COLORS.get(name), ls=STYLES.get(name, "-"), lw=1.6)

    # ---------- figura 1: trajetória (mesmo zero) + resumo das métricas
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax.plot(G['G'][:, 0], G['G'][:, 1], color=COLORS["GT"], lw=2.6, label="GT")
    for name, e in rows.items():
        ax.plot(e['P'][:, 0], e['P'][:, 1], label=f"{name}  (ATE {e['ate']:.1f} m)", **style(name))
        ax.plot(e['P'][-1, 0], e['P'][-1, 1], 'o', ms=8, color=COLORS.get(name), mec='white', mew=1.5)
    ref = rows.get("fusão suavizada")
    if ref is not None:
        for ok in (True, False):
            sel = [L for L in loops if L['final_ok'] == ok and L['t_a'] >= ref['t'][0]]
            for k, L in enumerate(sel):
                ia, ib = (int(np.argmin(np.abs(ref['t'] - L[key]))) for key in ('t_a', 't_b'))
                ax.plot(ref['P'][[ia, ib], 0], ref['P'][[ia, ib], 1], color=LOOP_COLORS[ok], lw=1.0,
                        ls='-' if ok else '--', marker='.', ms=5,
                        label=(f"laços {'aceitos' if ok else 'rejeitados'} ({len(sel)})" if k == 0 else None))
    ax.plot(0, 0, 'g^', ms=12, mec='black', label="início (0, 0)")
    ax.plot(G['G'][-1, 0], G['G'][-1, 1], 's', ms=9, color=COLORS["GT"], mec='white', label="fim (GT)")
    ax.axhline(0, color='0.6', lw=0.8); ax.axvline(0, color='0.6', lw=0.8)
    ax.set_aspect('equal', adjustable='datalim'); ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc='best')
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"{title}: trajetórias alinhadas no início (todas partem de 0,0)")

    names = list(rows)
    x = np.arange(len(names))
    w = 0.26
    for k, (key, lab) in enumerate((("ate", "ATE (Umeyama) [m]"), ("ate0", "RMSE ancorado [m]"),
                                    ("final", "erro final [m]"))):
        vals = [rows[n][key] for n in names]
        b = axb.bar(x + (k - 1) * w, vals, w * 0.92, label=lab, color=["#52514e", "#8a8984", "#c3c2b7"][k])
        axb.bar_label(b, fmt="%.1f", fontsize=8, padding=2)
    axb.set_xticks(x, [f"{n}\nderiva {rows[n]['drift']:.2f}%" for n in names], fontsize=9)
    axb.set_ylabel("m"); axb.grid(True, axis='y', alpha=0.3); axb.legend(fontsize=9)
    axb.set_title(f"Métricas (percurso do GT: {G['length']:.0f} m)")
    fig.tight_layout()
    figs.append(save_fig(fig, f"{out}_trajetoria"))

    # ---------- figura 2: erros e estados ao longo do tempo / da distância
    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    (a1, a2), (a3, a4), (a5, a6) = axs
    for name, e in rows.items():
        st = style(name)
        a1.plot(e['t'] - t0, e['err'], label=name, **st)
        a2.plot(e['dist'], e['err'], label=name, **st)
        a3.plot(e['t'] - t0, e['dyaw'], label=name, **st)
        a5.plot(e['t'] - t0, e['P'][:, 2], label=name, **st)
        a6.plot(np.sort(e['err']), np.linspace(0, 100, len(e['err'])), label=name, **st)
    a5.plot(G['t'] - t0, G['G'][:, 2], color=COLORS["GT"], lw=2.2, label="GT")
    for t_l in vo_lost_t:
        a1.axvline(t_l - t0, color='#e34948', alpha=0.15, lw=1)
    for k, L in enumerate([L for L in loops if L['final_ok']]):
        a1.axvline(L['t_b'] - t0, color=LOOP_COLORS[True], alpha=0.35, lw=1, ls=':',
                   label="laço aceito" if k == 0 else None)
    a1.set(xlabel="t [s]", ylabel="erro de posição [m]", title="Erro de translação ao longo do tempo"
           + (" (faixas vermelhas: VO perdido)" if len(vo_lost_t) else ""))
    a2.set(xlabel="distância percorrida (GT) [m]", ylabel="erro de posição [m]", title="Erro vs distância percorrida")
    a3.set(xlabel="t [s]", ylabel="erro de guinada [°]", title="Erro de orientação (yaw)")
    a5.set(xlabel="t [s]", ylabel="z [m]", title="Altura relativa ao início")
    a6.set(xlabel="erro de posição [m]", ylabel="% das amostras", title="Distribuição acumulada do erro (CDF)")
    if len(scale_t):
        a4.plot(np.asarray(scale_t) - t0, scale_s, color=COLORS["fusão suavizada"], lw=1.6, label="escala no grafo")
        if VO_PURE in rows:          # com realimentação o VO já está em ~metros: a escala Sim3 dele não é comparável
            a4.axhline(rows[VO_PURE]['scale'], color=COLORS[VO_PURE], ls='--', lw=1.4,
                       label=f"escala ótima Sim3 do VO = {rows[VO_PURE]['scale']:.3f}")
        a4.set(xlabel="t [s]", ylabel="m / unid. VO", title="Escala do VO estimada pelo grafo")
    else:
        a4.set_axis_off()
    for a in (a1, a2, a3, a4, a5, a6):
        if a.axison:
            a.grid(True, alpha=0.3)
            a.legend(fontsize=8)
    fig.suptitle(f"{title}: análise do erro (alinhamento ancorado em t0, zero comum)")
    fig.tight_layout()
    figs.append(save_fig(fig, f"{out}_erros"))

    # ---------- figura 3: diagnóstico dos laços (distância real entre os dois instantes, pelo GT)
    if loops:
        fig, (b1, b2) = plt.subplots(1, 2, figsize=(16, 5.5))
        marks = {'aceito': ('o', LOOP_COLORS[True]), 'rejeitado': ('X', LOOP_COLORS[False]),
                 'porta': ('s', '#8a8984'), 'outro': ('d', '#6250d6')}
        for key, (mk, col) in marks.items():
            sel = [L for L in loops if L['kind'] == key]
            if sel:
                b1.scatter([L['t_b'] - t0 for L in sel], [L['meas_err'] for L in sel], marker=mk, s=40, color=col,
                           edgecolor='white', lw=0.6, label=f"{key} ({len(sel)})")
                b2.scatter([L.get('dpos', np.nan) for L in sel], [L['meas_err'] for L in sel], marker=mk, s=40,
                           color=col, edgecolor='white', lw=0.6, label=f"{key} ({len(sel)})")
        for b in (b1, b2):
            b.axhline(LOOP_GT_TOL, color='0.4', ls='--', lw=1, label=f"tolerância ({LOOP_GT_TOL:.0f} m)")
            b.grid(True, alpha=0.3); b.legend(fontsize=8)
        b1.set(xlabel="t do KF atual [s]", ylabel="|dist. medida - dist. GT| [m]",
               title="Erro da medida de cada laço (corretos abaixo da linha)")
        b2.set(xlabel="desvio laço vs. estimativa no momento da detecção [m]",
               ylabel="|dist. medida - dist. GT| [m]", title="Deriva acumulada que cada laço corrige")
        fig.tight_layout()
        figs.append(save_fig(fig, f"{out}_lacos"))
    return plt, figs


def evaluate_loops(loops, t_gt, T_gt):
    """Classifica cada laço pelo GT (distância real entre os dois instantes) e imprime precisão por etapa."""
    out = []
    s_gt = np.r_[0, np.cumsum(np.linalg.norm(np.diff(T_gt[:, :2, 3], axis=0), axis=1))]
    for L in loops:
        (ia, ib), _ = _match_gt(np.array([L['t_a'], L['t_b']]), t_gt, tol=0.1)
        st = L['status']
        kind = 'aceito' if st.startswith('aceito') or st == 'ativo' else 'rejeitado' if (
            'rejeitado' in st or 'PCM' in st) else 'porta' if st == 'porta' else 'outro'
        gt_dist = float(np.linalg.norm(T_gt[ia, :3, 3] - T_gt[ib, :3, 3]))
        err = abs(L['z_dist'] - gt_dist) if 'z_dist' in L else np.nan
        travel = float(abs(s_gt[ib] - s_gt[ia]))         # m rodados entre as passagens: ~0 = parada, não revisita
        out.append(dict(L, gt_dist=gt_dist, meas_err=err, correct=bool(err < LOOP_GT_TOL), kind=kind,
                        final_ok=kind == 'aceito', gt_travel=travel, stop=travel < REVISIT['min_travel']))
    if not out:
        print("\nNenhum laço chegou ao grafo.")
        return out
    print(f"\nLaços (correto = distância medida pelo laço difere < {LOOP_GT_TOL:.0f} m da distância no GT):")
    print(f"{'resultado':12s} {'n':>4s} {'corretos':>11s} {'errados':>7s}")
    for kind in ('aceito', 'rejeitado', 'porta', 'outro'):
        sel = [L for L in out if L['kind'] == kind]
        if sel:
            tp = sum(L['correct'] for L in sel)
            print(f"{kind:12s} {len(sel):4d} {tp:11d} {len(sel) - tp:7d}")
    acc = [L for L in out if L['kind'] == 'aceito']
    if acc:
        print(f"Precisão dos laços aceitos: {100 * np.mean([L['correct'] for L in acc]):.0f}% | erro da distância "
              f"medida: mediana {np.nanmedian([L['meas_err'] for L in acc]):.2f} m")
        stops = [L for L in acc if L['stop']]
        if stops:
            print(f"  dos quais {len(stops)} sem rodar (< {REVISIT['min_travel']:.0f} m entre as passagens no GT: o carro "
                  f"parou, não é revisita): t = " + ", ".join(f"{L['t_b'] - t_gt[0]:.0f}" for L in stops) + " s")
    return out


# ------------------------------------------------------------------ revisitas (GT) e cobertura dos laços
def gt_revisits(t_gt, T_gt, radius, min_travel, same_deg, opp_deg, ds=1.0, **_):
    """Amostra o GT a cada `ds` m rodados e marca cada amostra como revisita no mesmo sentido (1), no sentido
    oposto (2) ou nenhuma (0): existe passagem anterior a < radius m, com >= min_travel m rodados entre elas."""
    from scipy.spatial import cKDTree
    s_all = np.r_[0, np.cumsum(np.linalg.norm(np.diff(T_gt[:, :2, 3], axis=0), axis=1))]
    idx = np.unique(np.searchsorted(s_all, np.arange(0, s_all[-1], ds)))
    p, s, yaw = T_gt[idx, :2, 3], s_all[idx], _yaw(T_gt[idx, :3, :3])
    cls = np.zeros(len(idx), int)
    for k, nb in enumerate(cKDTree(p).query_ball_point(p, radius)):
        nb = np.asarray(nb, int)
        nb = nb[s[nb] < s[k] - min_travel]
        if len(nb):
            dy = np.degrees(np.abs(np.angle(np.exp(1j * (yaw[k] - yaw[nb])))))
            cls[k] = 1 if (dy < same_deg).any() else 2 if (dy > opp_deg).any() else 0
    return dict(t=t_gt[idx], s=s, P=p, cls=cls)


def _runs(mask, s, max_gap=5.0):
    """Trechos contíguos (índices início/fim) onde mask é True; quebra se o salto em s passa de max_gap m."""
    i = np.flatnonzero(mask)
    if not len(i):
        return []
    cut = np.flatnonzero((np.diff(i) > 1) | (np.diff(s[i]) > max_gap))
    return list(zip(i[np.r_[0, cut + 1]], i[np.r_[cut, len(i) - 1]]))


REVISIT_NAMES = {1: "mesmo sentido", 2: "sentido oposto"}
REVISIT_COLORS = {1: "#2a78d6", 2: "#eb6834"}


def loop_coverage(rv, loops, cover):
    """Recall por distância: fração do trecho de revisita com um laço aceito E correto a até `cover` m."""
    good = [L for L in loops if L['kind'] == 'aceito' and L['correct'] and not L.get('stop', False)]
    s_loop = np.interp([L['t_b'] for L in good], rv['t'], rv['s']) if good else np.zeros(0)
    covered = (np.abs(rv['s'][:, None] - s_loop[None, :]) <= cover).any(axis=1) if len(s_loop) else \
        np.zeros(len(rv['s']), bool)
    print(f"\nRevisitas no GT (raio {REVISIT['radius']:.0f} m, >= {REVISIT['min_travel']:.0f} m rodados entre passagens)"
          f" e cobertura por laços aceitos corretos de revisita (sem paradas; a até {cover:.0f} m):")
    rec, w = {}, np.gradient(rv['s'])          # m representados por cada amostra
    for c, name in REVISIT_NAMES.items():
        m = rv['cls'] == c
        if not m.any():
            print(f"  {name:15s}: nenhum trecho")
            continue
        rec[c] = float(w[m & covered].sum() / w[m].sum())
        print(f"  {name:15s}: {w[m].sum():5.0f} m de revisita | coberto {100 * rec[c]:5.1f}%")
        for a, b in _runs(m, rv['s']):
            n = int(((s_loop >= rv['s'][a] - cover) & (s_loop <= rv['s'][b] + cover)).sum())
            print(f"      t {rv['t'][a] - rv['t'][0]:7.1f}-{rv['t'][b] - rv['t'][0]:7.1f} s  "
                  f"{rv['s'][b] - rv['s'][a]:5.0f} m  laços corretos: {n}")
    return covered, rec


def plot_revisits(rv, covered, rec, loops, t0, origin, title, out):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
    ax, at = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    P = rv['P'] - origin[:2]
    ax.plot(P[:, 0], P[:, 1], color='0.75', lw=1.2, label="GT (sem revisita)")
    for c, name in REVISIT_NAMES.items():
        m = rv['cls'] == c
        if m.any():
            ax.plot(np.where(m, P[:, 0], np.nan), np.where(m, P[:, 1], np.nan), color=REVISIT_COLORS[c], lw=4,
                    alpha=0.8, solid_capstyle='butt', label=f"revisita {name} ({np.gradient(rv['s'])[m].sum():.0f} m)")
    G_at = lambda t: P[int(np.argmin(np.abs(rv['t'] - t)))]
    marks = [("aceito e correto", lambda L: L['kind'] == 'aceito' and L['correct'] and not L.get('stop'), 'o',
              LOOP_COLORS[True], 7),
             ("aceito, sem rodar (parada)", lambda L: L['kind'] == 'aceito' and L['correct'] and L.get('stop'), 's',
              '#8a8984', 7),
             ("aceito e errado", lambda L: L['kind'] == 'aceito' and not L['correct'], 'X', LOOP_COLORS[False], 9),
             ("não aceito", lambda L: L['kind'] != 'aceito', 'x', '0.35', 5)]
    for lab, f, mk, col, ms in marks:
        sel = [L for L in loops if f(L)]
        for k, L in enumerate(sel):
            a, b = G_at(L['t_a']), G_at(L['t_b'])
            edge = dict(mec='white', mew=0.6) if mk != 'x' else dict(mew=1.2)       # 'x' não tem face
            ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=0.8, ls='-' if mk != 'x' else ':',
                    alpha=1.0 if mk != 'x' else 0.6)
            ax.plot(b[0], b[1], mk, color=col, ms=ms, label=f"laço {lab} ({len(sel)})" if k == 0 else None, **edge)
            at.plot(L['t_b'] - t0, 0, mk, color=col, ms=ms, **edge)
    ax.plot(P[0, 0], P[0, 1], 'g^', ms=12, mec='black', label="início")
    ax.set_aspect('equal', adjustable='datalim'); ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc='best')
    ax.set(xlabel="x [m]", ylabel="y [m]",
           title=f"{title}: onde havia revisita (GT) e onde houve laço (posições reais no GT)")

    # faixa no tempo: trecho de revisita (claro) e parte coberta por laço correto (escuro)
    rows = {1: 2, 2: 1}
    for c, y in rows.items():
        for a, b in _runs(rv['cls'] == c, rv['s']):
            at.broken_barh([(rv['t'][a] - t0, max(rv['t'][b] - rv['t'][a], 0.5))], (y - 0.35, 0.7),
                           color=REVISIT_COLORS[c], alpha=0.3)
        for a, b in _runs((rv['cls'] == c) & covered, rv['s']):
            at.broken_barh([(rv['t'][a] - t0, max(rv['t'][b] - rv['t'][a], 0.5))], (y - 0.35, 0.7),
                           color=REVISIT_COLORS[c], alpha=0.9)
    at.set_yticks([2, 1, 0], [f"{REVISIT_NAMES[c]}\ncoberto {100 * rec[c]:.0f}%" if c in rec else REVISIT_NAMES[c]
                              for c in (1, 2)] + ["laços (t atual)"], fontsize=9)
    at.set_ylim(-0.6, 2.6); at.set_xlim(rv['t'][0] - t0, rv['t'][-1] - t0)
    at.grid(True, axis='x', alpha=0.3); at.set_xlabel("t [s]")
    at.set_title(f"Cobertura: escuro = laço aceito correto a até {REVISIT['cover']:.0f} m; claro = revisita sem laço",
                 fontsize=10)
    fig.tight_layout()
    return save_fig(fig, f"{out}_revisitas")


# ------------------------------------------------------------------ RPE por comprimento de segmento (KITTI)
def rpe_segments(e, lengths=RPE_LENGTHS, step=5.0):
    """Erro relativo em segmentos de L m (distância no GT), com início a cada `step` m: translação em % de L e
    rotação em °/100 m (protocolo do KITTI odometry). Usa as poses já alinhadas de describe()."""
    d, P, G, Rw, Rg = e['dist'], e['P'], e['G'], e['Rw'], e['Rg']
    i0 = np.unique(np.searchsorted(d, np.arange(0, d[-1], step)))
    out = {}
    for L in lengths:
        j = np.searchsorted(d, d[i0] + L)
        ok = j < len(d)
        if ok.sum() < 3:
            continue
        i, j = i0[ok], j[ok]
        dT = np.einsum('nji,nj->ni', Rw[i], P[j] - P[i])
        dG = np.einsum('nji,nj->ni', Rg[i], G[j] - G[i])
        dRg = np.einsum('nji,njk->nik', Rg[i], Rg[j])           # Rg_i^T Rg_j
        dRw = np.einsum('nji,njk->nik', Rw[i], Rw[j])
        ER = np.einsum('nji,njk->nik', dRg, dRw)                 # erro relativo: dRg^T dRw
        ang = np.degrees(np.arccos(np.clip((np.trace(ER, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        Ld = d[j] - d[i]
        out[L] = dict(t=100 * np.linalg.norm(dT - dG, axis=1) / Ld, r=100 * ang / Ld)
    return out


def plot_rpe(rows, title, out):
    import matplotlib.pyplot as plt
    rpe = {n: rpe_segments(e) for n, e in rows.items()}
    print(f"\nRPE (KITTI, segmentos de {RPE_LENGTHS[0]}-{RPE_LENGTHS[-1]} m): média sobre todos os segmentos")
    print(f"{'':30s} {'transl. [%]':>11s} {'rot. [°/100 m]':>14s}")
    for n, r in rpe.items():
        if r:
            print(f"{n:30s} {np.mean(np.concatenate([v['t'] for v in r.values()])):11.2f} "
                  f"{np.mean(np.concatenate([v['r'] for v in r.values()])):14.3f}")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5))
    for n, r in rpe.items():
        if not r:
            continue
        Ls = np.array(list(r))
        for a, key in ((a1, 't'), (a2, 'r')):
            med = np.array([np.mean(r[L][key]) for L in Ls])
            lo, hi = (np.array([np.percentile(r[L][key], p) for L in Ls]) for p in (25, 75))
            a.plot(Ls, med, marker='o', ms=5, label=n, color=COLORS.get(n), ls=STYLES.get(n, "-"), lw=1.6)
            a.fill_between(Ls, lo, hi, color=COLORS.get(n), alpha=0.12, lw=0)
    a1.set(xlabel="comprimento do segmento [m]", ylabel="erro de translação [%]",
           title="RPE de translação (linha: média; faixa: quartis 25-75%)")
    a2.set(xlabel="comprimento do segmento [m]", ylabel="erro de rotação [°/100 m]", title="RPE de rotação")
    for a in (a1, a2):
        a.grid(True, alpha=0.3); a.legend(fontsize=9); a.set_ylim(bottom=0)
    fig.suptitle(f"{title}: erro relativo por comprimento de segmento")
    fig.tight_layout()
    return save_fig(fig, f"{out}_rpe"), rpe


# ------------------------------------------------------------------ pasta de resultados + dados das figuras
def rasterize_clouds(fig, min_points=2000):
    """Nuvens de pontos grandes viram imagem dentro do PDF (eixos/texto/linhas continuam vetoriais): sem isso o
    PDF do mapa fica com dezenas de MB e lento p/ abrir."""
    for ax in fig.axes:
        if ax.name == '3d':             # mplot3d não rasteriza coleções: rasteriza o eixo 3D inteiro (texto fora fica vetorial)
            ax.set_rasterized(True)
            continue
        for c in ax.collections:
            if len(c.get_offsets()) >= min_points:
                c.set_rasterized(True)
        for ln in ax.lines:                     # trajetórias/séries longas (centenas de milhares de vértices)
            if len(ln.get_xdata()) >= 10 * min_points:
                ln.set_rasterized(True)


def save_fig(fig, path_noext, dpi=300):
    path = f"{path_noext}.{FIG_FORMAT}"
    rasterize_clouds(fig)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    return path
dataset_name = base.dataset_name


def _slug(name):
    import unicodedata
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return "".join(c if c.isalnum() else "_" for c in s.lower()).strip("_").replace("__", "_")


def _csv(path, header, cols, fmt="%.6f"):
    np.savetxt(path, np.column_stack(cols), delimiter=",", header=",".join(header), comments="", fmt=fmt)


def save_figure_data(dados_dir, rows, rpe, rv, covered, loops, t0, origin, scale_t, scale_s, vo_lost_t):
    """CSVs com tudo o que as figuras desenham (p/ refazer os gráficos sem rodar de novo, ex.: pgfplots).
    Tempos relativos a t0; posições no referencial local com origem no GT em t0 (as mesmas das figuras)."""
    import csv
    p = lambda n: os.path.join(dados_dir, n)
    with open(p("metricas.csv"), "w", newline="") as fh:            # tabela-resumo (fig. trajetória + RPE)
        w = csv.writer(fh)
        w.writerow(["metodo", "ate_m", "rmse_ancorado_m", "erro_final_m", "deriva_pct", "yaw_medio_abs_deg",
                    "escala_sim3", "percurso_m", "rpe_transl_pct", "rpe_rot_deg_100m"])
        for n, e in rows.items():
            r = rpe.get(n) or {}
            rt = np.mean(np.concatenate([v['t'] for v in r.values()])) if r else np.nan
            rr = np.mean(np.concatenate([v['r'] for v in r.values()])) if r else np.nan
            w.writerow([n] + [f"{x:.4f}" for x in (e['ate'], e['ate0'], e['final'], e['drift'],
                                                   np.mean(np.abs(e['dyaw'])), e['scale'], e['length'], rt, rr)])
    for n, e in rows.items():                                         # séries das figuras de trajetória e de erro
        _csv(p(f"traj_{_slug(n)}.csv"),
             ["t", "x", "y", "z", "gt_x", "gt_y", "gt_z", "erro_m", "dist_gt_m", "erro_yaw_deg"],
             [e['t'] - t0, e['P'], e['G'], e['err'], e['dist'], e['dyaw']])
    with open(p("rpe.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metodo", "L_m", "n_segmentos", "transl_media_pct", "transl_p25", "transl_p75",
                    "rot_media_deg_100m", "rot_p25", "rot_p75"])
        for n, r in rpe.items():
            for L, v in r.items():
                w.writerow([n, L, len(v['t'])] + [f"{x:.5f}" for x in (
                    np.mean(v['t']), *np.percentile(v['t'], [25, 75]), np.mean(v['r']), *np.percentile(v['r'], [25, 75]))])
    _csv(p("revisitas_gt.csv"), ["t", "dist_m", "x", "y", "tipo", "coberto"],
         [rv['t'] - t0, rv['s'], rv['P'] - origin[:2], rv['cls'], covered.astype(int)],
         fmt=["%.4f", "%.3f", "%.3f", "%.3f", "%d", "%d"])
    keys = ["t_a", "t_b", "status", "kind", "correct", "stop", "gt_travel", "z_dist", "gt_dist", "meas_err", "dpos"]
    with open(p("lacos.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for L in loops:
            w.writerow([f"{L[k] - t0:.3f}" if k in ("t_a", "t_b") else L.get(k, "") for k in keys])
    _csv(p("escala_grafo.csv"), ["t", "escala_m_por_unid"], [np.asarray(scale_t) - t0, scale_s])
    _csv(p("vo_perdido.csv"), ["t"], [np.asarray(vo_lost_t) - t0])
    print(f"Dados das figuras: {dados_dir}/ (metricas, traj_*, rpe, revisitas_gt [tipo 1 = mesmo sentido, "
          f"2 = oposto], lacos, escala_grafo, vo_perdido)")


# ------------------------------------------------------------------ funil do detector de laço avaliado c/ GT
LOOP_CAND_TOL = 30.0                # m: candidato "verdadeiro" se distam < isso no GT (a câmera casa KFs até ~20 m à frente)
FUNNEL = ["consultado", "com candidato BoW", "temporalmente consistente", "passou F (2D-2D)", "pontos 3D ok",
          "verificado (PnP)", "aceito no grafo"]
STAGE_IDX = dict(sem_elegivel=0, pouco_informativa=0, sem_candidato=0, inconsistente=1, intervalo_laco=1,
                 falha_f=2, falha_3d=3, falha_pnp=4, verificado=5)


def loop_funnel(log, loops, t_gt, T_gt, rv, t0, title, out, dados_dir, min_f_inliers, node_dt):
    """Cada consulta do detector: até onde chegou no funil, se o lugar era revisita (GT) e se o candidato escolhido
    era o lugar certo (GT). Responde: as etapas barram laços verdadeiros ou só falsos?"""
    import csv
    import matplotlib.pyplot as plt
    if not log:
        return None
    qt = np.array([r['q_t'] for r in log])
    ct = np.array([r['c_t'] for r in log])
    pos = lambda t: np.column_stack([np.interp(t, t_gt, T_gt[:, i, 3]) for i in range(2)])
    has_c = np.isfinite(ct)
    dist = np.full(len(log), np.nan)
    dist[has_c] = np.linalg.norm(pos(qt[has_c]) - pos(ct[has_c]), axis=1)
    s_gt = np.r_[0, np.cumsum(np.linalg.norm(np.diff(T_gt[:, :2, 3], axis=0), axis=1))]
    travel = np.full(len(log), np.nan)
    travel[has_c] = np.abs(np.interp(qt[has_c], t_gt, s_gt) - np.interp(ct[has_c], t_gt, s_gt))
    cand_stop = has_c & (travel < REVISIT['min_travel'])         # quase não rodou entre os dois: o carro PAROU
    cand_ok = has_c & ~cand_stop & (dist < LOOP_CAND_TOL)
    cand_bad = has_c & ~cand_stop & ~cand_ok
    k = np.clip(np.searchsorted(rv['t'], qt), 0, len(rv['t']) - 1)
    place = np.where(np.abs(rv['t'][k] - qt) < 1.0, rv['cls'][k], 0)       # 0 sem revisita, 1 mesmo, 2 oposto
    idx = np.array([STAGE_IDX[r['stage']] for r in log])
    acc_t = np.array([L['t_b'] for L in loops if L['kind'] == 'aceito'])
    if len(acc_t):
        idx[(idx == 5) & (np.abs(qt[:, None] - acc_t[None, :]).min(axis=1) <= node_dt + 1e-3)] = 6

    with open(os.path.join(dados_dir, "funil_lacos.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "t_candidato", "etapa_final", "etapa_idx", "lugar_gt", "dist_gt_candidato_m",
                    "rodado_entre_m", "candidato_verdadeiro", "candidato_parada", "score_bow", "n_casamentos", "n_inliers_f", "n_pontos_3d", "n_inliers_pnp"])
        for r, i, p, d, tr, ok, sp in zip(log, idx, place, dist, travel, cand_ok, cand_stop):
            w.writerow([f"{r['q_t'] - t0:.3f}", f"{r['c_t'] - t0:.3f}" if np.isfinite(r['c_t']) else "",
                        FUNNEL[i] if i == 6 else r['stage'], i, int(p), f"{d:.2f}" if np.isfinite(d) else "",
                        f"{tr:.1f}" if np.isfinite(tr) else "", int(ok), int(sp), f"{r['score']:.4f}" if np.isfinite(r['score']) else "",
                        r['n_match'], r['n_f'], r['n_3d'], r['n_pnp']])

    reach = lambda m: [int(((idx >= s) & m).sum()) for s in range(len(FUNNEL))]
    groups_place = [("revisita mesmo sentido", place == 1, REVISIT_COLORS[1]),
                    ("revisita sentido oposto", place == 2, REVISIT_COLORS[2]),
                    ("sem revisita (GT)", place == 0, "#c3c2b7")]
    groups_cand = [(f"revisita verdadeira (< {LOOP_CAND_TOL:.0f} m no GT)", cand_ok, LOOP_COLORS[True]),
                   (f"mesmo lugar sem rodar (parada, < {REVISIT['min_travel']:.0f} m)", cand_stop, "#8a8984"),
                   ("candidato falso", cand_bad, LOOP_COLORS[False])]
    print(f"\nFunil do detector de laço x GT (candidato verdadeiro: < {LOOP_CAND_TOL:.0f} m no GT):")
    print(f"{'etapa':28s} {'total':>6s} {'mesmo s.':>8s} {'oposto':>7s} {'s/ rev.':>7s} {'cand. V':>8s} "
          f"{'parada':>7s} {'cand. F':>8s}")
    cols = [reach(np.ones(len(idx), bool))] + [reach(m) for _, m, _ in groups_place + groups_cand]
    for s, name in enumerate(FUNNEL):
        print(f"{name:28s} " + " ".join(f"{c[s]:{wd}d}" for c, wd in zip(cols, (6, 8, 7, 7, 8, 7, 8))))

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(19, 6), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
    y = np.arange(len(FUNNEL))[::-1]
    for ax, groups, ttl in ((a1, groups_place, "Até onde chega cada consulta, por tipo de lugar (GT)"),
                            (a2, groups_cand, "Candidato escolhido: lugar certo ou não (GT)")):
        left = np.zeros(len(FUNNEL))
        for lab, m, col in groups:
            v = np.array(reach(m), float)
            b = ax.barh(y, v, left=left, color=col, label=lab, height=0.7)
            ax.bar_label(b, labels=[f"{int(x)}" if x else "" for x in v], label_type='center', fontsize=8)
            left += v
        ax.set_yticks(y, FUNNEL if ax is a1 else [""] * len(FUNNEL), fontsize=9)
        ax.set_xscale('symlog', linthresh=10)
        ax.set(xlabel="nº de consultas (escala log)", title=ttl)
        ax.grid(True, axis='x', alpha=0.3); ax.legend(fontsize=8, loc='center right')
    tried = idx >= 2
    bins = np.arange(0, max(int(np.max([r['n_f'] for r in log] + [min_f_inliers])) + 10, 60), 5)
    nf = np.array([r['n_f'] for r in log])
    a3.hist([nf[tried & cand_ok], nf[tried & cand_stop], nf[tried & cand_bad]], bins=bins, stacked=True,
            color=[LOOP_COLORS[True], "#8a8984", LOOP_COLORS[False]],
            label=["revisita verdadeira", "parada (sem rodar)", "candidato falso"])
    a3.axvline(min_f_inliers, color='0.2', ls='--', lw=1.2, label=f"limiar min_f_inliers = {min_f_inliers}")
    a3.set(xlabel="inliers da matriz F", ylabel="nº de consultas verificadas",
           title="Inliers da F nas consultas consistentes")
    a3.grid(True, alpha=0.3); a3.legend(fontsize=8)
    fig.suptitle(f"{title}: funil do detector de laço avaliado com o GT")
    fig.tight_layout()
    return save_fig(fig, f"{out}_funil_lacos")


# ------------------------------------------------------------------ métricas de processamento (tempo real)
PROC_BLOCK = 200                    # frames por bloco nas curvas (média / p95 de cada bloco)
PROC_COMPONENTS = [("vo", "VO", "#2a78d6"), ("fusao", "fusão", "#1baf7a"), ("laco", "laço (envio/recepção)", "#eda100"),
                   ("viz", "visualização", "#8a8984"), ("io", "espera de imagem", "#c3c2b7")]


def _blocks(x, n):
    """Divide em blocos de n amostras; devolve a lista de índices de cada bloco."""
    return [np.arange(i, min(i + n, len(x))) for i in range(0, len(x), n)]


def processing_report(timing, ba_log, be_log, loop_log, node_dt, extra, title, out, dados_dir):
    """Tempo de processamento: por frame no processo principal (VO, fusão, laço, visualização, E/S), por KF no
    detector de laço, por nó no grafo e por BA. Compara com o orçamento de tempo real (período da câmera) e mostra
    como cada parte cresce com o tamanho do mapa."""
    import csv
    import matplotlib.pyplot as plt
    T = np.array(timing, float)
    t, n_pts, n_kf, n_nodes = T[:, 0] - T[0, 0], T[:, 6], T[:, 7], T[:, 8]
    lag = T[:, 9] if T.shape[1] > 9 else np.zeros(len(T))
    comp = dict(io=T[:, 1], vo=T[:, 2], fusao=T[:, 3], laco=T[:, 4], viz=T[:, 5])
    total = sum(comp.values())
    period = float(np.median(np.diff(T[:, 0])))                 # orçamento: período da câmera
    duration = t[-1] + period
    proc = total.sum()
    _csv(os.path.join(dados_dir, "tempos_frames.csv"),
         ["t", "espera_io_ms", "vo_ms", "fusao_ms", "laco_ms", "visualizacao_ms", "total_ms", "n_landmarks",
          "n_kfs", "n_nos", "atraso_grafo_nos", "fonte_online"],
         [t, *(1000 * comp[k] for k in ("io", "vo", "fusao", "laco", "viz")), 1000 * total, n_pts, n_kf, n_nodes, lag,
          T[:, 10] if T.shape[1] > 10 else -np.ones(len(T))],
         fmt=["%.3f"] + ["%.3f"] * 6 + ["%d"] * 5)

    # ---- tabela-resumo (início = primeiros 10% dos frames, fim = últimos 10%: mostra o crescimento)
    k10 = max(len(t) // 10, 1)
    rows = []
    print(f"\nProcessamento (orçamento de tempo real = período da câmera = {1000 * period:.1f} ms/frame):")
    print(f"{'componente':24s} {'média':>7s} {'p50':>7s} {'p95':>7s} {'máx':>8s} {'início':>7s} {'fim':>7s}  [ms/frame]")
    for key, name, _ in PROC_COMPONENTS + [("total", "TOTAL", None)]:
        x = 1000 * (total if key == "total" else comp[key])
        r = [name, x.mean(), np.median(x), np.percentile(x, 95), x.max(), x[:k10].mean(), x[-k10:].mean()]
        rows.append(r)
        print(f"{name:24s} " + " ".join(f"{v:7.1f}" for v in r[1:5]) + f" {r[5]:7.1f} {r[6]:7.1f}")
    over = 100 * np.mean(total > period)
    print(f"Tempo real: {proc:.0f} s de processamento p/ {duration:.0f} s de dados -> fator {proc / duration:.2f} "
          f"(< 1 = mais rápido que o tempo real) | {over:.0f}% dos frames acima do orçamento | taxa média "
          f"{len(t) / proc:.1f} FPS (câmera {1 / period:.1f} FPS)")
    print(f"Mapa no fim: {int(n_pts[-1])} landmarks, {int(n_kf[-1])} KFs, {int(n_nodes[-1])} nós do grafo | atraso do "
          f"grafo: máx {lag.max():.0f} nós ({lag.max() * node_dt:.0f} s), no fim {lag[-1]:.0f} nós")

    # ---- processos em paralelo
    L = [r for r in loop_log if 'dt_query' in r]
    kf_period = duration / max(len(L), 1)
    if L:
        lt = 1000 * np.array([r['dt_orb'] + r['dt_query'] for r in L])
        print(f"Detector de laço (outro processo): {lt.mean():.1f} ms/KF (p95 {np.percentile(lt, 95):.1f}, máx "
              f"{lt.max():.0f}) | início {lt[:max(len(lt) // 10, 1)].mean():.1f} -> fim "
              f"{lt[-max(len(lt) // 10, 1):].mean():.1f} ms | 1 KF a cada {1000 * kf_period:.0f} ms em média")
    B = np.array([b for b in be_log if np.isfinite(b[1])], float).reshape(-1, 4)
    if len(B):
        bt = 1000 * B[:, 1]
        node = B[:, 2] == 0
        print(f"Grafo (outro processo): {bt[node].mean():.1f} ms/nó (p95 {np.percentile(bt[node], 95):.1f}) | "
              f"início {bt[node][:max(node.sum() // 10, 1)].mean():.1f} -> fim "
              f"{bt[node][-max(node.sum() // 10, 1):].mean():.1f} ms | laço: {bt[~node].mean() if (~node).any() else 0:.0f} "
              f"ms médio | orçamento {1000 * node_dt:.0f} ms/nó")
    A = np.array(ba_log, float).reshape(-1, 3)
    if len(A):
        print(f"BA local (worker): {1000 * A[:, 1].mean():.0f} ms/BA (p95 {1000 * np.percentile(A[:, 1], 95):.0f}) | "
              f"{len(A)} BAs, janela {A[:, 2].mean():.0f} KFs")
    print(f"No fim da sequência: detector de laço terminando a fila {extra['laco_fim']:.1f} s | LM em lote "
          f"{extra['lote']:.1f} s")

    with open(os.path.join(dados_dir, "tempos_resumo.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["componente", "media_ms", "p50_ms", "p95_ms", "max_ms", "inicio_ms", "fim_ms"])
        for r in rows:
            w.writerow([r[0]] + [f"{v:.3f}" for v in r[1:]])
        w.writerow([])
        w.writerow(["periodo_camera_ms", f"{1000 * period:.3f}"])
        w.writerow(["fator_tempo_real", f"{proc / duration:.4f}"])
        w.writerow(["pct_frames_acima_orcamento", f"{over:.2f}"])
        w.writerow(["laco_fim_s", f"{extra['laco_fim']:.2f}"])
        w.writerow(["lm_lote_s", f"{extra['lote']:.2f}"])
    if L:
        _csv(os.path.join(dados_dir, "tempos_laco.csv"), ["t", "n_kfs", "orb_bow_ms", "consulta_verif_ms"],
             [np.array([r['q_t'] for r in L]) - T[0, 0], [r['n_kfs'] for r in L],
              [1000 * r['dt_orb'] for r in L], [1000 * r['dt_query'] for r in L]], fmt=["%.3f", "%d", "%.3f", "%.3f"])
    if len(B):
        _csv(os.path.join(dados_dir, "tempos_grafo.csv"), ["no", "dt_ms", "laco", "poses_completas"],
             [B[:, 0], 1000 * B[:, 1], B[:, 2], B[:, 3]], fmt=["%d", "%.3f", "%d", "%d"])
    if len(A):
        _csv(os.path.join(dados_dir, "tempos_ba.csv"), ["frame", "dt_ms", "kfs_janela"],
             [A[:, 0], 1000 * A[:, 1], A[:, 2]], fmt=["%d", "%.3f", "%d"])

    # ---- figura
    blk = _blocks(t, PROC_BLOCK)
    tb = np.array([t[i].mean() for i in blk])
    kb = np.array([n_kf[i].mean() for i in blk])
    mean_b = {k: np.array([1000 * comp[k][i].mean() for i in blk]) for k in comp}
    p95_b = np.array([1000 * np.percentile(total[i], 95) for i in blk])
    fig, axs = plt.subplots(3, 2, figsize=(16, 14))
    (a1, a2), (a3, a4), (a5, a6) = axs
    a1.stackplot(tb, *[mean_b[k] for k, _, _ in PROC_COMPONENTS], labels=[n for _, n, _ in PROC_COMPONENTS],
                 colors=[c for _, _, c in PROC_COMPONENTS], alpha=0.85)
    a1.plot(tb, p95_b, color='black', lw=1, ls=':', label="total, p95 do bloco")
    a1.axhline(1000 * period, color='#e34948', ls='--', lw=1.4, label=f"orçamento ({1000 * period:.0f} ms = câmera)")
    a1.set(xlabel="t [s]", ylabel="ms/frame", title=f"Processo principal: tempo por frame (média a cada {PROC_BLOCK} frames)")
    for k, name, col in PROC_COMPONENTS:
        a2.plot(kb, mean_b[k], color=col, lw=1.6, label=name)
    a2.plot(kb, sum(mean_b.values()), color='black', lw=1.8, label="total")
    a2.axhline(1000 * period, color='#e34948', ls='--', lw=1.4)
    a2.set(xlabel="nº de keyframes no mapa", ylabel="ms/frame", title="Crescimento com o tamanho do mapa")
    fps = np.array([len(i) / total[i].sum() for i in blk])
    a3.plot(tb, fps, color='black', lw=1.6, label="processamento")
    a3.axhline(1 / period, color='#e34948', ls='--', lw=1.4, label=f"câmera ({1 / period:.0f} FPS)")
    a3b = a3.twinx()
    a3b.plot(t, n_pts / 1000, color='#6250d6', lw=1.2, label="landmarks (mil)")
    a3b.plot(t, n_kf / 1000, color='#6250d6', lw=1.2, ls='--', label="KFs (mil)")
    a3b.set_ylabel("tamanho do mapa [mil]", color='#6250d6')
    a3.set(xlabel="t [s]", ylabel="FPS", title=f"Taxa de processamento (fator de tempo real {proc / duration:.2f})")
    h1, l1 = a3.get_legend_handles_labels(); h2, l2 = a3b.get_legend_handles_labels()
    a3.legend(h1 + h2, l1 + l2, fontsize=8, loc='best')
    if L:
        lk = np.array([r['n_kfs'] for r in L]); lo = 1000 * np.array([r['dt_orb'] for r in L])
        lq = 1000 * np.array([r['dt_query'] for r in L])
        a4.scatter(lk, lo + lq, s=3, color='#eda100', alpha=0.4, rasterized=True, label="por KF")
        bl = _blocks(lk, 50)
        a4.plot([lk[i].mean() for i in bl], [lo[i].mean() for i in bl], color='#2a78d6', lw=1.6, label="ORB + BoW (média)")
        a4.plot([lk[i].mean() for i in bl], [lq[i].mean() for i in bl], color='#e34948', lw=1.6,
                label="consulta + verificação (média)")
        a4.axhline(1000 * kf_period, color='0.3', ls='--', lw=1.2, label=f"intervalo médio entre KFs ({1000 * kf_period:.0f} ms)")
        a4.set_yscale('log')
    a4.set(xlabel="nº de KFs no banco", ylabel="ms/KF", title="Detector de laço (processo separado)")
    if len(B):
        a5.scatter(B[node, 0], bt[node], s=3, color='#1baf7a', alpha=0.4, rasterized=True, label="nó novo")
        a5.scatter(B[~node, 0], bt[~node], s=25, marker='x', color='#e34948', label="laço")
        bl = _blocks(B[node, 0], 50)
        a5.plot([B[node, 0][i].mean() for i in bl], [bt[node][i].mean() for i in bl], color='black', lw=1.4,
                label="média (nós)")
        a5.axhline(1000 * node_dt, color='0.3', ls='--', lw=1.2, label=f"intervalo entre nós ({1000 * node_dt:.0f} ms)")
        a5.set_yscale('log')
    a5b = a5.twinx()
    a5b.plot(n_nodes, lag, color='#6250d6', lw=1.2, label="atraso (nós enviados e ainda não otimizados)")
    a5b.set_ylabel("atraso [nós]", color='#6250d6')
    a5b.legend(fontsize=8, loc='upper center')
    a5.set(xlabel="nº de nós no grafo", ylabel="ms", title="Grafo (iSAM2, processo separado)")
    if len(A):
        tf = np.interp(A[:, 0], np.arange(len(t)), t)
        a6.scatter(tf, 1000 * A[:, 1], s=4, color='#2a78d6', alpha=0.5, rasterized=True, label="por BA")
        bl = _blocks(tf, 30)
        a6.plot([tf[i].mean() for i in bl], [1000 * A[i, 1].mean() for i in bl], color='black', lw=1.4, label="média")
    a6.set(xlabel="t [s]", ylabel="ms/BA", title="BA local (worker)")
    for a in axs.ravel():
        a.grid(True, alpha=0.3)
        if a is not a3 and a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8, loc='best')
    fig.suptitle(f"{title}: processamento ({len(t)} frames, {duration:.0f} s de dados)")
    fig.tight_layout()
    return save_fig(fig, f"{out}_processamento")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=dataset_path)
    ap.add_argument("--dataset-root", default=None, help="raiz com data_sensor/ e poses/ (padrão: procura)")
    ap.add_argument("--sensors", default=",".join(SENSORS))
    ap.add_argument("--node-dt", type=float, default=NODE_DT)
    ap.add_argument("--sim-imu", action="store_true", default=SIM_IMU)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--inline-backend", action="store_true", help="grafo na thread principal (depuração)")
    ap.add_argument("--out", default=None, help="pasta de saída (padrão: resultados/<dataset>)")
    ap.add_argument("--no-loop", action="store_true", help="desliga o fechamento de laço")
    ap.add_argument("--vocab", default=VOCAB_FILE)
    ap.add_argument("--no-feedback", action="store_true", help="não realimenta a escala do grafo no VO")
    ap.add_argument("--vo-view", action="store_true", help="3D no mundo do VO (escala arbitrária), não no fundido")
    ap.add_argument("--init-dist", type=float, default=None,
                    help="m de roda p/ estimar a escala inicial do VO (padrão: 5 no KAIST, 0.5 na Inomi)")
    ap.add_argument("--camera", default="auto", choices=["auto"] + list(base.CAMERAS),
                    help="calibração (auto: 'inomi' p/ pastas dataset_inomi / nomes AAAAMMDD_HHMMSS_micro)")
    args = ap.parse_args()
    sensors = [s.strip() for s in args.sensors.split(",") if s.strip()]
    gui = not args.no_gui
    out_dir = args.out or os.path.join(RESULTS_DIR, dataset_name(args.images))
    dados_dir = os.path.join(out_dir, "dados")
    os.makedirs(dados_dir, exist_ok=True)
    args.out = os.path.join(out_dir, OUT_PREFIX)          # prefixo das figuras
    dados = lambda name: os.path.join(dados_dir, name)
    print(f"Resultados em {out_dir}/ (figuras) e {dados_dir}/ (dados)")

    # ---------------- dados dos sensores (selecionados) + GT
    root, files = (args.dataset_root, None) if args.dataset_root else find_dataset_files(args.images)
    if files is None:
        _, files = find_dataset_files(os.path.join(root, "x"))
    files.setdefault("odom", None)
    print(f"Dataset: {root}\n  encoder: {files['encoder']}\n  odometria (pose): {files['odom']}\n  IMU: {files['imu']}\n"
          f"  GT: {files['gt']}")
    t_gt = T_gt = None
    if files["gt"]:
        t_gt, T_gt = load_gt(files["gt"])
    wheel = imu = None
    if "wheel" in sensors:
        if files["encoder"]:
            wheel = WheelOdometry.from_csv(files["encoder"], time_offset=TIME_OFFSETS["wheel"])
        elif files["odom"]:
            wheel = PoseOdometry.from_csv(files["odom"], time_offset=TIME_OFFSETS["wheel"])
        else:
            sensors = [s for s in sensors if s != "wheel"]
            print("AVISO: 'wheel' selecionado, mas não há encoder.csv nem odometria_dataset.csv -> sem roda "
                  f"(sensores: {sensors}; sem sensor métrico a escala fica a do VO)")
    if "imu" in sensors:
        if files["imu"]:
            imu = ImuData.from_kaist_xsens(files["imu"], time_offset=TIME_OFFSETS["imu"])
        elif args.sim_imu and T_gt is not None:
            imu = simulate_imu(t_gt, T_gt)
            print("  IMU SIMULADA a partir do GT (só p/ validar o código; resultado otimista)")
        else:
            raise SystemExit("'imu' selecionado, mas não há xsens_imu.csv (use --sim-imu p/ testar com IMU simulada)")

    image_paths = sorted(glob.glob(os.path.join(args.images, "*.png")) + glob.glob(os.path.join(args.images, "*.jpg")))
    if args.max_frames:
        image_paths = image_paths[:args.max_frames]
    cam = args.camera
    if cam == "auto":
        inomi = "inomi" in os.path.abspath(args.images).lower() or (
            image_paths and not os.path.splitext(os.path.basename(image_paths[0]))[0].isdigit())
        cam = "inomi" if inomi else "urban"
    K_cam, dist_cam = base.CAMERAS[cam]["K"], base.CAMERAS[cam]["dist"]
    if T_gt is not None:
        t_gt, T_gt = adapt_gt(t_gt, T_gt, [image_stamp_ns(p) * 1e-9 for p in image_paths],
                              body_yaw_deg=GT_BODY_YAW_DEG.get(cam, 0.0))
    print(f"Câmera: {cam} (fx={K_cam[0, 0]:.1f}, cx={K_cam[0, 2]:.1f}{', com distorção' if dist_cam is not None else ''})")

    vo = VisualOdometry(camera_matrix=K_cam, dist_coeffs=dist_cam, orb_params=base.orb_params,
                        lk_params=base.lk_params, keyframe_params=base.keyframe_params, sliding_window_size=8,
                        enable_plot=False, verbose=False, async_ba=base.ASYNC_BA)
    feedback = SCALE_FEEDBACK and not args.no_feedback and ("wheel" in sensors or "imu" in sensors)
    fused_view = FUSED_VIEW and not args.vo_view
    init_dist = args.init_dist if args.init_dist is not None else (0.5 if cam == "inomi" else 5.0)
    fusion = OnlineFusion(sensors, wheel=wheel, imu=imu, node_dt=args.node_dt, cfg=GPOConfig(),
                          imu_R_vi=IMU_R_VI, backend_process=not args.inline_backend, scale_feedback=feedback,
                          init_wheel_dist=init_dist)
    frame_view = FrameViewer(max_fps=base.WINDOW_MAX_FPS) if gui and SHOW_WINDOW else None
    traj_view = TrajViewer(f"Fusão {'+'.join(sensors)}") if gui and SHOW_TRAJ else None
    view3d = None
    if gui and SHOW_3D:
        view3d = Open3DViewer(K=K_cam, max_fps=base.VIEW3D_MAX_FPS, points_every=base.VIEW3D_POINTS_EVERY,
                              **(dict(world="vehicle", title="Fusao 3D", snap_fn=fusion.world_snapshot)
                                 if fused_view else {}))
    gt_view = GtAligner(t_gt, T_gt) if t_gt is not None else None
    lc = None
    if LOOP_CLOSURE and not args.no_loop:
        if os.path.isfile(args.vocab):
            lc = LoopClosure(K_cam, args.vocab, LoopConfig(), process=not args.inline_backend)
        else:
            print(f"AVISO: vocabulário {args.vocab} não encontrado -> sem fechamento de laço "
                  f"(treine com: python3 LoopClosure.py train --images <pastas> --out {args.vocab})")

    print(f"\nProcessando {len(image_paths)} imagens | sensores: {sensors} | nó a cada {args.node_dt}s | "
          f"laço: {'sim' if lc is not None else 'não'} | escala do grafo -> VO: {'sim' if feedback else 'não'} | "
          f"3D no mundo {'fundido' if fused_view else 'do VO'}")
    t_vo = t_fus = 0.0
    n = 0
    t_wall = time.time()
    timing = []                 # por frame: (t, espera E/S, VO, fusão, laço, visualização, landmarks, KFs, nós)
    t_end = time.perf_counter()
    for idx, (img_path, frame, prepared) in enumerate(base.prefetch(vo, image_paths)):
        a0 = time.perf_counter()
        if frame is None:
            continue
        ts_text = os.path.splitext(os.path.basename(img_path))[0]
        stamp = image_stamp_ns(img_path)           # KAIST: ns no nome; Inomi: AAAAMMDD_HHMMSS_micro
        t = stamp * 1e-9

        a = time.perf_counter()
        vo.process_frame_pnp(frame, timestamp=stamp, prepared=prepared, return_traj=False)
        b = time.perf_counter()
        fusion.on_frame(vo, t)
        b2 = time.perf_counter()
        if lc is not None:
            lc.on_frame(vo, prepared[1])
            for lp in lc.poll():
                fusion.add_loop(vo, lp)
        c = time.perf_counter()
        t_vo += b - a
        t_fus += c - b
        n += 1

        if frame_view is not None:
            extra = ""
            if fusion.fused:
                p = fusion.fused[-1][:3, 3]
                extra = f"fusao x={p[0]:.0f} y={p[1]:.0f} m  esc.VO={np.exp(fusion.log_scale) / vo.frame_units(len(vo.poses_est) - 1):.2f}"
            frame_view.show(vo, frame, ts_text, extra)
            if frame_view.stop_requested:
                break
        if view3d is not None:
            view3d.update(vo, with_points=True)
        if traj_view is not None and idx % PLOT_EVERY == 0 and fusion.fused:
            snap = fusion.snapshot()
            if gt_view is not None:
                g = gt_view.line(fusion.fused_t, fusion.fused)
                if g is not None:
                    snap['lines'] = dict(GT=g, **snap['lines'])
            snap['text'] = f"frame {idx} | escala VO = {snap['scale']:.3f} m/unid. | nós = {len(fusion.node_t)}"
            traj_view.update(snap)
        d = time.perf_counter()
        timing.append((t, a0 - t_end, b - a, b2 - b, c - b2, d - c, len(vo.Map.points), len(vo.Map.poses),
                       len(fusion.node_t), len(fusion.node_t) - 1 - fusion.last_n,     # atraso do grafo [nós]
                       fusion.fused_src[-1] if len(fusion.fused_src) and fusion.fused_t and fusion.fused_t[-1] == t
                       else -1))                                                       # fonte da pose online
        t_end = d

    vo.finish()
    t_wall = time.time() - t_wall
    print(f"\nFrames: {n} | VO {t_vo:.1f}s ({n / max(t_vo, 1e-9):.1f} FPS) | fusão no loop principal "
          f"{1000 * t_fus / max(n, 1):.2f} ms/frame | total {t_wall:.1f}s ({n / t_wall:.1f} FPS)")

    t_lc_finish = 0.0
    if lc is not None:
        a = time.perf_counter()
        for lp in lc.finish(vo):
            fusion.add_loop(vo, lp)
        t_lc_finish = time.perf_counter() - a       # > 0: o detector ficou para trás e terminou a fila no fim
        st = lc.stats
        print(f"Detector de laço: {st.get('kfs', 0)} KFs | {st.get('candidates', 0)} consultas c/ candidato | "
              f"{st.get('consistent', 0)} temporalmente consistentes | falha 2D-2D {st.get('fail_f', 0)}, "
              f"poucos pontos 3D {st.get('fail_3d', 0)}, falha PnP {st.get('fail_pnp', 0)} | "
              f"{st.get('loops', 0)} laços verificados")
    if fusion.fb_log:
        f = np.array([x[1] for x in fusion.fb_log])
        print(f"Escala grafo -> VO: {len(f)} correções (fator acumulado x{np.prod(f):.3f}; "
              f"{sum(x[2] for x in fusion.fb_log)} de segmento inteiro) | unidades finais do VO = "
              f"x{vo.frame_units(len(vo.poses_est) - 1):.3f} das originais")
    a = time.perf_counter()
    try:
        res = fusion.finish(batch=True)
    except Exception as e:      # grafo não terminou: salva ao menos as métricas de processamento
        print(f"[ERRO] o grafo não terminou ({type(e).__name__}); salvando só as métricas de processamento")
        processing_report(timing, vo.ba_log, fusion.be_log, lc.stats.get('log', []) if lc is not None else [],
                          args.node_dt, dict(laco_fim=t_lc_finish, lote=np.nan, total=t_wall),
                          f"Fusão {'+'.join(sensors)}", args.out, dados_dir)
        raise
    t_batch = time.perf_counter() - a
    print(f"Grafo final: {len(res['poses'])} nós, fatores {res.get('n_factors')} | LM em lote {t_batch:.1f}s")
    print(f"Reinícios do VO c/ escala re-semeada (roda/VO): {res.get('n_reseed', 0)} | frames em que a pose online "
          f"usou a roda (VO x escala incoerente): {fusion.n_guard}")
    if len(res['poses']):          # mundo fundido (3D final / mapa) com as poses e escalas suavizadas
        fusion.est = {k: P for k, P in enumerate(res['poses'])}
        fusion.est_ls = {k: float(v) for k, v in enumerate(res['log_scales'])}
        fusion.last_n = len(res['poses']) - 1

    for v in (frame_view, traj_view):
        if v is not None:
            v.close()
    if view3d is not None:
        view3d.final(vo)            # estado final do mapa (após o último BA)

    # ---------------- salva e avalia
    F = np.array(fusion.fused)
    save_tum(dados("online_tum.txt"), fusion.fused_t, F)
    save_tum(dados("suavizado_tum.txt"), res['node_t'], res['poses'])
    vo.save_trajectory(dados("vo_tum.txt"))
    vo.save_map(dados("mapa_vo.npz"))
    with open(dados("execucao.txt"), "w") as fh:
        fh.write(f"data: {time.strftime('%Y-%m-%d %H:%M:%S')}\nimagens: {args.images}\nsensores: {','.join(sensors)}\n"
                 f"realimentação de escala: {feedback}\nfechamento de laço: {not args.no_loop}\n"
                 + "".join(f"{k}: {v}\n" for k, v in sorted(vars(args).items())))
    print(f"Trajetórias salvas em {dados_dir}/: online_tum.txt (causal, por frame), suavizado_tum.txt (nós, "
          f"após otimização final), vo_tum.txt (VO puro) | mapa: mapa_vo.npz")

    import matplotlib
    if not gui:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.ioff()

    if t_gt is not None and len(F):
        # zero comum: 1º nó do grafo a partir do qual existe pose fundida (nós caem em instantes de frames, então
        # VO/fusão/roda têm amostra exatamente em t0); tudo é recortado em [t0, fim] e ancorado no GT em t0
        nt = np.array(res.get('t_nodes', res['node_t']))
        t0 = float(nt[np.argmax(nt >= fusion.fused_t[0] - 1e-6)])
        i0, ok0 = _match_gt(np.array([t0]), t_gt)
        origin = T_gt[i0[0], :3, 3]
        t_frames = np.array([float(x) * 1e-9 for x in vo.timestamps])
        T_vo = np.array([fusion.T_vc @ P @ fusion.T_cv for P in vo.poses_est])
        ft = np.array(fusion.fused_t)
        cand = [(VO_FB if feedback else VO_PURE, t_frames, T_vo, True), ("fusão online", ft, F, False),
                ("fusão suavizada", nt, res['poses'], False)]
        if fusion.wheel_dr:
            cand.append(("só roda", ft, np.array(fusion.wheel_dr), False))
        rows = {}
        for name, t, T, sc in cand:
            m = t >= t0 - 1e-6
            if m.sum() >= 3:
                rows[name] = describe(t[m], T[m], t_gt, T_gt, origin, with_scale=sc)

        print(f"\nPercurso (GT, a partir de t0): {rows['fusão online']['length']:.0f} m")
        print(f"{'':18s} {'ATE [m]':>8s} {'RMSE anc. [m]':>13s} {'erro final [m]':>14s} {'deriva %':>8s} "
              f"{'|yaw| méd [°]':>13s}")
        for name, e in rows.items():
            print(f"{name:18s} {e['ate']:8.2f} {e['ate0']:13.2f} {e['final']:14.2f} {e['drift']:8.2f} "
                  f"{np.mean(np.abs(e['dyaw'])):13.2f}")
        print("(ATE = Umeyama na média, padrão da literatura; RMSE anc. = alinhado com o início coincidente, como nos "
              "gráficos)\n(online = pose disponível no instante do frame; suavizada = após a otimização final em lote)")

        sc = np.exp(res['log_scales'])
        vo_row = rows.get(VO_PURE)
        print(f"Escala do VO no grafo (m/unid. original): início {sc[0]:.3f} | p10/50/90 "
              f"{np.percentile(sc, 10):.3f}/{np.median(sc):.3f}/{np.percentile(sc, 90):.3f} | fim {sc[-1]:.3f}"
              + (f" | ótima Sim3 do VO puro {vo_row['scale']:.3f}" if vo_row else ""))
        loops = evaluate_loops(res.get('loops', []), t_gt, T_gt)
        ok = np.array(fusion.frame_ok[:len(t_frames)])
        vo_lost_t = t_frames[(~ok) & (t_frames >= t0)]
        _, figs = plot_results(rows, sensors, t0, nt, np.exp(res['log_scales']), vo_lost_t,
                               args.out, gui, loops)
        title = f"Fusão {'+'.join(sensors)}"
        fig_rpe, rpe = plot_rpe(rows, title, args.out)
        figs.append(fig_rpe)
        mg = (t_gt >= t0) & (t_gt <= fusion.fused_t[-1])       # só o trecho processado: laço só liga KFs dele
        rv = gt_revisits(t_gt[mg], T_gt[mg], **REVISIT)
        covered, rec = loop_coverage(rv, loops, REVISIT['cover'])
        figs.append(plot_revisits(rv, covered, rec, loops, t0, origin, title, args.out))
        print("Figuras: " + ", ".join(figs))
        save_figure_data(dados_dir, rows, rpe, rv, covered, loops, t0, origin, nt, sc, vo_lost_t)
        if lc is not None and lc.stats.get('log'):
            fig_f = loop_funnel(lc.stats['log'], loops, t_gt, T_gt, rv, t0, title, args.out, dados_dir,
                                LoopConfig().min_f_inliers, args.node_dt)
            print(f"Funil dos laços: {fig_f} | {dados_dir}/funil_lacos.csv")

    # mapa do VO (mesmas figuras do testE_vo_time_satmp.py)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    draw_snapshot(ax, plot_snapshot(vo, with_points=True),
                  unit="~m" if feedback else "unid.",
                  title="Odometria Visual Monocular" + (" (escala realimentada pela fusão)" if feedback else
                                                       " (escala arbitrária)"))
    save_fig(fig, f"{args.out}_vo_trajetoria_xz")
    # o mesmo mapa no mundo fundido (Sim3 de cada nó, após a otimização final) + GT
    if len(res['poses']):
        ws = fusion.world_snapshot(vo, with_points=True)
        if ws is not None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            draw_snapshot(ax, ws, axes=(0, 1), unit="m", title="Mapa do VO no mundo fundido (após a otimização final)")
            if gt_view is not None:
                g = gt_view.line(fusion.fused_t, fusion.fused)
                if g is not None:
                    ax.plot(g[:, 0], g[:, 1], color=COLORS["GT"], lw=1.2, alpha=0.7, label="GT")
                    ax.legend(loc='upper right')
            print(f"Mapa no mundo fundido: {save_fig(fig, f'{args.out}_mapa_fundido_xy')}")
        fusion.save_world_map(vo, dados("mapa_fundido.npz"))      # p/ o view_map.py (mundo fundido, metros)
    if fusion.save_loop_snapshots(dados("grafo_lacos.npz")):     # p/ o make_gifs.py (grafo antes/depois dos laços)
        print(f"Estados do grafo nos laços: {dados('grafo_lacos.npz')}")
    vo.Map.plot_map_points(mode='3D', save=f"{args.out}_mapa_3d.{FIG_FORMAT}", show=False)
    print(f"Mapa do VO: {args.out}_vo_trajetoria_xz.{FIG_FORMAT}, {args.out}_mapa_3d.{FIG_FORMAT}")
    if timing:
        fig_p = processing_report(timing, vo.ba_log, fusion.be_log, lc.stats.get('log', []) if lc is not None else [],
                                  args.node_dt, dict(laco_fim=t_lc_finish, lote=t_batch, total=t_wall),
                                  f"Fusão {'+'.join(sensors)}", args.out, dados_dir)
        print(f"Processamento: {fig_p} | {dados_dir}/tempos_*.csv")
    if gui:
        plt.show(block=True)
    plt.close('all')

    if view3d is not None:
        print("Janelas 3D (Open3D) continuam abertas: feche-as para encerrar.")
        view3d.proc.join()
        view3d.close()
    vo.cleanup_plots()


if __name__ == "__main__":
    main()
