"""
Teste da otimização em grafo (GPO.py): funde a trajetória do VO com odometria de roda e/ou IMU e compara com o GT.

Entrada: a trajetória salva pelo VO (trajectory_tum.txt, poses T_wc da câmera) + pasta do dataset KAIST
(data_sensor/encoder.csv, poses/global_pose.csv e, se houver, data_sensor/xsens_imu.csv).

Uso:
    python3 fusao_offline.py --vo trajectory_tum.txt
    python3 fusao_offline.py --vo trajectory_tum.txt --sim-imu          # IMU simulada a partir do GT (sem xsens_imu.csv)
    python3 fusao_offline.py --vo trajectory_tum.txt --modes vo+wheel --no-batch

Modos: wheel | vo+wheel | vo+imu | wheel+imu | vo+wheel+imu   (o VO puro é sempre avaliado como referência)
"""
import argparse
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import savgol_filter
import gtsam
from gtsam import Pose3

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from GPO import GraphPoseOptimizer, GPOConfig, WheelOdometry, ImuData, visual_inertial_scale_init

DATASET = "/home/joseigo/Documentos/TCC/Datasets/urban26"


# ------------------------------------------------------------------ dados
def load_tum(path):
    d = np.loadtxt(path)
    T = np.tile(np.eye(4), (len(d), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(d[:, 4:8]).as_matrix()
    T[:, :3, 3] = d[:, 1:4]
    return d[:, 0] * 1e-9, T


def load_gt(path):
    """t, R00 R01 R02 tx R10 ... (3x4 por linha). KAIST: sem cabeçalho, t em ns. Inomi (poses_gt_unix.csv): com
    cabeçalho, t em s."""
    with open(path) as fh:
        first = fh.readline().split(",")[0].strip()
    head = not first.replace(".", "", 1).replace("-", "", 1).isdigit()
    d = np.loadtxt(path, delimiter=",", skiprows=1 if head else 0, ndmin=2)
    T = np.tile(np.eye(4), (len(d), 1, 1))
    T[:, :3, :4] = d[:, 1:13].reshape(-1, 3, 4)
    t = d[:, 0] * (1e-9 if d[0, 0] > 1e12 else 1.0)
    return t, T


def lost_flags(T_vo, tol=1e-12):
    """Frames em que o VO não atualizou a pose (inicializando / PnP falhou): pose idêntica à anterior."""
    same = np.r_[True, np.all(np.abs(np.diff(T_vo, axis=0)) < tol, axis=(1, 2))]
    return same


def simulate_imu(t_gt, T_gt, rate=100.0, acc_noise=0.02, gyro_noise=0.002, seed=0):
    """IMU SINTÉTICA a partir do GT (só p/ validar o código quando o dataset não tem IMU).
    Acelerômetro = força específica R^T (a + g); giroscópio = Log(R_k^T R_k+1)/dt. + ruído branco + viés constante."""
    rng = np.random.default_rng(seed)
    t = np.arange(t_gt[0], t_gt[-1], 1.0 / rate)
    p = np.column_stack([np.interp(t, t_gt, T_gt[:, i, 3]) for i in range(3)])
    R = Slerp(t_gt, Rotation.from_matrix(T_gt[:, :3, :3]))(t)
    dt = 1.0 / rate
    a = savgol_filter(p, 41, 3, deriv=2, delta=dt, axis=0)            # suaviza o GT antes de derivar 2x
    Rm = R.as_matrix()
    acc = np.einsum('nji,nj->ni', Rm, a + np.array([0, 0, 9.81]))
    rv = (R[:-1].inv() * R[1:]).as_rotvec() / dt
    gyro = savgol_filter(np.vstack([rv, rv[-1:]]), 21, 3, axis=0)
    ba, bg = rng.normal(0, 0.05, 3), rng.normal(0, 0.002, 3)
    acc += ba + rng.normal(0, acc_noise * np.sqrt(rate), acc.shape)
    gyro += bg + rng.normal(0, gyro_noise * np.sqrt(rate), gyro.shape)
    return ImuData(t, gyro, acc)


# ------------------------------------------------------------------ avaliação
def align(A, B, with_scale):
    """Umeyama: s R A + t ~ B. Retorna A alinhado e s."""
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, Sv, Vt = np.linalg.svd(B0.T @ A0 / len(A))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(Sv) @ D) / A0.var(0).sum() if with_scale else 1.0
    return (s * A0 @ R.T) + mb, s


def evaluate(t, P, t_gt, T_gt, with_scale):
    i = np.clip(np.searchsorted(t_gt, t), 1, len(t_gt) - 1)
    i = np.where(np.abs(t_gt[i - 1] - t) < np.abs(t_gt[i] - t), i - 1, i)
    ok = np.abs(t_gt[i] - t) < 0.02
    G = T_gt[i[ok], :3, 3]
    A, s = align(P[ok], G, with_scale)
    e = np.linalg.norm(A - G, axis=1)
    L = np.linalg.norm(np.diff(G, axis=0), axis=1).sum()
    return dict(ate=float(np.sqrt(np.mean(e ** 2))), final=float(e[-1]), drift=100 * float(e[-1]) / max(L, 1e-9),
                scale=float(s), length=float(L), aligned=A, gt=G)


# ------------------------------------------------------------------ GPO
def run_gpo(mode, nodes, t_vo, T_vo, lost, wheel, imu, cfg, batch, verbose=True):
    use_vo, use_wheel, use_imu = "vo" in mode, "wheel" in mode, "imu" in mode
    gpo = GraphPoseOptimizer(cfg, use_imu=use_imu)
    T_vc = gpo.T_vc.matrix()
    tn = t_vo[nodes]
    Tv = T_vc @ T_vo[nodes] @ np.linalg.inv(T_vc)          # poses do VO como veículo (unid. do VO)

    def lost_between(a, b):
        return bool(lost[nodes[a] + 1:nodes[b] + 1].any())

    # ---- escala inicial (log) e velocidade inicial
    s0, v0 = 1.0, np.zeros(3)
    if use_vo and use_wheel:
        dw = dv = 0.0
        for k in range(min(40, len(nodes) - 1)):
            if not lost_between(k, k + 1):
                dw += abs(wheel.relative(tn[k], tn[k + 1])[1])
                dv += np.linalg.norm(Tv[k + 1, :3, 3] - Tv[k, :3, 3])
        s0 = dw / dv if dv > 1e-9 and dw > 1.0 else 1.0
    elif use_vo and use_imu:
        W = min(30, len(nodes))                              # ~15 s de alinhamento visual-inercial
        pims = [imu.integrate(gpo.new_preintegration(), tn[k], tn[k + 1]) for k in range(W - 1)]
        s0, vels = visual_inertial_scale_init([T[:3, :3] for T in Tv[:W]], [T[:3, 3] for T in Tv[:W]],
                                              pims, np.diff(tn[:W]))
        v0 = vels[0]
        if verbose:
            print(f"  [{mode}] alinhamento visual-inercial: escala inicial = {s0:.3f} m/unid.")
        if not np.isfinite(s0) or s0 <= 0:
            s0 = 1.0
    if use_wheel and use_imu:
        v0 = np.array([wheel.relative(tn[0], tn[1])[1] / max(tn[1] - tn[0], 1e-3), 0, 0])

    gpo.add_first_node(tn[0], np.log(s0), v0)
    gpo.update()
    t_upd = []
    for j in range(1, len(nodes)):
        i = j - 1
        lost_ij = lost_between(i, j)
        Pi = gpo.pose(i)
        rel_vo = gpo.vo_rel_vehicle(T_vo[nodes[i]], T_vo[nodes[j]])
        s = float(np.exp(gpo.current_log_scale(i)))
        rel_w = dist = dyaw = None
        if use_wheel:
            rel_w, dist, dyaw = wheel.relative(tn[i], tn[j])
        pim = None
        if use_imu:
            pim = imu.integrate(gpo.new_preintegration(), tn[i], tn[j])

        # chute inicial do nó novo
        if use_wheel:
            guess = Pi.compose(rel_w)
        elif use_imu and (lost_ij or not use_vo):
            nav = pim.predict(gtsam.NavState(Pi, gpo.velocity(i)), gpo.bias)
            guess = nav.pose()
        else:
            guess = Pi.compose(Pose3(rel_vo.rotation(), s * np.asarray(rel_vo.translation())))
        vel_guess = None
        if use_imu:
            vel_guess = (np.asarray(guess.translation()) - np.asarray(Pi.translation())) / max(tn[j] - tn[i], 1e-3)
        gpo.add_node(tn[j], guess, vel_guess, vo_lost=lost_ij or not use_vo)

        if use_vo and not lost_ij:
            step = abs(dist) if use_wheel else s * np.linalg.norm(rel_vo.translation())
            gpo.add_vo(i, j, rel_vo, step)
        if use_wheel:
            gpo.add_wheel(i, j, rel_w, dist, dyaw)
        if use_imu:
            gpo.add_imu(i, j, pim)
        a = time.perf_counter()
        gpo.update()
        t_upd.append(time.perf_counter() - a)

    a = time.perf_counter()
    gpo.finalize(batch=batch)
    t_batch = time.perf_counter() - a
    return dict(t=tn, poses=gpo.poses(), scales=gpo.scales(), t_upd=np.array(t_upd), t_batch=t_batch,
                factors=dict(gpo.n_factors))


def save_tum(path, t, T):
    with open(path, "w") as f:
        for ti, P in zip(t, T):
            q = Rotation.from_matrix(P[:3, :3]).as_quat()
            f.write(f"{int(round(ti * 1e9))} {P[0, 3]:.6f} {P[1, 3]:.6f} {P[2, 3]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def plot(results, vo_eval, out_png, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw=dict(width_ratios=[1.4, 1]))
    G = vo_eval['gt']
    ax.plot(G[:, 0], G[:, 1], 'k-', lw=2.5, label='GT')
    ax.plot(vo_eval['aligned'][:, 0], vo_eval['aligned'][:, 1], '--', color='gray',
            label=f"VO puro (Sim3) ATE={vo_eval['ate']:.1f} m")
    for mode, r in results.items():
        A = r['eval']['aligned']
        ax.plot(A[:, 0], A[:, 1], lw=1.4, label=f"{mode} ATE={r['eval']['ate']:.1f} m")
        if "vo" in mode:
            ax2.plot(r['t'] - r['t'][0], r['scales'], label=mode)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(True); ax.legend(fontsize=8)
    ax.set_title("Trajetórias alinhadas ao GT (SE3; VO puro: Sim3)")
    ax2.set_xlabel("t [s]"); ax2.set_ylabel("escala estimada [m / unid. VO]"); ax2.grid(True); ax2.legend(fontsize=8)
    ax2.set_title("Escala do VO estimada no grafo")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"Figura salva em {out_png}")
    if show:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vo", default="trajectory_tum.txt", help="trajetória do VO (TUM, T_wc da câmera)")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--modes", default="wheel,vo+wheel,vo+imu,vo+wheel+imu")
    ap.add_argument("--node-every", type=int, default=5, help="1 nó a cada N frames do VO")
    ap.add_argument("--sim-imu", action="store_true", help="IMU sintética a partir do GT se não houver xsens_imu.csv")
    ap.add_argument("--no-batch", action="store_true", help="só ISAM2 (sem o LM em lote no final)")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--out", default="gpo")
    args = ap.parse_args()

    t_vo, T_vo = load_tum(args.vo)
    t_gt, T_gt = load_gt(os.path.join(args.dataset, "poses", "global_pose.csv"))
    wheel = WheelOdometry.from_csv(os.path.join(args.dataset, "data_sensor", "encoder.csv"))
    imu_path = os.path.join(args.dataset, "data_sensor", "xsens_imu.csv")
    imu = None
    if os.path.isfile(imu_path):
        imu = ImuData.from_kaist_xsens(imu_path)
        print("IMU: xsens_imu.csv")
    elif args.sim_imu:
        imu = simulate_imu(t_gt, T_gt)
        print("IMU: SIMULADA a partir do GT (só p/ validar o código; resultado otimista)")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if imu is None and any("imu" in m for m in modes):
        print("Sem IMU (não há xsens_imu.csv; use --sim-imu p/ testar): modos com IMU ignorados.")
        modes = [m for m in modes if "imu" not in m]

    # nós: frames do VO a cada N, onde VO, roda, GT (e IMU) estão disponíveis
    lost = lost_flags(T_vo)
    t_start = max(t_vo[np.argmax(~lost)], wheel.t[0], t_gt[0], imu.t[0] if imu is not None else -np.inf)
    t_end = min(t_vo[-1], wheel.t[-1], t_gt[-1], imu.t[-1] if imu is not None else np.inf)
    idx = np.where((t_vo >= t_start) & (t_vo <= t_end))[0]
    nodes = idx[::args.node_every]
    print(f"{len(nodes)} nós ({t_vo[nodes[-1]] - t_vo[nodes[0]]:.0f} s), {lost[idx].sum()} frames do VO sem pose "
          f"válida no intervalo")

    # referência: VO puro (escala arbitrária -> alinhamento Sim3)
    T_vc = GraphPoseOptimizer().T_vc.matrix()
    P_vo = (T_vc @ T_vo[nodes] @ np.linalg.inv(T_vc))[:, :3, 3]
    vo_eval = evaluate(t_vo[nodes], P_vo, t_gt, T_gt, with_scale=True)

    cfg = GPOConfig()
    results = {}
    for mode in modes:
        a = time.perf_counter()
        r = run_gpo(mode, nodes, t_vo, T_vo, lost, wheel, imu, cfg, batch=not args.no_batch)
        r['wall'] = time.perf_counter() - a
        r['eval'] = evaluate(r['t'], r['poses'][:, :3, 3], t_gt, T_gt, with_scale=False)
        results[mode] = r
        save_tum(f"{args.out}_{mode.replace('+', '_')}.txt", r['t'], r['poses'])

    L = vo_eval['length']
    print(f"\nPercurso do GT no trecho: {L:.0f} m")
    print(f"{'modo':14s} {'ATE [m]':>8s} {'erro final':>10s} {'deriva %':>8s} {'escala*':>8s} "
          f"{'ISAM2 ms/nó':>11s} {'LM lote s':>9s}  fatores")
    print(f"{'VO puro':14s} {vo_eval['ate']:8.2f} {vo_eval['final']:10.2f} {vo_eval['drift']:8.2f} "
          f"{vo_eval['scale']:8.3f} {'-':>11s} {'-':>9s}  (alinhado em Sim3)")
    for mode, r in results.items():
        e = r['eval']
        sc = evaluate(r['t'], r['poses'][:, :3, 3], t_gt, T_gt, with_scale=True)['scale']
        print(f"{mode:14s} {e['ate']:8.2f} {e['final']:10.2f} {e['drift']:8.2f} {sc:8.3f} "
              f"{1000 * r['t_upd'].mean():11.2f} {r['t_batch']:9.2f}  {r['factors']}")
    print("* escala do alinhamento Sim3 com o GT: ~1.0 = trajetória em escala métrica correta")

    plot(results, vo_eval, f"{args.out}_resultado.png", show=not args.no_show)


if __name__ == "__main__":
    main()
