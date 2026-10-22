import time
import numpy as np
import gtsam
from gtsam import Cal3_S2, Pose3, Rot3, PriorFactorPose3
from gtsam import NonlinearFactorGraph, Values
from gtsam import symbol_shorthand as sh   # sh.X(i) poses, sh.L(j) landmarks
from typing import List, Optional


# O BA é dividido em 3 etapas p/ poder rodar em OUTRO PROCESSO (em paralelo com o tracking):
#   snapshot_ba_problem (thread principal) -> solve_ba (worker, só dados "picklable") -> apply_ba_result (thread principal)
# run_bundle_adjustment encadeia as três (modo síncrono, comportamento original).

def warmup() -> bool:
    """Chamada no worker logo ao criar o processo: paga o custo de importar o gtsam antes do 1º BA."""
    return True


def snapshot_ba_problem(map_obj, active_kf_ids: List[int], n_fixed: int = 2) -> Optional[dict]:
    """Copia do mapa só o necessário p/ o BA da janela (poses + landmarks observados por >=2 KFs da janela e
    por >=1 KF livre). O resultado é independente do mapa: o tracking pode continuar mexendo nele."""
    active = [k for k in active_kf_ids if k in map_obj.poses]
    if len(active) < n_fixed + 1:
        return None
    free = set(active[n_fixed:])
    active_set = set(active)
    cand = set()
    for kf in active:
        cand |= map_obj.kf_obs.get(kf, set())

    points = []        # (pid, posição, [(kf, uv), ...])
    for pid in cand:
        pt = map_obj.points.get(pid)
        if pt is None or not pt.is_valid:
            continue
        obs = [(kf, uv) for kf, uv in pt.observations if kf in active_set]
        # uma observação por keyframe
        obs = list({kf: (kf, uv) for kf, uv in obs}.values())
        if len(obs) < 2 or not any(kf in free for kf, _ in obs):
            continue
        points.append((pid, pt.position.copy(), obs))
    if not points:
        return None
    return dict(active=active, n_fixed=n_fixed,
                poses={kf: np.asarray(map_obj.poses[kf], dtype=np.float64).copy() for kf in active},
                points=points)


def solve_ba(problem: dict, K: np.ndarray, *args, **kwargs) -> Optional[dict]:
    """_solve_ba + tempo de CPU do worker em res['dt'] (métrica de processamento)."""
    t0 = time.perf_counter()
    res = _solve_ba(problem, K, *args, **kwargs)
    if res is not None:
        res['dt'] = time.perf_counter() - t0
        res['n_active'] = len(problem.get('active', ()))
    return res


def _solve_ba(problem: dict, K: np.ndarray, max_iters: int = 20, reproj_reject_px: float = 4.0,
             verbose: bool = True, odom_rot_sigma: float = 0.05,
             odom_trans_frac: float = 0.25, max_step_change: float = 2.0,
             min_parallax_deg: float = 1.0, point_prior_frac: float = 0.2) -> Optional[dict]:
    """
    Bundle Adjustment local (janela deslizante), monocular. Função pura (pode rodar em outro processo).
    Retorna None se falhou / foi descartado, senão {'poses', 'points', 'rej_obs', 'invalid'}.

    Condicionamento dos landmarks: um ponto visto com paralaxe < `min_parallax_deg` na janela tem profundidade
    quase inobservável; o otimizador a ajusta "de graça" ao longo do raio e arrasta poses/escala junto (ou deixa o
    sistema singular). Esses pontos recebem um prior FRACO na posição (sigma = `point_prior_frac` x profundidade):
    ainda restringem a rotação, mas não conseguem mais puxar a escala.

    Gauge (por que fixar DOIS keyframes): em monocular existem 7 graus de liberdade não observáveis
    (6 de pose global + escala). Fixar só a primeira pose deixa a escala livre e a janela pode
    "respirar" (o mapa encolhe/estica). Fixando as duas poses mais antigas (que já foram otimizadas
    em janelas anteriores) a escala fica travada e continua consistente ao longo da trajetória.
    """
    active, n_fixed, T_old = problem['active'], problem['n_fixed'], problem['poses']
    K = np.asarray(K, dtype=np.float64)
    calib = Cal3_S2(K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2])

    graph = NonlinearFactorGraph()
    initial = Values()

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4] * 6))
    meas_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Isotropic.Sigma(2, 1.0))

    # ---- poses ----
    for kf in active:
        initial.insert(sh.X(kf), Pose3(T_old[kf]))
    fixed = active[:n_fixed]
    free = set(active[n_fixed:])
    for kf in fixed:
        graph.add(PriorFactorPose3(sh.X(kf), Pose3(T_old[kf]), prior_noise))

    # ---- odometria fraca entre KFs consecutivos ----
    # Em rotação (curvas) a translação entre KFs é quase inobservável: o custo de reprojeção fica plano ao
    # longo da baseline e o BA pode colapsar KFs vizinhos (baseline -> 0), destruindo a escala. Um prior FRACO
    # na pose relativa (sigma de translação = odom_trans_frac * |t|) segura a baseline nesses casos, mas
    # perde para os dados quando a geometria é bem condicionada.
    for a, b in zip(active[:-1], active[1:]):
        if a not in free and b not in free:
            continue
        rel = np.linalg.inv(T_old[a]) @ T_old[b]
        sig_t = max(odom_trans_frac * np.linalg.norm(rel[:3, 3]), 1e-4)
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([odom_rot_sigma] * 3 + [sig_t] * 3))
        graph.add(gtsam.BetweenFactorPose3(sh.X(a), sh.X(b), Pose3(rel), noise))

    # ---- landmarks ----
    # paralaxe de cada landmark (vetorizado): menor ângulo entre os raios câmera->ponto das suas observações
    pts = problem['points']
    pids = np.array([pid for pid, _, _ in pts], np.int64)
    pos = np.array([X for _, X, _ in pts], np.float64)
    nobs = np.array([len(obs) for _, _, obs in pts])
    o_pt = np.repeat(np.arange(len(pts)), nobs)                               # observação -> índice do ponto
    o_kf = np.array([kf for _, _, obs in pts for kf, _ in obs], np.int64)
    o_uv = np.array([uv for _, _, obs in pts for _, uv in obs], np.float64).reshape(-1, 2)
    centers = {kf: T_old[kf][:3, 3] for kf in active}
    rays = pos[o_pt] - np.array([centers[kf] for kf in o_kf])
    nr = np.linalg.norm(rays, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        rays = rays / nr[:, None]
    ok_nr = np.logical_and.reduceat(nr > 1e-9, np.r_[0, np.cumsum(nobs)[:-1]])
    mean_nr = np.add.reduceat(nr, np.r_[0, np.cumsum(nobs)[:-1]]) / nobs
    min_cos = np.ones(len(pts))
    start = np.r_[0, np.cumsum(nobs)[:-1]]
    for k in np.unique(nobs):                                                  # agrupa por nº de observações
        sel = np.where(nobs == k)[0]
        R = rays[start[sel][:, None] + np.arange(k)]                            # (n, k, 3)
        min_cos[sel] = np.einsum('nik,njk->nij', R, R).reshape(len(sel), -1).min(axis=1)
    ang = np.degrees(np.arccos(np.clip(min_cos, -1.0, 1.0)))
    weak = ok_nr & (ang < min_parallax_deg)

    for i, (pid, X, obs) in enumerate(pts):
        initial.insert(sh.L(pid), gtsam.Point3(*X))
        if weak[i]:
            graph.add(gtsam.PriorFactorPoint3(
                sh.L(pid), gtsam.Point3(*X),
                gtsam.noiseModel.Isotropic.Sigma(3, max(point_prior_frac * float(mean_nr[i]), 1e-6))))
        for kf, uv in obs:
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                np.array([uv[0], uv[1]], dtype=np.float64), meas_noise, sh.X(kf), sh.L(pid), calib))

    if verbose:
        print(f"[BA] {len(active)} KFs ({len(free)} livres), {len(problem['points'])} landmarks, "
              f"{graph.size()} fatores")

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(max_iters)
    params.setRelativeErrorTol(1e-4)
    try:
        result = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()
    except Exception as e:      # ex.: IndeterminantLinearSystemException
        print(f"[BA] Falhou ({type(e).__name__}); mantendo estimativa anterior.")
        return None

    # ---- checagem de sanidade: BA não pode "explodir" as poses ----
    new_poses = {}
    for kf in free:
        p = result.atPose3(sh.X(kf))
        T = np.eye(4)
        T[:3, :3] = p.rotation().matrix()
        T[:3, 3] = p.translation()
        old = T_old[kf]
        step = np.linalg.norm(T[:3, 3] - old[:3, 3])
        scale = max(np.linalg.norm(old[:3, 3] - T_old[fixed[-1]][:3, 3]), 1e-6)
        if not np.isfinite(T).all() or step > 5.0 * scale:
            print("[BA] Resultado suspeito (pose divergiu); descartando.")
            return None
        new_poses[kf] = T

    # baseline entre KFs consecutivos não pode encolher/crescer absurdamente
    P = {kf: (new_poses[kf][:3, 3] if kf in new_poses else T_old[kf][:3, 3]) for kf in active}
    for a, b in zip(active[:-1], active[1:]):
        o = np.linalg.norm(T_old[b][:3, 3] - T_old[a][:3, 3])
        n = np.linalg.norm(P[b] - P[a])
        if o > 1e-6 and not (1.0 / max_step_change <= n / o <= max_step_change):
            print(f"[BA] Baseline KF{a}->{b} mudou {n / o:.2f}x; descartando resultado.")
            return None

    # ---- landmarks e outliers pelo erro de reprojeção pós-BA (vetorizado) ----
    T_cw = {kf: np.linalg.inv(new_poses.get(kf, T_old[kf])) for kf in active}   # inverte cada pose UMA vez
    Xn = np.array([result.atPoint3(sh.L(int(pid))) for pid in pids], np.float64).reshape(-1, 3)
    finite = np.isfinite(Xn).all(axis=1)
    Rs = np.array([T_cw[kf][:3, :3] for kf in o_kf])
    ts = np.array([T_cw[kf][:3, 3] for kf in o_kf])
    Xc = np.einsum('mij,mj->mi', Rs, Xn[o_pt]) + ts
    with np.errstate(divide='ignore', invalid='ignore'):
        uv_hat = (Xc / Xc[:, 2:3]) @ K.T
        err = np.linalg.norm(uv_hat[:, :2] - o_uv, axis=1)
    behind = np.zeros(len(pts), bool)
    np.logical_or.at(behind, o_pt, ~(Xc[:, 2] > 1e-9))
    bad = ~finite | (behind & finite)
    rej = (err > reproj_reject_px) & ~bad[o_pt]
    points = {int(pids[i]): Xn[i] for i in np.where(finite)[0]}
    rej_obs = [(int(pids[o_pt[m]]), int(o_kf[m])) for m in np.where(rej)[0]]
    invalid = [int(p) for p in pids[bad]]
    return dict(poses=new_poses, points=points, rej_obs=rej_obs, invalid=invalid)


def apply_ba_result(map_obj, keyframes: dict, res: dict, verbose: bool = True):
    """Escreve o resultado do BA no mapa (thread principal)."""
    for kf, T in res['poses'].items():
        if kf in map_obj.poses:
            map_obj.poses[kf] = T
        if kf in keyframes:
            keyframes[kf].pose = T
    for pid, X in res['points'].items():
        pt = map_obj.points.get(pid)
        if pt is not None:
            pt.position = X
    for pid, kf in res['rej_obs']:
        map_obj.remove_observation(pid, kf)
    bad = set(res['invalid'])
    for pid in res['points']:
        pt = map_obj.points.get(pid)
        if pt is not None and len(pt.observations) < 2:
            bad.add(pid)
    for pid in bad:
        map_obj.invalidate_point(pid)
    if verbose:
        print(f"[BA] ok. obs rejeitadas: {len(res['rej_obs'])}, landmarks invalidados: {len(bad)}")


def run_bundle_adjustment(map_obj, keyframes: dict, K: np.ndarray, active_kf_ids: List[int],
                          n_fixed: int = 2, verbose: bool = True, **kw) -> bool:
    """BA síncrono (snapshot + solve + apply na mesma thread)."""
    problem = snapshot_ba_problem(map_obj, active_kf_ids, n_fixed)
    if problem is None:
        return False
    res = solve_ba(problem, K, verbose=verbose, **kw)
    if res is None:
        return False
    apply_ba_result(map_obj, keyframes, res, verbose=verbose)
    return True
