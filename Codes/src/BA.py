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
#
# Gauge da janela, dois modos (keyframe_params['ba_mode']):
#   'fix'  (padrão): os n_fixed KFs mais antigos da janela recebem prior forte; o KF que sai da janela é descartado.
#   'marg': o KF que sai é MARGINALIZADO (complemento de Schur) num prior linear sobre os KFs que ficam, com o
#           Jacobiano fixo no ponto de linearização (FEJ). Só o 1º BA do segmento fixa n_fixed KFs (define o gauge);
#           depois o gauge e a escala vêm do prior. Os landmarks não são estados persistentes do BA (são relidos do
#           mapa a cada janela), então o prior é a informação sobre as poses que sai junto com o KF:
#               Lambda = S(G_total) - S(G_readd),  eta idem,
#           S(.) = complemento de Schur sobre as poses que ficam (elimina landmarks e KFs que saem), G_readd = fatores
#           que voltam no próximo BA (só envolvem KFs que ficam). Lambda >= 0 porque S(.) é monótono em fatores.

def warmup() -> bool:
    """Chamada no worker logo ao criar o processo: paga o custo de importar o gtsam antes do 1º BA."""
    return True


def snapshot_ba_problem(map_obj, active_kf_ids: List[int], n_fixed: int = 2, prior: Optional[dict] = None,
                        marg: bool = False) -> Optional[dict]:
    """Copia do mapa só o necessário p/ o BA da janela (poses + landmarks observados por >=2 KFs da janela e
    por >=1 KF livre). O resultado é independente do mapa: o tracking pode continuar mexendo nele.
    marg=True com `prior` (do BA anterior): os KFs do prior que saíram da janela entram como variáveis a marginalizar
    e nenhum KF é fixo (o gauge vem do prior)."""
    active = [k for k in active_kf_ids if k in map_obj.poses]
    if len(active) < n_fixed + 1:
        return None
    if not marg:
        prior = None
    leaving = [k for k in prior['keys'] if k not in set(active)] if prior is not None else []
    if prior is not None:
        n_fixed = 0
    kfs = sorted(set(active) | set(leaving))
    free = set(kfs) - set(active[:n_fixed])
    active_set = set(kfs)
    cand = set()
    for kf in kfs:
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
    poses = {kf: np.asarray(map_obj.poses[kf] if kf in map_obj.poses else prior['lin'][kf], dtype=np.float64).copy()
             for kf in kfs}
    return dict(active=active, n_fixed=n_fixed, poses=poses, points=points, leaving=leaving, prior=prior, marg=marg)


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
             min_parallax_deg: float = 1.0, point_prior_frac: float = 0.2,
             min_depth_frac: float = 0.01) -> Optional[dict]:
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
    leaving, prior, marg = problem.get('leaving', []), problem.get('prior'), problem.get('marg', False)
    kfs = sorted(set(active) | set(leaving))
    stay = set(active)
    K = np.asarray(K, dtype=np.float64)
    calib = Cal3_S2(K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2])

    graph = NonlinearFactorGraph()
    initial = Values()

    tagged = []                        # (marg) (fator, landmark ou None, volta?) p/ refazer os grafos sem pontos ruins

    def add(f, again=True, pid=None):
        graph.add(f)
        if marg:
            tagged.append((f, pid, again))

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4] * 6))
    meas_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Isotropic.Sigma(2, 1.0))

    # ---- poses ----
    for kf in kfs:
        initial.insert(sh.X(kf), Pose3(T_old[kf]))
    fixed = active[:n_fixed]
    free = set(kfs) - set(fixed)
    for kf in fixed:
        add(PriorFactorPose3(sh.X(kf), Pose3(T_old[kf]), prior_noise), again=False)
    if prior is not None:
        add(marginal_prior_factor(prior), again=False)

    # ---- odometria fraca entre KFs consecutivos ----
    # Em rotação (curvas) a translação entre KFs é quase inobservável: o custo de reprojeção fica plano ao
    # longo da baseline e o BA pode colapsar KFs vizinhos (baseline -> 0), destruindo a escala. Um prior FRACO
    # na pose relativa (sigma de translação = odom_trans_frac * |t|) segura a baseline nesses casos, mas
    # perde para os dados quando a geometria é bem condicionada.
    for a, b in zip(kfs[:-1], kfs[1:]):
        if a not in free and b not in free:
            continue
        rel = np.linalg.inv(T_old[a]) @ T_old[b]
        sig_t = max(odom_trans_frac * np.linalg.norm(rel[:3, 3]), 1e-4)
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([odom_rot_sigma] * 3 + [sig_t] * 3))
        add(gtsam.BetweenFactorPose3(sh.X(a), sh.X(b), Pose3(rel), noise), again=a in stay and b in stay)

    # ---- landmarks ----
    # paralaxe de cada landmark (vetorizado): menor ângulo entre os raios câmera->ponto das suas observações
    pts = problem['points']
    pids = np.array([pid for pid, _, _ in pts], np.int64)
    pos = np.array([X for _, X, _ in pts], np.float64)
    nobs = np.array([len(obs) for _, _, obs in pts])
    o_pt = np.repeat(np.arange(len(pts)), nobs)                               # observação -> índice do ponto
    o_kf = np.array([kf for _, _, obs in pts for kf, _ in obs], np.int64)
    o_uv = np.array([uv for _, _, obs in pts for _, uv in obs], np.float64).reshape(-1, 2)
    centers = {kf: T_old[kf][:3, 3] for kf in kfs}
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
            add(gtsam.PriorFactorPoint3(
                sh.L(pid), gtsam.Point3(*X),
                gtsam.noiseModel.Isotropic.Sigma(3, max(point_prior_frac * float(mean_nr[i]), 1e-6))), pid=pid)
        for kf, uv in obs:
            add(gtsam.GenericProjectionFactorCal3_S2(
                np.array([uv[0], uv[1]], dtype=np.float64), meas_noise, sh.X(kf), sh.L(pid), calib),
                again=kf in stay, pid=pid)

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
    anchor = fixed[-1] if fixed else active[0]
    # (marg) a âncora também é livre: a distância até ela não serve de escala p/ ela mesma -> baseline média da janela
    base = 1e-6 if fixed else float(np.mean([np.linalg.norm(T_old[b][:3, 3] - T_old[a][:3, 3])
                                             for a, b in zip(active[:-1], active[1:])]))
    for kf in free & stay:
        p = result.atPose3(sh.X(kf))
        T = np.eye(4)
        T[:3, :3] = p.rotation().matrix()
        T[:3, 3] = p.translation()
        old = T_old[kf]
        step = np.linalg.norm(T[:3, 3] - old[:3, 3])
        scale = max(np.linalg.norm(old[:3, 3] - T_old[anchor][:3, 3]), base)
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
    T_cw = {kf: np.linalg.inv(new_poses.get(kf, T_old[kf])) for kf in kfs}   # inverte cada pose UMA vez
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
    out = dict(poses=new_poses, points=points, rej_obs=rej_obs, invalid=invalid)
    if marg:
        # Pontos inválidos, com observação rejeitada ou colados a uma câmera (profundidade ~0: Jacobiano da projeção
        # explode e a eliminação fica indeterminada) saem do mapa / são outliers: não entram no prior.
        drop = bad.copy()
        drop[o_pt[rej]] = True
        depth = np.where(np.isfinite(Xc[:, 2]), Xc[:, 2], 0.0)
        med = float(np.median(depth[depth > 0])) if (depth > 0).any() else np.inf
        np.logical_or.at(drop, o_pt, depth < min_depth_frac * med)
        drop_pids = set(int(p) for p in pids[drop])
        keep_pt = ~drop
        # Landmark com um fator só num grafo (1 observação, ou só o prior de baixa paralaxe) não informa nada sobre as
        # poses depois de eliminado (o Jacobiano 2x3 da projeção tem posto 2: o complemento de Schur é zero). Mas o
        # bloco 3x3 dele fica com posto 2 + o amortecimento ao longo do raio, e num ponto perto da câmera (|J| ~ f/z)
        # a razão passa de 1e7 e a eliminação acusa sistema indeterminado. Tirá-lo daquele grafo é exato.
        sel = [(f, pid, again) for f, pid, again in tagged if pid is None or pid not in drop_pids]
        n_full, n_readd = {}, {}
        for _, pid, again in sel:
            if pid is not None:
                n_full[pid] = n_full.get(pid, 0) + 1
                n_readd[pid] = n_readd.get(pid, 0) + again
        full, readd = NonlinearFactorGraph(), NonlinearFactorGraph()
        for f, pid, again in sel:
            if pid is None or n_full[pid] > 1:
                full.add(f)
            if again and (pid is None or n_readd[pid] > 1):
                readd.add(f)
        try:
            out['prior'] = _marginalize(full, readd, result, active, leaving, pids[keep_pt], mean_nr[keep_pt])
        except RuntimeError as e:      # ex.: IndeterminantLinearSystemException -> próximo BA volta a fixar KFs
            print(f"[BA] Marginalização falhou ({type(e).__name__}: {e}); reiniciando o gauge.")
            out['prior'] = None
    return out


# ------------------------------------------------------------------ marginalização (ba_mode='marg')
def _marginalize(graph, readd, result, active, leaving, pids, mean_nr, rel_eig=1e-12) -> dict:
    """Prior linear {keys, lin, R, r} sobre as poses de `active`: 0.5 ||R dx - r||^2, dx = lin.localCoordinates(X)."""
    keep = [sh.X(k) for k in active]
    hess = []
    for g, elim in ((graph, [sh.X(k) for k in leaving]), (readd, [])):
        lin = g.linearize(result)
        # amortecimento desprezível: landmark visto por 1 KF só em G_readd e poses sem fator não quebram a eliminação
        for pid, d in zip(pids, mean_nr):
            sig = max(float(d), 1e-6)
            lin.add(gtsam.JacobianFactor(sh.L(int(pid)), np.eye(3) / sig, np.zeros(3), gtsam.noiseModel.Unit.Create(3)))
        for k in keep:
            lin.add(gtsam.JacobianFactor(k, 1e-6 * np.eye(6), np.zeros(6), gtsam.noiseModel.Unit.Create(6)))
        order = gtsam.Ordering()
        for k in [sh.L(int(p)) for p in pids] + elim:
            order.push_back(k)
        _, rem = lin.eliminatePartialMultifrontal(order)
        order_keep = gtsam.Ordering()
        for k in keep:
            order_keep.push_back(k)
        hess.append(rem.hessian(order_keep))
    L = hess[0][0] - hess[1][0]
    eta = hess[0][1] - hess[1][1]
    w, V = np.linalg.eigh(0.5 * (L + L.T))
    ok = w > rel_eig * max(w.max(), 1e-300)
    sq = np.sqrt(w[ok])
    R = sq[:, None] * V[:, ok].T
    r = (V[:, ok].T @ eta) / sq
    lin_poses = {k: _pose_matrix(result.atPose3(sh.X(k))) for k in active}
    return dict(keys=list(active), lin=lin_poses, R=R, r=r)


def _pose_matrix(p: Pose3) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = p.rotation().matrix()
    T[:3, 3] = p.translation()
    return T


def marginal_prior_factor(prior: dict) -> gtsam.CustomFactor:
    """Fator do prior marginal: e = R [lin_k.localCoordinates(X_k)]_k - r, Jacobiano constante R (FEJ)."""
    keys = [sh.X(k) for k in prior['keys']]
    lin = [Pose3(prior['lin'][k]) for k in prior['keys']]
    R, r = prior['R'], prior['r']
    blocks = [np.asfortranarray(R[:, 6 * i:6 * i + 6]) for i in range(len(keys))]

    def err(this, values, H):
        d = np.concatenate([lin[i].localCoordinates(values.atPose3(k)) for i, k in enumerate(keys)])
        if H is not None:
            for i in range(len(keys)):
                H[i] = blocks[i]
        return R @ d - r
    return gtsam.CustomFactor(gtsam.noiseModel.Unit.Create(R.shape[0]), keys, err)


def scale_prior(prior: Optional[dict], kfs, c0: np.ndarray, s: float) -> Optional[dict]:
    """Acompanha uma semelhança aplicada às poses `kfs` (centros c0 + s (t - c0), rotações iguais): move o ponto de
    linearização e troca a unidade das colunas de translação dessas poses (dx_t' = s dx_t)."""
    if prior is None:
        return None
    kfs = set(kfs)
    lin, R = dict(prior['lin']), prior['R'].copy()
    for i, k in enumerate(prior['keys']):
        if k in kfs:
            T = lin[k].copy()
            T[:3, 3] = c0 + s * (T[:3, 3] - c0)
            lin[k] = T
            R[:, 6 * i + 3:6 * i + 6] /= s
    return dict(prior, lin=lin, R=R)


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
                          n_fixed: int = 2, verbose: bool = True, prior: Optional[dict] = None, marg: bool = False,
                          **kw) -> Optional[dict]:
    """BA síncrono (snapshot + solve + apply na mesma thread). Devolve o resultado (None se falhou)."""
    problem = snapshot_ba_problem(map_obj, active_kf_ids, n_fixed, prior=prior, marg=marg)
    if problem is None:
        return None
    res = solve_ba(problem, K, verbose=verbose, **kw)
    if res is None:
        return None
    apply_ba_result(map_obj, keyframes, res, verbose=verbose)
    return res
