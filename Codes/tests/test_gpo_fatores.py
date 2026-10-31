"""
GPO.py: fatores do grafo isolados.

- Jacobianas analíticas do fator do VO / laço (_vo_edge, _vo_factor_error) contra diferenças finitas centrais.
  Uma Jacobiana errada não quebra nada visivelmente: só converge mais devagar ou para um ponto pior.
- Fator-bloco do VO (_vo_block, vo_chunk > 1) equivalente às arestas individuais com Huber (mesma linearização).
- GraphPoseOptimizer com vo_chunk=1 e vo_chunk=20 dão a mesma estimativa.
- Convenção câmera -> veículo (vo_rel_vehicle).
"""
import numpy as np
import gtsam
from gtsam import Pose3, Rot3
from gtsam.symbol_shorthand import X, S

from _comum import metrica, rand_pose, make_T, planar_trajectory, relative, noisy_rel
from GPO import _vo_edge, _vo_factor_error, GraphPoseOptimizer, GPOConfig


def _num_jac(f, x0, retract, dim, eps=1e-6):
    J = np.zeros((6, dim))
    for k in range(dim):
        d = np.zeros(dim)
        d[k] = eps
        J[:, k] = (f(retract(x0, d)) - f(retract(x0, -d))) / (2 * eps)
    return J


def _pose_retract(P, d):
    return P.compose(Pose3.Expmap(d))


def test_jacobianas_vo_edge():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(50):
        Xi, Xj, Z = rand_pose(rng), rand_pose(rng), rand_pose(rng)
        lam = float(rng.normal(0, 1))
        zr, zt = Z.rotation(), np.asarray(Z.translation())
        H = [None] * 3
        _vo_edge(zr, zt, Xi, Xj, lam, H)
        Ni = _num_jac(lambda P: _vo_edge(zr, zt, P, Xj, lam), Xi, _pose_retract, 6)
        Nj = _num_jac(lambda P: _vo_edge(zr, zt, Xi, P, lam), Xj, _pose_retract, 6)
        Nl = _num_jac(lambda l: _vo_edge(zr, zt, Xi, Xj, float(l[0])), np.array([lam]), lambda a, d: a + d, 1)
        for A, N in ((H[0], Ni), (H[1], Nj), (H[2], Nl)):
            worst = max(worst, float(np.abs(A - N).max() / max(1.0, np.abs(N).max())))
    print(f"maior erro relativo das Jacobianas (50 casos): {worst:.2e}")
    metrica("jacobiana_erro_rel_max", worst)
    assert worst < 1e-6


def test_vo_factor_error_igual_vo_edge():
    """O CustomFactor do laço/VO (_vo_factor_error) calcula o mesmo erro/Jacobianas que _vo_edge."""
    rng = np.random.default_rng(1)
    Xi, Xj, Z = rand_pose(rng), rand_pose(rng), rand_pose(rng)
    lam = 0.4
    zr, zt = Z.rotation(), np.asarray(Z.translation())
    v = gtsam.Values()
    v.insert(X(0), Xi)
    v.insert(X(1), Xj)
    v.insert(S(0), np.array([lam]))
    f = gtsam.CustomFactor(gtsam.noiseModel.Unit.Create(6), [X(0), X(1), S(0)],
                           lambda this, vv, H: _vo_factor_error(zr, zt, this, vv, H))
    He, Hf = [None] * 3, [None] * 3
    e1 = _vo_edge(zr, zt, Xi, Xj, lam, He)
    e2 = _vo_factor_error(zr, zt, f, v, Hf)
    assert np.allclose(e1, e2, atol=1e-12)
    for a, b in zip(He, Hf):
        assert np.allclose(a, b, atol=1e-12)
    # erro zero quando a medida é exatamente o movimento relativo (com a escala)
    Zs = Xi.between(Xj)
    e0 = _vo_edge(Zs.rotation(), np.asarray(Zs.translation()) / np.exp(lam), Xi, Xj, lam)
    assert np.abs(e0).max() < 1e-9


def _hessian(graph, values, ordering):
    return graph.linearize(values).augmentedHessian(ordering)


def test_bloco_vo_equivale_arestas_individuais():
    """_vo_block (1 CustomFactor com K arestas) == K fatores robustos (Huber) individuais: mesma Hessiana aumentada,
    inclusive com arestas dentro e fora da região quadrática do Huber."""
    rng = np.random.default_rng(2)
    cfg = GPOConfig(vo_chunk=1)
    g_ind = GraphPoseOptimizer(cfg)
    n = 8
    T = planar_trajectory(n + 1, speed=3.0)
    g_ind.add_first_node(0.0, 0.0)
    for j in range(1, n + 1):
        g_ind.add_node(j * 0.5, Pose3(T[j]))
    vals = gtsam.Values()
    edges = []
    for k in range(n):
        rel = relative(T[k], T[k + 1])
        # metade das arestas com erro grande (Huber ativo)
        big = k % 2 == 0
        rel_m = noisy_rel(rel, rng, 0.05 if big else 0.0005, 0.3 if big else 0.002)
        st = cfg.vo_trans_frac * 1.5 + cfg.vo_trans_min
        edges.append(dict(i=k, j=k + 1, s=k, z_rot=Rot3(rel_m[:3, :3]), z_t=rel_m[:3, 3].copy(),
                          sig=np.array([cfg.vo_rot_sigma] * 3 + [st] * 3), idx=None))
    for k in range(n + 1):
        vals.insert(X(k), Pose3(T[k]) if k == 0 else Pose3(T[k]).compose(Pose3.Expmap(rng.normal(0, 0.001, 6))))
        vals.insert(S(k), np.array([rng.normal(0, 0.005)]))

    ind = gtsam.NonlinearFactorGraph()
    for e in edges:
        noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(cfg.vo_huber),
                                               gtsam.noiseModel.Diagonal.Sigmas(e['sig']))
        ind.add(gtsam.CustomFactor(noise, [X(e['i']), X(e['j']), S(e['s'])],
                                   lambda this, v, H, zr=e['z_rot'], zt=e['z_t']: _vo_factor_error(zr, zt, this, v, H)))
    blk = gtsam.NonlinearFactorGraph()
    blk.add(g_ind._vo_block(edges))
    keys = [X(k) for k in range(n + 1)] + [S(k) for k in range(n)]
    ordering = gtsam.Ordering()
    for k in keys:
        ordering.push_back(k)
    Hi, Hb = _hessian(ind, vals, ordering), _hessian(blk, vals, ordering)
    rel_diff = np.abs(Hi - Hb).max() / np.abs(Hi).max()
    n_huber = sum(np.linalg.norm(_vo_edge(e['z_rot'], e['z_t'], vals.atPose3(X(e['i'])), vals.atPose3(X(e['j'])),
                                          float(vals.atVector(S(e['s']))[0])) / e['sig']) > cfg.vo_huber
                  for e in edges)
    print(f"arestas com Huber ativo: {n_huber}/{n}; diferença relativa das Hessianas: {rel_diff:.2e}")
    metrica("hessiana_diff_rel", rel_diff)
    metrica("arestas_huber_ativo", n_huber)
    assert 2 <= n_huber < n
    assert rel_diff < 1e-9


def _run_gpo(vo_chunk, T, vo, wheel):
    g = GraphPoseOptimizer(GPOConfig(vo_chunk=vo_chunk))
    g.add_first_node(0.0, 0.0)
    for j in range(1, len(T)):
        i = j - 1
        guess = g.pose(i).compose(Pose3(wheel[i]))
        g.add_node(j * 0.5, guess)
        g.add_vo(i, j, Pose3(vo[i]), float(np.linalg.norm(wheel[i][:3, 3])))
        g.add_wheel(i, j, Pose3(wheel[i]), float(np.linalg.norm(wheel[i][:3, 3])), float(np.arctan2(wheel[i][1, 0],
                                                                                                  wheel[i][0, 0])))
        g.update()
    online = g.poses().copy()
    g.finalize(batch=True)
    return online, g.poses(), g.n_vo_chunks


def test_gpo_chunk_equivale_sem_chunk():
    """Agrupar o VO em blocos no ISAM2 (otimização de tempo) não muda a solução."""
    rng = np.random.default_rng(3)
    n = 70
    T = planar_trajectory(n)
    vo = [noisy_rel(relative(T[k], T[k + 1]), rng, 0.003, 0.02) for k in range(n - 1)]
    wh = [noisy_rel(relative(T[k], T[k + 1]), rng, 0.002, 0.01) for k in range(n - 1)]
    on1, b1, _ = _run_gpo(1, T, vo, wh)
    on20, b20, nch = _run_gpo(20, T, vo, wh)
    d_on = np.abs(on1[:, :3, 3] - on20[:, :3, 3]).max()
    d_b = np.abs(b1[:, :3, 3] - b20[:, :3, 3]).max()
    print(f"blocos criados: {nch}; diferença máx. de posição: online {d_on:.2e} m, lote {d_b:.2e} m")
    metrica("diff_pos_online", d_on, "m")
    metrica("diff_pos_lote", d_b, "m")
    assert nch >= 2
    assert d_b < 1e-4
    assert d_on < 0.05          # online: o ISAM2 relineariza em momentos diferentes (tolerância folgada)


def test_vo_rel_vehicle_convencao():
    """Câmera andando p/ frente (+z_cam) = veículo andando p/ frente (+x_veh); câmera virando p/ a direita
    (+yaw em torno do y_cam, que aponta p/ baixo) = veículo virando p/ a direita (yaw negativo em z_veh)."""
    g = GraphPoseOptimizer()
    rel = g.vo_rel_vehicle(np.eye(4), make_T(t=[0, 0, 2.0]))
    assert np.allclose(rel.translation(), [2.0, 0, 0], atol=1e-12)
    a = 0.1
    c, s = np.cos(a), np.sin(a)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    rel = g.vo_rel_vehicle(np.eye(4), make_T(R=Ry))
    yaw = Rot3.Logmap(rel.rotation())
    assert np.allclose(yaw, [0, 0, -a], atol=1e-12)
    # câmera andando p/ a direita (+x_cam) = veículo p/ a direita (-y_veh)
    rel = g.vo_rel_vehicle(np.eye(4), make_T(t=[1.0, 0, 0]))
    assert np.allclose(rel.translation(), [0, -1.0, 0], atol=1e-12)
