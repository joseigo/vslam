"""
BA.py: bundle adjustment local isolado, numa janela sintética (KFs e landmarks conhecidos).

- Converge: poses livres perturbadas voltam ao valor verdadeiro (os 2 KFs fixos travam gauge e escala).
- Observações outlier aparecem em rej_obs; os KFs fixos não se mexem.
- snapshot_ba_problem: só entram landmarks válidos, com >= 2 observações na janela e >= 1 KF livre.
- run_bundle_adjustment escreve o resultado no mapa.
- Sanidade: resultado que muda a baseline > max_step_change é descartado.
"""
import numpy as np
from gtsam import Pose3, Rot3

from _comum import metrica, figuras_ativas, nova_figura, salvar_figura, COR, K_URBAN, IMG_W, IMG_H, make_T, cam_yaw_R, rot_err_deg
from Map import Map
from KeyFrame import KeyFrame
from BA import snapshot_ba_problem, solve_ba, run_bundle_adjustment


def _window(rng, n_kf=6, n_pts=400, noise_px=0.5, n_outliers=10):
    T_true = [make_T(cam_yaw_R(np.radians(1.0 * k)), [0.1 * k, 0.0, 1.0 * k]) for k in range(n_kf)]
    X = np.column_stack([rng.uniform(-10, 10, n_pts), rng.uniform(-3, 2, n_pts), rng.uniform(8, 35, n_pts)])
    m = Map()
    for k, T in enumerate(T_true):
        m.add_keyframe(k, float(k), T)
    outliers = []
    for X_i in X:
        obs = []
        for k, T in enumerate(T_true):
            Tc = np.linalg.inv(T)
            Xc = Tc[:3, :3] @ X_i + Tc[:3, 3]
            uv = (K_URBAN @ Xc)[:2] / Xc[2]
            if Xc[2] > 0 and 0 <= uv[0] < IMG_W and 0 <= uv[1] < IMG_H:
                obs.append((k, tuple(uv + rng.normal(0, noise_px, 2))))
        if len(obs) >= 2:
            m.add_point(X_i, None, obs)
    pids = list(m.points)
    for pid in rng.choice(pids, n_outliers, replace=False):
        p = m.points[int(pid)]
        kf, uv = p.observations[-1]
        p.observations[-1] = (kf, (uv[0] + 15.0, uv[1] - 12.0))
        outliers.append((int(pid), kf))
    return m, T_true, X, outliers


def _perturb(m, rng, free, rot_deg=0.5, trans=0.05, pt_sigma=0.15):
    for k in free:
        dR = Rot3.Expmap(np.radians(rot_deg) * rng.normal(0, 1, 3) / np.sqrt(3)).matrix()
        m.poses[k] = m.poses[k] @ make_T(dR, rng.normal(0, trans, 3))
    for p in m.points.values():
        p.position = p.position + rng.normal(0, pt_sigma, 3)


def _pose_errs(poses, T_true, kfs):
    et = [np.linalg.norm(poses[k][:3, 3] - T_true[k][:3, 3]) for k in kfs]
    er = [rot_err_deg(poses[k][:3, :3], T_true[k][:3, :3]) for k in kfs]
    return max(et), max(er)


def _fig_ba(free, before, after, T_true):
    fig, axs = nova_figura(ncols=2, w=3.6, h=3.0)
    x = np.arange(len(free))
    w = 0.38
    for ax, idx, unit, nome in ((axs[0], 0, 100, "translação [cm]"), (axs[1], 1, 1, "rotação [°]")):
        e0 = [unit * _pose_errs(before, T_true, [k])[idx] for k in free]
        e1 = [unit * _pose_errs(after, T_true, [k])[idx] for k in free]
        ax.bar(x - w / 2 - 0.01, e0, w, color=COR['neutro'], label="perturbada")
        ax.bar(x + w / 2 + 0.01, e1, w, color=COR['vo'], label="após o BA")
        ax.set_xticks(x)
        ax.set_xticklabels([f"KF{k}" for k in free])
        ax.set_ylabel(f"erro de {nome}")
        ax.grid(axis="x", visible=False)
    axs[0].legend(loc="upper left")
    fig.suptitle("BA local: KFs livres (KF0 e KF1 fixos)", fontsize=9.5)
    fig.tight_layout()
    salvar_figura(fig, "ba_erro_por_kf")


def test_ba_converge_e_rejeita_outliers():
    rng = np.random.default_rng(40)
    m, T_true, X, outliers = _window(rng)
    active = list(range(6))
    free = active[2:]
    _perturb(m, rng, free)
    et0, er0 = _pose_errs(m.poses, T_true, free)
    before = dict(m.poses)
    prob = snapshot_ba_problem(m, active, n_fixed=2)
    res = solve_ba(prob, K_URBAN, verbose=False)
    assert res is not None
    poses = dict(m.poses)
    poses.update(res['poses'])
    et1, er1 = _pose_errs(poses, T_true, free)
    if figuras_ativas():
        _fig_ba(free, before, poses, T_true)
    rej = set(res['rej_obs'])
    hit = sum(o in rej for o in outliers)
    print(f"erro das poses livres: {100 * et0:.1f} cm / {er0:.2f}° -> {100 * et1:.2f} cm / {er1:.3f}°; "
          f"outliers rejeitados {hit}/{len(outliers)}, obs. boas rejeitadas {len(rej) - hit}; "
          f"{len(res['points'])} landmarks; solve {1000 * res['dt']:.0f} ms")
    metrica("pose_erro_t_antes", et0, "m")
    metrica("pose_erro_t_depois", et1, "m")
    metrica("pose_erro_r_depois", er1, "deg")
    metrica("outliers_rejeitados", hit)
    metrica("obs_boas_rejeitadas", len(rej) - hit)
    metrica("tempo_solve", res['dt'], "s")
    assert set(res['poses']) == set(free)                # fixos não são devolvidos (não mudam)
    assert et1 < 0.03 and er1 < 0.05
    assert et1 < 0.3 * et0
    assert hit == len(outliers)
    assert len(rej) - hit <= 0.005 * sum(len(p.observations) for p in m.points.values())


def test_ba_pontos_recuperados():
    """Landmarks perturbados (15 cm) voltam ao piso de ruído: o erro do BA partindo da perturbação fica igual ao do
    BA partindo da verdade (o que sobra é a incerteza de profundidade com 0.5 px, ~z^2 / (f b) x 0.5 px)."""
    rng = np.random.default_rng(41)
    m, T_true, X, _ = _window(rng, n_outliers=0)
    X_true = {pid: p.position.copy() for pid, p in m.points.items()}
    ref = solve_ba(snapshot_ba_problem(m, list(range(6))), K_URBAN, verbose=False)
    _perturb(m, rng, [2, 3, 4, 5])
    e0 = np.median([np.linalg.norm(m.points[pid].position - X_true[pid]) for pid in m.points])
    res = solve_ba(snapshot_ba_problem(m, list(range(6))), K_URBAN, verbose=False)
    e1 = np.median([np.linalg.norm(res['points'][pid] - X_true[pid]) for pid in res['points']])
    e_ref = np.median([np.linalg.norm(ref['points'][pid] - X_true[pid]) for pid in ref['points']])
    print(f"erro mediano dos landmarks: perturbado {100 * e0:.1f} cm -> BA {100 * e1:.1f} cm "
          f"(BA partindo da verdade: {100 * e_ref:.1f} cm)")
    metrica("landmark_erro_mediano_antes", e0, "m")
    metrica("landmark_erro_mediano_depois", e1, "m")
    metrica("landmark_erro_mediano_ref", e_ref, "m")
    assert e1 < 0.7 * e0
    assert e1 < 1.15 * e_ref


def test_snapshot_filtra_landmarks():
    rng = np.random.default_rng(42)
    m, _, _, _ = _window(rng, n_kf=5, n_pts=50, n_outliers=0)
    so_fixos = m.add_point([0, 0, 10.0], None, [(0, (600.0, 260.0)), (1, (601.0, 260.0))])
    uma_obs = m.add_point([0, 0, 12.0], None, [(3, (600.0, 260.0))])
    invalido = m.add_point([0, 0, 14.0], None, [(2, (600.0, 260.0)), (3, (601.0, 260.0))])
    m.invalidate_point(invalido.id)
    fora = m.add_point([0, 0, 16.0], None, [(2, (600.0, 260.0)), (9, (601.0, 260.0))])
    prob = snapshot_ba_problem(m, [0, 1, 2, 3, 4], n_fixed=2)
    ids = {pid for pid, _, _ in prob['points']}
    assert so_fixos.id not in ids                         # só KFs fixos: não ajuda os livres
    assert uma_obs.id not in ids and invalido.id not in ids and fora.id not in ids
    assert len(ids) > 0
    assert snapshot_ba_problem(m, [0, 1], n_fixed=2) is None


def test_run_bundle_adjustment_escreve_no_mapa():
    rng = np.random.default_rng(43)
    m, T_true, _, outliers = _window(rng)
    _perturb(m, rng, [2, 3, 4, 5])
    kfs = {k: KeyFrame(k, float(k), k, m.poses[k]) for k in m.poses}
    assert run_bundle_adjustment(m, kfs, K_URBAN, list(range(6)), verbose=False)
    et, er = _pose_errs(m.poses, T_true, [2, 3, 4, 5])
    assert et < 0.03 and er < 0.05
    for k in kfs:
        assert np.allclose(kfs[k].pose, m.poses[k])
    for pid, kf in outliers:                              # observação rejeitada saiu do mapa
        assert all(o[0] != kf for o in m.points[pid].observations) or not m.points[pid].is_valid


def test_ba_descarta_resultado_que_muda_baseline():
    """Uma janela cuja inicialização tem baseline 3x a real (escala errada) é 'corrigida' pelo BA além de
    max_step_change -> o resultado é descartado (proteção contra colapso/explosão da escala)."""
    rng = np.random.default_rng(44)
    m, T_true, _, _ = _window(rng, n_outliers=0)
    for k in (3, 4, 5):                                   # KFs 0, 1 fixos; 3-5 com baseline 3x desde o KF2
        m.poses[k] = m.poses[k].copy()
        m.poses[k][:3, 3] = m.poses[2][:3, 3] + 3.0 * (T_true[k][:3, 3] - T_true[2][:3, 3])
    res = solve_ba(snapshot_ba_problem(m, list(range(6))), K_URBAN, verbose=False, max_step_change=2.0)
    assert res is None
    res = solve_ba(snapshot_ba_problem(m, list(range(6))), K_URBAN, verbose=False, max_step_change=10.0)
    assert res is not None
