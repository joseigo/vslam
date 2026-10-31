"""
Métricas de avaliação isoladas (scripts/test_gpo.py, scripts/test_vo_fusion_rt.py) contra resposta conhecida e o evo.

- evaluate / align (Umeyama): trajetória = Sim3 do GT -> ATE 0 e escala certa; ATE igual ao APE do evo.
- align_at_start: o 1º ponto coincide; deriva linear dá erro final conhecido.
- rpe_segments: estimativa perfeita -> 0; erro de escala de 2% -> RPE de translação = 2%.
"""
import numpy as np

from _comum import metrica, planar_trajectory, yaw_R, skip


def _gt(n=600):
    T = planar_trajectory(n, dt=0.5, speed=4.0)
    t = 1e3 + np.arange(n) * 0.5
    return t, T


def test_ate_umeyama_sim3():
    from test_gpo import evaluate
    t, T = _gt()
    s, R, off = 0.37, yaw_R(0.8), np.array([10.0, -4.0, 2.0])
    P = (T[:, :3, 3] @ R.T) * s + off
    r = evaluate(t, P, t, T, with_scale=True)
    assert r['ate'] < 1e-9
    assert abs(r['scale'] - 1 / s) < 1e-9
    r = evaluate(t, P, t, T, with_scale=False)          # sem escala: erro grande
    assert r['ate'] > 10


def test_ate_igual_ape_do_evo():
    from test_gpo import evaluate
    try:
        from evo.core import metrics
        from evo.core.trajectory import PoseTrajectory3D
    except ImportError:
        skip("evo não instalado")
    rng = np.random.default_rng(70)
    t, T = _gt()
    P = T[:, :3, 3] + np.cumsum(rng.normal(0, 0.05, (len(t), 3)), axis=0)   # passeio aleatório (deriva)
    for with_scale in (False, True):
        Pi = P * (0.5 if with_scale else 1.0)
        ours = evaluate(t, Pi, t, T, with_scale)['ate']
        q = np.tile([1.0, 0, 0, 0], (len(t), 1))
        ref = PoseTrajectory3D(positions_xyz=T[:, :3, 3], orientations_quat_wxyz=q, timestamps=t)
        est = PoseTrajectory3D(positions_xyz=Pi, orientations_quat_wxyz=q, timestamps=t)
        est.align(ref, correct_scale=with_scale)
        ape = metrics.APE(metrics.PoseRelation.translation_part)
        ape.process_data((ref, est))
        evo_rmse = ape.get_statistic(metrics.StatisticsType.rmse)
        print(f"{'Sim3' if with_scale else 'SE3 '}: ATE do pipeline {ours:.6f} m, APE do evo {evo_rmse:.6f} m")
        metrica(f"diff_ate_evo_{'sim3' if with_scale else 'se3'}", abs(ours - evo_rmse), "m")
        assert abs(ours - evo_rmse) < 1e-6 * max(1.0, evo_rmse)


def _fusion_module():
    try:
        import test_vo_fusion_rt as m
    except Exception as e:                               # (Viewer/open3d etc.)
        skip(f"test_vo_fusion_rt não importa: {type(e).__name__}: {e}")
    return m


def test_align_at_start_e_rpe():
    m = _fusion_module()
    t, T = _gt()
    origin = T[0, :3, 3]
    e = m.describe(t, T, t, T, origin, with_scale=False)
    assert e['ate0'] < 1e-9 and e['final'] < 1e-9
    rpe = m.rpe_segments(e, lengths=(100, 200))
    for L, v in rpe.items():
        assert np.abs(v['t']).max() < 1e-9 and np.abs(v['r']).max() < 1e-6
    # erro de escala de 2% (sem alinhar escala) num trecho RETO: RPE de translação = 2% em qualquer segmento
    # (em curva o RPE fica menor: o erro é proporcional à corda e o RPE divide pelo comprimento percorrido)
    T = planar_trajectory(len(t), dt=0.5, speed=4.0, yaw_rate=lambda _: 0.0)
    T2 = T.copy()
    T2[:, :3, 3] *= 1.02
    e2 = m.describe(t, T2, t, T, origin, with_scale=False)
    rpe2 = m.rpe_segments(e2, lengths=(100, 200))
    med = {L: float(np.median(v['t'])) for L, v in rpe2.items()}
    print(f"RPE de translação com escala 1.02: {med} (esperado 2 %); erro final {e2['final']:.2f} m")
    for L, v in rpe2.items():
        assert np.allclose(v['t'], 2.0, atol=0.05)
    # erro final = 2% da distância em linha reta até o fim (origem coincidente)
    assert abs(e2['final'] - 0.02 * np.linalg.norm(T[-1, :3, 3] - T[0, :3, 3])) < 1e-6
