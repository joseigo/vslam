"""
GPO.py: modelos de sensor isolados, com resposta analítica.

- WheelOdometry: reta, arco de raio conhecido, frações nas bordas (t0/t1 entre amostras), composição.
- PoseOdometry: movimento relativo no referencial do robô.
- ImuData.integrate: parado (só gravidade), rotação pura, aceleração constante.
- visual_inertial_scale_init: recupera a escala do VO de uma trajetória sintética.
- image_stamp_ns: KAIST (ns no nome) e Inomi (horário local UTC-3).
"""
import numpy as np
import gtsam
from gtsam import Pose3, Rot3

from _comum import metrica, planar_trajectory, yaw_R
from GPO import WheelOdometry, PoseOdometry, ImuData, image_stamp_ns, visual_inertial_scale_init

RES, D, BASE = 4096, 0.623803, 1.52683          # calibração do KAIST (resolução, diâmetro, bitola)


def _ticks(dist_m):
    return np.round(np.asarray(dist_m) * RES / (np.pi * D)).astype(np.int64)


def _wheel(dl, dr, rate=100.0):
    """Encoder sintético a partir das distâncias ACUMULADAS de cada roda (m)."""
    t_ns = (np.arange(len(dl)) / rate * 1e9).astype(np.int64)
    return WheelOdometry(t_ns, _ticks(dl), _ticks(dr), resolution=RES, d_left=D, d_right=D, wheel_base=BASE)


def test_roda_reta():
    n = 501                                       # 5 s a 100 Hz, 2 m/s
    s = np.linspace(0, 10.0, n)
    w = _wheel(s, s)
    rel, dist, dyaw = w.relative(0.0, 5.0)
    assert abs(dist - 10.0) < 1e-3
    assert abs(dyaw) < 1e-9
    assert np.allclose(rel.translation(), [10.0, 0, 0], atol=1e-3)


def test_roda_arco_raio_conhecido():
    """Curva à esquerda de raio R por um ângulo th: x = R sin th, y = R (1 - cos th)."""
    R, th = 20.0, np.pi / 2
    n = 1001
    a = np.linspace(0, th, n)
    w = _wheel((R - BASE / 2) * a, (R + BASE / 2) * a)
    rel, dist, dyaw = w.relative(0.0, (n - 1) / 100.0)
    exp = np.array([R * np.sin(th), R * (1 - np.cos(th)), 0.0])
    err = np.linalg.norm(np.asarray(rel.translation()) - exp)
    print(f"arco R={R} m, 90°: erro de posição {err * 100:.2f} cm, yaw {np.degrees(dyaw):.3f}°, "
          f"distância {dist:.3f} m (esperado {R * th:.3f})")
    metrica("arco_erro_pos", err, "m")
    metrica("arco_erro_yaw", np.degrees(dyaw - th), "deg")
    assert abs(dyaw - th) < 1e-3
    assert abs(dist - R * th) < 1e-2
    assert err < 0.02


def test_roda_fracao_nas_bordas_e_composicao():
    """t0/t1 entre amostras: só a fração do incremento que cai no intervalo entra; rel(a,b)∘rel(b,c) = rel(a,c)."""
    rate, v = 10.0, 3.0                           # encoder lento (10 Hz) p/ a fração importar
    n = 101
    t = np.arange(n) / rate
    s = v * t
    yaw_rate = 0.2
    dth = yaw_rate * t
    w = _wheel(s - dth * BASE / 2, s + dth * BASE / 2, rate)
    _, dist, dyaw = w.relative(1.03, 2.57)        # bordas no meio das amostras
    assert abs(dist - v * (2.57 - 1.03)) < 2e-3
    assert abs(dyaw - yaw_rate * (2.57 - 1.03)) < 2e-3
    a, b, c = 0.37, 4.21, 8.88
    Tab = w.relative(a, b)[0]
    Tbc = w.relative(b, c)[0]
    Tac = w.relative(a, c)[0]
    d = Tab.compose(Tbc).between(Tac)
    print(f"composição: erro {np.linalg.norm(d.translation()) * 1000:.3f} mm")
    metrica("composicao_erro", np.linalg.norm(d.translation()), "m")
    assert np.linalg.norm(d.translation()) < 5e-3
    assert np.linalg.norm(Rot3.Logmap(d.rotation())) < 1e-6
    assert w.covers(0.0, 10.0) and not w.covers(-0.1, 1.0) and not w.covers(0.0, 10.1)


def test_roda_ignora_dados_fora_do_intervalo():
    """Um pulo grande nos ticks DEPOIS de t1 não pode vazar p/ dentro de [t0, t1]."""
    n = 201
    s = np.linspace(0, 4.0, n)
    s[150:] += 100.0
    w = _wheel(s, s)
    _, dist, _ = w.relative(0.0, 1.0)
    assert abs(dist - 2.0) < 1e-3


def test_pose_odometry_relativa():
    """Robô em (2, 1) virado 90°, anda 1 m p/ frente e gira +30°: no ref. dele, (1, 0) e yaw 30°."""
    t = np.array([0.0, 1.0])
    po = PoseOdometry(t, [2.0, 2.0], [1.0, 2.0], [np.pi / 2, np.pi / 2 + np.radians(30)])
    rel, dist, dyaw = po.relative(0.0, 1.0)
    assert np.allclose(rel.translation(), [1.0, 0, 0], atol=1e-12)
    assert abs(dist - 1.0) < 1e-12
    assert abs(dyaw - np.radians(30)) < 1e-12
    # ré: distância negativa
    po = PoseOdometry(t, [0.0, -1.0], [0.0, 0.0], [0.0, 0.0])
    _, dist, _ = po.relative(0.0, 1.0)
    assert dist < 0
    # theta com salto de 2pi (unwrap)
    po = PoseOdometry(t, [0.0, 0.0], [0.0, 0.0], [np.pi - 0.05, -np.pi + 0.05])
    assert abs(po.relative(0.0, 1.0)[2] - 0.1) < 1e-12


def _pim(g=9.81):
    p = gtsam.PreintegrationParams.MakeSharedU(g)
    p.setAccelerometerCovariance(np.eye(3) * 1e-4)
    p.setGyroscopeCovariance(np.eye(3) * 1e-6)
    p.setIntegrationCovariance(np.eye(3) * 1e-8)
    return gtsam.PreintegratedImuMeasurements(p, gtsam.imuBias.ConstantBias())


def _predict(pim, v0=np.zeros(3)):
    return pim.predict(gtsam.NavState(Pose3(), v0), gtsam.imuBias.ConstantBias())


def test_imu_parado_so_gravidade():
    """Parado e nivelado: acelerômetro mede +g em z (força específica); a posição não pode derivar."""
    t = np.arange(0, 3.0, 0.01)
    imu = ImuData(t, np.zeros((len(t), 3)), np.tile([0, 0, 9.81], (len(t), 1)))
    st = _predict(imu.integrate(_pim(), 0.5, 2.5))
    assert np.linalg.norm(st.position()) < 1e-9
    assert np.linalg.norm(st.velocity()) < 1e-9


def test_imu_rotacao_e_aceleracao():
    t = np.arange(0, 3.0, 0.01)
    gyro = np.tile([0, 0, 0.5], (len(t), 1))
    imu = ImuData(t, gyro, np.tile([0, 0, 9.81], (len(t), 1)))
    st = _predict(imu.integrate(_pim(), 0.0, 2.0))
    assert abs(Rot3.Logmap(st.attitude())[2] - 1.0) < 1e-6
    # aceleração constante de 1 m/s² p/ frente: x = a t² / 2
    acc = np.tile([1.0, 0, 9.81], (len(t), 1))
    imu = ImuData(t, np.zeros((len(t), 3)), acc)
    st = _predict(imu.integrate(_pim(), 0.0, 2.0))
    assert abs(st.position()[0] - 2.0) < 1e-6
    assert abs(st.velocity()[0] - 2.0) < 1e-6
    # fração nas bordas: [0.005, 1.005] = exatamente 1 s de integração
    st = _predict(imu.integrate(_pim(), 0.005, 1.005))
    assert abs(st.velocity()[0] - 1.0) < 1e-9
    # slice: o pedaço enviado ao back-end integra igual ao todo
    sl = imu.slice(0.333, 1.777)
    st_a = _predict(imu.integrate(_pim(), 0.333, 1.777))
    st_b = _predict(sl.integrate(_pim(), 0.333, 1.777))
    assert np.allclose(st_a.position(), st_b.position(), atol=1e-12)


def test_inicializacao_visual_inercial_escala():
    """VO = posição real / s_true; a IMU (sintética, sem ruído) deve devolver s_true."""
    from fusao_offline import simulate_imu
    dt_gt = 0.01
    n = 3001                                       # 30 s
    t_gt = np.arange(n) * dt_gt
    # velocidade variando (acelera e freia): a escala só é observável com aceleração
    speed = 5.0 + 3.0 * np.sin(2 * np.pi * t_gt / 10.0)
    yaw = 0.3 * np.sin(2 * np.pi * t_gt / 15.0)
    T = np.tile(np.eye(4), (n, 1, 1))
    pos = np.zeros(3)
    for k in range(1, n):
        pos = pos + speed[k] * dt_gt * np.array([np.cos(yaw[k]), np.sin(yaw[k]), 0])
        T[k, :3, :3] = yaw_R(yaw[k])
        T[k, :3, 3] = pos
    imu = simulate_imu(t_gt, T, acc_noise=0.0, gyro_noise=0.0)     # (mantém o viés constante pequeno do simulador)
    s_true = 7.5
    nodes = np.arange(50, 2000, 50)                          # 0.5 s entre nós, 20 s
    pims, dts = [], []
    for a, b in zip(nodes[:-1], nodes[1:]):
        pims.append(imu.integrate(_pim(), t_gt[a], t_gt[b]))
        dts.append(t_gt[b] - t_gt[a])
    T0inv = np.linalg.inv(T[nodes[0]])
    Tn = [T0inv @ T[k] for k in nodes]                        # mundo do VO = veículo no 1º nó (nivelado)
    s, vels = visual_inertial_scale_init([x[:3, :3] for x in Tn], [x[:3, 3] / s_true for x in Tn], pims, dts)
    v_true = speed[nodes[0]]
    print(f"escala estimada {s:.3f} (verdadeira {s_true}); |v0| {np.linalg.norm(vels[0]):.2f} m/s "
          f"(verdadeira {v_true:.2f})")
    metrica("escala_vi_erro_rel", s / s_true - 1)
    metrica("v0_erro", np.linalg.norm(vels[0]) - v_true, "m/s")
    assert abs(s / s_true - 1) < 0.03
    assert abs(np.linalg.norm(vels[0]) - v_true) < 0.3


def test_image_stamp_ns():
    assert image_stamp_ns("/a/b/1544590798702227700.png") == 1544590798702227700
    # Inomi: 2024-01-01 12:00:00.500000 local (UTC-3) = 15:00:00.5 UTC
    import calendar
    import datetime
    utc = calendar.timegm(datetime.datetime(2024, 1, 1, 15, 0, 0).timetuple())
    assert image_stamp_ns("20240101_120000_500000.jpg") == utc * 10 ** 9 + 500_000_000


def test_acha_encoder_em_sensor_data():
    """KAIST: as sequências mais novas trazem o encoder em sensor_data/ (as antigas em data_sensor/); o pipeline acha
    os dois (antes, urban31/37 rodavam sem roda, em silêncio)."""
    import os
    import tempfile
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from rodar_sistema import find_dataset_files
    for sub in ("sensor_data", "data_sensor"):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, sub))
            os.makedirs(os.path.join(d, "stereo_left"))
            open(os.path.join(d, sub, "encoder.csv"), "w").write("0,0,0\n")
            open(os.path.join(d, "global_pose.csv"), "w").write("0" + ",0" * 12 + "\n")
            _, f = find_dataset_files(os.path.join(d, "stereo_left"))
            print(f"{sub}: encoder={os.path.relpath(f['encoder'], d) if f['encoder'] else None}")
            assert f["encoder"] == os.path.join(d, sub, "encoder.csv")
            assert f["gt"] == os.path.join(d, "global_pose.csv")
