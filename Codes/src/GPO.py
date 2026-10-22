"""
Otimização em grafo de poses (GPO) fundindo a odometria visual monocular com outros sensores (GTSAM).

Estado (um nó por instante t_i, pose do VEÍCULO no mundo):
    X(i)  Pose3   pose do veículo (mundo = referencial do veículo no 1º nó: x frente, y esquerda, z cima)
    S(i)  Vector1 log da escala do VO (m / unidade do VO) -- o VO monocular não tem escala e ela deriva:
                  em vez de "consertar" a escala antes, ela é estimada no grafo, com passeio aleatório
    V(i)  Vector3 velocidade no mundo        } só se houver IMU
    B(i)  imuBias viés do acelerômetro/giro  }

Fatores:
    VO     (X_i, X_j, S_i): movimento relativo do VO (convertido p/ o veículo), translação x exp(S_i). Robusto (Huber).
    escala (S_i, S_j)     : passeio aleatório da escala (maior depois de uma perda de tracking do VO)
    roda   (X_i, X_j)     : odometria de roda (modelo diferencial, plano) -> métrica
    IMU    (X_i, V_i, X_j, V_j, B_i): pré-integração (Forster et al.) + passeio aleatório do viés
    laço   (X_a, X_b, S_a): fechamento de laço (LoopClosure.py), mesmo modelo do VO (translação x exp(S_a)), mas
           4-DoF como no VINS-Mono (Qin et al. 2018): x, y, z e yaw; roll/pitch quase livres.
           Robustez (esquema do Kimera-RPGO, Rosinol et al. ICRA 2020):
             (1) porta de deriva: laço incompatível com a odometria além da deriva possível no percurso -> descartado;
             (2) PCM incremental (Mangelson et al., ICRA 2018): só entram no ISAM2 os laços do maior conjunto
                 mutuamente consistente (clique máximo, >= loop_min_clique laços);
             (3) GNC-TLS no lote final (Yang, Antonante, Tzoumas & Carlone, RA-L 2020) sobre TODOS os candidatos.
           (Um kernel DCS online foi testado e descartado: como a odometria subestima a própria deriva, o resíduo de um
            laço verdadeiro fica enorme e o DCS o desliga.)

Solver: ISAM2 (incremental: cada nó novo custa pouco -> uso online). `finalize(batch=True)` refina tudo com LM no fim.

Convenção dos referenciais: câmera (x dir., y baixo, z frente) -> veículo (x frente, y esq., z cima) via T_vc.
O braço de alavanca câmera-veículo é ignorado (só rotação), o que é razoável p/ um carro (erro ~ cm em curvas).
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import gtsam
from gtsam import Pose3, Rot3
from gtsam.symbol_shorthand import X, S, V, B

# câmera -> veículo (nominal): z_cam = x_veh, x_cam = -y_veh, y_cam = -z_veh
R_VC_NOMINAL = np.array([[0., 0., 1.],
                         [-1., 0., 0.],
                         [0., -1., 0.]])


@dataclass
class GPOConfig:
    # VO
    vo_rot_sigma: float = 0.01            # rad por aresta
    vo_trans_frac: float = 0.10           # sigma translação = frac x passo métrico esperado + min
    vo_trans_min: float = 0.05            # m
    vo_huber: float = 1.345
    # escala do VO (log)
    scale_rw_sigma: float = 0.02          # por sqrt(s): ~2%/sqrt(s) de deriva permitida
    scale_reset_sigma: float = 1.0        # depois de perda de tracking (escala pode pular)
    scale_prior_sigma: float = 0.5        # prior na escala inicial (log)
    # roda (proporcional à distância percorrida d [m] e à variação de yaw dth [rad])
    wheel_x_frac: float = 0.02
    wheel_y_frac: float = 0.01
    wheel_z_frac: float = 0.05            # modelo plano: rampas violam -> folgado
    wheel_rp_frac: float = 0.01           # roll/pitch [rad/m]
    wheel_yaw_frac: float = 0.02          # [rad/rad]
    wheel_yaw_dist: float = 0.0005        # [rad/m]
    wheel_min: float = 0.005
    # IMU (densidades contínuas; valores ~ Xsens MTi com margem p/ vibração do carro)
    gravity: float = 9.81
    acc_noise: float = 0.02               # m/s^2/sqrt(Hz)
    gyro_noise: float = 0.002             # rad/s/sqrt(Hz)
    acc_bias_rw: float = 0.002            # m/s^3/sqrt(Hz)
    gyro_bias_rw: float = 0.0002          # rad/s^2/sqrt(Hz)
    integration_sigma: float = 1e-4
    bias_prior_acc: float = 0.1
    bias_prior_gyro: float = 0.01
    vel_prior_sigma: float = 1.0
    # laço (sigmas do fator 4-DoF; roll/pitch folgados)
    loop_rp_sigma: float = 0.5            # rad
    loop_yaw_sigma: float = 0.02          # rad
    loop_trans_sigma: float = 0.5         # m (+ frac x distância relativa medida)
    loop_trans_frac: float = 0.05
    loop_gate_min: float = 15.0           # m: porta de deriva = min + frac x percurso entre os nós (estimado)
    loop_gate_frac: float = 0.10
    loop_gate_rot: float = 0.35           # rad: porta de rotação = min + frac x percurso
    loop_gate_rot_frac: float = 0.0005    # rad/m (~0.03°/m)
    loop_pcm_frac: float = 0.05           # PCM: deriva relativa da odometria entre laços (fração da distância)
    loop_pcm_rot_frac: float = 0.0005     # rad/m
    loop_pcm_chi2: float = 13.28          # chi2(4 g.l.: x, y, z, yaw) a 99%
    loop_min_clique: int = 2              # nº mínimo de laços consistentes p/ inserir online
    loop_extra_iters: int = 3             # iterações extras do ISAM2 quando o conjunto de laços muda
    loop_gnc: bool = True                 # GNC-TLS no lote final (False = LM só com os laços ativos do PCM)
    loop_gnc_chi2: float = 16.81          # limiar TLS: chi2(6 g.l.) a 99%
    loop_gnc_mu_step: float = 1.4
    # nó inicial
    prior_rp_sigma: float = 0.05          # roll/pitch iniciais desconhecidos (IMU estima a gravidade)
    prior_sigma: float = 1e-3
    # solver
    isam_relin_thresh: float = 0.1        # 0.01 -> 0.1: ~30% menos tempo no ISAM2, mesmo ATE (replay urban38)
    isam_relin_skip: int = 10            # relineariza a cada 10 updates (mesma precisão, ~2x mais rápido)
    # VO no ISAM2 em BLOCOS de vo_chunk arestas (1 = um CustomFactor por aresta, como antes).
    # Motivo: o ISAM2 avalia um CustomFactor passando o Values inteiro, que o pybind COPIA a cada chamada -> O(N) por
    # fator relinearizado (urban38: 2 -> 260 ms/nó, grafo 190 s atrasado). Um bloco de K arestas = 1 chamada (K vezes
    # menos cópias); os termos de erro e o Huber POR ARESTA são os mesmos. O lote final usa as arestas individuais.
    # (Testado e descartado: VO como LinearContainerFactor relinearizado no Python -- instável com a escala exp(S),
    #  que no urban38 salta ~15x a cada reinício do VO.)
    vo_chunk: int = 20
    isam_factorization: str = "QR"        # QR é mais robusto que Cholesky a mau condicionamento (escala x IMU)


def _vo_edge(z_rot: Rot3, z_t: np.ndarray, Xi: Pose3, Xj: Pose3, lam: float, H: Optional[list] = None) -> np.ndarray:
    """Erro (não branqueado) de uma aresta do VO; H (lista de 3) recebe d/dX_i, d/dX_j, d/dS_i."""
    s = float(np.exp(lam))
    Z = Pose3(z_rot, s * z_t)
    Hi, Hj = np.zeros((6, 6), order='F'), np.zeros((6, 6), order='F')
    Hz, Hh, Hl = (np.zeros((6, 6), order='F') for _ in range(3))
    h = Xi.between(Xj, Hi, Hj)
    d = Z.between(h, Hz, Hh)
    e = Pose3.Logmap(d, Hl)
    if H is not None:
        dZ = np.zeros((6, 1))
        dZ[3:, 0] = z_rot.matrix().T @ (s * z_t)
        H[0] = Hl @ Hh @ Hi
        H[1] = Hl @ Hh @ Hj
        H[2] = Hl @ Hz @ dZ
    return e


def _vo_factor_error(z_rot: Rot3, z_t: np.ndarray, this: gtsam.CustomFactor, values: gtsam.Values,
                     H: Optional[List[np.ndarray]]):
    """e = Log( Z(s)^-1 * (X_i^-1 X_j) ),  Z(s) = (R_z, exp(lambda) * t_z).   Jacobianas analíticas."""
    ki, kj, ks = this.keys()
    Xi, Xj = values.atPose3(ki), values.atPose3(kj)
    s = float(np.exp(values.atVector(ks)[0]))
    Z = Pose3(z_rot, s * z_t)
    Hi, Hj = np.zeros((6, 6), order='F'), np.zeros((6, 6), order='F')
    Hz, Hh, Hl = (np.zeros((6, 6), order='F') for _ in range(3))
    h = Xi.between(Xj, Hi, Hj)
    d = Z.between(h, Hz, Hh)
    e = Pose3.Logmap(d, Hl)
    if H is not None:
        dZ = np.zeros((6, 1))
        dZ[3:, 0] = z_rot.matrix().T @ (s * z_t)            # d(translação)/d(lambda) no espaço tangente de Z
        H[0] = Hl @ Hh @ Hi
        H[1] = Hl @ Hh @ Hj
        H[2] = Hl @ Hz @ dZ
    return e


def levenberg_marquardt(graph, values, max_iters=50, lam=1e-5, rel_tol=1e-5, abs_tol=1e-5):
    """LM (amortecimento lambda*I, como o padrão do GTSAM) conduzido do Python.

    Por que não gtsam.LevenbergMarquardtOptimizer: quando o C++ chama um CustomFactor com um Values criado por ele
    (os valores tentativos do LM), o pybind COPIA o Values inteiro a cada fator -> O(N^2) por iteração (urban38,
    3886 nós: ~77 s por otimização). Aqui linearize/error recebem um Values do próprio Python (sem cópia) e o
    custo fica O(N): mesma solução, ~15x mais rápido com 2400 nós (a diferença cresce com N)."""
    v = gtsam.Values(values)
    err = graph.error(v)
    zero = v.zeroVectors()
    keys = list(v.keys())
    damp = [(k, zero.at(k).shape[0]) for k in keys]
    for _ in range(max_iters):
        lin = graph.linearize(v)
        while True:
            damped = lin.clone()
            sl = np.sqrt(lam)
            for k, d in damp:
                damped.add(gtsam.JacobianFactor(k, sl * np.eye(d), np.zeros(d), gtsam.noiseModel.Unit.Create(d)))
            new = v.retract(damped.optimize())
            new_err = graph.error(new)
            if new_err <= err:
                done = (err - new_err) < max(abs_tol, rel_tol * err)
                v, err = new, new_err
                lam = max(lam / 10, 1e-10)
                break
            lam *= 10
            if lam > 1e8:               # não consegue mais diminuir o erro: convergiu
                return v
        if done:
            break
    return v


class GraphPoseOptimizer:
    def __init__(self, cfg: GPOConfig = GPOConfig(), R_vc: np.ndarray = R_VC_NOMINAL, use_imu: bool = False,
                 imu_R_vi: np.ndarray = np.eye(3)):
        self.cfg = cfg
        self.T_vc = Pose3(Rot3(R_vc), np.zeros(3))
        self.use_imu = use_imu
        p = cfg
        self.isam = gtsam.ISAM2(self._isam_params())
        self.graph = gtsam.NonlinearFactorGraph()     # fatores novos (ainda não enviados ao ISAM2)
        self.initial = gtsam.Values()
        self.vo_edges: List[dict] = []                # arestas do VO ainda individuais no ISAM2 (vo_chunk > 1)
        self._vo_pos: Dict[int, int] = {}             # posição em self.graph -> aresta (p/ achar o índice no ISAM2)
        self.n_vo_chunks = 0
        self.all_factors = gtsam.NonlinearFactorGraph()   # cópia de tudo (p/ o LM em lote no final)
        self.estimate = gtsam.Values()
        self.t: List[float] = []
        self.n_factors: Dict[str, int] = dict(vo=0, wheel=0, imu=0, loop=0)
        self.loops: List[dict] = []                    # candidatos a laço (passaram na porta); fora de all_factors
        self.loop_isam_idx: Dict[int, int] = {}        # candidato ativo -> índice do fator no ISAM2
        self.pcm_adj: Dict[int, set] = {}              # grafo de consistência par a par (PCM)

        if use_imu:
            ip = gtsam.PreintegrationParams.MakeSharedU(p.gravity)
            ip.setAccelerometerCovariance(np.eye(3) * p.acc_noise ** 2)
            ip.setGyroscopeCovariance(np.eye(3) * p.gyro_noise ** 2)
            ip.setIntegrationCovariance(np.eye(3) * p.integration_sigma ** 2)
            if not np.allclose(imu_R_vi, np.eye(3)):
                ip.setBodyPSensor(Pose3(Rot3(imu_R_vi), np.zeros(3)))
            self.imu_params = ip
            self.bias = gtsam.imuBias.ConstantBias()

    def _isam_params(self):
        prm = gtsam.ISAM2Params()
        prm.setRelinearizeThreshold(self.cfg.isam_relin_thresh)
        prm.relinearizeSkip = self.cfg.isam_relin_skip
        prm.setFactorization(self.cfg.isam_factorization)
        return prm

    # ---------------------------------------------------------------- helpers
    def vo_rel_vehicle(self, T_c1: np.ndarray, T_c2: np.ndarray) -> Pose3:
        """Movimento relativo do VO (poses T_wc da câmera, unidades do VO) expresso no referencial do veículo."""
        rel_c = Pose3(np.linalg.inv(T_c1) @ T_c2)
        return self.T_vc.compose(rel_c).compose(self.T_vc.inverse())

    def new_preintegration(self):
        return gtsam.PreintegratedImuMeasurements(self.imu_params, self.bias)

    def _add(self, f):
        self.graph.add(f)
        self.all_factors.add(f)

    def _sigma_step(self, dt):
        return self.cfg.scale_rw_sigma * np.sqrt(max(dt, 1e-3))

    # ---------------------------------------------------------------- VO em blocos (vo_chunk)
    def _isam_update(self, graph, values, remove=(), vo_pos=None):
        """isam.update + guarda o índice no ISAM2 de cada aresta do VO enviada (p/ trocá-las por um bloco depois)."""
        res = self.isam.update(graph, values, list(remove))
        if vo_pos:
            idx = res.getNewFactorsIndices()
            for pos, k in vo_pos.items():
                self.vo_edges[k]['idx'] = int(idx[pos])
        return res

    def _flush(self):
        """Envia os fatores/nós pendentes (self.graph/self.initial) ao ISAM2."""
        self._isam_update(self.graph, self.initial, vo_pos=self._vo_pos)
        self.graph, self.initial, self._vo_pos = gtsam.NonlinearFactorGraph(), gtsam.Values(), {}

    def _chunk_vo(self):
        """Troca, no ISAM2, as K arestas individuais mais antigas por um único fator-bloco equivalente."""
        K = self.cfg.vo_chunk
        while len(self.vo_edges) >= K and all(e['idx'] is not None for e in self.vo_edges[:K]):
            E, self.vo_edges = self.vo_edges[:K], self.vo_edges[K:]
            g = gtsam.NonlinearFactorGraph()
            g.add(self._vo_block(E))
            self._isam_update(g, gtsam.Values(), [e['idx'] for e in E])
            self._vo_pos = {p: k - K for p, k in self._vo_pos.items()}     # (vazio aqui: chamado após _flush)
            self.n_vo_chunks += 1

    def _vo_block(self, E):
        """K arestas do VO num CustomFactor só. Cada aresta: erro de _vo_factor_error, branqueado pelos seus sigmas e
        reponderado pelo Huber dela (como o noiseModel.Robust faz na linearização: sqrt(w) no erro e nas Jacobianas)."""
        keys, col = [], {}
        for e in E:
            for k in (X(e['i']), X(e['j']), S(e['s'])):
                if k not in col:
                    col[k] = len(keys)
                    keys.append(k)
        slots = [(col[X(e['i'])], col[X(e['j'])], col[S(e['s'])]) for e in E]
        huber = self.cfg.vo_huber
        dims = [6 if gtsam.Symbol(k).chr() == ord('x') else 1 for k in keys]
        m = 6 * len(E)

        def err(this, values, H):
            out = np.zeros(m)
            if H is not None:
                J = [np.zeros((m, d), order='F') for d in dims]
            for r, (e, (a, b, c)) in enumerate(zip(E, slots)):
                he = [None] * 3 if H is not None else None
                ev = _vo_edge(e['z_rot'], e['z_t'], values.atPose3(keys[a]), values.atPose3(keys[b]),
                              float(values.atVector(keys[c])[0]), he)
                w = ev / e['sig']
                n = np.linalg.norm(w)
                sw = 1.0 if n <= huber else np.sqrt(huber / n)
                out[6 * r:6 * r + 6] = sw * w
                if H is not None:
                    sc = (sw / e['sig'])[:, None]
                    J[a][6 * r:6 * r + 6] += sc * he[0]
                    J[b][6 * r:6 * r + 6] += sc * he[1]
                    J[c][6 * r:6 * r + 6] += sc * he[2]
            if H is not None:
                for q in range(len(keys)):
                    H[q] = J[q]
            return out
        return gtsam.CustomFactor(gtsam.noiseModel.Unit.Create(m), keys, err)

    # ---------------------------------------------------------------- nós
    def add_first_node(self, t: float, log_scale0: float = 0.0, vel0: Optional[np.ndarray] = None):
        c = self.cfg
        self.t.append(t)
        pose0 = Pose3()
        sig = np.array([c.prior_rp_sigma if self.use_imu else c.prior_sigma] * 2 + [c.prior_sigma] * 4)
        self._add(gtsam.PriorFactorPose3(X(0), pose0, gtsam.noiseModel.Diagonal.Sigmas(sig)))
        self.initial.insert(X(0), pose0)
        self._add(gtsam.PriorFactorVector(S(0), np.array([log_scale0]),
                                          gtsam.noiseModel.Isotropic.Sigma(1, c.scale_prior_sigma)))
        self.initial.insert(S(0), np.array([log_scale0]))
        if self.use_imu:
            v0 = np.zeros(3) if vel0 is None else np.asarray(vel0, float)
            self._add(gtsam.PriorFactorVector(V(0), v0, gtsam.noiseModel.Isotropic.Sigma(3, c.vel_prior_sigma)))
            self.initial.insert(V(0), v0)
            self._add(gtsam.PriorFactorConstantBias(
                B(0), self.bias, gtsam.noiseModel.Diagonal.Sigmas(
                    np.array([c.bias_prior_acc] * 3 + [c.bias_prior_gyro] * 3))))
            self.initial.insert(B(0), self.bias)
        return 0

    def add_node(self, t: float, guess: Pose3, vel_guess: Optional[np.ndarray] = None,
                 vo_lost: bool = False, log_scale_guess: Optional[float] = None) -> int:
        """Cria o nó j = (último + 1) com chute inicial. A escala segue passeio aleatório a partir do nó anterior.
        log_scale_guess: chute de S(j) (padrão: a escala do nó anterior) -- usado quando o VO reinicia."""
        c = self.cfg
        j = len(self.t)
        i = j - 1
        dt = t - self.t[i]
        self.t.append(t)
        self.initial.insert(X(j), guess)
        s_prev = self.current_log_scale(i)
        self.initial.insert(S(j), np.array([s_prev if log_scale_guess is None else log_scale_guess]))
        sig = c.scale_reset_sigma if vo_lost else self._sigma_step(dt)
        self._add(gtsam.BetweenFactorVector(S(i), S(j), np.zeros(1), gtsam.noiseModel.Isotropic.Sigma(1, sig)))
        if self.use_imu:
            self.initial.insert(V(j), np.zeros(3) if vel_guess is None else np.asarray(vel_guess, float))
            self.initial.insert(B(j), self.bias)
            sb = np.sqrt(max(dt, 1e-3))
            self._add(gtsam.BetweenFactorConstantBias(
                B(i), B(j), gtsam.imuBias.ConstantBias(),
                gtsam.noiseModel.Diagonal.Sigmas(np.array([c.acc_bias_rw * sb] * 3 + [c.gyro_bias_rw * sb] * 3))))
        return j

    # ---------------------------------------------------------------- fatores
    def add_vo(self, i: int, j: int, rel_v: Pose3, expected_step_m: float, scale_node: Optional[int] = None):
        """rel_v: movimento relativo do VO no referencial do veículo (unidades do VO). expected_step_m: passo
        métrico esperado (p/ o ruído da translação). scale_node: nó cuja escala multiplica a translação (padrão i;
        j na 1ª aresta depois de um reinício do VO, ver FusionBackend)."""
        c = self.cfg
        st = c.vo_trans_frac * expected_step_m + c.vo_trans_min
        noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(c.vo_huber),
            gtsam.noiseModel.Diagonal.Sigmas(np.array([c.vo_rot_sigma] * 3 + [st] * 3)))
        z_rot, z_t = rel_v.rotation(), np.asarray(rel_v.translation(), float)
        sn = i if scale_node is None else scale_node
        f = gtsam.CustomFactor(noise, [X(i), X(j), S(sn)],
                               lambda this, v, H: _vo_factor_error(z_rot, z_t, this, v, H))
        self.n_factors['vo'] += 1
        if self.cfg.vo_chunk > 1:
            self._vo_pos[self.graph.size()] = len(self.vo_edges)
            self.vo_edges.append(dict(i=i, j=j, s=sn, z_rot=z_rot, z_t=z_t,
                                      sig=np.array([c.vo_rot_sigma] * 3 + [st] * 3), idx=None))
        self._add(f)

    def add_wheel(self, i: int, j: int, rel: Pose3, dist: float, dyaw: float):
        c = self.cfg
        d = abs(dist)
        sig = np.array([c.wheel_rp_frac * d, c.wheel_rp_frac * d,
                        c.wheel_yaw_frac * abs(dyaw) + c.wheel_yaw_dist * d,
                        c.wheel_x_frac * d, c.wheel_y_frac * d, c.wheel_z_frac * d]) + c.wheel_min
        self._add(gtsam.BetweenFactorPose3(X(i), X(j), rel, gtsam.noiseModel.Diagonal.Sigmas(sig)))
        self.n_factors['wheel'] += 1

    def add_imu(self, i: int, j: int, pim):
        self._add(gtsam.ImuFactor(X(i), V(i), X(j), V(j), B(i), pim))
        self.n_factors['imu'] += 1

    def add_loop(self, a: int, b: int, rel_v: Pose3, gate: bool = True):
        """Candidato a laço entre os nós a (antigo) e b: rel_v = pose de b em a (veículo, unidades do VO na escala
        de a). Porta de deriva -> PCM -> (se no clique máximo) ISAM2. Retorna (ativo, info)."""
        c = self.cfg
        s = float(np.exp(self.current_log_scale(a)))
        z_rot, z_t = rel_v.rotation(), np.asarray(rel_v.translation(), float)
        d = Pose3(z_rot, s * z_t).between(self.pose(a).between(self.pose(b)))
        cum = self._cum_dist()
        path = float(abs(cum[b] - cum[a]))
        dpos, drot = float(np.linalg.norm(d.translation())), float(np.linalg.norm(Rot3.Logmap(d.rotation())))
        gate_t = c.loop_gate_min + c.loop_gate_frac * path
        gate_r = c.loop_gate_rot + c.loop_gate_rot_frac * path
        info = dict(a=a, b=b, t_a=self.t[a], t_b=self.t[b], dpos=dpos, drot_deg=float(np.degrees(drot)),
                    z_dist=float(s * np.linalg.norm(z_t)),
                    path=path, gate_t=gate_t, status='porta', w=0.0)
        if gate and (dpos > gate_t or drot > gate_r):
            return False, info
        sig = np.array([c.loop_rp_sigma, c.loop_rp_sigma, c.loop_yaw_sigma] +
                       [c.loop_trans_sigma + c.loop_trans_frac * s * np.linalg.norm(z_t)] * 3)
        k = len(self.loops)
        self.loops.append(dict(a=a, b=b, z_rot=z_rot, z_t=z_t, sig=sig, info=info))
        self.pcm_adj[k] = set()
        for j in range(k):
            if self._pcm_consistent(self.loops[j], self.loops[k], cum):
                self.pcm_adj[k].add(j)
                self.pcm_adj[j].add(k)
        self._pcm_update()
        return k in self.loop_isam_idx, info

    def _cum_dist(self):
        P = np.array([self.pose(k).translation() for k in range(len(self.t))])
        return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]

    def _loop_Z(self, L):
        return Pose3(L['z_rot'], float(np.exp(self.current_log_scale(L['a']))) * L['z_t'])

    def _pcm_consistent(self, L1, L2, cum) -> bool:
        """PCM (Mangelson et al. 2018): o ciclo laço1 -> odometria a1..a2 -> laço2 -> odometria b2..b1 deve fechar
        na identidade. Mahalanobis em (x, y, z, yaw), com a incerteza da odometria crescendo com a distância."""
        c = self.cfg
        X = self.pose
        E = self._loop_Z(L1).inverse().compose(X(L1['a']).between(X(L2['a']))).compose(
            self._loop_Z(L2)).compose(X(L2['b']).between(X(L1['b'])))
        dist = abs(cum[L1['a']] - cum[L2['a']]) + abs(cum[L1['b']] - cum[L2['b']])
        st = np.hypot(np.hypot(L1['sig'][3], L2['sig'][3]), c.loop_pcm_frac * dist)
        sy = np.hypot(np.hypot(L1['sig'][2], L2['sig'][2]), c.loop_pcm_rot_frac * dist)
        yaw = float(Rot3.Logmap(E.rotation())[2])
        chi2 = float(np.sum(np.asarray(E.translation()) ** 2)) / st ** 2 + (yaw / sy) ** 2
        return chi2 <= c.loop_pcm_chi2

    def _pcm_update(self):
        """Atualiza o conjunto ativo = clique máximo do grafo de consistência; insere/remove fatores no ISAM2."""
        import networkx as nx
        c = self.cfg
        active = set(self.loop_isam_idx)
        new = len(self.loops) - 1
        if active and active <= self.pcm_adj[new]:
            clique = active | {new}                      # incremental: consistente com todo o conjunto ativo
        else:
            G = nx.Graph()
            G.add_nodes_from(self.pcm_adj)
            G.add_edges_from((i, j) for i, nb in self.pcm_adj.items() for j in nb if i < j)
            clique, _ = nx.max_weight_clique(G, weight=None)
            clique = set(clique) if len(clique) >= c.loop_min_clique else set()
        for k, L in enumerate(self.loops):
            L['info']['status'] = 'ativo' if k in clique else 'fora do clique (PCM)'
            L['info']['w'] = 1.0 if k in clique else 0.0
        if clique == active:
            return
        add, rem = sorted(clique - active), sorted(active - clique)
        if self.graph.size() or self.initial.size():   # nós/fatores pendentes primeiro
            self._flush()
        g = gtsam.NonlinearFactorGraph()
        for k in add:
            g.add(self._loop_factor(self.loops[k]))
        res = self.isam.update(g, gtsam.Values(), [self.loop_isam_idx.pop(k) for k in rem])
        for k, idx in zip(add, res.getNewFactorsIndices()):
            self.loop_isam_idx[k] = int(idx)
        for _ in range(c.loop_extra_iters):
            self.isam.update()
        self.estimate = self.isam.calculateEstimate()
        self.n_factors['loop'] = len(self.loop_isam_idx)

    def _loop_factor(self, L, w=1.0):
        """Fator do laço gaussiano com peso w (GNC): erro e Jacobianas x sqrt(w)."""
        sw, z_rot, z_t = np.sqrt(w), L['z_rot'], L['z_t']

        def err(this, values, H):
            e = _vo_factor_error(z_rot, z_t, this, values, H)
            if H is not None:
                for k in range(3):
                    H[k] = H[k] * sw
            return e * sw
        return gtsam.CustomFactor(gtsam.noiseModel.Diagonal.Sigmas(L['sig']), [X(L['a']), X(L['b']), S(L['a'])], err)

    def _gnc_tls(self, values, max_iters=50, inner_iters=8, max_outer=60):
        """GNC com Truncated Least Squares (Yang et al. 2020, Alg. 1). Fatores de odometria = inliers conhecidos;
        só os laços recebem peso. Começa no problema convexo (mu pequeno) e vai apertando até os pesos ficarem 0/1."""
        c = self.cfg
        cb2 = c.loop_gnc_chi2

        def solve(w, init, iters):
            g = gtsam.NonlinearFactorGraph(self.all_factors)
            for L, wi in zip(self.loops, w):
                if wi > 1e-6:
                    g.add(self._loop_factor(L, wi))
            return levenberg_marquardt(g, init, iters)

        def r2(vals):
            return np.array([2.0 * self._loop_factor(L).error(vals) for L in self.loops])

        w = np.ones(len(self.loops))                   # GNC parte de todos os candidatos (inclusive fora do clique)
        vals = solve(w, values, max_iters)
        res = r2(vals)
        if res.max() > cb2:
            mu = cb2 / max(2 * res.max() - cb2, 1e-9)
            for _ in range(max_outer):
                lo, hi = mu / (mu + 1) * cb2, (mu + 1) / mu * cb2
                w_new = np.where(res >= hi, 0.0, np.where(res <= lo, 1.0,
                                                          np.sqrt(cb2 * mu * (mu + 1) / np.maximum(res, 1e-12)) - mu))
                vals = solve(w_new, vals, inner_iters)
                res = r2(vals)
                done = np.abs(w_new - w).max() < 1e-3 and np.all((w_new < 1e-3) | (w_new > 1 - 1e-3))
                w = w_new
                if done:
                    break
                mu *= c.loop_gnc_mu_step
            w = np.round(w)
            vals = solve(w, vals, max_iters)
        for L, wi, ri in zip(self.loops, w, r2(vals)):
            L['info']['w'] = float(wi)
            L['info']['r2'] = float(ri)
            L['info']['status'] = 'aceito (GNC)' if wi >= 0.5 else 'rejeitado (GNC)'
        return vals

    # ---------------------------------------------------------------- solver
    def update(self, extra_iters: int = 0):
        """Envia os fatores/nós novos ao ISAM2 (incremental). extra_iters: iterações extras (após um laço)."""
        self._flush()
        for _ in range(extra_iters):
            self.isam.update()
        self.estimate = self.isam.calculateEstimate()
        if self.cfg.vo_chunk > 1:
            self._chunk_vo()
        if self.use_imu:
            self.bias = self.estimate.atConstantBias(B(len(self.t) - 1))

    def finalize(self, batch: bool = True, max_iters: int = 50):
        if self.graph.size() or self.initial.size():
            self.update()
        if batch:
            if self.loops and self.cfg.loop_gnc:
                self.estimate = self._gnc_tls(self.estimate, max_iters)
                return
            g = gtsam.NonlinearFactorGraph(self.all_factors)
            for k in self.loop_isam_idx:
                g.add(self._loop_factor(self.loops[k]))
            self.estimate = levenberg_marquardt(g, self.estimate, max_iters)

    def loop_report(self) -> List[dict]:
        return [dict(L['info']) for L in self.loops]

    # ---------------------------------------------------------------- consulta
    def _val(self, key):
        if self.estimate.exists(key):
            return self.estimate
        return self.initial

    def pose(self, i) -> Pose3:
        return self._val(X(i)).atPose3(X(i))

    def current_log_scale(self, i) -> float:
        return float(self._val(S(i)).atVector(S(i))[0])

    def velocity(self, i) -> np.ndarray:
        return np.asarray(self._val(V(i)).atVector(V(i)), float)

    def poses(self) -> np.ndarray:
        return np.array([self.pose(i).matrix() for i in range(len(self.t))])

    def scales(self) -> np.ndarray:
        return np.exp([self.current_log_scale(i) for i in range(len(self.t))])


# ====================================================================== sensores
INOMI_UTC_OFFSET_H = -3.0      # nomes das imagens da Inomi (AAAAMMDD_HHMMSS_micro) estão em horário local (UTC-3)


def image_stamp_ns(name) -> int:
    """Instante (ns, Unix) a partir do nome da imagem. KAIST: o nome já é o tempo em ns. Inomi: AAAAMMDD_HHMMSS_micro
    em horário local (INOMI_UTC_OFFSET_H) -- assim bate com poses_gt_unix.csv."""
    s = os.path.splitext(os.path.basename(str(name)))[0]
    if s.isdigit():
        return int(s)
    import calendar
    import datetime
    dt = datetime.datetime.strptime(s, "%Y%m%d_%H%M%S_%f")
    sec = calendar.timegm(dt.timetuple()) - int(INOMI_UTC_OFFSET_H * 3600)
    return int(sec) * 10 ** 9 + dt.microsecond * 1000


class PoseOdometry:
    """Odometria dada como POSE acumulada (x, y, θ) no tempo (robô da Inomi: odometria_dataset.csv), com a mesma
    interface de WheelOdometry (relative / covers / t)."""

    def __init__(self, t, x, y, th, time_offset: float = 0.0):
        o = np.argsort(t)
        self.t = np.asarray(t, np.float64)[o] + time_offset
        self.x, self.y = np.asarray(x, np.float64)[o], np.asarray(y, np.float64)[o]
        self.th = np.unwrap(np.asarray(th, np.float64)[o])

    @classmethod
    def from_csv(cls, path, time_offset: float = 0.0):
        """odometria_dataset.csv: timestamp (AAAAMMDD_HHMMSS_micro),x,y,theta,image."""
        import csv
        rows = list(csv.DictReader(open(path)))
        t = [image_stamp_ns(r["timestamp"]) * 1e-9 for r in rows]
        return cls(t, [float(r["x"]) for r in rows], [float(r["y"]) for r in rows],
                   [float(r["theta"]) for r in rows], time_offset)

    def _at(self, t):
        return np.interp(t, self.t, self.x), np.interp(t, self.t, self.y), np.interp(t, self.t, self.th)

    def relative(self, t0: float, t1: float):
        """Movimento de t0 a t1 no referencial do robô em t0. Retorna (Pose3, distância percorrida, dyaw)."""
        x0, y0, a0 = self._at(t0)
        x1, y1, a1 = self._at(t1)
        c, s = np.cos(a0), np.sin(a0)
        dx, dy = x1 - x0, y1 - y0
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        m = (self.t > t0) & (self.t < t1)
        P = np.vstack([[x0, y0], np.column_stack([self.x[m], self.y[m]]), [x1, y1]])
        dist = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) * (1.0 if lx >= 0 else -1.0)
        return Pose3(Rot3.Yaw(a1 - a0), np.array([lx, ly, 0.0])), dist, float(a1 - a0)

    def covers(self, t0, t1):
        return self.t[0] <= t0 and t1 <= self.t[-1]


class WheelOdometry:
    """Odometria diferencial a partir de ticks acumulados de encoder (formato KAIST: t, ticks_esq, ticks_dir)."""

    def __init__(self, t_ns, left, right, resolution=4096, d_left=0.623803, d_right=0.623095, wheel_base=1.52683,
                 time_offset: float = 0.0):
        self.t = np.asarray(t_ns, np.float64) * 1e-9 + time_offset     # offset: alinha o relógio do sensor ao da câmera
        self.dl = np.diff(np.asarray(left, np.int64)) * np.pi * d_left / resolution
        self.dr = np.diff(np.asarray(right, np.int64)) * np.pi * d_right / resolution
        self.base = wheel_base
        self.ds = (self.dl + self.dr) / 2
        self.dth = (self.dr - self.dl) / wheel_base

    @classmethod
    def from_csv(cls, path, time_offset: float = 0.0, **kw):
        """Lê encoder.csv; se houver EncoderParameter.txt na mesma pasta (KAIST), usa a calibração dele."""
        par = os.path.join(os.path.dirname(path), "EncoderParameter.txt")
        if os.path.isfile(par) and not kw:
            names = {"resolution": "resolution", "left wheel diameter": "d_left",
                     "right wheel diameter": "d_right", "wheel base": "wheel_base"}
            for line in open(par):
                for k, arg in names.items():
                    if k in line.lower() and ":" in line:
                        kw[arg] = float(line.split(":")[1])
            if "resolution" in kw:
                kw["resolution"] = int(kw["resolution"])
        e = np.loadtxt(path, delimiter=",", dtype=np.int64)
        return cls(e[:, 0], e[:, 1], e[:, 2], time_offset=time_offset, **kw)

    def relative(self, t0: float, t1: float):
        """Integra (plano, ponto médio) entre t0 e t1 [s]. Retorna (Pose3 relativa, distância, dyaw).
        Alinhamento temporal: os incrementos das bordas entram só com a fração que cai dentro de [t0, t1]
        (interpolação linear dos ticks), então t0/t1 podem ser os instantes exatos dos frames."""
        a = max(int(np.searchsorted(self.t, t0, side='right')) - 1, 0)
        b = min(int(np.searchsorted(self.t, t1, side='left')), len(self.t) - 1)
        x = y = th = 0.0
        dist = 0.0
        for k in range(a, b):
            span = self.t[k + 1] - self.t[k]
            w = (min(t1, self.t[k + 1]) - max(t0, self.t[k])) / span if span > 0 else 0.0
            if w <= 0:
                continue
            ds, dth = self.ds[k] * min(w, 1.0), self.dth[k] * min(w, 1.0)
            x += ds * np.cos(th + dth / 2)
            y += ds * np.sin(th + dth / 2)
            th += dth
            dist += ds
        return Pose3(Rot3.Yaw(th), np.array([x, y, 0.0])), dist, th

    def covers(self, t0, t1):
        return self.t[0] <= t0 and t1 <= self.t[-1]


class ImuData:
    """IMU (t [s], giro [rad/s], acel [m/s^2]) no referencial do veículo (ou ajuste imu_R_vi no GPO)."""

    def __init__(self, t, gyro, acc, time_offset: float = 0.0):
        self.t = np.asarray(t, float) + time_offset
        self.gyro, self.acc = np.asarray(gyro, float), np.asarray(acc, float)

    @classmethod
    def from_kaist_xsens(cls, path, time_offset: float = 0.0):
        """xsens_imu.csv do KAIST Complex Urban: t_ns, q(4), euler(3), giro(3), acel(3), mag(3)."""
        d = np.loadtxt(path, delimiter=",")
        return cls(d[:, 0] * 1e-9, d[:, 8:11], d[:, 11:14], time_offset)

    def slice(self, t0: float, t1: float):
        """Medidas que afetam [t0, t1] (p/ mandar ao back-end em outro processo)."""
        a = max(int(np.searchsorted(self.t, t0, side='right')) - 1, 0)
        b = min(int(np.searchsorted(self.t, t1, side='left')) + 1, len(self.t))
        return ImuData(self.t[a:b], self.gyro[a:b], self.acc[a:b])

    def covers(self, t0, t1):
        return self.t[0] <= t0 and t1 <= self.t[-1]

    def integrate(self, pim, t0: float, t1: float):
        """Acrescenta à pré-integração as medidas entre t0 e t1 (segura o valor anterior até a próxima amostra)."""
        a, b = np.searchsorted(self.t, [t0, t1])
        a = max(a - 1, 0)
        for k in range(a, min(b, len(self.t))):
            ta, tb = max(self.t[k], t0), min(self.t[k + 1] if k + 1 < len(self.t) else t1, t1)
            if tb > ta:
                pim.integrateMeasurement(self.acc[k], self.gyro[k], tb - ta)
        return pim


def visual_inertial_scale_init(rot_w: List[np.ndarray], p_vo: List[np.ndarray], pims, dts, g=9.81):
    """Alinhamento visual-inercial linear (estilo VINS-Mono, sem viés): resolve escala s e velocidades v_i com
        s p_j = s p_i + v_i dt + 1/2 g dt^2 + R_i dP_ij       v_j = v_i + g dt + R_i dV_ij
    rot_w: rotações (mundo<-veículo) do VO (independem da escala). Retorna (s, velocidades (n,3))."""
    n = len(p_vo)
    gv = np.array([0, 0, -g])
    A, rhs = [], []
    for k in range(n - 1):
        dt, pim, R = dts[k], pims[k], rot_w[k]
        dP, dV = np.asarray(pim.deltaPij()), np.asarray(pim.deltaVij())
        row = np.zeros((3, 1 + 3 * n))
        row[:, 0] = p_vo[k + 1] - p_vo[k]
        row[:, 1 + 3 * k:4 + 3 * k] = -np.eye(3) * dt
        A.append(row); rhs.append(0.5 * gv * dt ** 2 + R @ dP)
        row = np.zeros((3, 1 + 3 * n))
        row[:, 1 + 3 * (k + 1):4 + 3 * (k + 1)] = np.eye(3)
        row[:, 1 + 3 * k:4 + 3 * k] = -np.eye(3)
        A.append(row); rhs.append(gv * dt + R @ dV)
    sol = np.linalg.lstsq(np.vstack(A), np.concatenate(rhs), rcond=None)[0]
    return float(sol[0]), sol[1:].reshape(n, 3)
