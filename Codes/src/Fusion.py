"""
Fusão ONLINE do VO com outros sensores (odometria de roda, IMU) num grafo de poses (GPO.py), em tempo real.

    [processo principal]                                  [processo do back-end]
    VisualOdometry.process_frame_pnp(frame)
    OnlineFusion.on_frame(vo, t)
      a cada NODE_DT s cria um nó:  ------- mensagem ------> FusionBackend.add(msg)
        - movimento do VO entre os nós (já corrigido pelo BA)      ISAM2.update()  (~1-70 ms, não trava o VO)
        - roda integrada entre os instantes dos nós
        - medidas da IMU entre os nós
      pose fundida A CADA FRAME:     <------ estimativas --------
        último nó otimizado ⊕ movimento do VO desde ele x escala estimada (ou roda, se o VO estiver perdido)

Fechamento de laço (opcional, LoopClosure.py): OnlineFusion.add_loop(vo, laço) converte o laço entre dois KFs numa
restrição entre os nós mais próximos (compondo com o movimento curto do VO até eles) e manda ao back-end, que aplica
porta de deriva + PCM e insere no ISAM2 (GPO.add_loop); no lote final, GNC-TLS sobre todos os candidatos.

Realimentação (fusão -> VO) e um só referencial:
    - escala: a escala do VO estimada no grafo (m / unid.) volta para o VO (VisualOdometry.request_scale), que
      re-escala o mapa -> o VO passa a trabalhar em ~metros. O 1º ajuste de cada segmento re-escala o segmento todo;
      depois, correções suaves da janela ativa (ganho + limite por passo). Cada correção muda as UNIDADES do VO a
      partir de um frame (vo.unit_events); aqui todo movimento do VO é convertido de volta às unidades ORIGINAIS antes
      de ir ao grafo (_cam_rel) -> o grafo não enxerga a realimentação (sem dupla contagem), só a deriva do VO.
    - pose: o mapa e a trajetória do VO são levados ao mundo fundido (métrico, veículo) por uma semelhança (Sim3) por
      nó do grafo: A_n = C_n · s_n · P_n^-1 (C_n = câmera do nó no mundo fundido, P_n = pose do VO no frame do nó).
      Cada KF/landmark usa a Sim3 do seu nó -> o mapa acompanha as correções do grafo (inclusive laços), como o
      "drift" do VINS-Mono. world_snapshot() devolve isso p/ os visualizadores.

Alinhamento dos sensores:
    temporal: cada sensor tem `time_offset` (relógio do sensor -> relógio da câmera) e é integrado exatamente entre os
              instantes dos frames dos nós (interpolação nas bordas). Só usa dados com t <= instante atual (causal).
    espacial: T_vc (câmera -> veículo) e imu_R_vi (IMU -> veículo). Os nós são poses do VEÍCULO.
"""
import multiprocessing as mp
import os
import queue
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import gtsam
from gtsam import Pose3

from GPO import GraphPoseOptimizer, GPOConfig, ImuData, R_VC_NOMINAL, visual_inertial_scale_init

VALID_SENSORS = ("vo", "wheel", "imu")


# ====================================================================== back-end
class FusionBackend:
    """Recebe uma mensagem por nó e mantém o grafo (ISAM2). Independe do VO (só dados numpy) -> roda em outro processo.

    msg = dict(t, frame,                         # instante do nó / índice do frame
               vo_rel = 4x4 | None,              # movimento do VO (veículo, unid. do VO) desde o nó anterior; None = VO inválido
               wheel = (4x4, dist, dyaw) | None, # odometria de roda desde o nó anterior
               imu = (t, gyro, acc) | None)      # medidas da IMU entre os nós
    A 1ª mensagem só tem t/frame (nó 0)."""

    def __init__(self, sensors: Sequence[str], cfg: GPOConfig = GPOConfig(), R_vc=R_VC_NOMINAL,
                 imu_R_vi=np.eye(3), init_wheel_dist: float = 5.0, init_vi_nodes: int = 30, full_every: int = 50):
        self.use_vo, self.use_wheel, self.use_imu = ("vo" in sensors), ("wheel" in sensors), ("imu" in sensors)
        self.cfg, self.R_vc, self.imu_R_vi = cfg, R_vc, imu_R_vi
        self.init_wheel_dist, self.init_vi_nodes, self.full_every = init_wheel_dist, init_vi_nodes, full_every
        self._vo_gap = False            # a aresta anterior não teve VO (perda): a próxima com VO semeia a escala
        self.n_reseed = 0
        self.gpo: Optional[GraphPoseOptimizer] = None
        self.buf: List[dict] = []
        self.frames: List[int] = []
        self.s0_info = ""
        self.loop_log: List[dict] = []

    # ---------------------------------------------------------- inicialização
    def _pim(self, imu, bias=None):
        pim = gtsam.PreintegratedImuMeasurements(self.gpo.imu_params if self.gpo else self._tmp.imu_params,
                                                 bias if bias is not None else gtsam.imuBias.ConstantBias())
        if imu is not None:
            t, g, a = imu
            d = ImuData(t, g, a)
            d.integrate(pim, self._interval[0], self._interval[1])
        return pim

    def _try_init(self):
        """Decide a escala inicial do VO (log) e a velocidade inicial. Retorna (ok, log_s0, v0)."""
        b = self.buf
        if self.use_vo and self.use_wheel:
            dw = sum(abs(m['wheel'][1]) for m in b[1:] if m['vo_rel'] is not None)
            dv = sum(np.linalg.norm(m['vo_rel'][:3, 3]) for m in b[1:] if m['vo_rel'] is not None)
            if dw < self.init_wheel_dist or dv < 1e-9:
                return False, 0.0, None
            s0 = dw / dv
            self.s0_info = f"escala inicial (roda/VO em {dw:.1f} m) = {s0:.3f} m/unid."
            v0 = None
            if self.use_imu:
                v0 = np.array([b[1]['wheel'][1] / max(b[1]['t'] - b[0]['t'], 1e-3), 0, 0])
            return True, float(np.log(s0)), v0
        if self.use_vo and self.use_imu:
            if any(m['vo_rel'] is None for m in b[1:]):          # janela precisa de VO contínuo: recomeça
                last_bad = max(k for k, m in enumerate(b) if k > 0 and m['vo_rel'] is None)
                self.buf = b = [dict(t=b[last_bad]['t'], frame=b[last_bad]['frame'])] + b[last_bad + 1:]
            if len(b) < self.init_vi_nodes:
                return False, 0.0, None
            self._tmp = GraphPoseOptimizer(self.cfg, self.R_vc, use_imu=True, imu_R_vi=self.imu_R_vi)
            T = [np.eye(4)]
            for m in b[1:]:
                T.append(T[-1] @ m['vo_rel'])
            pims = []
            for k in range(1, len(b)):
                self._interval = (b[k - 1]['t'], b[k]['t'])
                pims.append(self._pim(b[k]['imu']))
            s0, vels = visual_inertial_scale_init([x[:3, :3] for x in T], [x[:3, 3] for x in T], pims,
                                                  np.diff([m['t'] for m in b]), self.cfg.gravity)
            if not np.isfinite(s0) or s0 <= 0:
                self.buf = b[-1:]
                return False, 0.0, None
            self.s0_info = f"escala inicial (alinhamento visual-inercial, {len(b)} nós) = {s0:.3f} m/unid."
            return True, float(np.log(s0)), vels[0]
        if self.use_wheel and self.use_imu:
            if len(b) < 2:
                return False, 0.0, None
            return True, 0.0, np.array([b[1]['wheel'][1] / max(b[1]['t'] - b[0]['t'], 1e-3), 0, 0])
        return True, 0.0, None

    # ---------------------------------------------------------- nós
    def add(self, msg: dict) -> List[dict]:
        if msg.get('type') == 'loop':
            return self._add_loop(msg)
        if self.gpo is None:
            self.buf.append(msg)
            ok, ls0, v0 = self._try_init()
            if not ok:
                return []
            self.gpo = GraphPoseOptimizer(self.cfg, self.R_vc, use_imu=self.use_imu, imu_R_vi=self.imu_R_vi)
            first, rest = self.buf[0], self.buf[1:]
            self.buf = []
            self.gpo.add_first_node(first['t'], ls0, v0)
            self.frames.append(first['frame'])
            for m in rest:
                self._add_node(m, update=False)
            self.gpo.update()
            return [self._result(full=True, info=self.s0_info)]
        self._add_node(msg)
        n = len(self.gpo.t) - 1
        return [self._result(full=(n % self.full_every == 0))]

    def _add_node(self, msg, update=True):
        g, c = self.gpo, self.cfg
        i = len(g.t) - 1
        t_i, t_j = g.t[i], msg['t']
        Pi = g.pose(i)
        s = float(np.exp(g.current_log_scale(i)))
        vo_rel = Pose3(msg['vo_rel']) if (self.use_vo and msg.get('vo_rel') is not None) else None
        wheel = msg.get('wheel') if self.use_wheel else None
        pim = None
        if self.use_imu and msg.get('imu') is not None:
            self._interval = (t_i, t_j)
            pim = self._pim(msg['imu'], g.bias)

        if wheel is not None:
            guess = Pi.compose(Pose3(wheel[0]))
        elif vo_rel is not None:
            guess = Pi.compose(Pose3(vo_rel.rotation(), s * np.asarray(vo_rel.translation())))
        elif pim is not None:
            guess = pim.predict(gtsam.NavState(Pi, g.velocity(i)), g.bias).pose()
        else:
            guess = Pi
        vel_guess = None
        if self.use_imu:
            vel_guess = (np.asarray(guess.translation()) - np.asarray(Pi.translation())) / max(t_j - t_i, 1e-3)
        # 1ª aresta do VO depois de uma perda: o VO reiniciou com OUTRA escala (urban38: ~10 -> ~92 m/unid.). Partir
        # da escala antiga faz o Gauss-Newton em exp(S) extrapolar (pose online saltou 2,5 km). Semeia a escala pelo
        # dado (distância da roda / do VO) no nó novo e usa S(j) nessa aresta: a linearização já começa perto.
        lam_obs = None
        if vo_rel is not None and self._vo_gap and wheel is not None:
            d_vo = float(np.linalg.norm(vo_rel.translation()))
            if abs(wheel[1]) > 0.5 and d_vo > 1e-9:
                lam_obs = float(np.log(abs(wheel[1]) / d_vo))
                self.n_reseed += 1
        j = g.add_node(t_j, guess, vel_guess, vo_lost=vo_rel is None, log_scale_guess=lam_obs)
        self.frames.append(msg['frame'])
        self._vo_gap = vo_rel is None

        if vo_rel is not None:
            step = abs(wheel[1]) if wheel is not None else s * np.linalg.norm(vo_rel.translation())
            g.add_vo(i, j, vo_rel, step, scale_node=j if lam_obs is not None else None)
        if wheel is not None:
            g.add_wheel(i, j, Pose3(wheel[0]), wheel[1], wheel[2])
        if pim is not None:
            g.add_imu(i, j, pim)
        elif self.use_imu:                    # sem dados da IMU no intervalo: segura velocidade/viés com prior fraco
            g._add(gtsam.BetweenFactorVector(gtsam.symbol_shorthand.V(i), gtsam.symbol_shorthand.V(j), np.zeros(3),
                                             gtsam.noiseModel.Isotropic.Sigma(3, 5.0)))
        if update:
            g.update()

    def _add_loop(self, msg) -> List[dict]:
        base = dict(t_a=msg['t_a'], t_b=msg['t_b'], n_pnp=msg['n_pnp'], n_f=msg['n_f'])
        if self.gpo is None:
            self.loop_log.append(dict(base, status='antes da inicialização do grafo', w=0.0))
            return []
        node = {f: n for n, f in enumerate(self.frames)}
        a, b = node.get(msg['frame_a']), node.get(msg['frame_b'])
        if a is None or b is None or a == b:
            self.loop_log.append(dict(base, status='nó inexistente', w=0.0))
            return []
        ok, info = self.gpo.add_loop(a, b, Pose3(msg['rel']))
        info.update(n_pnp=msg['n_pnp'], n_f=msg['n_f'])
        self.loop_log.append(info)                 # mesmo objeto do GPO: o status final (PCM/GNC) aparece aqui
        r = self._result(full=True)
        r['loop'] = dict(info)
        r['loops_active'] = [(self.gpo.loops[k]['a'], self.gpo.loops[k]['b']) for k in self.gpo.loop_isam_idx]
        return [r]

    def _result(self, full=False, info=""):
        g = self.gpo
        n = len(g.t) - 1
        r = dict(n=n, t=g.t[n], frame=self.frames[n], pose=g.pose(n).matrix(), log_scale=g.current_log_scale(n),
                 info=info)
        if full:
            r['poses'] = g.poses()
            r['frames'] = list(self.frames)
            r['log_scales'] = np.log(g.scales())
        return r

    def finalize(self, batch=True) -> dict:
        if self.gpo is None:
            return dict(n=-1, poses=np.zeros((0, 4, 4)), frames=[], t=[], log_scales=np.zeros(0), final=True,
                        loops=[dict(i) for i in self.loop_log])
        self.gpo.finalize(batch=batch)
        r = self._result(full=True)
        r['final'] = True
        r['t_nodes'] = list(self.gpo.t)
        r['n_factors'] = dict(self.gpo.n_factors)
        r['n_reseed'] = self.n_reseed
        r['loops'] = [dict(i) for i in self.loop_log]
        return r


def _backend_loop(q_in, q_out, args, kwargs):
    import os
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    # diagnóstico (opcional): FUSION_RECORD=arquivo grava as mensagens (p/ reprocessar só o grafo com
    # replay_backend.py); FUSION_PROFILE=prefixo grava o perfil (cProfile) do online e do lote final
    rec = prof = None
    if os.environ.get("FUSION_RECORD"):
        import pickle
        rec = open(os.environ["FUSION_RECORD"], "wb")
        pickle.dump((args, kwargs), rec)
    if os.environ.get("FUSION_PROFILE"):
        import cProfile
        prof = cProfile.Profile()
        prof.enable()
    be = FusionBackend(*args, **kwargs)
    while True:
        msg = q_in.get()
        if msg is None:
            break
        if msg == "finalize":
            if prof is not None:
                prof.disable()
                prof.dump_stats(os.environ["FUSION_PROFILE"] + "_online.prof")
                prof = cProfile.Profile()
                prof.enable()
            if rec is not None:
                rec.close()
                rec = None
            q_out.put(be.finalize())
            if prof is not None:
                prof.disable()
                prof.dump_stats(os.environ["FUSION_PROFILE"] + "_final.prof")
                prof = None
            continue
        if rec is not None:
            pickle.dump(msg, rec)
        t0 = time.perf_counter()
        out = be.add(msg)
        dt = time.perf_counter() - t0
        for r in out:
            r['dt_be'] = dt                     # tempo do grafo p/ esta mensagem (métrica de processamento)
            q_out.put(r)


class BackendProcess:
    """Back-end em outro processo (não trava o VO). inline=True roda na mesma thread (depuração)."""

    def __init__(self, *args, inline=False, **kwargs):
        self.inline = inline
        if inline:
            self.be = FusionBackend(*args, **kwargs)
            self._out: List[dict] = []
            return
        self.late: List[dict] = []
        ctx = mp.get_context("spawn")
        self.q_in, self.q_out = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=_backend_loop, args=(self.q_in, self.q_out, args, kwargs), daemon=True)
        self.proc.start()

    def send(self, msg):
        if self.inline:
            t0 = time.perf_counter()
            out = self.be.add(msg)
            for r in out:
                r['dt_be'] = time.perf_counter() - t0
            self._out.extend(out)
        else:
            self.q_in.put(msg)

    def poll(self) -> List[dict]:
        if self.inline:
            out, self._out = self._out, []
            return out
        out = []
        while True:
            try:
                out.append(self.q_out.get_nowait())
            except queue.Empty:
                return out

    def finalize(self, batch=True, timeout=3600) -> dict:
        if self.inline:
            return self.be.finalize(batch)
        self.q_in.put("finalize")
        while True:
            r = self.q_out.get(timeout=timeout)
            if r.get('final'):
                self.q_in.put(None)
                self.proc.join(10)
                return r
            self.late.append(r)          # respostas que chegaram depois do fim do laço principal (não descartar)


# ====================================================================== front-end
class OnlineFusion:
    def __init__(self, sensors: Sequence[str], wheel=None, imu=None, node_dt: float = 0.5,
                 cfg: GPOConfig = GPOConfig(), R_vc=R_VC_NOMINAL, imu_R_vi=np.eye(3), backend_process: bool = True,
                 verbose: bool = True, scale_feedback: bool = True, fb_min_nodes: int = 4, fb_gain: float = 0.5,
                 fb_max_step: float = 1.10, fb_deadband: float = 0.02, init_wheel_dist: float = 5.0):
        sensors = [s.lower() for s in sensors]
        bad = [s for s in sensors if s not in VALID_SENSORS]
        if bad:
            raise ValueError(f"sensores inválidos {bad}; opções: {VALID_SENSORS}")
        if "wheel" in sensors and wheel is None:
            raise ValueError("'wheel' selecionado mas sem dados de encoder")
        if "imu" in sensors and imu is None:
            raise ValueError("'imu' selecionado mas sem dados de IMU")
        if len([s for s in sensors if s != "vo"]) == 0:
            raise ValueError("selecione pelo menos um sensor além do VO ('wheel' e/ou 'imu')")
        self.sensors, self.wheel, self.imu = sensors, wheel, imu
        self.use_vo = "vo" in sensors
        self.node_dt, self.verbose = node_dt, verbose
        self.T_vc = np.eye(4)
        self.T_vc[:3, :3] = R_vc
        self.T_cv = np.linalg.inv(self.T_vc)
        self.backend = BackendProcess(sensors, cfg, R_vc, imu_R_vi, inline=not backend_process,
                                      init_wheel_dist=init_wheel_dist)

        self.frame_ok: List[bool] = []
        self.frame_seg: List[int] = []
        self.node_t: List[float] = []
        self.node_frame: List[int] = []
        self.est: Dict[int, np.ndarray] = {}       # nó -> pose (veículo no mundo), última estimativa recebida
        self.est_ls: Dict[int, float] = {}
        self.last_n = -1
        self.log_scale = 0.0
        # saídas
        self.fused_t: List[float] = []
        self.fused: List[np.ndarray] = []
        self.wheel_dr: List[np.ndarray] = []       # só roda (dead reckoning), p/ comparação
        self._wheel_T = np.eye(4)
        self._pending_loops: List[dict] = []
        self.loops_active: List[tuple] = []        # pares de nós (a, b) dos laços ativos no ISAM2
        self.loop_events: List[dict] = []
        self.n_guard = 0                    # frames em que a pose online usou a roda (VO x escala incoerente)
        self.loop_snaps: List[dict] = []    # estado do grafo antes/depois de cada laço (p/ animações)
        self.fused_src: List[int] = []      # por frame: 0 VO x escala, 1 roda (proteção), 2 roda (sem VO), 3 nada
        self.be_log: List[tuple] = []       # (nó, tempo do grafo [s], laço?, poses completas?) p/ métricas
        # realimentação de escala p/ o VO
        self.scale_feedback = scale_feedback and self.use_vo
        self.fb_min_nodes, self.fb_gain, self.fb_max_step, self.fb_deadband = fb_min_nodes, fb_gain, fb_max_step, fb_deadband
        self._fb_node = -1
        self.fb_log: List[tuple] = []              # (t, fator pedido, segmento inteiro?)

    # ---------------------------------------------------------- utilidades
    def _covers(self, t0, t1):
        for s in self.sensors:
            d = self.wheel if s == "wheel" else self.imu if s == "imu" else None
            if d is not None and not (d.t[0] <= t0 and t1 <= d.t[-1]):
                return False
        return True

    def _vo_ok(self, f0, f1):
        return (self.use_vo and all(self.frame_ok[f0:f1 + 1]) and self.frame_seg[f0] == self.frame_seg[f1])

    @staticmethod
    def _cam_rel(vo, f0, f1):
        """Movimento da câmera f0 -> f1 (P_f0^-1 P_f1) nas unidades ORIGINAIS do VO (desfaz as escalas externas).
        As poses (vo.poses_est) já vêm re-sincronizadas após BA / correções de escala; cada escala externa muda as
        unidades dos frames >= pivô dela, então o trecho é quebrado nos pivôs e cada pedaço é dividido pelo seu fator."""
        if f1 < f0:
            return np.linalg.inv(OnlineFusion._cam_rel(vo, f1, f0))
        P = vo.poses_est
        fp = vo.unit_arrays()[0]
        cuts = np.unique(fp[(fp > f0) & (fp < f1)]).tolist()
        T = np.eye(4)
        for a, b in zip([f0] + cuts, cuts + [f1]):
            r = np.linalg.inv(P[a]) @ P[b]
            r[:3, 3] /= vo.frame_units(a)         # o pivô em b não re-escala a -> b (só o que vem depois de b)
            T = T @ r
        return T

    def _vo_rel(self, vo, f0, f1):
        return self.T_vc @ self._cam_rel(vo, f0, f1) @ self.T_cv

    # ---------------------------------------------------------- por frame
    def on_frame(self, vo, t: float) -> Optional[np.ndarray]:
        """Chamar logo após vo.process_frame_pnp(). Retorna a pose fundida do veículo (4x4) ou None (ainda iniciando)."""
        fi = len(vo.poses_est) - 1
        self.frame_ok.append(bool(vo.initialized and vo.lost == 0))
        self.frame_seg.append(vo.segment)

        if not self.node_t:
            if (self.use_vo and not self.frame_ok[-1]) or not self._covers(t, t):
                return None
            self._new_node(dict(t=t, frame=fi))
        elif t - self.node_t[-1] >= self.node_dt - 1e-6 and self._covers(self.node_t[-1], t):
            fp, tp = self.node_frame[-1], self.node_t[-1]
            msg = dict(t=t, frame=fi, vo_rel=self._vo_rel(vo, fp, fi) if self._vo_ok(fp, fi) else None)
            if "wheel" in self.sensors:
                rel, dist, dyaw = self.wheel.relative(tp, t)
                msg['wheel'] = (rel.matrix(), dist, dyaw)
            if "imu" in self.sensors:
                d = self.imu.slice(tp, t)
                msg['imu'] = (d.t, d.gyro, d.acc) if len(d.t) >= 2 else None
            self._new_node(msg)

        if self._pending_loops:
            self._flush_loops(vo)
        self._poll()
        self._feedback(vo, fi)
        return self._fused_now(vo, t, fi)

    # ---------------------------------------------------------- realimentação de escala (grafo -> VO)
    def _feedback(self, vo, fi):
        """Uma vez por estimativa nova do grafo: leva as unidades do VO p/ metros (escala do último nó)."""
        k = self.last_n
        if not self.scale_feedback or k < 0 or k == self._fb_node or k < self.fb_min_nodes:
            return
        if not self._vo_ok(self.node_frame[k - self.fb_min_nodes], fi):
            return                                 # escala do segmento atual ainda não observada pelo grafo
        self._fb_node = k
        s = float(np.exp(self.est_ls[k])) / vo.frame_units(fi)      # m/unid. original -> fator sobre as unid. atuais
        whole = not any(seg == vo.segment for _, _, seg in vo.unit_events)
        if not whole:
            if abs(np.log(s)) < self.fb_deadband:
                return
            s = float(np.clip(s ** self.fb_gain, 1 / self.fb_max_step, self.fb_max_step))
        vo.request_scale(s, whole_segment=whole)
        self.fb_log.append((self.node_t[k], s, whole))
        if whole and self.verbose:
            print(f"[FUSÃO->VO] escala métrica aplicada ao segmento {vo.segment}: x{s:.3f}")

    # ---------------------------------------------------------- laços
    def add_loop(self, vo, lp: dict):
        """lp: laço do LoopClosure (frames dos KFs q/c + T_cq = pose da câmera q no referencial da câmera c)."""
        self._pending_loops.append(lp)
        self._flush_loops(vo)

    def _flush_loops(self, vo):
        keep = []
        for lp in self._pending_loops:
            fq, fc = lp['q_frame'], lp['c_frame']
            if not self.node_frame or fq > self.node_frame[-1]:
                keep.append(lp)                    # o nó do instante do KF atual ainda não foi criado
                continue
            nf = np.asarray(self.node_frame)
            fa, fb = int(nf[np.argmin(np.abs(nf - fc))]), int(nf[np.argmin(np.abs(nf - fq))])
            if not (self._vo_ok(min(fa, fc), max(fa, fc)) and self._vo_ok(min(fb, fq), max(fb, fq))):
                continue                           # trecho curto KF->nó sem VO válido: descarta
            T_cq = lp['T_cq'].copy()
            T_cq[:3, 3] /= lp.get('c_units', 1.0)  # unidades do mapa do candidato quando a pose dele foi enviada
            T_ab = self._cam_rel(vo, fa, fc) @ T_cq @ self._cam_rel(vo, fq, fb)
            self.backend.send(dict(type='loop', frame_a=fa, frame_b=fb, rel=self.T_vc @ T_ab @ self.T_cv,
                                   t_a=lp['c_t'], t_b=lp['q_t'], n_pnp=lp['n_pnp'], n_f=lp['n_f']))
        self._pending_loops = keep

    def _new_node(self, msg):
        self.node_t.append(msg['t'])
        self.node_frame.append(msg['frame'])
        self.backend.send(msg)

    def _poll(self):
        for r in self.backend.poll():
            self._handle(r)

    def _handle(self, r):
        self.be_log.append((int(r.get('n', -1)), float(r.get('dt_be', np.nan)), 'loop' in r, 'poses' in r))
        if r.get('info') and self.verbose:
            print(f"[FUSÃO] {r['info']}")
        if 'loop' in r and 'poses' in r:             # snapshot antes/depois (posições dos nós)
            m = min(len(r['poses']), self.last_n + 1)
            if m > 1 and all(k in self.est for k in (0, m - 1)):
                self.loop_snaps.append(dict(
                    t_a=r['loop']['t_a'], t_b=r['loop']['t_b'], status=r['loop']['status'],
                    before=np.array([self.est[k][:3, 3] for k in range(m)], np.float32),
                    after=np.asarray(r['poses'][:m, :3, 3], np.float32)))
        if 'loop' in r:
            L = r['loop']
            self.loop_events.append(L)
            self.loops_active = r['loops_active']
            if self.verbose:
                print(f"[LAÇO] nós {L['a']}<->{L['b']} ({L['t_b'] - L['t_a']:.0f} s depois) | PnP {L['n_pnp']} | "
                      f"desvio {L['dpos']:.1f} m / {L['drot_deg']:.1f}° (porta {L['gate_t']:.0f} m) -> "
                      f"{L['status']} | ativos: {len(self.loops_active)}")
        if 'poses' in r:
            for k, P in enumerate(r['poses']):
                self.est[k] = P
            for k, ls in enumerate(r['log_scales']):
                self.est_ls[k] = float(ls)
        self.est[r['n']] = r['pose']
        self.est_ls[r['n']] = r['log_scale']
        self.last_n = max(self.last_n, r['n'])
        self.log_scale = r['log_scale']

    def _fused_now(self, vo, t, fi):
        if self.last_n < 0:
            return None
        k = self.last_n
        T_k, f_k, t_k = self.est[k], self.node_frame[k], self.node_t[k]
        has_wheel = self.wheel is not None and "wheel" in self.sensors and self._covers(t_k, t)
        src = 3
        if self._vo_ok(f_k, fi):
            src = 0
            rel = self._vo_rel(vo, f_k, fi)
            rel[:3, 3] *= np.exp(self.est_ls.get(k, self.log_scale))
            if has_wheel:          # proteção: escala do grafo em transitório (ex.: VO acabou de reiniciar) -> roda
                w_rel, w_d, _ = self.wheel.relative(t_k, t)
                d_v, d_w = float(np.linalg.norm(rel[:3, 3])), abs(float(w_d))
                if max(d_v, d_w) > 2.0 and (d_v > 3 * d_w or d_w > 3 * d_v):
                    rel = w_rel.matrix()
                    self.n_guard += 1
                    src = 1
        elif has_wheel:
            rel = self.wheel.relative(t_k, t)[0].matrix()
            src = 2
        else:
            rel = np.eye(4)
        self.fused_src.append(src)
        T = T_k @ rel
        self.fused_t.append(t)
        self.fused.append(T)
        if self.wheel is not None:
            if len(self.fused_t) >= 2:
                self._wheel_T = self._wheel_T @ self.wheel.relative(self.fused_t[-2], t)[0].matrix()
            self.wheel_dr.append(self._wheel_T.copy())
        return T

    # ---------------------------------------------------------- VO no mundo fundido
    def node_sim3(self, vo):
        """Sim3 (4x4, bloco 3x3 = s·R) que leva o mundo do VO ao mundo fundido, uma por nó estimado (0..last_n)."""
        n = self.last_n + 1
        if n <= 0:
            return None
        F = np.array([self.est[k] for k in range(n)])
        f = np.asarray(self.node_frame[:n])
        s = np.exp([self.est_ls.get(k, self.log_scale) for k in range(n)]) / np.array([vo.frame_units(x) for x in f])
        P_inv = np.linalg.inv(np.array([vo.poses_est[x] for x in f]))
        A = F @ self.T_vc @ P_inv
        A[:, :3, :3] *= s[:, None, None]
        A[:, :3, 3] = F[:, :3, 3] + s[:, None] * (F[:, :3, :3] @ self.T_vc[:3, :3] @ P_inv[:, :3, 3:])[:, :, 0]
        return A

    def world_snapshot(self, vo, with_points=True) -> Optional[dict]:
        """Trajetória da câmera + mapa do VO no mundo FUNDIDO (métrico, x frente / y esq. / z cima), no formato de
        Viewer.plot_snapshot. Cada frame/landmark usa a Sim3 do nó do grafo anterior a ele (landmark: KF de origem)."""
        A = self.node_sim3(vo)
        if A is None or not self.fused:
            return None
        nf = np.asarray(self.node_frame[:len(A)])
        f = np.arange(nf[0], len(vo.poses_est))
        idx = np.clip(np.searchsorted(nf, f, side='right') - 1, 0, len(A) - 1)
        Pv = np.array(vo.poses_est[nf[0]:])
        traj = np.einsum('nij,nj->ni', A[idx, :3, :3], Pv[:, :3, 3]) + A[idx, :3, 3]
        snap = dict(traj=traj.astype(np.float32), pose=self.fused[-1] @ self.T_vc, P=None, C=None)
        if with_points:
            pts = vo.Map.get_all_points()
            if pts:
                kfs = vo.keyframes_dict
                X = np.array([p.position for p in pts])
                fk = np.array([kfs[p.observations[0][0]].frame_idx if p.observations and p.observations[0][0] in kfs
                               else nf[0] for p in pts])
                C = np.array([p.color / 255.0 if p.color is not None else [1.0, 0.0, 0.0] for p in pts], np.float32)
                k = np.clip(np.searchsorted(nf, fk, side='right') - 1, 0, len(A) - 1)
                snap['P'] = (np.einsum('nij,nj->ni', A[k, :3, :3], X) + A[k, :3, 3]).astype(np.float32)
                snap['C'] = C
        return snap

    def save_loop_snapshots(self, filename: str) -> int:
        """Estado do grafo (posições dos nós, mundo do grafo) antes/depois de cada laço -> make_gifs.py."""
        S = self.loop_snaps
        if not S:
            return 0
        arr = {f"antes_{k}": s['before'] for k, s in enumerate(S)}
        arr.update({f"depois_{k}": s['after'] for k, s in enumerate(S)})
        np.savez_compressed(filename, node_t=np.asarray(self.node_t), t_a=np.array([s['t_a'] for s in S]),
                            t_b=np.array([s['t_b'] for s in S]), status=np.array([s['status'] for s in S]), **arr)
        return len(S)

    def save_world_map(self, vo, filename: str, ply: bool = True) -> bool:
        """Mapa do VO no mundo FUNDIDO, no formato de VisualOdometry.save_map (abre no view_map.py): pontos, cores,
        nº de observações, poses dos keyframes e da câmera por frame (rígidas, em metros). Cada frame/landmark usa a
        Sim3 do nó do grafo anterior a ele (como world_snapshot). Referencial: veículo no 1º nó (x frente, y esq.,
        z cima) -> up='z'. Chamar depois da otimização final (est = poses suavizadas)."""
        from VisualOdometry import write_ply
        A = self.node_sim3(vo)
        if A is None:
            return False
        nf = np.asarray(self.node_frame[:len(A)])
        sim = lambda f: A[np.clip(np.searchsorted(nf, f, side='right') - 1, 0, len(A) - 1)]

        def rigid(As, T):            # Sim3 (sR, t) aplicada a poses de câmera -> pose rígida no mundo fundido
            s = np.cbrt(np.linalg.det(As[:, :3, :3]))[:, None, None]
            out = np.tile(np.eye(4), (len(T), 1, 1))
            out[:, :3, :3] = (As[:, :3, :3] / s) @ T[:, :3, :3]
            out[:, :3, 3] = np.einsum('nij,nj->ni', As[:, :3, :3], T[:, :3, 3]) + As[:, :3, 3]
            return out
        f = np.arange(nf[0], len(vo.poses_est))
        poses = rigid(sim(f), np.array(vo.poses_est[nf[0]:], np.float64))
        kfs = vo.keyframes_dict
        kf_ids = np.array([k for k in sorted(kfs) if kfs[k].frame_idx >= nf[0]], np.int64)
        kf_poses = rigid(sim(np.array([kfs[k].frame_idx for k in kf_ids])),
                         np.array([kfs[k].pose for k in kf_ids], np.float64).reshape(-1, 4, 4))
        pts = vo.Map.get_all_points()
        X = np.array([p.position for p in pts], np.float64).reshape(-1, 3)
        fk = np.array([kfs[p.observations[0][0]].frame_idx if p.observations and p.observations[0][0] in kfs
                       else nf[0] for p in pts], np.int64)
        Ak = sim(fk)
        P = np.einsum('nij,nj->ni', Ak[:, :3, :3], X) + Ak[:, :3, 3]
        C = np.array([p.color if p.color is not None else (255, 0, 0) for p in pts], np.uint8).reshape(-1, 3)
        n_obs = np.array([len(p.observations) for p in pts], np.int32)
        np.savez_compressed(filename, points=P, colors=C, n_obs=n_obs, kf_ids=kf_ids, kf_poses=kf_poses, poses=poses,
                            timestamps=np.array([float(t) for t in vo.timestamps[nf[0]:]]), K=vo.K, up='z',
                            vehicle_poses=np.array([self.est[k] for k in range(len(A))]),
                            node_t=np.asarray(self.node_t[:len(A)]))
        print(f"Mapa fundido salvo em {filename} ({len(P)} pontos, {len(kf_ids)} keyframes, metros, z p/ cima)")
        if ply:
            write_ply(os.path.splitext(filename)[0] + ".ply", P, C)
        return True

    # ---------------------------------------------------------- fim
    def snapshot(self) -> dict:
        """Trajetórias p/ plot ao vivo (XY em metros)."""
        lines = {}
        if self.fused:
            F = np.array([T[:3, 3] for T in self.fused])
            lines['fusão (online)'] = F[:, :2]
        if self.est:
            N = np.array([self.est[k][:3, 3] for k in sorted(self.est)])
            lines['nós (suavizado)'] = N[:, :2]
        if self.wheel_dr:
            W = np.array([T[:3, 3] for T in self.wheel_dr])
            lines['só roda'] = W[:, :2]
        seg = [np.array([self.est[a][:2, 3], self.est[b][:2, 3], [np.nan, np.nan]])
               for a, b in self.loops_active if a in self.est and b in self.est]
        if seg:
            lines['laços'] = np.vstack(seg)
        return dict(lines=lines, scale=float(np.exp(self.log_scale)))

    def finish(self, batch=True) -> dict:
        self._poll()
        r = self.backend.finalize(batch=batch)
        for x in getattr(self.backend, 'late', []):     # laços que terminaram junto com a sequência
            self._handle(x)
        r['node_t'] = list(self.node_t[:len(r['poses'])])
        return r
