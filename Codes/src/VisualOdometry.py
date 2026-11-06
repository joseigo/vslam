"""
Odometria visual monocular (front-end KLT + PnP, back-end BA local com GTSAM).

Pipeline
--------
1. INICIALIZAÇÃO: detecta features, rastreia com KLT até haver paralaxe suficiente, estima a
   matriz Essencial, recupera (R, t) e triangula o mapa inicial (escala arbitrária = `init_scale`).
2. TRACKING (todo frame): rastreia com KLT (+ checagem forward-backward). Cada track carrega o id do
   landmark 3D associado -> PnP-RANSAC 2D-3D direto, sem casar descritores. Pose = T_wc (câmera->mundo).
3. KEYFRAME: quando a razão de landmarks rastreados cai / há paralaxe / passou tempo demais:
   registra observações, triangula novos tracks (origem = KF onde nasceram), roda BA local
   (janela deslizante, 2 KFs mais antigos fixos p/ travar a escala) e repõe features.
   Com `async_ba=True` o BA roda em OUTRO PROCESSO; enquanto ele roda, a thread principal detecta as features
   novas do KF. Com `ba_max_lag=0` (padrão) o tracking espera o BA antes do próximo frame -> mesmo resultado do
   modo síncrono. Com `ba_max_lag>=1` o tracking segue sem esperar (mais FPS), e a correção do último KF otimizado
   é propagada aos KFs/landmarks criados nesse meio-tempo -- mas em testes (urban26) isso piorou o ATE (~56 -> 80+ m):
   os landmarks novos são usados no PnP antes de serem refinados.
4. PERDA: se o PnP falha vários frames seguidos, reinicializa o mapa a partir da última pose,
   reaproveitando a escala típica dos passos anteriores.
5. ESCALA (anti-deriva): em monocular a escala não é observável, então ela deriva. Três camadas:
   (a) geometria: paralaxe DEROTADA p/ decidir keyframes (rotação pura não gera baseline), gate de ângulo de
       triangulação mais rígido, não cria KF quando o passo foi limitado (pose inconsistente com o mapa);
   (b) BA: landmarks de baixa paralaxe ganham prior fraco (não "escorregam" ao longo do raio);
   (c) âncora de escala pelo PLANO DO CHÃO: a altura da câmera sobre o chão é constante; estimamos essa
       altura a cada KF e re-escalamos a janela ativa quando ela se afasta da referência.
"""
import os
import time
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import matplotlib

if not os.environ.get("MPLBACKEND"):
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
import matplotlib.pyplot as plt

from Map import Map
from KeyFrame import KeyFrame
from Viewer import plot_snapshot, draw_snapshot
from BA import run_bundle_adjustment, snapshot_ba_problem, solve_ba, apply_ba_result, scale_prior, warmup as _ba_warmup



def write_ply(path, P, C):
    """Nuvem de pontos em PLY binário (abre no MeshLab / CloudCompare / Open3D)."""
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.zeros(len(P), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
        rec['x'], rec['y'], rec['z'] = np.asarray(P).T
        rec['r'], rec['g'], rec['b'] = np.asarray(C).T
        f.write(rec.tobytes())
    print(f"Nuvem de pontos (PLY) salva em {path}")

class _Tracks:
    """Features rastreadas (arrays paralelos)."""

    def __init__(self):
        self.pts = np.zeros((0, 2), np.float32)       # posição atual (px)
        self.lm = np.zeros(0, np.int64)               # id do landmark (-1 = ainda não triangulado)
        self.org_kf = np.zeros(0, np.int64)           # KF onde o track nasceu
        self.org_uv = np.zeros((0, 2), np.float32)    # posição no KF de origem
        self.kf_uv = np.zeros((0, 2), np.float32)     # posição no último KF (p/ medir paralaxe)
        self.age = np.zeros(0, np.int32)              # nº de KFs desde que nasceu

    def __len__(self):
        return len(self.pts)

    def keep(self, mask):
        mask = np.asarray(mask, bool)
        self.pts, self.lm, self.org_kf = self.pts[mask], self.lm[mask], self.org_kf[mask]
        self.org_uv, self.kf_uv, self.age = self.org_uv[mask], self.kf_uv[mask], self.age[mask]

    def add(self, pts, org_kf, lm=None):
        n = len(pts)
        if n == 0:
            return
        pts = np.asarray(pts, np.float32).reshape(-1, 2)
        self.pts = np.vstack([self.pts, pts])
        self.lm = np.concatenate([self.lm, np.full(n, -1, np.int64) if lm is None else np.asarray(lm, np.int64)])
        self.org_kf = np.concatenate([self.org_kf, np.full(n, org_kf, np.int64)])
        self.org_uv = np.vstack([self.org_uv, pts])
        self.kf_uv = np.vstack([self.kf_uv, pts])
        self.age = np.concatenate([self.age, np.zeros(n, np.int32)])


class VisualOdometry:

    def __init__(self, camera_matrix: np.ndarray, orb_params: Dict, lk_params: Dict, keyframe_params: Dict,
                 dist_coeffs: Optional[np.ndarray] = None, sliding_window_size: int = 8,
                 enable_plot: bool = True, verbose: bool = True, async_ba: bool = True):

        self.K = np.asarray(camera_matrix, dtype=np.float64)
        self.dist_coeffs = None if dist_coeffs is None else np.asarray(dist_coeffs, np.float64).reshape(-1)
        if self.dist_coeffs is not None and not np.any(self.dist_coeffs):
            self.dist_coeffs = None
        self._undist_maps = None
        self._undist_lock = threading.Lock()   # _prep pode ser chamado pelas threads de prefetch

        self.orb = cv2.ORB_create(**orb_params)
        self.lk_params = dict(lk_params)
        self.verbose = verbose

        kp = keyframe_params or {}
        self.min_pnp_inliers = int(kp.get('min_inliers', 20))
        self.min_tracked_ratio = float(kp.get('min_tracked_ratio', 0.7))
        self.min_dist = int(kp.get('min_dist_inliers', 12))            # espaçamento entre features (px)
        self.kf_parallax_px = float(kp.get('parallax_threshold', 15))  # paralaxe mediana p/ novo KF (px)
        self.max_frames_no_kf = int(kp.get('max_frames_sem_keyframe', 30))
        self.min_kf_gap = int(kp.get('min_kf_gap', 2))
        self.max_features = int(kp.get('max_features', 800))
        self.detector = kp.get('detector', 'shitomasi')                # 'shitomasi' | 'orb'
        self.init_parallax_px = float(kp.get('init_parallax', 20))
        self.min_init_landmarks = int(kp.get('min_init_landmarks', 40))
        self.min_tri_angle = float(kp.get('min_tri_angle_deg', 1.5))
        self.reproj_thr = float(kp.get('reproj_thr', 2.0))
        self.fb_thr = float(kp.get('fb_thr', 1.0))                     # erro forward-backward (px)
        self.max_track_age = int(kp.get('max_track_age', 6))           # KFs p/ desistir de triangular
        self.max_lost = int(kp.get('max_lost', 5))
        self.init_scale = float(kp.get('init_scale', 1.0))
        self.ba_fixed = int(kp.get('ba_fixed', 2))
        # gauge do BA: 'fix' = fixa os ba_fixed KFs mais antigos da janela; 'marg' = marginaliza o KF que sai (BA.py)
        self.ba_marg = str(kp.get('ba_mode', 'fix')) == 'marg'
        self.ba_prior: Optional[dict] = None         # (marg) prior marginal do último BA
        # limitador de passo (anti-inflação de escala em curvas): |passo| <= max_step_ratio * velocidade_ref
        self.max_step_ratio = float(kp.get('max_step_ratio', 1.6))
        self.speed_min_inliers = int(kp.get('speed_min_inliers', 40))   # só atualiza a vel. ref. com PnP saudável
        self.speed_rate = float(kp.get('speed_rate', 0.2))              # EMA da velocidade de referência
        self.clamp_growth = float(kp.get('clamp_growth', 0.03))        # crescimento da vel. ref. por frame limitado
        self.moving_frac = float(kp.get('moving_frac', 0.3))            # só aprende a vel. ref. se passo >= frac*ref
        self.use_flow_prior = bool(kp.get('use_flow_prior', True))      # semeia o KLT com o fluxo previsto
        self.flow_prior_min_px = float(kp.get('flow_prior_min_px', 8))  # ...quando o fluxo previsto mediano > isto
        self.subpix = bool(kp.get('subpix', False))                    # refino sub-pixel (lento, ganho pequeno)

        self.ba_min_parallax_deg = float(kp.get('ba_min_parallax_deg', 1.0))   # abaixo disso o landmark recebe prior no BA
        self.ba_point_prior_frac = float(kp.get('ba_point_prior_frac', 0.2))   # sigma do prior = frac x profundidade
        self.block_kf_on_clamp = bool(kp.get('block_kf_on_clamp', True))       # não cria KF (não urgente) com pose limitada
        # BA assíncrono: nº máx. de KFs ainda sem BA incorporado. Ao passar disso o tracking ESPERA o BA.
        # 0 = mesma precisão do síncrono; maior = mais FPS, mas landmarks novos ficam mais tempo sem refino (pior ATE).
        self.ba_max_lag = int(kp.get('ba_max_lag', 0))
        # âncora de escala pelo plano do chão
        self.use_ground_scale = bool(kp.get('use_ground_scale', True))
        self.camera_height = kp.get('camera_height', None)                     # altura real da câmera (None = auto)
        self.ground_row_frac = float(kp.get('ground_row_frac', 0.05))          # só pontos abaixo de cy + frac*altura_img
        self.ground_min_pts = int(kp.get('ground_min_pts', 25))
        self.ground_min_cos = float(kp.get('ground_min_cos', 0.85))            # normal do plano vs eixo y da câmera
        self.ground_thr_rel = float(kp.get('ground_thr_rel', 0.02))            # dist. ao plano / profundidade
        self.ground_init_n = int(kp.get('ground_init_n', 5))                   # estimativas p/ fixar a referência (modo auto)
        self.ground_win = int(kp.get('ground_win', 5))                         # estimativas (mediana) por correção
        self.ground_deadband = float(kp.get('ground_deadband', 0.04))          # ignora desvios < 4%
        self.ground_gain = float(kp.get('ground_gain', 0.7))                   # fração do erro corrigida por passo (filtra ruído)
        self.ground_max_corr = float(kp.get('ground_max_corr', 1.15))          # correção máx. por passo (após o 1º alinhamento)

        self.sliding_window_size = max(int(sliding_window_size), self.ba_fixed + 2)

        # --- estado ---
        self.Map = Map()
        self.keyframes_dict: Dict[int, KeyFrame] = {}
        self.window: List[int] = []
        self.kf_counter = 0
        self.cur_kf_id: Optional[int] = None

        self.tracks = _Tracks()
        self.prev_gray: Optional[np.ndarray] = None
        self.initialized = False
        self.init_ref = None
        self.anchor_pose = np.eye(4)
        self.T_cur = np.eye(4)
        self.vel: Optional[np.ndarray] = None
        self.lost = 0
        self.frames_since_kf = 0
        self.n_lm_kf = 0
        self.segment = 0
        self.speed_ref: Optional[float] = None     # velocidade robusta (unidades do mapa / frame)
        self.speed_hi: Optional[float] = None      # velocidade 'de cruzeiro' (decai muito devagar, só andando): piso do limitador
        self.gap = 1                                # frames desde a última pose válida (p/ o limitador)
        self.init_speed: Optional[float] = None    # |t| por frame do segmento anterior (continuidade de escala)

        self.step_clamped = False                   # o último passo foi limitado? (pose inconsistente com o mapa)
        self.img_h = None
        # frames -> (KF de referência, pose relativa), p/ re-sincronizar a trajetória inteira após BA/escala
        self.kf_frames: Dict[int, List[list]] = {}
        # âncora de escala (chão)
        self.ground_ref: Optional[float] = None if self.camera_height is None else float(self.camera_height)
        self.ground_aligned = False
        self._ground_buf: List[float] = []
        self.ground_hist: List[Tuple[int, float]] = []   # (kf_id, altura estimada em unid. do mapa) p/ diagnóstico
        self.scale_corrections: List[Tuple[int, float]] = []   # (kf_id, fator aplicado)
        # escala EXTERNA (ex.: escala métrica do grafo de fusão, request_scale): muda as UNIDADES do mapa a partir do
        # frame do pivô. unit_events = [(frame do pivô, log do fator, segmento)] p/ quem consome o VO converter
        # qualquer movimento de volta às unidades originais (frame_units)
        self.unit_events: List[Tuple[int, float, int]] = []
        self._ext_scale: Optional[Tuple[float, bool]] = None
        self.seg_kf0: Optional[int] = None          # 1º KF do segmento atual

        # BA assíncrono: 1 processo dedicado, no máximo 1 BA em andamento
        self.async_ba = bool(async_ba)
        self._ba_kw = dict(min_parallax_deg=self.ba_min_parallax_deg, point_prior_frac=self.ba_point_prior_frac)
        self._ba_pool = None
        self._ba_job = None          # (future, info do snapshot)
        self._ba_pending = False     # houve KF novo enquanto o BA rodava -> roda outro ao terminar
        self.ba_stats = dict(submitted=0, merged=0, failed=0, wait_s=0.0)
        self.ba_log = []                 # (frame, tempo do solve no worker [s], KFs na janela) p/ métricas
        if self.async_ba:
            self._ba_pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
            self._ba_pool.submit(_ba_warmup)

        # uma pose por frame, alinhada com self.timestamps
        self.timestamps: List = []
        self.poses_est: List[np.ndarray] = []
        self.trajectory_est: List[np.ndarray] = []

        # plot
        self.enable_plot = enable_plot
        self.fig = self.ax = None
        if enable_plot:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
            plt.ion()
            plt.show(block=False)

        self._log("Odometria Visual inicializada.")

    # ------------------------------------------------------------------ util
    def _log(self, *a):
        if self.verbose:
            print(*a)

    @property
    def current_keyframe(self) -> Optional[KeyFrame]:
        return self.keyframes_dict.get(self.cur_kf_id)

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(t).flatten()
        return T

    def _append_pose(self, T):
        idx = len(self.poses_est)
        self.poses_est.append(np.array(T, dtype=np.float64))
        self.trajectory_est.append(np.array(T[:3, 3], dtype=np.float64))
        # guarda a pose relativa ao KF corrente: assim, quando o BA / a correção de escala mexer no KF,
        # os frames intermediários acompanham (antes só as poses dos próprios KFs eram corrigidas).
        if self.initialized and self.cur_kf_id is not None and self.cur_kf_id in self.keyframes_dict:
            rel = np.linalg.inv(self.keyframes_dict[self.cur_kf_id].pose) @ np.asarray(T, np.float64)
            self.kf_frames.setdefault(self.cur_kf_id, []).append([idx, rel])

    def _prep(self, frame):
        """Remove distorção (se houver) e devolve (bgr, gray). Todo o resto assume imagem sem distorção."""
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if self.dist_coeffs is not None:
            if self._undist_maps is None:
                with self._undist_lock:
                    if self._undist_maps is None:
                        h, w = frame.shape[:2]
                        self._undist_maps = cv2.initUndistortRectifyMap(
                            self.K, self.dist_coeffs, None, self.K, (w, h), cv2.CV_16SC2)
            frame = cv2.remap(frame, *self._undist_maps, interpolation=cv2.INTER_LINEAR)
        return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    prepare = _prep   # público: pode ser chamado em outra thread (prefetch) e passado em process_frame_pnp(prepared=...)

    def _color_at(self, img, uv):
        x, y = int(round(uv[0])), int(round(uv[1]))
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            return img[y, x][::-1]
        return None

    # ------------------------------------------------------------- front-end
    def _detect(self, gray, existing: Optional[np.ndarray], n_needed: int) -> np.ndarray:
        if n_needed <= 0:
            return np.zeros((0, 2), np.float32)
        h, w = gray.shape[:2]
        mask = np.full((h, w), 255, np.uint8)
        b = 8
        mask[:b], mask[-b:], mask[:, :b], mask[:, -b:] = 0, 0, 0, 0
        if existing is not None:
            for p in existing:
                cv2.circle(mask, (int(p[0]), int(p[1])), self.min_dist, 0, -1)

        if self.detector == 'orb':
            kps = [k for k in self.orb.detect(gray, mask) if k.octave <= 1]   # níveis grossos são imprecisos p/ KLT
            kps.sort(key=lambda k: -k.response)
            out = []
            for k in kps:
                if len(out) >= n_needed:
                    break
                x, y = int(round(k.pt[0])), int(round(k.pt[1]))
                if 0 <= y < h and 0 <= x < w and mask[y, x]:
                    out.append(k.pt)
                    cv2.circle(mask, (x, y), self.min_dist, 0, -1)
            pts = np.array(out, np.float32).reshape(-1, 2)
        else:
            c = cv2.goodFeaturesToTrack(gray, maxCorners=n_needed, qualityLevel=0.01,
                                        minDistance=self.min_dist, mask=mask, blockSize=7)
            pts = np.zeros((0, 2), np.float32) if c is None else c.reshape(-1, 2).astype(np.float32)

        if len(pts) and self.subpix:
            try:
                cv2.cornerSubPix(gray, pts.reshape(-1, 1, 2), (5, 5), (-1, -1),
                                 (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
            except cv2.error:
                pass
        return pts

    def _klt_core(self, prev_gray, gray, p0, init=None):
        """KLT + checagem forward-backward. `init` (opcional) = posições previstas (fluxo inicial)."""
        if init is not None:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, init.copy(),
                                                 flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk_params)
        else:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **self.lk_params)
        pb, stb, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **self.lk_params)
        fb = np.linalg.norm(p0 - pb, axis=2).ravel()
        p1 = p1.reshape(-1, 2)
        h, w = gray.shape[:2]
        ok = (st.ravel() == 1) & (stb.ravel() == 1) & (fb < self.fb_thr)
        ok &= (p1[:, 0] >= 2) & (p1[:, 0] < w - 2) & (p1[:, 1] >= 2) & (p1[:, 1] < h - 2)
        return p1, ok

    def _klt(self, prev_gray, gray, pts, init=None):
        """Retorna (novas posições (N,2), máscara de sucesso). Com `init`, tenta primeiro com o fluxo previsto
        (essencial em curvas: o deslocamento por rotação pode passar de 100 px) e refaz sem ele o que falhar."""
        if len(pts) == 0:
            return pts.reshape(-1, 2), np.zeros(0, bool)
        p0 = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        if init is None:
            return self._klt_core(prev_gray, gray, p0)
        p1, ok = self._klt_core(prev_gray, gray, p0, np.asarray(init, np.float32).reshape(-1, 1, 2))
        bad = ~ok
        if bad.any():
            p1b, okb = self._klt_core(prev_gray, gray, p0[bad])
            idx = np.where(bad)[0]
            p1[idx[okb]] = p1b[okb]
            ok[idx[okb]] = True
        return p1, ok

    def _predict_tracks(self, T_pred):
        """Onde cada track deve estar no frame atual segundo o modelo de movimento.
        Landmarks: projeção 3D com a pose prevista. Demais: só a rotação (homografia no infinito)."""
        tr = self.tracks
        T_cw = np.linalg.inv(T_pred)
        R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
        pred = tr.pts.astype(np.float64).copy()
        Kinv = np.linalg.inv(self.K)
        # rotação-apenas p/ todos: raio na câmera anterior -> câmera prevista
        T_prev_cw = np.linalg.inv(self.T_cur)
        R_rel = R_cw @ np.linalg.inv(T_prev_cw)[:3, :3]        # cam anterior -> cam atual (só rotação)
        rays = (Kinv @ np.hstack([tr.pts, np.ones((len(tr), 1))]).T).T
        rr = rays @ R_rel.T
        with np.errstate(divide='ignore', invalid='ignore'):
            uv = (self.K @ rr.T).T
            uv = uv[:, :2] / uv[:, 2:3]
        good = np.isfinite(uv).all(axis=1) & (rr[:, 2] > 1e-6)
        pred[good] = uv[good]
        # landmarks: projeção completa (inclui translação)
        idx = np.where(tr.lm >= 0)[0]
        if len(idx):
            X = np.array([self.Map.points[int(tr.lm[i])].position for i in idx])
            Xc = X @ R_cw.T + t_cw
            with np.errstate(divide='ignore', invalid='ignore'):
                uvl = (self.K @ Xc.T).T
                uvl = uvl[:, :2] / uvl[:, 2:3]
            okl = np.isfinite(uvl).all(axis=1) & (Xc[:, 2] > 1e-6)
            pred[idx[okl]] = uvl[okl]
        return pred.astype(np.float32)

    # ---------------------------------------------------------- inicialização
    def _process_init(self, color, gray, timestamp, frame_idx):
        if self.init_ref is None:
            self._start_init_ref(gray, frame_idx)
            self._append_pose(self.anchor_pose)
            return

        ref = self.init_ref
        new, ok = self._klt(self.prev_gray, gray, ref['pts'])
        ref['pts0'], ref['pts'] = ref['pts0'][ok], new[ok]

        if len(ref['pts']) < 40:               # perdeu quase tudo -> recomeça daqui
            self._start_init_ref(gray, frame_idx)
            self._append_pose(self.anchor_pose)
            return

        disp = np.median(np.linalg.norm(ref['pts'] - ref['pts0'], axis=1))
        if disp >= self.init_parallax_px:
            if self._initialize_map(color, timestamp, frame_idx):
                return
            if disp > 3 * self.init_parallax_px:   # muito movimento e não deu certo -> novo par de referência
                self._start_init_ref(gray, frame_idx)
        self._append_pose(self.anchor_pose)

    def _start_init_ref(self, gray, frame_idx):
        pts = self._detect(gray, None, self.max_features)
        if len(pts) < 50:
            self.init_ref = None
            return
        self.init_ref = {'pts0': pts.copy(), 'pts': pts.copy(), 'frame_idx': frame_idx}

    def _initialize_map(self, color, timestamp, frame_idx) -> bool:
        ref = self.init_ref
        pts0, pts1 = ref['pts0'].astype(np.float64), ref['pts'].astype(np.float64)
        E, mask = cv2.findEssentialMat(pts0, pts1, self.K, cv2.RANSAC, 0.999, 1.0)
        if E is None or mask is None:
            return False
        E = E[:3]
        if int(mask.sum()) < self.min_init_landmarks:
            return False
        n, R, t, mask_pose = cv2.recoverPose(E, pts0, pts1, self.K, mask=mask.copy())
        sel = mask_pose.ravel() > 0
        if sel.sum() < self.min_init_landmarks:
            return False

        # triangula em coordenadas da câmera 0
        P0 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = self.K @ np.hstack([R, t])
        X = cv2.triangulatePoints(P0, P1, pts0[sel].T, pts1[sel].T)
        X = (X[:3] / X[3]).T
        good, ang = self._check_triangulation(X, np.eye(3), np.zeros(3), R, t.ravel(), pts0[sel], pts1[sel])
        if good.sum() < self.min_init_landmarks:
            self._log(f"[INIT] triangulação fraca ({good.sum()} pts bons). Aguardando mais paralaxe.")
            return False

        # escala: 1ª inicialização usa `init_scale`; após uma perda, assume velocidade por frame ~ constante
        if self.init_speed is not None:
            s = self.init_speed * max(1, frame_idx - ref['frame_idx'])
        else:
            s = self.init_scale
        X, t = X * s, t * s
        T_wc0 = self.anchor_pose.copy()
        T_10 = self._form_transf(R, t)                   # x_cam1 = R x_cam0 + t
        T_wc1 = T_wc0 @ np.linalg.inv(T_10)
        Xw = (T_wc0[:3, :3] @ X.T).T + T_wc0[:3, 3]

        # keyframes 0 e 1
        kf0 = self._new_kf(timestamp=None, frame_idx=ref['frame_idx'], pose=T_wc0)
        kf1 = self._new_kf(timestamp, frame_idx, T_wc1)
        self.window = [kf0, kf1]
        self.seg_kf0 = kf0
        self.kf_frames[kf0] = [[ref['frame_idx'], np.eye(4)]]

        uv0, uv1 = pts0[sel][good], pts1[sel][good]
        ids = []
        for Xi, a, b in zip(Xw[good], uv0, uv1):
            p = self.Map.add_point(Xi, None, [(kf0, (float(a[0]), float(a[1]))), (kf1, (float(b[0]), float(b[1])))],
                                   color=self._color_at(color, b))
            ids.append(p.id)

        self.tracks = _Tracks()
        self.tracks.add(uv1, kf1, lm=ids)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        self.tracks.add(self._detect(gray, self.tracks.pts, self.max_features - len(self.tracks)), kf1)

        self.speed_ref = float(np.linalg.norm(t)) / max(1, frame_idx - ref['frame_idx'])
        if self.speed_hi is None:
            self.speed_hi = self.speed_ref
        self.gap = 1
        self.step_clamped = False
        self._ground_buf = []
        self.initialized = True
        self.init_ref = None
        self.T_cur = T_wc1.copy()
        self.vel = None
        self.lost = 0
        self.frames_since_kf = 0
        self.n_lm_kf = len(ids)
        self._append_pose(T_wc1)
        self._log(f"[INIT] Mapa inicializado: {len(ids)} landmarks, ângulo mediano "
                  f"{np.median(ang[good]):.2f}°, baseline={np.linalg.norm(t):.3f}")
        return True

    def _check_triangulation(self, X, R0, t0, R1, t1, uv0, uv1):
        """
        X (N,3) no referencial comum. Câmera i: x_i = R_i X + t_i.
        Retorna máscara de pontos bons (profundidade>0, reprojeção baixa, ângulo de paralaxe suficiente).
        """
        good = np.isfinite(X).all(axis=1)
        errs, depths = [], []
        for R, t, uv in ((R0, t0, uv0), (R1, t1, uv1)):
            Xc = X @ R.T + t
            z = Xc[:, 2]
            with np.errstate(divide='ignore', invalid='ignore'):
                proj = (Xc @ self.K.T)
                proj = proj[:, :2] / proj[:, 2:3]
            errs.append(np.linalg.norm(proj - uv, axis=1))
            depths.append(z)
        good &= (depths[0] > 0) & (depths[1] > 0) & (errs[0] < self.reproj_thr) & (errs[1] < self.reproj_thr)
        C0, C1 = -R0.T @ t0, -R1.T @ t1
        a, b = X - C0, X - C1
        with np.errstate(divide='ignore', invalid='ignore'):
            cosang = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        good &= np.nan_to_num(ang) > self.min_tri_angle
        return good, ang

    # ---------------------------------------------------------------- tracking
    def _lm_ok(self, l):
        if l < 0:
            return True
        p = self.Map.points.get(int(l))
        return p is not None and p.is_valid

    def _purge_invalid_tracks(self):
        tr = self.tracks
        if len(tr):
            tr.keep([self._lm_ok(l) for l in tr.lm.tolist()])

    def _pnp(self, obj, img, T_pred):
        """PnP robusto. Usa a predição de movimento se ela já explica os dados; senão RANSAC."""
        n = len(obj)
        if n < 6:
            return False, None, None
        thr = self.reproj_thr
        T_cw = np.linalg.inv(T_pred)
        rvec, _ = cv2.Rodrigues(T_cw[:3, :3])
        tvec = T_cw[:3, 3:4].copy()

        def reproj_err(rv, tv):
            proj, _ = cv2.projectPoints(obj, rv, tv, self.K, None)
            R, _ = cv2.Rodrigues(rv)
            z = (obj @ R.T + tv.ravel())[:, 2]
            e = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
            e[z <= 0] = 1e9
            return e

        inl = reproj_err(rvec, tvec) < 1.5 * thr
        if inl.sum() < max(self.min_pnp_inliers, 0.6 * n):
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj, img, self.K, None, iterationsCount=200, reprojectionError=thr,
                confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
            if not ok or inliers is None or len(inliers) < 6:
                return False, None, None
            inl = np.zeros(n, bool)
            inl[inliers.ravel()] = True

        for _ in range(2):   # refina só com inliers e reavalia
            if inl.sum() < 6:
                return False, None, None
            rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], self.K, None, rvec, tvec)
            inl = reproj_err(rvec, tvec) < 1.5 * thr

        if inl.sum() < self.min_pnp_inliers:
            return False, None, None

        R, _ = cv2.Rodrigues(rvec)
        T_new = np.linalg.inv(self._form_transf(R, tvec))
        return True, T_new, inl

    def _process_tracking(self, color, gray, timestamp, frame_idx):
        tr = self.tracks
        T_pred = self.T_cur @ self.vel if self.vel is not None else self.T_cur
        init = None
        if self.vel is not None and self.use_flow_prior and len(tr):
            pred = self._predict_tracks(T_pred)
            # só vale a pena semear se o movimento previsto é grande (curva / alta velocidade)
            if float(np.median(np.linalg.norm(pred - tr.pts, axis=1))) > self.flow_prior_min_px:
                init = pred
        p1, ok = self._klt(self.prev_gray, gray, tr.pts, init)
        tr.pts = p1
        tr.keep(ok)
        self._purge_invalid_tracks()

        idx = np.where(tr.lm >= 0)[0]
        ok_pose, T_new, inl = False, None, None
        if len(idx) >= 6:
            obj = np.array([self.Map.points[int(tr.lm[i])].position for i in idx])
            ok_pose, T_new, inl = self._pnp(obj, tr.pts[idx].astype(np.float64), T_pred)

        if not ok_pose:
            self.lost += 1
            self.gap += 1
            self.step_clamped = True
            self.vel = None
            self._log(f"[TRACK] frame {frame_idx}: PnP falhou ({len(idx)} landmarks rastreados) "
                      f"[{self.lost}/{self.max_lost}]")
            self._append_pose(self.T_cur)
            if self.lost >= self.max_lost or len(tr) < 10:
                self._reset_map()
            return

        # tracks que eram landmarks mas são outliers do PnP saem
        drop = np.zeros(len(tr), bool)
        drop[idx[~inl]] = True
        tr.keep(~drop)

        self.step_clamped = False
        T_new = self._limit_step(T_new, int(inl.sum()), frame_idx)
        self.vel = np.linalg.inv(self.T_cur) @ T_new
        self.T_cur = T_new
        self.lost = 0
        self.gap = 1
        self.frames_since_kf += 1

        if self._need_keyframe():
            self._create_keyframe(color, gray, timestamp, frame_idx)
        self._append_pose(self.T_cur)

    def _limit_step(self, T_new, n_inl, frame_idx):
        """
        Em curvas o PnP fica mal condicionado (poucos landmarks, a maioria distante: rotação e translação
        lateral se confundem) e a translação estimada "explode". Como o mapa novo é triangulado com essas
        poses, o erro de escala vira permanente. Limitamos o passo a `max_step_ratio` x velocidade de
        referência (x nº de frames sem pose válida).

        IMPORTANTE: a referência só APRENDE quando o veículo está andando (passo >= `moving_frac` x ref).
        Parado num cruzamento o passo é ~0; se a referência decaísse junto, ao arrancar o limitador só
        deixaria o carro andar a uma fração da velocidade real (trajetória "travada").
        """
        if self.speed_ref is None:
            return T_new
        step_vec = T_new[:3, 3] - self.T_cur[:3, 3]
        step = float(np.linalg.norm(step_vec))
        ref = max(self.speed_ref, 0.3 * self.speed_hi)       # piso: velocidade de cruzeiro
        lim = self.max_step_ratio * ref * self.gap
        if step > lim and step > 1e-12:
            T_new = T_new.copy()
            self.step_clamped = True
            T_new[:3, 3] = self.T_cur[:3, 3] + step_vec * (lim / step)
            self._log(f"[TRACK] frame {frame_idx}: passo limitado ({step / self.gap:.3f} -> {lim / self.gap:.3f} "
                      f"un/frame; ref={ref:.3f}; inliers={n_inl})")
            # Limitar repetidamente também indica aceleração legítima (ex.: voltando à velocidade normal depois
            # da curva). Deixa a referência subir devagar, senão o limitador prende o carro a uma velocidade baixa.
            # Um surto curto de inflação (poucos frames) só a eleva um pouco.
            if self.gap == 1:
                self.speed_ref *= 1.0 + self.clamp_growth
                self.speed_hi = max(self.speed_hi * 0.999, self.speed_ref)
        elif n_inl >= self.speed_min_inliers and self.gap == 1 and step >= self.moving_frac * ref:
            self.speed_ref += self.speed_rate * (step - self.speed_ref)
            self.speed_hi = max(self.speed_hi * 0.999, self.speed_ref)   # pico com decaimento lento
        return T_new

    def _derotated_parallax(self) -> float:
        """Paralaxe mediana (px) desde o último KF DESCONTANDO a rotação. Só a translação gera baseline útil p/
        triangular: em curva o fluxo bruto é enorme mas quase todo de rotação, e criar KF por causa dele gera
        landmarks com profundidade ruim (principal fonte de encolhimento/inflação de escala em curvas)."""
        tr = self.tracks
        if len(tr) == 0 or self.current_keyframe is None:
            return 0.0
        R_rel = self.T_cur[:3, :3].T @ self.current_keyframe.pose[:3, :3]      # cam_KF -> cam_atual
        rays = (np.linalg.inv(self.K) @ np.hstack([tr.kf_uv, np.ones((len(tr), 1))]).T).T @ R_rel.T
        with np.errstate(divide='ignore', invalid='ignore'):
            uv = (self.K @ rays.T).T
            uv = uv[:, :2] / uv[:, 2:3]
        ok = np.isfinite(uv).all(axis=1) & (rays[:, 2] > 1e-6)
        if not ok.any():
            return float(np.median(np.linalg.norm(tr.pts - tr.kf_uv, axis=1)))
        return float(np.median(np.linalg.norm(tr.pts[ok] - uv[ok], axis=1)))

    def _need_keyframe(self) -> bool:
        tr = self.tracks
        n_lm = int((tr.lm >= 0).sum())
        disp = self._derotated_parallax()
        ratio = n_lm / max(self.n_lm_kf, 1)

        weak = n_lm < 1.5 * self.min_pnp_inliers                       # urgente: repor landmarks
        want = (ratio < self.min_tracked_ratio) or (disp > self.kf_parallax_px) or \
               (self.frames_since_kf >= self.max_frames_no_kf and disp > 2.0)   # parado -> não cria KF à toa
        # pose limitada = inconsistente com o mapa: triangular a partir dela fixaria o erro de escala p/ sempre
        if self.block_kf_on_clamp and self.step_clamped and not weak:
            return False
        return weak or (want and self.frames_since_kf >= self.min_kf_gap)

    # ---------------------------------------------------------------- keyframes
    def _new_kf(self, timestamp, frame_idx, pose) -> int:
        kid = self.kf_counter
        self.kf_counter += 1
        kf = KeyFrame(kid, timestamp, frame_idx, pose)
        self.keyframes_dict[kid] = kf
        self.Map.add_keyframe(kid, timestamp, pose)
        self.cur_kf_id = kid
        return kid

    def _create_keyframe(self, color, gray, timestamp, frame_idx):
        if self.async_ba:
            # a triangulação abaixo usa as poses dos KFs de origem: elas precisam já ter o BA incorporado
            # (corrigir depois só com um deslocamento rígido deixa o mapa novo inconsistente -> deriva de escala)
            while self._ba_job is not None and self._ba_lag() + 1 > self.ba_max_lag:
                self._poll_ba(wait=True)
        tr = self.tracks
        kid = self._new_kf(timestamp, frame_idx, self.T_cur)

        # 1) observações dos landmarks rastreados
        for i in np.where(tr.lm >= 0)[0]:
            self.Map.add_observation(int(tr.lm[i]), kid, tr.pts[i])

        # 2) triangula tracks ainda sem landmark
        n_new = self._triangulate_new(color, kid)

        # 3) envelhece / descarta tracks que nunca ganharam paralaxe
        tr.age += 1
        tr.keep(~((tr.lm < 0) & (tr.age > self.max_track_age)))
        tr.kf_uv = tr.pts.copy()

        # 4) janela deslizante + BA local
        self.window.append(kid)
        if len(self.window) > self.sliding_window_size:
            self.window.pop(0)
        if self.async_ba:
            # BA + âncora de escala rodam quando o resultado do worker chegar (_poll_ba)
            if self._ba_job is None:
                self._submit_ba()
            else:
                self._ba_pending = True
            # 5) repõe features ENQUANTO o BA roda no outro processo (não depende do resultado dele)
            tr.add(self._detect(gray, tr.pts, self.max_features - len(tr)), kid)
            while self._ba_job is not None and self._ba_lag() > self.ba_max_lag:
                self._poll_ba(wait=True)     # incorpora e já reenvia com a janela atual (pendente)
        else:
            if len(self.window) >= self.ba_fixed + 1:
                res = run_bundle_adjustment(self.Map, self.keyframes_dict, self.K, list(self.window),
                                            n_fixed=self.ba_fixed, verbose=self.verbose, prior=self.ba_prior,
                                            marg=self.ba_marg, **self._ba_kw)
                if res is not None:
                    if self.ba_marg:
                        self.ba_prior = res.get('prior')
                    self._sync_poses_after_ba()
            self._purge_invalid_tracks()
            self._ground_scale_update(kid)
            # 5) repõe features
            tr.add(self._detect(gray, tr.pts, self.max_features - len(tr)), kid)
        self.n_lm_kf = int((tr.lm >= 0).sum())
        self.frames_since_kf = 0
        self._log(f"[KF {kid}] frame {frame_idx}: +{n_new} landmarks, {self.n_lm_kf} rastreados, "
                  f"{len(tr)} tracks, mapa={self.Map.count_points()} pts")

    def _sync_poses_after_ba(self, kfs: Optional[List[int]] = None, rel_cur: Optional[np.ndarray] = None):
        """Propaga as poses (refinadas pelo BA / pela correção de escala) dos KFs para keyframes_dict e
        para TODOS os frames que usam esses KFs como referência (não só os próprios KFs).
        `rel_cur` = pose do frame atual relativa ao KF corrente ANTES da correção (None = frame atual é o KF)."""
        for k in (self.window if kfs is None else kfs):
            kf = self.keyframes_dict[k]
            kf.pose = self.Map.poses[k].copy()
            for fi, rel in self.kf_frames.get(k, ()):
                if fi < len(self.poses_est):
                    P = kf.pose @ rel
                    self.poses_est[fi] = P
                    self.trajectory_est[fi] = P[:3, 3].copy()
        self.T_cur = self.keyframes_dict[self.cur_kf_id].pose.copy()
        if rel_cur is not None:
            self.T_cur = self.T_cur @ rel_cur

    def _rel_cur(self) -> np.ndarray:
        return np.linalg.inv(self.keyframes_dict[self.cur_kf_id].pose) @ self.T_cur

    # ------------------------------------------------------- BA assíncrono
    def _submit_ba(self):
        self._ba_pending = False
        if len(self.window) < self.ba_fixed + 1:
            return
        problem = snapshot_ba_problem(self.Map, list(self.window), self.ba_fixed, prior=self.ba_prior,
                                      marg=self.ba_marg)
        if problem is None:
            return
        fut = self._ba_pool.submit(solve_ba, problem, self.K, verbose=self.verbose, **self._ba_kw)
        self._ba_job = (fut, dict(active=problem['active'], segment=self.segment, next_pid=self.Map.next_id))
        self.ba_stats['submitted'] += 1

    def _ba_lag(self) -> int:
        """KFs cujo BA ainda não foi incorporado (o BA em andamento + os criados depois do snapshot)."""
        return self.cur_kf_id - self._ba_job[1]['active'][-1] + 1

    def _poll_ba(self, wait: bool = False):
        """Incorpora o resultado do BA se já estiver pronto (ou espera, com wait=True)."""
        if self._ba_job is None:
            return
        fut, info = self._ba_job
        if not fut.done():
            if not wait:
                return
            t0 = time.perf_counter()
            fut.result()
            self.ba_stats['wait_s'] += time.perf_counter() - t0
        self._ba_job = None
        try:
            res = fut.result()
        except Exception as e:
            print(f"[BA] erro no worker: {type(e).__name__}: {e}")
            res = None
        same_seg = info['segment'] == self.segment
        if res is None:
            self.ba_stats['failed'] += 1
        else:
            self.ba_stats['merged'] += 1
            self.ba_log.append((len(self.poses_est) - 1, float(res.get('dt', np.nan)), int(res.get('n_active', 0))))
            if self.ba_marg and same_seg:
                self.ba_prior = res.get('prior')
            self._merge_ba(res, info, same_seg)
        self._purge_invalid_tracks()
        if same_seg and self.initialized:
            self._ground_scale_update(self.cur_kf_id)
            if self._ba_pending:
                self._submit_ba()
        else:
            self._ba_pending = False

    def _merge_ba(self, res, info, same_seg):
        """Aplica o BA calculado sobre um snapshot antigo. KFs e landmarks criados DEPOIS do snapshot foram
        estimados com as poses pré-BA: recebem a mesma correção rígida do último KF otimizado."""
        active = info['active']
        if not same_seg:       # mapa foi reiniciado nesse meio-tempo: só atualiza o segmento antigo
            apply_ba_result(self.Map, self.keyframes_dict, res, verbose=self.verbose)
            for k in active:
                kf = self.keyframes_dict[k]
                for fi, rel in self.kf_frames.get(k, ()):
                    if fi < len(self.poses_est):
                        P = kf.pose @ rel
                        self.poses_est[fi] = P
                        self.trajectory_est[fi] = P[:3, 3].copy()
            return
        last = active[-1]
        T_last_old = self.Map.poses[last].copy()
        rel_cur = self._rel_cur()
        apply_ba_result(self.Map, self.keyframes_dict, res, verbose=self.verbose)
        delta = self.Map.poses[last] @ np.linalg.inv(T_last_old)
        newer = [k for k in self.Map.poses if k > last]
        for k in newer:
            self.Map.poses[k] = delta @ self.Map.poses[k]
        if newer:
            R, t = delta[:3, :3], delta[:3, 3]
            for pid in range(info['next_pid'], self.Map.next_id):
                p = self.Map.points.get(pid)
                if p is not None:
                    p.position = R @ p.position + t
        self._sync_poses_after_ba(active + newer, rel_cur=rel_cur)

    def finish(self):
        """Espera o BA em andamento (e o pendente) e encerra o processo do BA."""
        while self._ba_job is not None:
            self._poll_ba(wait=True)
        if self._ba_pool is not None:
            self._ba_pool.shutdown(wait=True)
            self._ba_pool = None

    # ------------------------------------------------- âncora de escala (plano do chão)
    def _ground_height_estimate(self) -> Optional[float]:
        """Altura da câmera sobre o plano do chão, em unidades do mapa (RANSAC nos landmarks abaixo do horizonte).
        Retorna None se não houver plano confiável (poucos pontos, normal inclinada demais, etc.)."""
        tr = self.tracks
        idx = np.where(tr.lm >= 0)[0]
        if len(idx) < self.ground_min_pts or self.img_h is None:
            return None
        X = np.array([self.Map.points[int(tr.lm[i])].position for i in idx])
        T_cw = np.linalg.inv(self.T_cur)
        Xc = X @ T_cw[:3, :3].T + T_cw[:3, 3]
        z = Xc[:, 2]
        front = z > 1e-6
        if front.sum() < self.ground_min_pts:
            return None
        z_med = np.median(z[front])
        below = tr.pts[idx, 1] > self.K[1, 2] + self.ground_row_frac * self.img_h
        sel = front & below & (z < 1.5 * z_med) & (Xc[:, 1] > 0)           # perto o bastante e abaixo da câmera
        Xc = Xc[sel]
        if len(Xc) < self.ground_min_pts:
            return None
        # RANSAC vetorizado: as 120 hipóteses de uma vez (mesmas amostras da versão em laço)
        rng = np.random.default_rng(len(self.ground_hist))
        S = Xc[np.array([rng.choice(len(Xc), 3, replace=False) for _ in range(120)])]   # (120, 3, 3)
        n = np.cross(S[:, 1] - S[:, 0], S[:, 2] - S[:, 0])
        nn = np.linalg.norm(n, axis=1)
        valid = nn >= 1e-12
        n[valid] /= nn[valid, None]
        n[n[:, 1] < 0] *= -1
        valid &= n[:, 1] >= self.ground_min_cos                # chão tem normal ~ eixo y da câmera
        if not valid.any():
            return None
        n, S0 = n[valid], S[valid, 0]
        inl = np.abs(Xc @ n.T - np.sum(S0 * n, axis=1)) < self.ground_thr_rel * Xc[:, 2:3]   # (N, hipóteses)
        cnt = inl.sum(axis=0)
        j = int(np.argmax(cnt))
        best, best_n = inl[:, j], int(cnt[j])
        if best_n < self.ground_min_pts or best_n < 0.4 * len(Xc):
            return None
        P = Xc[best]
        c = P.mean(axis=0)
        n = np.linalg.svd(P - c)[2][-1]
        if n[1] < 0:
            n = -n
        if n[1] < self.ground_min_cos:
            return None
        h = float(n @ c)                                       # distância câmera -> plano
        return h if h > 1e-9 else None

    def _ground_scale_update(self, kid):
        """Mede a altura da câmera sobre o chão; se divergiu da referência, re-escala a janela ativa."""
        if not self.use_ground_scale or not self.initialized or len(self.window) < self.ba_fixed + 1:
            return
        h = self._ground_height_estimate()
        if h is None:
            return
        self.ground_hist.append((kid, h))
        self._ground_buf.append(h)
        if self.ground_ref is None:                            # modo auto: referência = altura logo após a inicialização
            if len(self._ground_buf) >= self.ground_init_n:
                v = np.array(self._ground_buf[-self.ground_init_n:])
                if (v.max() - v.min()) / np.median(v) < 0.25:
                    self.ground_ref = float(np.median(v))
                    self.ground_aligned = True
                    self._log(f"[ESCALA] altura de referência (auto) = {self.ground_ref:.4f} un. do mapa")
                    self._ground_buf = []
                else:
                    self._ground_buf = self._ground_buf[-(self.ground_init_n - 1):]
            return
        if len(self._ground_buf) < self.ground_win:
            return
        hm = float(np.median(self._ground_buf[-self.ground_win:]))
        s = self.ground_ref / hm
        if abs(s - 1.0) < self.ground_deadband:
            self._ground_buf = self._ground_buf[-self.ground_win:]
            return
        full = not self.ground_aligned                         # 1º alinhamento (altura real dada): histórico todo
        if not full:
            s = float(np.clip(s ** self.ground_gain, 1.0 / self.ground_max_corr, self.ground_max_corr))
        self.ground_aligned = True
        self._apply_scale(s, kid, full=full)
        self._ground_buf = []

    def _apply_scale(self, s: float, kid, full: bool = False, whole_window: bool = False):
        """Semelhança (escala s) em torno do centro do último KF fixo da janela: poses e landmarks da janela ativa
        são re-escalados; tudo antes do pivô fica intacto -> trajetória contínua, sem 'degrau'.
        full=True (1º alinhamento): re-escala o SEGMENTO ATUAL todo, em torno do 1º KF dele (segmentos anteriores
        ficam intactos). whole_window=True: pivô = KF mais antigo da janela (os KFs fixos do BA também são re-escalados,
        senão a linha de base entre eles mantém a escala antiga e o BA desfaz parte da correção). Retorna o pivô."""
        if full:
            ks = sorted(k for k in self.Map.poses if k >= self.seg_kf0)
            piv, scaled = ks[0], ks[1:]
            ids = set()
            for k in ks:
                ids |= self.Map.kf_obs.get(k, set())
            ids |= set(int(l) for l in self.tracks.lm if l >= 0)
        else:
            n_keep = 1 if whole_window else self.ba_fixed
            piv = self.window[n_keep - 1]
            scaled = self.window[n_keep:]
            ids = set(int(l) for l in self.tracks.lm if l >= 0)
            for k in self.window:
                ids |= self.Map.kf_obs.get(k, set())
        c0 = self.Map.poses[piv][:3, 3].copy()
        rel_cur = self._rel_cur()
        rel_cur[:3, 3] *= s
        for pid in ids:
            p = self.Map.points.get(pid)
            if p is not None:
                p.position = c0 + s * (p.position - c0)
        for k in scaled:
            T = self.Map.poses[k].copy()
            T[:3, 3] = c0 + s * (T[:3, 3] - c0)
            self.Map.poses[k] = T
        for k in [piv] + scaled:                               # frames intermediários: offset relativo também escala
            for item in self.kf_frames.get(k, ()):
                item[1] = item[1].copy()
                item[1][:3, 3] *= s
        self._sync_poses_after_ba([piv] + scaled, rel_cur=rel_cur)
        self.ba_prior = scale_prior(self.ba_prior, scaled, c0, s)
        if self.vel is not None:
            self.vel = self.vel.copy()
            self.vel[:3, 3] *= s
        for name in ('speed_ref', 'speed_hi', 'init_speed'):
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, v * s)
        self.scale_corrections.append((kid, s))
        self._log(f"[ESCALA] KF {kid}: correção x{s:.3f} (altura câmera ref={self.ground_ref})")
        return piv

    # ------------------------------------------------- escala externa (fusão -> VO)
    def request_scale(self, s: float, whole_segment: bool = False):
        """Pede uma correção de escala vinda de fora (ex.: escala métrica estimada pelo grafo com a roda/IMU).
        É aplicada no próximo ponto seguro (sem BA em andamento), com o mesmo mecanismo da âncora do chão
        (_apply_scale): whole_segment=True re-escala o segmento atual inteiro, senão só a janela ativa.
        Um pedido novo substitui o pendente (o fator é sempre relativo às unidades atuais)."""
        if np.isfinite(s) and s > 0:
            self._ext_scale = (float(s), bool(whole_segment))
            self._try_ext_scale()

    def _try_ext_scale(self):
        if (self._ext_scale is None or not self.initialized or self._ba_job is not None
                or len(self.window) < self.ba_fixed + 1):
            return
        s, whole = self._ext_scale
        self._ext_scale = None
        piv = self._apply_scale(s, self.cur_kf_id, full=whole, whole_window=True)
        # a âncora do chão passa a seguir a escala externa (não briga com ela), mas continua agindo: as alturas já
        # medidas são convertidas p/ as unidades novas (zerar o buffer a cada correção a deixaria inativa).
        # Altura real dada (camera_height) e ainda sem o 1º alinhamento: ground_ref está em METROS, não em unidades do
        # mapa -> não acompanha a escala externa (senão a âncora passa a buscar uma altura s vezes maior e briga com
        # a fusão: urban30 teve escala x8 / x0.1 / x3.6 em segmentos seguidos)
        if self.ground_ref is not None and (self.camera_height is None or self.ground_aligned):
            self.ground_ref *= s
        self._ground_buf = [h * s for h in self._ground_buf]
        self.unit_events.append((self.keyframes_dict[piv].frame_idx, float(np.log(s)), self.segment))

    def frame_units(self, f: int) -> float:
        """Fator (unidades atuais / unidades originais) das poses do frame f, acumulado pelas escalas externas
        (cada uma re-escala exatamente os frames >= frame do seu pivô)."""
        fp, ls = self.unit_arrays()
        return float(np.exp(ls[fp <= f].sum())) if len(fp) else 1.0

    def unit_arrays(self):
        """unit_events como arrays (frame do pivô, log do fator), refeitos só quando chega evento novo: frame_units é
        chamado várias vezes por frame e a lista cresce a sessão toda (~700 eventos em 12 km)."""
        if getattr(self, '_ue_n', -1) != len(self.unit_events):
            self._ue_fp = np.array([e[0] for e in self.unit_events], np.int64)
            self._ue_ls = np.array([e[1] for e in self.unit_events], np.float64)
            self._ue_n = len(self.unit_events)
        return self._ue_fp, self._ue_ls

    def _triangulate_new(self, color, kid) -> int:
        tr = self.tracks
        idx = np.where(tr.lm < 0)[0]
        if len(idx) == 0:
            return 0
        T_c = self.keyframes_dict[kid].pose
        T_cw_c = np.linalg.inv(T_c)
        P_c = self.K @ T_cw_c[:3]
        added = 0
        for okf in np.unique(tr.org_kf[idx]):
            kf_o = self.keyframes_dict.get(int(okf))
            sel = idx[tr.org_kf[idx] == okf]
            if kf_o is None or okf == kid or len(sel) == 0:
                continue
            T_cw_o = np.linalg.inv(kf_o.pose)
            P_o = self.K @ T_cw_o[:3]
            uv_o = tr.org_uv[sel].astype(np.float64)
            uv_c = tr.pts[sel].astype(np.float64)
            X = cv2.triangulatePoints(P_o, P_c, uv_o.T, uv_c.T)
            X = (X[:3] / X[3]).T
            good, _ = self._check_triangulation(X, T_cw_o[:3, :3], T_cw_o[:3, 3],
                                                T_cw_c[:3, :3], T_cw_c[:3, 3], uv_o, uv_c)
            for j in np.where(good)[0]:
                i = sel[j]
                p = self.Map.add_point(
                    X[j], None,
                    [(int(okf), (float(uv_o[j, 0]), float(uv_o[j, 1]))), (kid, (float(uv_c[j, 0]), float(uv_c[j, 1])))],
                    color=self._color_at(color, uv_c[j]))
                tr.lm[i] = p.id
                added += 1
        return added

    def _reset_map(self):
        """Perda de tracking: reinicia o mapa a partir da última pose.
        A escala do novo segmento é ancorada na velocidade (por frame) dos últimos keyframes."""
        if self.speed_ref is not None and self.speed_ref > 1e-9:
            self.init_speed = float(self.speed_ref)      # robusta: não depende dos últimos KFs (contaminados pela curva)
        self.segment += 1
        self._log(f"[RESET] Tracking perdido. Reinicializando mapa (segmento {self.segment}, "
                  f"velocidade de referência={self.init_speed}).")
        self.anchor_pose = self.T_cur.copy()
        self._ground_buf = []
        self.initialized = False
        self.init_ref = None
        self.tracks = _Tracks()
        self.window = []
        self.ba_prior = None
        self._ba_pending = False
        self._ext_scale = None
        self.vel = None
        self.lost = 0

    # ------------------------------------------------------------------ API
    def process_frame_pnp(self, frame_img: Optional[np.ndarray], timestamp: Optional[float] = None,
                          prepared: Optional[Tuple[np.ndarray, np.ndarray]] = None, return_traj: bool = True):
        """Processa um frame. Retorna (trajetória (N,3), poses T_wc (N,4,4)), uma entrada por frame.
        `prepared` = saída de `prepare(frame)` já calculada (ex.: em thread de prefetch).
        `return_traj=False` evita copiar a trajetória inteira a cada frame (custo cresce com N)."""
        if self.async_ba:
            self._poll_ba()
        self._try_ext_scale()
        if timestamp is None:
            timestamp = time.time()
        self.timestamps.append(timestamp)
        frame_idx = len(self.timestamps) - 1

        color, gray = prepared if prepared is not None else self._prep(frame_img)
        self.img_h = gray.shape[0]
        if not self.initialized:
            self._process_init(color, gray, timestamp, frame_idx)
        else:
            self._process_tracking(color, gray, timestamp, frame_idx)
        self.prev_gray = gray
        if not return_traj:
            return None
        return np.array(self.trajectory_est), np.array(self.poses_est)

    process_frame = process_frame_pnp   # alias

    # ------------------------------------------------------------ visualização
    def draw_tracks(self, frame_img: np.ndarray) -> np.ndarray:
        """Frame com os tracks: verde = com landmark 3D, vermelho = ainda sem triangulação."""
        img = frame_img.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if self.initialized:
            tr = self.tracks
            for p, l in zip(tr.pts, tr.lm):
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 255, 0) if l >= 0 else (0, 0, 255), -1)
            txt = f"KFs:{len(self.keyframes_dict)} lm:{int((tr.lm >= 0).sum())} tracks:{len(tr)}"
            if self.lost:
                txt += f"  PERDIDO({self.lost})"
        else:
            if self.init_ref is not None:
                for p, p0 in zip(self.init_ref['pts'], self.init_ref['pts0']):
                    cv2.line(img, (int(p0[0]), int(p0[1])), (int(p[0]), int(p[1])), (0, 255, 255), 1)
            txt = "Inicializando..."
        cv2.putText(img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
        return img

    def update_plot(self, plot_3d: bool = False):
        """Trajetória e mapa no plano X-Z. (plot_3d=True desenha também os landmarks.)"""
        if not self.enable_plot or self.ax is None:
            return
        draw_snapshot(self.ax, plot_snapshot(self, with_points=plot_3d))
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def cleanup_plots(self):
        self.finish()
        plt.ioff()
        plt.close('all')

    # ------------------------------------------------------------------ saída
    def save_keyframes_info(self, filename: str):
        """CSV: timestamp,R00,R01,R02,tx,R10,...,tz  (T_wc por frame, alinhado com o timestamp)."""
        with open(filename, 'w') as f:
            f.write("timestamp,R00,R01,R02,tx,R10,R11,R12,ty,R20,R21,R22,tz\n")
            for ts, pose in zip(self.timestamps, self.poses_est):
                vals = [f"{ts}"] + [f"{v:.6f}" for v in pose[:3, :4].flatten()]
                f.write(",".join(vals) + "\n")
        print(f"Poses salvas em {filename}")

    def save_trajectory(self, filename: str):
        """Formato TUM: timestamp tx ty tz qx qy qz qw (útil p/ evo)."""
        from scipy.spatial.transform import Rotation
        with open(filename, 'w') as f:
            for ts, pose in zip(self.timestamps, self.poses_est):
                q = Rotation.from_matrix(pose[:3, :3]).as_quat()  # x y z w
                t = pose[:3, 3]
                f.write(f"{ts} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
        print(f"Trajetória (TUM) salva em {filename}")

    def save_map(self, filename: str, ply: bool = True):
        """Salva o mapa p/ visualizar depois (view_map.py), sem rodar o VO de novo.
        .npz: pontos válidos (N,3) + cores RGB + nº de observações, poses dos keyframes, trajetória (uma pose por
        frame + timestamps) e K. Com ply=True grava também um .ply (abre no MeshLab / CloudCompare / Open3D).
        Referencial: o do mundo do VO (= 1ª câmera: x direita, y p/ baixo, z p/ frente), escala do mapa."""
        pts = self.Map.get_all_points()
        P = np.array([p.position for p in pts], np.float64).reshape(-1, 3)
        C = np.array([p.color if p.color is not None else (255, 0, 0) for p in pts], np.uint8).reshape(-1, 3)
        n_obs = np.array([len(p.observations) for p in pts], np.int32)
        kf_ids = np.array(sorted(self.keyframes_dict), np.int64)
        kf_poses = np.array([self.keyframes_dict[k].pose for k in kf_ids], np.float64).reshape(-1, 4, 4)
        np.savez_compressed(
            filename, points=P, colors=C, n_obs=n_obs, kf_ids=kf_ids, kf_poses=kf_poses,
            poses=np.array(self.poses_est, np.float64).reshape(-1, 4, 4),
            timestamps=np.array([float(t) for t in self.timestamps]), K=self.K,
            camera_height=np.nan if self.ground_ref is None else float(self.ground_ref))
        print(f"Mapa salvo em {filename} ({len(P)} pontos, {len(kf_ids)} keyframes)")
        if ply:
            write_ply(os.path.splitext(filename)[0] + ".ply", P, C)
