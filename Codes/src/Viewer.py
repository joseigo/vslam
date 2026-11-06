"""
Visualização em PROCESSOS separados: o loop do VO só enfileira dados (sem bloquear) e segue.
Se o visualizador estiver ocupado, a mensagem é descartada (mostra sempre o estado mais recente).
Os processos rodam com prioridade baixa (nice), p/ não roubar CPU do VO e do BA.

- FrameViewer: janela do OpenCV com os tracks (tecla 'q' -> `stop_requested`).
- PlotViewer : trajetória + landmarks no plano X-Z (matplotlib), mesma figura de VisualOdometry.update_plot.
"""
import multiprocessing as mp
import os
import queue
import time

import numpy as np


def _drain(q, timeout):
    """Bloqueia até chegar algo e devolve só a mensagem mais recente (None = encerrar)."""
    msg = q.get(timeout=timeout)
    while True:
        try:
            msg = q.get_nowait()
        except queue.Empty:
            return msg


def _lower_priority(n=10):
    try:
        os.nice(n)
    except OSError:
        pass


class _Proc:
    def __init__(self, target, *args):
        ctx = mp.get_context("spawn")
        self.q = ctx.Queue(maxsize=2)
        self.stop_event = ctx.Event()
        self.proc = ctx.Process(target=target, args=(self.q, self.stop_event) + args, daemon=True)
        self.proc.start()

    @property
    def stop_requested(self) -> bool:
        return self.stop_event.is_set()

    def _send(self, msg) -> bool:
        try:
            self.q.put_nowait(msg)
            return True
        except queue.Full:
            return False

    def busy(self) -> bool:
        return self.q.full()

    def close(self, timeout=5.0):
        if self.proc.is_alive():
            try:
                self.q.put(None, timeout=timeout)
            except queue.Full:
                pass
            self.proc.join(timeout)
            if self.proc.is_alive():
                self.proc.terminate()
        self.q.cancel_join_thread()


# ------------------------------------------------------------------ janela OpenCV
def _frame_loop(q, stop_event, win):
    _lower_priority()
    import cv2
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        try:
            msg = _drain(q, 0.05)
        except queue.Empty:
            msg = False
        if msg is None:
            break
        if msg is not False:
            img, pts, lm, txt, ts = msg
            for p, l in zip(pts, lm):
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 255, 0) if l >= 0 else (0, 0, 255), -1)
            cv2.putText(img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
            cv2.putText(img, f"ts: {ts}", (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(win, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
    cv2.destroyAllWindows()


class FrameViewer(_Proc):
    def __init__(self, win="Visual Odometry Frame", max_fps=25.0):
        super().__init__(_frame_loop, win)
        self.min_dt = 1.0 / max_fps if max_fps else 0.0
        self._t_last = 0.0

    def show(self, vo, frame, ts_text, extra="") -> bool:
        """Copia só o necessário do estado do VO (pontos + flags) e manda p/ o processo desenhar (`extra`: texto
        adicional, ex.: pose fundida).
        Limitado a `max_fps`: mostrar mais rápido que isso só gasta CPU (cada frame são ~2 MB copiados)."""
        now = time.perf_counter()
        if now - self._t_last < self.min_dt or self.busy():
            return False
        self._t_last = now
        img = frame if frame.ndim == 3 else np.dstack([frame] * 3)
        if vo.initialized:
            tr = vo.tracks
            pts, lm = tr.pts.copy(), tr.lm.copy()
            txt = f"KFs:{len(vo.keyframes_dict)} lm:{int((lm >= 0).sum())} tracks:{len(lm)}"
            if vo.lost:
                txt += f"  PERDIDO({vo.lost})"
            if extra:
                txt += f" | {extra}"
        else:
            pts, lm = np.zeros((0, 2), np.float32), np.zeros(0, np.int64)
            txt = "Inicializando..."
            if vo.init_ref is not None:     # linhas do fluxo de inicialização: desenha aqui mesmo (raro)
                import cv2
                img = img.copy()
                for p, p0 in zip(vo.init_ref['pts'], vo.init_ref['pts0']):
                    cv2.line(img, (int(p0[0]), int(p0[1])), (int(p[0]), int(p[1])), (0, 255, 255), 1)
        return self._send((img, pts, lm, txt, ts_text))


# ------------------------------------------------------------------ plot matplotlib
def plot_snapshot(vo, with_points=True) -> dict:
    """Estado mínimo p/ desenhar o plot (arrays numpy, 'picklable')."""
    traj = np.array(vo.trajectory_est) if vo.trajectory_est else np.zeros((0, 3))
    snap = dict(traj=traj, pose=(vo.poses_est[-1].copy() if vo.poses_est else None), P=None, C=None)
    if with_points:
        pts = vo.Map.get_all_points()
        if pts:
            snap['P'] = np.array([p.position for p in pts], np.float32)
            snap['C'] = np.array([p.color / 255.0 if p.color is not None else [1.0, 0.0, 0.0] for p in pts],
                                 np.float32)
    return snap


def draw_snapshot(ax, snap, axes=(0, 2), title="Odometria Visual Monocular (escala arbitrária)", unit="unid."):
    """Mesmo desenho de VisualOdometry.update_plot, a partir de um snapshot. axes = eixos do plano desenhado
    ((0, 2) = X-Z do mundo do VO; (0, 1) = X-Y do mundo fundido, visto de cima)."""
    import matplotlib.pyplot as plt
    a, b = axes
    ax.cla()
    traj = snap['traj']
    if snap['P'] is not None:
        P = snap['P']
        ax.scatter(P[:, a], P[:, b], c=snap['C'], s=3, alpha=0.8, rasterized=True, label='Landmarks')
    if len(traj):
        ax.plot(traj[:, a], traj[:, b], '-', color='magenta', lw=1.5, label='Trajetória')
        ax.plot(traj[0, a], traj[0, b], 'gx', ms=8, mew=2, label='Início')
        xs, zs = traj[:, a], traj[:, b]
        extent = max(np.ptp(xs), np.ptp(zs), 1.0)
        m = 0.3 * extent
        cx, cz = (xs.min() + xs.max()) / 2, (zs.min() + zs.max()) / 2
        half = extent / 2 + m
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cz - half, cz + half)
        T = snap['pose']
        c = np.array([T[a, 3], T[b, 3]])
        h = np.array([T[a, 2], T[b, 2]])
        n = np.linalg.norm(h)
        if n > 1e-9:
            h /= n
            perp = np.array([-h[1], h[0]])
            s = 0.04 * (2 * half)
            tri = np.array([c + h * s, c - h * s * 0.6 + perp * s * 0.5, c - h * s * 0.6 - perp * s * 0.5])
            ax.add_patch(plt.Polygon(tri, closed=True, fc='red', ec='black', alpha=0.9))
    ax.set_xlabel(f"{'XYZ'[a]} [{unit}]")
    ax.set_ylabel(f"{'XYZ'[b]} [{unit}]")
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    if len(traj):
        ax.legend(loc='upper right')


class _LivePlot:
    """Mesma figura de draw_snapshot, mas com artistas persistentes (só atualiza os dados):
    evita reconstruir eixos/legenda/grade a cada redesenho (~2x mais rápido com alguns milhares de pontos)."""

    def __init__(self, ax):
        import matplotlib.pyplot as plt
        self.ax = ax
        self.sc = ax.scatter(np.zeros(0), np.zeros(0), s=3, alpha=0.8, rasterized=True, label='Landmarks')
        self.traj, = ax.plot([], [], '-', color='magenta', lw=1.5, label='Trajetória')
        self.start, = ax.plot([], [], 'gx', ms=8, mew=2, label='Início')
        self.robot = plt.Polygon(np.zeros((3, 2)), closed=True, fc='red', ec='black', alpha=0.9, visible=False)
        ax.add_patch(self.robot)
        ax.set_xlabel("X [unid.]")
        ax.set_ylabel("Z [unid.]")
        ax.set_title("Odometria Visual Monocular (escala arbitrária)")
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper right')

    def update(self, snap):
        if snap['P'] is not None:
            self.sc.set_offsets(snap['P'][:, [0, 2]])
            self.sc.set_facecolors(snap['C'])
        traj = snap['traj']
        if not len(traj):
            return
        self.traj.set_data(traj[:, 0], traj[:, 2])
        self.start.set_data([traj[0, 0]], [traj[0, 2]])
        xs, zs = traj[:, 0], traj[:, 2]
        extent = max(np.ptp(xs), np.ptp(zs), 1.0)
        cx, cz = (xs.min() + xs.max()) / 2, (zs.min() + zs.max()) / 2
        half = extent / 2 + 0.3 * extent
        self.ax.set_xlim(cx - half, cx + half)
        self.ax.set_ylim(cz - half, cz + half)
        T = snap['pose']
        c = np.array([T[0, 3], T[2, 3]])
        h = np.array([T[0, 2], T[2, 2]])
        n = np.linalg.norm(h)
        if n > 1e-9:
            h /= n
            perp = np.array([-h[1], h[0]])
            s = 0.04 * (2 * half)
            self.robot.set_xy(np.array([c + h * s, c - h * s * 0.6 + perp * s * 0.5, c - h * s * 0.6 - perp * s * 0.5]))
            self.robot.set_visible(True)


def _plot_loop(q, stop_event):
    _lower_priority()
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    live = _LivePlot(ax)
    plt.ion()
    plt.show(block=False)
    while plt.fignum_exists(fig.number):
        try:
            msg = _drain(q, 0.05)
        except queue.Empty:
            plt.pause(0.05)          # mantém a janela responsiva
            continue
        if msg is None:
            break
        live.update(msg)
        fig.canvas.draw_idle()
        plt.pause(0.001)
    plt.close('all')


class PlotViewer(_Proc):
    def __init__(self):
        super().__init__(_plot_loop)

    def update(self, vo, with_points=True) -> bool:
        if self.busy():               # ainda desenhando o anterior: nem monta o snapshot
            return False
        return self._send(plot_snapshot(vo, with_points))


# ------------------------------------------------------------------ várias trajetórias (fusão)
def _traj_loop(q, stop_event, title):
    _lower_priority()
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(True); ax.set_aspect('equal', adjustable='datalim')
    ax.set_title(title)
    styles = {'GT': dict(color='k', lw=2.5), 'fusão (online)': dict(color='tab:blue', lw=1.5),
              'nós (suavizado)': dict(color='tab:orange', lw=1.2, ls='--'), 'odometria de rodas': dict(color='tab:gray', lw=1),
              'laços': dict(color='tab:green', lw=1.2)}
    lines = {}
    plt.ion()
    plt.show(block=False)
    while plt.fignum_exists(fig.number):
        try:
            msg = _drain(q, 0.05)
        except queue.Empty:
            plt.pause(0.05)
            continue
        if msg is None:
            break
        for name, xy in msg['lines'].items():
            if name not in lines:
                lines[name], = ax.plot([], [], label=name, **styles.get(name, {}))
                ax.legend(loc='upper right')
            lines[name].set_data(xy[:, 0], xy[:, 1])
        if msg.get('text'):
            ax.set_title(f"{title}\n{msg['text']}")
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()
        plt.pause(0.001)
    plt.close('all')


class TrajViewer(_Proc):
    """Plot ao vivo de várias trajetórias XY: update(dict(lines={nome: (N,2)}, text=str))."""

    def __init__(self, title="Fusão VO + sensores"):
        super().__init__(_traj_loop, title)

    def update(self, msg) -> bool:
        if self.busy():
            return False
        return self._send(msg)


# ------------------------------------------------------------------ Open3D 3D ao vivo (1ª e 3ª pessoa)
_UP = np.array([0.0, -1.0, 0.0])    # "cima" do mundo do VO (= 1ª câmera, y p/ baixo)
# referencial do mundo -> ("cima", direção de onde a 3ª pessoa olha: de cima e de trás da 1ª pose)
_WORLDS = {"camera": (_UP, np.array([0.0, -0.75, -0.66])),          # mundo do VO: x dir., y baixo, z frente
           "vehicle": (np.array([0.0, 0.0, 1.0]), np.array([-0.66, 0.0, 0.75]))}   # fundido: x frente, z cima


def _look_at(eye, target, up):
    """Extrínseca (mundo->câmera, convenção OpenCV/Open3D: x dir., y p/ baixo, z p/ frente)."""
    z = target - eye
    z /= max(np.linalg.norm(z), 1e-12)
    x = np.cross(-up, z)
    if np.linalg.norm(x) < 1e-9:                 # olhando exatamente ao longo do "up"
        x = np.cross(np.array([0.0, 0.0, 1.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    E = np.eye(4)
    E[:3, :3] = np.stack([x, y, z])
    E[:3, 3] = -E[:3, :3] @ eye
    return E


def _frustum_pts(T, K, size):
    if K is not None:
        w, h, f = 2 * K[0, 2], 2 * K[1, 2], K[0, 0]
    else:
        w, h, f = 2.0, 1.0, 1.0
    c = np.array([[0, 0, 0], [-w / 2, -h / 2, f], [w / 2, -h / 2, f], [w / 2, h / 2, f], [-w / 2, h / 2, f]]) / f * size
    return (T[:3, :3] @ c.T).T + T[:3, 3]


_FRUSTUM_LINES = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]], np.int32)


class _O3DWindow:
    """Uma janela Open3D com nuvem + trajetória + câmera atual. A vista é recalculada a cada atualização
    (enquanto `follow` estiver ligado; tecla F alterna -> com ele desligado, o mouse navega livremente)."""

    def __init__(self, o3d, name, left, top, width, height, point_size, bg):
        self.o3d = o3d
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=name, width=width, height=height, left=left, top=top)
        opt = self.vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.asarray(bg, float)
        opt.line_width = 2.0
        self.pcd = o3d.geometry.PointCloud()
        self.traj = o3d.geometry.LineSet()
        self.cam = o3d.geometry.LineSet()
        # O ViewControl do Open3D calcula a vista a partir da bounding box da cena: vazia ou degenerada (ex.: VO ainda
        # não inicializado, trajetória toda em 0) -> extrínseca NaN -> tela preta. Fixa uma bbox não degenerada com
        # uma geometria auxiliar (removida em seguida, a bbox fica); a vista em si é sempre definida por set_view.
        dummy = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])))
        self.vis.add_geometry(dummy, reset_bounding_box=True)
        self.vis.remove_geometry(dummy, reset_bounding_box=False)
        for g in (self.pcd, self.traj, self.cam):
            self.vis.add_geometry(g, reset_bounding_box=False)
        self.follow = True
        self.alive = True
        self.eye = self.target = None

        def toggle_follow(_):
            self.follow = not self.follow
            print(f"[{name}] seguir câmera: {'ligado' if self.follow else 'desligado (mouse livre)'}")
            return False
        self.vis.register_key_callback(ord('F'), toggle_follow)

    def set_points(self, P, C):
        V = self.o3d.utility.Vector3dVector
        self.pcd.points = V(P.astype(np.float64))
        self.pcd.colors = V(C.astype(np.float64))
        self.vis.update_geometry(self.pcd)

    def set_traj(self, traj, T, K, cam_size):
        V, I = self.o3d.utility.Vector3dVector, self.o3d.utility.Vector2iVector
        if len(traj) > 1:
            self.traj.points = V(traj.astype(np.float64))
            self.traj.lines = I(np.c_[np.arange(len(traj) - 1), np.arange(1, len(traj))].astype(np.int32))
            self.traj.paint_uniform_color([1.0, 0.0, 1.0])
            self.vis.update_geometry(self.traj)
        if T is not None:
            self.cam.points = V(_frustum_pts(T, K, cam_size))
            self.cam.lines = I(_FRUSTUM_LINES)
            self.cam.paint_uniform_color([1.0, 0.2, 0.2])
            self.vis.update_geometry(self.cam)

    def set_view(self, eye, target, up, near, far, smooth=0.0):
        if not self.follow:
            return
        if self.eye is not None and smooth > 0:      # suaviza o tremor da pose estimada
            eye = smooth * self.eye + (1 - smooth) * eye
            target = smooth * self.target + (1 - smooth) * target
        self.eye, self.target = eye, target
        vc = self.vis.get_view_control()
        vc.set_constant_z_near(near)
        vc.set_constant_z_far(far)
        prm = vc.convert_to_pinhole_camera_parameters()
        prm.extrinsic = _look_at(eye, target, up)
        vc.convert_from_pinhole_camera_parameters(prm, allow_arbitrary=True)

    def tick(self):
        if self.alive:
            self.alive = self.vis.poll_events()
            self.vis.update_renderer()
            if not self.alive:
                self.vis.destroy_window()


def _o3d_loop(q, stop_event, K, point_size, size, world, title):
    _lower_priority()
    import open3d as o3d
    W, H = size
    up, view_dir = _WORLDS[world]
    bg = [0.08, 0.08, 0.1]
    fp = _O3DWindow(o3d, f"{title} - 1a pessoa (F: seguir/livre)", 50, 50, W, H, point_size, bg)
    tp = _O3DWindow(o3d, f"{title} - 3a pessoa / trajetoria (F: seguir/livre)", 60 + W, 50, W, H, point_size, bg)
    wins = (fp, tp)
    depth = None              # profundidade típica da cena (un. do mapa): dá a escala da vista em 1ª pessoa
    P = None
    while any(w.alive for w in wins):
        msg = False
        try:                  # junta tudo que chegou: pose/trajetória mais recentes + últimos pontos enviados
            while True:
                m = q.get(timeout=0.02) if msg is False else q.get_nowait()
                if m is None:
                    msg = None
                    break
                if msg and m.get('P') is None and msg.get('P') is not None:
                    m['P'], m['C'] = msg['P'], msg['C']
                msg = m
        except queue.Empty:
            pass
        if msg is None:
            break
        if msg is not False and msg['pose'] is not None:
            traj, T = msg['traj'], msg['pose']
            pos, R = T[:3, 3], T[:3, :3]
            if msg.get('P') is not None and len(msg['P']):
                P = msg['P']
                for w in wins:
                    if w.alive:
                        w.set_points(P, msg['C'])
            if P is not None and len(P) > 10:
                d = np.linalg.norm(P - pos, axis=1)
                depth = float(np.median(d))
            ext = float(np.ptp(traj, axis=0).max()) if len(traj) > 1 else 0.0
            s = depth or max(ext, 1.0)

            # 1ª pessoa: logo atrás e um pouco acima da câmera, olhando p/ onde ela olha (rola junto com ela)
            fwd, down = R[:, 2], R[:, 1]
            eye = pos - 0.05 * s * fwd - 0.02 * s * down
            if fp.alive:
                fp.set_traj(traj, T, K, 0.015 * s)
                fp.set_view(eye, pos + s * fwd, -down, 0.005 * s, 200 * s, smooth=0.3)

            # 3ª pessoa: enquadra a trajetória inteira + câmera atual, de cima e de trás (mundo fixo)
            if tp.alive:
                lo, hi = traj.min(axis=0), traj.max(axis=0)
                center = (lo + hi) / 2
                r = max(0.5 * float(np.linalg.norm(hi - lo)), 0.5 * s, 1e-3)
                vd = view_dir / np.linalg.norm(view_dir)            # de cima e de trás da 1ª pose
                tp.set_traj(traj, T, K, max(0.04 * r, 0.03 * s))
                tp.set_view(center + vd * 2.4 * r, center, up, 0.01 * r, 100 * r + 200 * s, smooth=0.6)
        for w in wins:
            w.tick()
    for w in wins:
        if w.alive:
            w.vis.destroy_window()


class Open3DViewer(_Proc):
    """Mapa 3D ao vivo no Open3D (processo separado), em duas janelas:
    - 1ª pessoa: a vista acompanha a câmera (posição + orientação);
    - 3ª pessoa: trajetória completa enquadrada automaticamente.
    Tecla F em cada janela alterna entre seguir automaticamente e navegar com o mouse.
    update() a cada frame é barato (só pose + trajetória); os pontos do mapa vão a cada `points_every` s.
    world='camera': mundo do VO (escala arbitrária); world='vehicle': mundo fundido (métrico, z p/ cima), com
    `snap_fn(vo, with_points)` -> dict(traj, pose, P, C) já nesse mundo (ex.: OnlineFusion.world_snapshot)."""

    def __init__(self, K=None, point_size=2.0, size=(800, 600), max_fps=30.0, points_every=0.5, world="camera",
                 title="VO 3D", snap_fn=None):
        super().__init__(_o3d_loop, None if K is None else np.asarray(K, float), point_size, size, world, title)
        self.min_dt = 1.0 / max_fps if max_fps else 0.0
        self.points_every = points_every
        self.snap_fn = snap_fn
        self._t_last = self._t_pts = 0.0

    def _snapshot(self, vo, with_points):
        if self.snap_fn is not None:
            return self.snap_fn(vo, with_points)
        if with_points:
            return plot_snapshot(vo, with_points=True)
        return dict(traj=np.array(vo.trajectory_est, np.float32), pose=vo.poses_est[-1].copy(), P=None, C=None)

    def update(self, vo, with_points=True) -> bool:
        now = time.perf_counter()
        if now - self._t_last < self.min_dt or self.busy() or not vo.poses_est:
            return False
        send_pts = with_points and now - self._t_pts >= self.points_every
        snap = self._snapshot(vo, send_pts)
        if snap is None:
            return False
        snap['traj'] = np.asarray(snap['traj'], np.float32)
        if not self._send(snap):
            return False
        self._t_last = now
        if send_pts and snap['P'] is not None:
            self._t_pts = now
        return True

    def final(self, vo, timeout=5.0):
        """Envia o estado final (trajetória + todos os pontos) esperando vaga na fila, sem descartar."""
        if not vo.poses_est or not self.proc.is_alive():
            return
        s = self._snapshot(vo, True)
        if s is None:
            return
        s['traj'] = np.asarray(s['traj'], np.float32)
        try:
            self.q.put(s, timeout=timeout)
        except queue.Full:
            pass
