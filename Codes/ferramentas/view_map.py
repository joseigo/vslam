"""
Navegador 3D do mapa salvo pelo VO (VisualOdometry.save_map) -- não roda o VO, só carrega o arquivo.

Uso:
    python3 view_map.py mapa_vo.npz                 # nuvem + trajetória + keyframes
    python3 view_map.py mapa_vo.npz --min-obs 3     # só pontos vistos por >= 3 keyframes (menos ruído)
    python3 view_map.py mapa_vo.ply                 # também abre só a nuvem (.ply)
    python3 view_map.py mapa_fundido.npz            # mapa no mundo FUNDIDO (metros, z p/ cima; Fusion.save_world_map)

Controles (janela Open3D):
    mouse esq. = girar | dir./shift+esq. = arrastar | roda = zoom | ctrl+esq. = rolar
    +/-  tamanho dos pontos         T  liga/desliga trajetória        K  liga/desliga keyframes
    C    cor: RGB da imagem <-> altura                                V  vista de cima / vista em perspectiva
    H    ajuda completa do Open3D (no terminal)                       Q/Esc  sair
"""
import argparse
import os
import sys

import numpy as np


def load_map(path):
    if path.endswith(".ply"):
        import open3d as o3d
        pc = o3d.io.read_point_cloud(path)
        return dict(points=np.asarray(pc.points), colors=(np.asarray(pc.colors) * 255).astype(np.uint8),
                    n_obs=None, poses=np.zeros((0, 4, 4)), kf_poses=np.zeros((0, 4, 4)), K=None)
    d = np.load(path)
    return {k: d[k] for k in d.files}


def filter_points(m, percentile, min_obs):
    P, C = m['points'], m['colors']
    keep = np.ones(len(P), bool)
    if min_obs and m.get('n_obs') is not None:
        keep &= m['n_obs'] >= min_obs
    if percentile < 100 and keep.sum() > 10:
        # pontos muito distantes (profundidade ruim) esticam a cena: corta o percentil mais longe
        dist = np.linalg.norm(P - np.median(P[keep], axis=0), axis=1)
        keep &= dist <= np.percentile(dist[keep], percentile)
    return P[keep], C[keep]


def z_up(m):
    """Mapa fundido (Fusion.save_world_map): z p/ cima. Mapa do VO: mundo da 1ª câmera, y p/ baixo."""
    return str(m.get('up', '')) == 'z'


def height_colors(P, zup=False):
    """Cor pela altura (z no mapa fundido; -y no mapa do VO, já que y da câmera aponta p/ baixo)."""
    import matplotlib.cm as cm
    h = P[:, 2] if zup else -P[:, 1]
    lo, hi = np.percentile(h, [2, 98])
    return cm.viridis(np.clip((h - lo) / max(hi - lo, 1e-9), 0, 1))[:, :3]


def frustums(kf_poses, K, size, every):
    """Pirâmides de câmera (LineSet) p/ os keyframes."""
    import open3d as o3d
    if K is not None:
        w, h, f = 2 * K[0, 2], 2 * K[1, 2], K[0, 0]
    else:
        w, h, f = 2.0, 1.0, 1.0
    c = np.array([[0, 0, 0], [-w / 2, -h / 2, f], [w / 2, -h / 2, f], [w / 2, h / 2, f], [-w / 2, h / 2, f]])
    c = c / f * size
    pts, lines = [], []
    for i, T in enumerate(kf_poses[::every]):
        base = len(pts)
        pts.extend((T[:3, :3] @ c.T).T + T[:3, 3])
        lines.extend([[base, base + j] for j in range(1, 5)] +
                     [[base + j, base + j % 4 + 1] for j in range(1, 5)])
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.array(pts).reshape(-1, 3)),
                              o3d.utility.Vector2iVector(np.array(lines, np.int32).reshape(-1, 2)))
    ls.paint_uniform_color([0.1, 0.4, 1.0])
    return ls


def view_open3d(m, args):
    import open3d as o3d
    P, C = filter_points(m, args.percentile, args.min_obs)
    rgb = C.astype(np.float64) / 255.0
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    zup = z_up(m)
    hcol = height_colors(P, zup) if len(P) else rgb

    traj = m['poses'][:, :3, 3] if len(m['poses']) else np.zeros((0, 3))
    extent = float(np.ptp(np.vstack([P, traj]) if len(traj) else P, axis=0).max()) if len(P) else 1.0
    geoms = {'pcd': pcd}
    if len(traj) > 1:
        tl = o3d.geometry.LineSet(o3d.utility.Vector3dVector(traj),
                                  o3d.utility.Vector2iVector(np.c_[np.arange(len(traj) - 1), np.arange(1, len(traj))]))
        tl.paint_uniform_color([1.0, 0.0, 1.0])
        start = o3d.geometry.TriangleMesh.create_sphere(radius=0.01 * max(float(np.ptp(traj, axis=0).max()), 1e-6))
        start.translate(traj[0])
        start.paint_uniform_color([0.0, 0.8, 0.0])
        geoms['traj'], geoms['start'] = tl, start
    if len(m['kf_poses']):
        geoms['kfs'] = frustums(m['kf_poses'], m.get('K'), 0.015 * extent, args.kf_every)

    print(f"{len(P)} pontos | {len(traj)} poses | {len(m['kf_poses'])} keyframes")
    print(__doc__.split("Controles")[1])

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Mapa {'fundido' if zup else 'VO'} - {os.path.basename(args.file)}", width=1400,
                      height=900)
    for g in geoms.values():
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0.08, 0.08, 0.1]) if args.dark else np.array([1.0, 1.0, 1.0])
    opt.line_width = 2.0

    state = dict(traj=True, kfs=True, height=False, top=False)

    def set_view(top):
        ctr = vis.get_view_control()
        ctr.set_lookat(np.median(P, axis=0) if len(P) else [0, 0, 0])
        if zup:                      # mundo fundido: x frente, y esq., z cima
            if top:
                ctr.set_front([0, 0, 1]); ctr.set_up([1, 0, 0])
            else:
                ctr.set_front([-0.94, 0, 0.35]); ctr.set_up([0, 0, 1])
        elif top:                    # de cima: olhando ao longo de +y (y da câmera aponta p/ baixo)
            ctr.set_front([0, -1, 0]); ctr.set_up([0, 0, 1])
        else:                        # perspectiva: de cima e de trás
            ctr.set_front([0, -0.35, -0.94]); ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.6)

    def toggle(name):
        def cb(v):
            if name not in geoms:
                return False
            state[name] = not state[name]
            (v.add_geometry if state[name] else v.remove_geometry)(geoms[name], reset_bounding_box=False)
            if name == 'traj' and 'start' in geoms:
                (v.add_geometry if state[name] else v.remove_geometry)(geoms['start'], reset_bounding_box=False)
            return True
        return cb

    def toggle_color(v):
        state['height'] = not state['height']
        pcd.colors = o3d.utility.Vector3dVector(hcol if state['height'] else rgb)
        v.update_geometry(pcd)
        return True

    def toggle_view(v):
        state['top'] = not state['top']
        set_view(state['top'])
        return True

    vis.register_key_callback(ord('T'), toggle('traj'))
    vis.register_key_callback(ord('K'), toggle('kfs'))
    vis.register_key_callback(ord('C'), toggle_color)
    vis.register_key_callback(ord('V'), toggle_view)
    set_view(False)
    vis.run()
    vis.destroy_window()


def view_matplotlib(m, args):
    """Alternativa sem Open3D (mais lenta, para nuvens pequenas)."""
    import matplotlib.pyplot as plt
    P, C = filter_points(m, args.percentile, args.min_obs)
    if len(P) > args.max_mpl:
        P, C = P[::len(P) // args.max_mpl + 1], C[::len(C) // args.max_mpl + 1]
    ax = plt.figure(figsize=(11, 8)).add_subplot(111, projection='3d')
    ax3 = (lambda Q: (Q[:, 0], Q[:, 1], Q[:, 2])) if z_up(m) else (lambda Q: (Q[:, 2], Q[:, 0], -Q[:, 1]))
    ax.scatter(*ax3(P), c=C / 255.0, s=1)          # VO: Z p/ frente, X p/ direita, -Y p/ cima; fundido: x, y, z
    if len(m['poses']):
        ax.plot(*ax3(m['poses'][:, :3, 3]), color='magenta', lw=1.5)
    ax.set_xlabel("x" if z_up(m) else "Z"); ax.set_ylabel("y" if z_up(m) else "X"); ax.set_zlabel("z" if z_up(m) else "-Y")
    ax.set_title(os.path.basename(args.file))
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="Navega pela nuvem de pontos salva pelo VO.")
    ap.add_argument("file", nargs="?", help=".npz (save_map / save_world_map) ou .ply",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "resultados", "urban38",
                                         "dados", "mapa_fundido.npz"))
    ap.add_argument("--percentile", type=float, default=98, help="descarta o percentil mais distante (100 = nada)")
    ap.add_argument("--min-obs", type=int, default=0, help="mín. de keyframes que viram o ponto")
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--kf-every", type=int, default=5, help="desenha 1 a cada N keyframes")
    ap.add_argument("--dark", action="store_true", help="fundo escuro")
    ap.add_argument("--mpl", action="store_true", help="força o visualizador matplotlib")
    ap.add_argument("--max-mpl", type=int, default=300000, help="máx. de pontos no modo matplotlib")
    args = ap.parse_args()

    if not os.path.isfile(args.file):
        sys.exit(f"Arquivo não encontrado: {args.file}")
    m = load_map(args.file)
    try:
        if args.mpl:
            raise ImportError
        import open3d  # noqa: F401
    except ImportError:
        if not args.mpl:
            print("Open3D não instalado (pip install open3d); usando matplotlib.")
        return view_matplotlib(m, args)
    view_open3d(m, args)


if __name__ == "__main__":
    main()
