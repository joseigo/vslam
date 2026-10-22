import numpy as np


class Point:
    """Landmark 3D no referencial do mundo."""

    def __init__(self, position, descriptor, observations, id, color=None):
        self.id = id
        self.position = np.array(position, dtype=np.float64)   # float64: o BA precisa da precisão
        self.descriptor = descriptor                             # opcional (não é mais necessário p/ tracking)
        self.observations = list(observations) if observations else []   # [(kf_id, (u, v)), ...]
        self.color = np.array(color, dtype=np.uint8) if color is not None else None
        self.is_valid = True

    def __repr__(self):
        return (f"Point(id={self.id}, position={self.position.tolist()}, "
                f"observations={len(self.observations)}, valid={self.is_valid})")


class Map:
    def __init__(self, hab_logs=False):
        self.keyframes = {}      # kf_id -> timestamp
        self.poses = {}          # kf_id -> T_wc (4x4, câmera -> mundo)
        self.points = {}         # point_id -> Point
        self.kf_obs = {}         # kf_id -> set(point_id)   (índice p/ o BA não varrer o mapa todo)
        self.next_id = 0
        self.logs = hab_logs

    def __len__(self):
        return len(self.keyframes) + len(self.points)

    def __repr__(self):
        return (f"Map(keyframes={len(self.keyframes)}, poses={len(self.poses)}, "
                f"points={len(self.points)}, valid={self.count_points()})")

    # ---------------- keyframes ----------------
    def add_keyframe(self, keyframe_id, timestamp, pose):
        self.keyframes[keyframe_id] = timestamp
        self.poses[keyframe_id] = np.array(pose, dtype=np.float64)
        self.kf_obs.setdefault(keyframe_id, set())

    def get_keyframe_timestamp(self, keyframe_id):
        return self.keyframes.get(keyframe_id, None)

    def get_keyframe(self, keyframe_id):
        return self.keyframes.get(keyframe_id, None)

    def get_pose(self, keyframe_id):
        return self.poses.get(keyframe_id, None)

    def remove_keyframe(self, keyframe_id):
        self.keyframes.pop(keyframe_id, None)
        self.poses.pop(keyframe_id, None)
        self.kf_obs.pop(keyframe_id, None)

    # ---------------- pontos ----------------
    def add_point(self, position, descriptor=None, observations=None, color=None):
        point = Point(position, descriptor, observations, id=self.next_id, color=color)
        self.points[point.id] = point
        for kf_id, _ in point.observations:
            self.kf_obs.setdefault(kf_id, set()).add(point.id)
        self.next_id += 1
        return point

    def add_observation(self, point_id, kf_id, uv):
        p = self.points.get(point_id)
        if p is None:
            return
        p.observations.append((kf_id, (float(uv[0]), float(uv[1]))))
        self.kf_obs.setdefault(kf_id, set()).add(point_id)

    def remove_observation(self, point_id, kf_id):
        p = self.points.get(point_id)
        if p is None:
            return
        p.observations = [o for o in p.observations if o[0] != kf_id]
        if kf_id in self.kf_obs:
            self.kf_obs[kf_id].discard(point_id)

    def invalidate_point(self, point_id):
        p = self.points.get(point_id)
        if p is not None:
            p.is_valid = False

    def remove_point(self, point_id):
        p = self.points.pop(point_id, None)
        if p is None:
            return
        for kf_id, _ in p.observations:
            if kf_id in self.kf_obs:
                self.kf_obs[kf_id].discard(point_id)

    def get_point(self, point_id):
        return self.points.get(point_id, None)

    def get_all_points(self):
        return [p for p in self.points.values() if p.is_valid]

    def get_points_observed_in(self, frame_id):
        return [self.points[i] for i in self.kf_obs.get(frame_id, ()) if i in self.points]

    def cull_points(self, min_observations=2):
        for pid, p in list(self.points.items()):
            if len(p.observations) < min_observations:
                self.remove_point(pid)

    def count_points(self):
        return sum(1 for p in self.points.values() if p.is_valid)

    def summary(self):
        print(f"Keyframes: {len(self.keyframes)}")
        print(f"Total de pontos no mapa: {len(self.points)}")
        print(f"Pontos válidos: {self.count_points()}")

    # ---------------- visualização ----------------
    def plot_map_points(self, mode="2D", percentile=98, save=None, show=True):
        """Plota o mapa. Ignora o percentil mais distante (pontos muito longe esticam a escala).
        save: caminho p/ salvar a figura; show=False não chama plt.show()."""
        import matplotlib.pyplot as plt

        points = self.get_all_points()
        if not points:
            print("[WARN] Nenhum ponto válido para plotar.")
            return
        P = np.array([p.position for p in points])
        C = np.array([p.color / 255.0 if p.color is not None else [1.0, 0.0, 0.0] for p in points])

        d = np.linalg.norm(P - np.median(P, axis=0), axis=1)
        keep = d <= np.percentile(d, percentile)
        P, C = P[keep], C[keep]

        if mode.upper() == "3D":
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(P[:, 2], P[:, 0], -P[:, 1], c=C, s=2)   # Z p/ frente, X p/ direita, -Y p/ cima
            ax.set_xlabel("Z"); ax.set_ylabel("X"); ax.set_zlabel("-Y")
            ax.set_title("Pontos 3D do Mapa")
        elif mode.upper() == "2D":
            plt.figure()
            plt.scatter(P[:, 0], P[:, 2], c=C, s=2)
            plt.xlabel("X"); plt.ylabel("Z")
            plt.title("Pontos 3D do Mapa (Vista XZ)")
            plt.axis('equal')
        else:
            print(f"[ERRO] Modo inválido: {mode}. Use '2D' ou '3D'.")
            return
        plt.grid(True)
        if save:
            for ax in plt.gcf().axes:          # nuvem grande vira imagem dentro do PDF (eixos/texto vetoriais)
                if ax.name == '3d':            # mplot3d não rasteriza coleções: rasteriza o eixo inteiro
                    ax.set_rasterized(True)
                for c in ([] if ax.name == '3d' else ax.collections):
                    if len(c.get_offsets()) >= 2000:
                        c.set_rasterized(True)
            plt.gcf().savefig(save, dpi=200, bbox_inches='tight')
            print(f"Figura salva em {save}")
        if show:
            plt.show()