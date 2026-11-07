"""
Utilidades dos testes isolados: caminhos do projeto, poses aleatórias, trajetórias sintéticas e câmera do KAIST.

Os testes usam dados SINTÉTICOS com resposta conhecida (não dependem dos datasets), exceto onde indicado.
"""
import os
import sys
import unittest

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CODES_DIR = os.path.dirname(TESTS_DIR)
VOCAB_DIR = os.path.join(CODES_DIR, "vocabularios")
for _d in ("src", "scripts", "ferramentas"):
    _p = os.path.join(CODES_DIR, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                      # noqa: E402
import gtsam                            # noqa: E402
from gtsam import Pose3, Rot3           # noqa: E402

# K e distorção do urban25-34 (left.yaml), as mesmas de testE_vo_time_satmp.CAMERAS["urban"]
K_URBAN = np.array([[816.9037899277, 0.505101667, 608.5072628169],
                    [0., 811.5680382849, 263.4759976444],
                    [0., 0., 1.]])
DIST_URBAN = np.array([-0.0561430278, 0.1395256320, -0.0012155907, -0.0009728139, -0.0808781688])
IMG_W, IMG_H = 1280, 560

LK_PARAMS = dict(winSize=(21, 21), maxLevel=4)


METRICAS = []          # (nome, valor, unidade) do teste em execução; o rodar_testes.py grava no CSV e limpa


def metrica(nome, valor, unidade=""):
    """Registra um número medido pelo teste (vai p/ resultados/testes_unitarios/historico.csv)."""
    METRICAS.append((nome, float(valor), unidade))


# ---------------------------------------------------------------- figuras
FIG_DIR = None         # o rodar_testes.py aponta p/ resultados/testes_unitarios/<execucao>/figuras (None = não gera figuras)

# mesmas cores das figuras do pipeline (test_vo_fusion_rt.COLORS / LOOP_COLORS)
COR = dict(gt="#0b0b0b", vo="#2a78d6", fusao="#eb6834", aceito="#008300", rejeitado="#e34948", neutro="#8a8984",
           texto="#52514e", grade="#e4e3df")


def figuras_ativas() -> bool:
    return FIG_DIR is not None


def nova_figura(ncols=1, nrows=1, w=5.0, h=3.4, **kw):
    """Figura no estilo das figuras da dissertação (eixos discretos, grade clara, linhas finas)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": COR["neutro"],
        "axes.labelcolor": COR["texto"], "xtick.color": COR["texto"], "ytick.color": COR["texto"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": COR["grade"],
        "grid.linewidth": 0.7, "axes.axisbelow": True, "lines.linewidth": 1.8, "legend.frameon": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False, **kw)
    return fig, (ax[0, 0] if ncols * nrows == 1 else ax.ravel())


def salvar_figura(fig, nome):
    """Salva em FIG_DIR/<nome>.pdf (e .png p/ visualizar rápido) e fecha a figura."""
    import matplotlib.pyplot as plt
    if FIG_DIR is None:
        plt.close(fig)
        return None
    os.makedirs(FIG_DIR, exist_ok=True)
    base = os.path.join(FIG_DIR, nome)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return base + ".pdf"


def skip(msg):
    """Pula o teste (pytest reconhece unittest.SkipTest; o rodar_testes.py também)."""
    raise unittest.SkipTest(msg)


def rand_pose(rng, rot_sigma=0.3, trans_sigma=2.0) -> Pose3:
    return Pose3(Rot3.Expmap(rng.normal(0, rot_sigma, 3)), rng.normal(0, trans_sigma, 3))


def rot_err_deg(Ra, Rb) -> float:
    """Ângulo (graus) de Ra^T Rb."""
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def make_T(R=np.eye(3), t=np.zeros(3)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def yaw_R(a) -> np.ndarray:
    """Rotação em torno de z (veículo: z p/ cima)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def cam_yaw_R(a) -> np.ndarray:
    """Rotação em torno do y da CÂMERA (y p/ baixo): yaw da câmera."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def planar_trajectory(n, dt=0.5, speed=5.0, yaw_rate=None):
    """Poses do veículo no mundo (n, 4x4), plano, x p/ frente. yaw_rate(t) [rad/s] (padrão: curvas suaves)."""
    if yaw_rate is None:
        def yaw_rate(t):
            return 0.15 * np.sin(2 * np.pi * t / 40.0)
    T = np.tile(np.eye(4), (n, 1, 1))
    x = y = th = 0.0
    for k in range(1, n):
        t = (k - 1) * dt
        dth = yaw_rate(t) * dt
        x += speed * dt * np.cos(th + dth / 2)
        y += speed * dt * np.sin(th + dth / 2)
        th += dth
        T[k, :3, :3] = yaw_R(th)
        T[k, :3, 3] = [x, y, 0.0]
    return T


def relative(Ta, Tb) -> np.ndarray:
    return np.linalg.inv(Ta) @ Tb


def noisy_rel(T, rng, rot_sigma, trans_frac, trans_min=0.0, yaw_bias=0.0) -> np.ndarray:
    """Pose relativa com ruído: rotação (rad) + translação proporcional (fração) + viés de yaw."""
    d = np.linalg.norm(T[:3, 3])
    R = T[:3, :3] @ Rot3.Expmap(rng.normal(0, rot_sigma, 3)).matrix() @ yaw_R(yaw_bias)
    t = T[:3, 3] + rng.normal(0, trans_frac * d + trans_min, 3)
    return make_T(R, t)


def ate(P, G) -> float:
    """RMSE de posição sem alinhamento (mesmo referencial)."""
    return float(np.sqrt(np.mean(np.sum((np.asarray(P) - np.asarray(G)) ** 2, axis=1))))


def textured_image(rng, h, w, n_shapes=400, blur=1.0) -> np.ndarray:
    """Imagem cinza com formas aleatórias (cantos p/ ORB/Shi-Tomasi) + ruído fino."""
    import cv2
    img = np.full((h, w), 128, np.uint8)
    for _ in range(n_shapes):
        c = int(rng.integers(0, 256))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        s = int(rng.integers(4, max(6, min(h, w) // 12)))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + s, y + int(rng.integers(4, s + 5))), c, -1)
        else:
            cv2.circle(img, (x, y), s // 2, c, -1)
    img = cv2.add(img, rng.integers(0, 12, (h, w), dtype=np.uint8))
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return img
