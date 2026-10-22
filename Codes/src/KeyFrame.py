import numpy as np


class KeyFrame:
    """
    Keyframe leve: guarda só o que o back-end (mapa/BA) precisa.
    O rastreamento frame-a-frame (KLT) fica em VisualOdometry; não guardamos mais
    imagem/descritores aqui (antes, `pts` e `desc` saíam de sincronia depois do 1º frame rastreado).
    """

    def __init__(self, index: int, timestamp, frame_idx: int, pose: np.ndarray):
        self.index = index               # id do keyframe
        self.timestamp = timestamp
        self.frame_idx = frame_idx       # posição em poses_est / timestamps
        self.pose = np.array(pose, dtype=np.float64)   # T_wc (câmera -> mundo)

    def __repr__(self):
        return f"KeyFrame(id={self.index}, frame={self.frame_idx}, t={self.pose[:3, 3].round(3).tolist()})"