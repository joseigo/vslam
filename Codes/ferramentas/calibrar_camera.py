"""
Calibra uma câmera com fotos de um tabuleiro de xadrez e grava o YAML que o pipeline lê (calibration/left.yaml:
camera_matrix, distortion_coefficients -- mesmo formato do KAIST).

Uso (câmera do robô do SLAM-Teaching: grave frames com o tabuleiro em várias posições/inclinações, cobrindo também
as bordas da imagem, ex. com 05_camera_robo.py --gravar mexendo o tabuleiro na frente do robô parado):
    python3 calibrar_camera.py ~/Documentos/SLAM/SLAM-Teaching/utilitarios/dados/camera_AAAAMMDD_HHMMSS \
        --cantos 9x6 --quadrado 0.025 --out camera_robo.yaml

--cantos = cantos INTERNOS (colunas x linhas) do tabuleiro; --quadrado = lado do quadrado em metros.
Usa no máximo --max-imagens (escolhidas espalhadas no tempo). Erro de reprojeção RMS < 0,5 px é bom; > 1 px, refazer.
"""
import argparse
import glob
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta")
    ap.add_argument("--cantos", default="9x6")
    ap.add_argument("--quadrado", type=float, default=0.025)
    ap.add_argument("--max-imagens", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    nx, ny = (int(v) for v in a.cantos.lower().split("x"))
    obj = np.zeros((nx * ny, 3), np.float32)
    obj[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * a.quadrado
    fs = sorted(glob.glob(os.path.join(a.pasta, "*.jpg")) + glob.glob(os.path.join(a.pasta, "*.png")))
    if not fs:
        raise SystemExit("nenhuma imagem")
    pts_o, pts_i, usadas, tam = [], [], [], None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    for f in fs[::max(1, len(fs) // (4 * a.max_imagens))]:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        tam = g.shape[::-1]
        ok, c = cv2.findChessboardCorners(g, (nx, ny), flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            pts_o.append(obj)
            pts_i.append(cv2.cornerSubPix(g, c, (5, 5), (-1, -1), crit))
            usadas.append(f)
    if len(pts_o) > a.max_imagens:                      # espalhadas no tempo
        idx = np.linspace(0, len(pts_o) - 1, a.max_imagens).round().astype(int)
        pts_o, pts_i = [pts_o[i] for i in idx], [pts_i[i] for i in idx]
    print(f"tabuleiro achado em {len(pts_o)} imagens (de {len(fs)})")
    if len(pts_o) < 10:
        raise SystemExit("menos de 10 imagens com o tabuleiro: grave mais (e confira --cantos)")
    rms, K, dist, _, _ = cv2.calibrateCamera(pts_o, pts_i, tam, None, None)
    print(f"RMS de reprojeção: {rms:.3f} px\nK =\n{K}\ndistorção = {dist.ravel()}")
    fsw = cv2.FileStorage(a.out, cv2.FILE_STORAGE_WRITE)
    fsw.write("image_width", int(tam[0]))
    fsw.write("image_height", int(tam[1]))
    fsw.write("camera_matrix", K)
    fsw.write("distortion_coefficients", dist.reshape(1, -1))
    fsw.write("rms_reprojecao_px", float(rms))
    fsw.write("n_imagens", len(pts_o))
    fsw.release()
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
