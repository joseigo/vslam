import os
# BLAS com 1 thread (ANTES de importar numpy): as matrizes aqui são minúsculas (4x4) e as threads extras do
# OpenBLAS ficam disputando CPU com o processo do BA -> np.linalg.inv chegava a 0.35 ms em vez de ~5 us.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
# estrutura do projeto: Codes/src (biblioteca), Codes/vocabularios, Dissert/resultados
CODES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CODES_DIR, "src"))
VOCAB_DIR = os.path.join(CODES_DIR, "vocabularios")
RESULTS_DIR = os.path.join(os.path.dirname(CODES_DIR), "resultados", "avulsos")   # execuções avulsas: resultados/avulsos/<dataset>/ (a bateria usa --out)
import cv2
import numpy as np
import glob
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from VisualOdometry import VisualOdometry
from Viewer import FrameViewer, Open3DViewer, plot_snapshot, draw_snapshot

# ------------------------------------------------------------------
# CÂMERA  (K e dist_coeffs DEVEM ser do mesmo conjunto/calibração)
# ------------------------------------------------------------------
# No arquivo antigo, K_camera (urban, cx=608) era usada junto com dist_coeffs calculados para
# OUTRA câmera (cx=366, dataset_inomi). Misturar os dois distorce a geometria.
# No KAIST a calibração vem de <dataset>/calibration/left.yaml (load_calibration); "urban" abaixo é só o fallback
# (= left.yaml do urban25-34; urban38/39 têm OUTRA calibração). Sem a distorção a escala do VO cresce sem parar
# (urban39: ATE 71 -> 17 m em 2.8 km só por corrigi-la).
CAMERAS = {
    "urban": dict(
        K=np.array([[816.9037899277, 0.505101667, 608.5072628169],
                    [0., 811.5680382849, 263.4759976444],
                    [0., 0., 1.]]),
        dist=np.array([-0.0561430278, 0.1395256320, -0.0012155907, -0.0009728139, -0.0808781688]),
        height=None),        # altura REAL da câmera sobre o chão (m). Se souber, coloque: o mapa fica em escala métrica
    "inomi": dict(
        K=np.array([[764.95638737, 0., 366.51531468],
                    [0., 766.5635544, 330.23956583],
                    [0., 0., 1.]]),
        dist=np.array([0.16414672, -0.50398383, 0.00763623, -0.00328063, 0.44156834]),
        height=None),        # idem (None = modo auto: a escala fica constante, mas em unidades arbitrárias)
}
CAMERA = "urban"

K_camera = CAMERAS[CAMERA]["K"]
dist_coeffs = CAMERAS[CAMERA]["dist"]
CAMERA_HEIGHT = CAMERAS[CAMERA]["height"]


def load_calibration(images_dir, name="left.yaml", max_up=4):
    """(K, dist, caminho) de <dataset>/calibration/<name> (formato OpenCV/ROS do KAIST), procurado subindo a partir
    da pasta das imagens (urban38/stereo_left/stereo_left -> urban38/calibration). None se não houver."""
    d = os.path.abspath(images_dir)
    for _ in range(max_up + 1):
        path = os.path.join(d, "calibration", name)
        if os.path.isfile(path):
            fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
            K = fs.getNode("camera_matrix").mat()
            dist = fs.getNode("distortion_coefficients").mat()
            fs.release()
            if K is None or K.shape != (3, 3):
                return None
            dist = None if dist is None or not np.any(dist) else dist.reshape(-1).astype(np.float64)
            return K.astype(np.float64), dist, path
        d = os.path.dirname(d)
    return None


def camera_height_for(images_dir, max_up=4):
    """Altura da câmera sobre o chão (m) = z de calibration/Vehicle2Stereo.txt (KAIST: o referencial do veículo fica
    NO CHÃO, sob o eixo traseiro, z p/ cima). None se não houver o arquivo."""
    d = os.path.abspath(images_dir)
    for _ in range(max_up + 1):
        path = os.path.join(d, "calibration", "Vehicle2Stereo.txt")
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    if line.startswith("T:"):
                        return float(line.split()[3])
            return None
        d = os.path.dirname(d)
    return None


def vo_params_for(images_dir):
    """keyframe_params com a altura real da câmera (escala métrica) quando CAMERA_HEIGHT não foi dada à mão."""
    kp = dict(keyframe_params)
    if kp.get('camera_height') is None:
        h = camera_height_for(images_dir)
        if h is not None:
            kp['camera_height'] = h
            print(f"Altura da câmera (Vehicle2Stereo): {h:.3f} m -> VO em escala métrica")
    return kp


def camera_for(images_dir, cam):
    """(K, dist) do dataset: calibration/left.yaml se existir, senão o perfil fixo CAMERAS[cam]."""
    calib = load_calibration(images_dir)
    if calib is not None:
        print(f"Calibração: {calib[2]}")
        return calib[0], calib[1]
    print(f"Calibração: perfil fixo '{cam}' (não achei calibration/left.yaml)")
    return CAMERAS[cam]["K"], CAMERAS[cam]["dist"]

# ------------------------------------------------------------------
# PARÂMETROS
# ------------------------------------------------------------------
orb_params = {
    'nfeatures': 3000, 'scaleFactor': 1.2, 'nlevels': 8, 'edgeThreshold': 31, 'firstLevel': 0,
    'WTA_K': 2, 'scoreType': cv2.ORB_FAST_SCORE, 'patchSize': 31, 'fastThreshold': 20
}

lk_params = dict(winSize=(21, 21), maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

keyframe_params = {
    'min_inliers': 20,              # mínimo de inliers do PnP p/ aceitar a pose
    'min_tracked_ratio': 0.70,      # novo KF se < 70% dos landmarks do último KF ainda são rastreados
    'min_dist_inliers': 25,         # espaçamento mínimo entre features (px)
    'parallax_threshold': 15,       # novo KF se a paralaxe mediana desde o último KF > 15 px
    'max_frames_sem_keyframe': 30,  # (antes: 2000) força KF por tempo, se houver movimento
    'max_features': 800,            # nº de features rastreadas
    'init_parallax': 20,            # paralaxe (px) exigida para inicializar o mapa
    'detector': 'shitomasi',        # 'shitomasi' (recomendado p/ KLT) ou 'orb'

    # --- anti-deriva de escala ---
    'min_tri_angle_deg': 1.5,       # ângulo mínimo de triangulação (antes 1.0): descarta pontos com profundidade ruim
    'ba_min_parallax_deg': 1.0,     # landmarks com menos paralaxe que isso ganham prior fraco no BA
    'ba_point_prior_frac': 0.2,     # sigma do prior = 20% da profundidade
    'block_kf_on_clamp': True,      # não cria KF (não urgente) quando o passo foi limitado
    'use_ground_scale': True,       # âncora de escala pela altura da câmera sobre o plano do chão
    'camera_height': CAMERA_HEIGHT, # None = auto (trava a escala inicial); número = escala métrica
    'ground_row_frac': 0.05,        # só usa pontos abaixo de cy + 5% da altura da imagem p/ achar o chão
    'ground_min_cos': 0.85,         # tolerância de inclinação da câmera (0.85 ~ 32°; baixe se a câmera olha p/ baixo)
    'ground_win': 5,                # nº de estimativas (mediana) por correção
    'ground_deadband': 0.04,        # ignora desvios de escala < 4%
    'ground_gain': 0.7,             # fração do erro corrigida por passo
    'ground_max_corr': 1.15,        # correção máxima por passo (depois do 1º alinhamento)
}

# dataset_path = "/home/joseigo/Documentos/TCC/Datasets/urban25/image/stereo_left"
# dataset_path = "/home/joseigo/Documentos/TCC/Datasets/urban26/stereo_left"
dataset_path = "/home/joseigo/Documentos/TCC/Datasets/urban27/stereo_left"
# dataset_path = "/home/joseigo/Documentos/TCC/Datasets/urban30/stereo_left"
# dataset_path = "/home/joseigo/Documentos/TCC/Datasets/urban32/urban32/urban32/image/stereo_left"
# dataset_path = "/home/joseigo/Documentos/TCC/Datasets/urban33/stereo_left"
# dataset_path = "/home/joseigo/Documentos/TCC/Datasets/urban34/stereo_left"
# dataset_path = "/home/joseigo/Documentos/TCC/Datasets/urban38/stereo_left/stereo_left"
# dataset_path = "/home/joseigo/Documentos/TCC/Datasets/urban39/stereo_left"

# dataset_path = "/home/joseigo/Documentos/TCC/dataset_inomi/dataset_08_08_16_0"
# dataset_path = "/home/joseigo/Documentos/TCC/dataset_inomi/dataset_16_0_0_16"
# dataset_path = "/home/joseigo/Documentos/TCC/dataset_inomi/dataset_16_0"
#dataset_path = "/home/joseigo/Documentos/TCC/dataset_inomi/dataset_16_16"
# dataset_path = "/home/joseigo/Documentos/TCC/dataset_inomi/dataset_16_16_n2" 

SHOW_WINDOW = True      # janela do OpenCV com os tracks (processo separado)
SHOW_3D = True          # mapa 3D ao vivo no Open3D (processo separado): janelas de 1ª pessoa e 3ª pessoa
VIEW3D_MAX_FPS = 30     # pose/trajetória enviadas no máx. N vezes/s (descarta se o visualizador estiver ocupado)
VIEW3D_POINTS_EVERY = 0.5  # nuvem de pontos reenviada a cada X s (montar o snapshot custa CPU, cresce com o mapa)
SAVE_PLOTS = True       # ao fim: matplotlib plota e salva os gráficos (PNG)
SHOW_PLOTS = True       # ...e mostra as janelas (False = só salva)
MAP_FILE = "mapa_vo.npz"  # mapa salvo no fim (+ mapa_vo.ply) em resultados/<dataset>/vo_puro/; veja com visualizar_mapa.py
WINDOW_MAX_FPS = 25     # a janela do OpenCV mostra no máx. N frames/s (o VO continua processando todos)
CV_THREADS = None       # threads internas do OpenCV no processo principal (None = padrão do OpenCV)

# --- paralelismo ---
ASYNC_BA = True         # BA local em outro processo (False = na thread principal, como antes)
                        # keyframe_params['ba_max_lag'] (padrão 0): >0 não espera o BA -> mais FPS, porém pior ATE
PREFETCH_WORKERS = 2    # threads que leem + pré-processam (undistort/cinza) os próximos frames
PREFETCH_DEPTH = 8      # nº de frames lidos adiantado


def _load(vo, img_path):
    """Roda numa thread de prefetch (cv2 libera o GIL): leitura + pré-processamento do frame."""
    frame = cv2.imread(img_path)
    if frame is None:
        return img_path, None, None
    return img_path, frame, vo.prepare(frame)


def prefetch(vo, image_paths, workers=PREFETCH_WORKERS, depth=PREFETCH_DEPTH):
    """Gera (path, frame, (color, gray)) NA ORDEM, mantendo `depth` frames sendo carregados em paralelo."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        it = iter(image_paths)
        q = deque(ex.submit(_load, vo, p) for _, p in zip(range(depth), it))
        while q:
            out = q.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                q.append(ex.submit(_load, vo, nxt))
            yield out


def dataset_name(images_dir):
    """Nome do dataset a partir da pasta das imagens (pula stereo_left/image/...): urban27/stereo_left -> urban27."""
    d = os.path.abspath(images_dir)
    while os.path.basename(d).lower() in ("stereo_left", "stereo_right", "image", "images", "left", "img", ""):
        d = os.path.dirname(d)
    return os.path.basename(d) or "dataset"


def run_dataset_test(path=dataset_path, enable_plot=True):
    if not os.path.isdir(path):
        print(f"Erro: diretório não encontrado: {path}")
        return
    out = os.path.join(RESULTS_DIR, dataset_name(path), "vo_puro")    # saídas do VO puro (não na pasta atual)
    os.makedirs(out, exist_ok=True)
    print(f"Saídas em {out}/")

    image_paths = sorted(glob.glob(os.path.join(path, "*.png")) +
                         glob.glob(os.path.join(path, "*.jpg")))
    if not image_paths:
        print(f"Nenhuma imagem encontrada em {path}")
        return

    K_cam, dist_cam = camera_for(path, CAMERA)
    vo = VisualOdometry(
        camera_matrix=K_cam,
        dist_coeffs=dist_cam,
        orb_params=orb_params,
        lk_params=lk_params,
        keyframe_params=vo_params_for(path),
        sliding_window_size=8,
        enable_plot=False,      # a visualização ao vivo roda no Open3DViewer (outro processo)
        verbose=True,
        async_ba=ASYNC_BA,
    )

    print(f"\nProcessando {len(image_paths)} imagens usando timestamp do nome do arquivo...")
    frame_view = FrameViewer(max_fps=WINDOW_MAX_FPS) if SHOW_WINDOW else None
    view3d = Open3DViewer(K=K_cam, max_fps=VIEW3D_MAX_FPS, points_every=VIEW3D_POINTS_EVERY) \
        if enable_plot and SHOW_3D else None

    if CV_THREADS is not None:
        cv2.setNumThreads(CV_THREADS)

    proc_time = 0.0
    n_ok = 0
    t_wall = time.time()
    for idx, (img_path, frame, prepared) in enumerate(prefetch(vo, image_paths)):
        if frame is None:
            print(f"Erro ao ler imagem {img_path}")
            continue

        timestamp_text = os.path.splitext(os.path.basename(img_path))[0]
        try:
            timestamp = int(timestamp_text)
        except ValueError:
            print(f"Erro ao converter timestamp: {timestamp_text}")
            continue

        t0 = time.time()
        vo.process_frame_pnp(frame, timestamp=timestamp, prepared=prepared, return_traj=False)
        proc_time += time.time() - t0
        n_ok += 1

        if frame_view is not None:
            frame_view.show(vo, frame, timestamp_text)
            if frame_view.stop_requested:       # 'q' na janela
                break

        if view3d is not None:
            view3d.update(vo, with_points=True)

    vo.finish()   # espera o último BA e incorpora o resultado antes de salvar
    t_wall = time.time() - t_wall
    fps = n_ok / proc_time if proc_time > 0 else 0
    print(f"\nProcessamento concluído.")
    print(f"Frames processados: {n_ok} | tempo só do VO: {proc_time:.2f} s | FPS médio: {fps:.2f} | "
          f"tempo total (com leitura/visualização): {t_wall:.2f} s ({n_ok / max(t_wall, 1e-9):.2f} FPS)")
    if ASYNC_BA:
        st = vo.ba_stats
        print(f"BA assíncrono: {st['submitted']} enviados, {st['merged']} incorporados, {st['failed']} falharam/descartados")
    print(f"Segmentos de mapa (1 = nunca perdeu o tracking): {vo.segment + 1}")

    # diagnóstico de escala: altura da câmera (em unid. do mapa) a cada KF. Constante = sem deriva.
    if vo.ground_hist:
        h = np.array([v for _, v in vo.ground_hist])
        print(f"Altura do chão (un. do mapa): {len(h)} estimativas | primeiras 5 (mediana)={np.median(h[:5]):.4f} | "
              f"últimas 5 (mediana)={np.median(h[-5:]):.4f} | correções de escala aplicadas: {len(vo.scale_corrections)}")
        np.savetxt(os.path.join(out, "ground_height_hist.csv"), np.array(vo.ground_hist), delimiter=",",
                   header="kf_id,altura_un_mapa", comments="")
    else:
        print("Aviso: nenhum plano de chão confiável encontrado (ajuste ground_row_frac / ground_min_cos).")

    vo.save_keyframes_info(os.path.join(out, "keyframes_info.csv"))
    vo.save_trajectory(os.path.join(out, "trajectory_tum.txt"))
    vo.save_map(os.path.join(out, MAP_FILE))        # depois: python3 ferramentas/visualizar_mapa.py <out>/mapa_vo.npz
    vo.Map.summary()

    if frame_view is not None:
        frame_view.close()
    if view3d is not None:
        view3d.final(vo)            # estado final (após o último BA), com todos os pontos
        print("Janelas 3D (Open3D) continuam abertas: feche-as para encerrar.")

    if enable_plot and (SAVE_PLOTS or SHOW_PLOTS):
        import matplotlib.pyplot as plt
        plt.ioff()
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        draw_snapshot(ax, plot_snapshot(vo, with_points=True))
        if SAVE_PLOTS:
            fig.savefig(os.path.join(out, "trajetoria_xz.png"), dpi=200, bbox_inches='tight')
            print(f"Figura salva em {out}/trajetoria_xz.png")
        vo.Map.plot_map_points(mode='3D', save=os.path.join(out, "mapa_3d.png") if SAVE_PLOTS else None, show=False)
        if SHOW_PLOTS:
            plt.show(block=True)
        plt.close('all')

    if view3d is not None:
        view3d.proc.join()          # espera o usuário fechar as janelas 3D
        view3d.close()

    vo.cleanup_plots()
    return vo


if __name__ == "__main__":
    run_dataset_test()