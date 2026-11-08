"""
Converte uma sessão gravada pela plataforma SLAM-Teaching (~/Documentos/SLAM/SLAM-Teaching/utilitarios: robô
ESP32-S3 + câmera do teto) para o formato que o rodar_sistema.py lê (o mesmo da Inomi).

Entrada (gravada com robo.record(True) ... robo.record(False) e a câmera do robô ligada, ex.:
    python3 exemplos/05_camera_robo.py --gravar      ou      python3 controle.py --gravar joystick):
    utilitarios/dados/recording.csv               sessões "Inicio" ... "Fim"; linhas odometria / ground_truth /
                                                  camera / velocidades, última coluna = time.time() do PC
    utilitarios/dados/camera_AAAAMMDD_HHMMSS/     <instante_da_captura_us>.jpg

Saída (--out):
    <out>/imagens/<t_ns>.jpg                      nome = instante da captura em ns (como o KAIST)
    <out>/data_sensor/odometria_dataset.csv       timestamp (ns), x, y, theta, image  -> PoseOdometry
    <out>/poses_gt_unix.csv                       t [s], R(3x3) e t(3) por linha       -> GT
    <out>/calibration/left.yaml                   se --calib for dado (OBRIGATÓRIO p/ resultados sérios)
    <out>/LEIA-ME.txt                             resumo da sessão (duração, percurso, fps, cobertura da GT)

Uso:
    python3 importar_slam_teaching.py --listar
    python3 importar_slam_teaching.py --sessao -1 --calib camera_robo.yaml --out ~/Documentos/TCC/slam_teaching/s01
    python3 rodar_sistema.py --images ~/Documentos/TCC/slam_teaching/s01/imagens --init-dist 0.5 --no-gui

Tempo: odometria e GT levam o time.time() do PC na CHEGADA; a imagem leva o instante da CAPTURA no robô (SNTP).
O atraso de rede entra como erro de sincronismo (a coluna 'atraso' do 05_camera_robo.py mostra a ordem de
grandeza); dá para compensar com --atraso-roda no rodar_sistema.py.
"""
import argparse
import glob
import os
import shutil

import numpy as np

SLAM_TEACHING = os.path.expanduser("~/Documentos/SLAM/SLAM-Teaching/utilitarios")


def ler_sessoes(path):
    """Lista de sessões; cada uma é um dict tipo -> array (colunas numéricas, a última é o tempo do PC)."""
    sessoes, atual = [], None
    for linha in open(path):
        p = linha.strip().split(",")
        if p[0] == "Inicio":
            atual = {}
        elif p[0] == "Fim":
            if atual is not None:
                sessoes.append({k: np.array(v, float) for k, v in atual.items()})
            atual = None
        elif atual is not None and len(p) > 1:
            try:
                atual.setdefault(p[0], []).append([float(v) for v in p[1:]])
            except ValueError:
                continue
    return sessoes


def resumo(s):
    def dur(k):
        return s[k][-1, -1] - s[k][0, -1] if k in s and len(s[k]) > 1 else 0.0
    odo = s.get("odometria")
    perc = float(np.linalg.norm(np.diff(odo[:, :2], axis=0), axis=1).sum()) if odo is not None and len(odo) > 1 else 0
    nc = len(s.get("camera", []))
    return dict(odometria=len(odo) if odo is not None else 0, gt=len(s.get("ground_truth", [])), camera=nc,
                duracao_s=max(dur("odometria"), dur("camera")), percurso_m=perc,
                fps=nc / dur("camera") if dur("camera") > 0 else 0.0)


def achar_pasta_camera(dados, t_captura):
    """A pasta camera_* que contém o 1º frame da sessão (nome = instante da captura em µs)."""
    alvo = f"{int(round(t_captura * 1e6))}.jpg"
    for d in sorted(glob.glob(os.path.join(dados, "camera_*"))):
        if os.path.isfile(os.path.join(d, alvo)):
            return d
    # tolerância de arredondamento: procura o arquivo mais próximo
    melhor, dmin = None, np.inf
    for d in sorted(glob.glob(os.path.join(dados, "camera_*"))):
        ts = [int(os.path.splitext(os.path.basename(f))[0]) for f in glob.glob(os.path.join(d, "*.jpg"))]
        if ts:
            dd = np.min(np.abs(np.array(ts) - t_captura * 1e6))
            if dd < dmin:
                melhor, dmin = d, dd
    return melhor if dmin < 1e5 else None                              # < 0,1 s


def pose_mat(x, y, th):
    c, s = np.cos(th), np.sin(th)
    return [c, -s, 0.0, x, s, c, 0.0, y, 0.0, 0.0, 1.0, 0.0]


def converter(s, pasta_cam, out, calib=None):
    os.makedirs(os.path.join(out, "imagens"), exist_ok=True)
    os.makedirs(os.path.join(out, "data_sensor"), exist_ok=True)
    jpgs = sorted(glob.glob(os.path.join(pasta_cam, "*.jpg")))
    t_ini = s["odometria"][0, -1] - 1.0 if "odometria" in s else -np.inf
    t_fim = s["odometria"][-1, -1] + 1.0 if "odometria" in s else np.inf
    n = 0
    for f in jpgs:
        t_us = int(os.path.splitext(os.path.basename(f))[0])
        if not t_ini <= t_us * 1e-6 <= t_fim:
            continue
        dst = os.path.join(out, "imagens", f"{t_us * 1000:d}.jpg")
        if not os.path.exists(dst):
            shutil.copy2(f, dst)
        n += 1
    odo = s["odometria"].copy()
    odo[:, 2] = np.unwrap(odo[:, 2])         # θ vem em [-π, π]; o PoseOdometry interpola θ linearmente
    with open(os.path.join(out, "data_sensor", "odometria_dataset.csv"), "w") as fh:
        fh.write("timestamp,x,y,theta,image\n")
        for x, y, th, t in odo[:, :4]:
            fh.write(f"{int(round(t * 1e9))},{x:.6f},{y:.6f},{th:.6f},\n")
    gt = s.get("ground_truth")
    if gt is not None and len(gt):
        with open(os.path.join(out, "poses_gt_unix.csv"), "w") as fh:
            fh.write("timestamp,R00,R01,R02,tx,R10,R11,R12,ty,R20,R21,R22,tz\n")
            for x, y, th, t in gt[:, :4]:
                fh.write(f"{t:.6f}," + ",".join(f"{v:.6f}" for v in pose_mat(x, y, th)) + "\n")
    if calib:
        os.makedirs(os.path.join(out, "calibration"), exist_ok=True)
        shutil.copy2(calib, os.path.join(out, "calibration", "left.yaml"))
    r = resumo(s)
    with open(os.path.join(out, "LEIA-ME.txt"), "w") as fh:
        fh.write(f"Sessão do SLAM-Teaching convertida por importar_slam_teaching.py\n"
                 f"frames da câmera: {n} (pasta {pasta_cam})\nodometria: {r['odometria']} poses\n"
                 f"ground truth: {r['gt']} poses\nduração: {r['duracao_s']:.1f} s\npercurso (odometria): "
                 f"{r['percurso_m']:.2f} m\nfps da câmera: {r['fps']:.1f}\n"
                 f"calibração: {'calibration/left.yaml' if calib else 'NENHUMA (o VO usará o perfil padrão: errado)'}\n")
    return n, r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", default=os.path.join(SLAM_TEACHING, "dados"), help="pasta dados/ do SLAM-Teaching")
    ap.add_argument("--listar", action="store_true", help="lista as sessões do recording.csv")
    ap.add_argument("--sessao", type=int, default=-1, help="índice da sessão (padrão: a última)")
    ap.add_argument("--calib", default=None, help="calibração da câmera do robô (YAML do OpenCV: camera_matrix, "
                                                  "distortion_coefficients)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sessoes = ler_sessoes(os.path.join(a.dados, "recording.csv"))
    if a.listar or a.out is None:
        for i, s in enumerate(sessoes):
            r = resumo(s)
            print(f"[{i:3d}] {r['duracao_s']:7.1f} s  {r['percurso_m']:6.2f} m  odometria {r['odometria']:5d}  "
                  f"GT {r['gt']:5d}  câmera {r['camera']:5d} ({r['fps']:.1f} fps)")
        if a.out is None:
            return
    s = sessoes[a.sessao]
    if "camera" not in s:
        raise SystemExit("sessão sem frames da câmera do robô (grave com a câmera ligada: robo.start_camera())")
    if "odometria" not in s:
        raise SystemExit("sessão sem odometria")
    pasta = achar_pasta_camera(a.dados, s["camera"][0, 1])
    if pasta is None:
        raise SystemExit("não achei a pasta camera_* com os frames desta sessão")
    n, r = converter(s, pasta, os.path.expanduser(a.out), a.calib)
    print(f"{n} imagens, {r['odometria']} poses de odometria, {r['gt']} de GT -> {a.out}")
    if not a.calib:
        print("AVISO: sem --calib. Calibre a câmera do robô (tabuleiro de xadrez) antes de usar os resultados.")
    if not r["gt"]:
        print("AVISO: sessão sem ground truth (câmera do teto desligada?) -> sem métricas de erro.")


if __name__ == "__main__":
    main()
