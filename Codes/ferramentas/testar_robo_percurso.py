"""
Teste LONGO do robô do SLAM-Teaching: voltas em volta da mesa do laboratório, nos dois sentidos, com a câmera do robô
gravando, como num conjunto de dados de verdade. Complementa testar_robo_slam_teaching.py (teste de bancada curto).

Percurso: retângulo (x 0,25-1,30 m, y 0-1,50 m no referencial da GT; ~5,1 m por volta) nos dois sentidos e depois uma
'gravata-borboleta' (diagonais cruzando no centro: revisita em direções diferentes), >= 37 cm dos obstáculos
(OBSTACULOS) e >= 40 cm das bordas da área vista. Controle de posição em malha fechada na GT, a V m/s.
SEGURANÇA: para se a GT ficar velha (> 0,5 s; aborta após 3 s), se o robô chegar a < 20 cm de um obstáculo ou < 15 cm da borda
da área vista. Ctrl+C para o robô. O robô para sempre no fim.

Mede:
  - erro da odometria x GT ao longo do percurso e no fim de cada volta (deriva em % da distância);
  - o mesmo erro com a odometria carimbada na CHEGADA ao PC (como o recording.csv faz) x no RELÓGIO DO ROBÔ
    (timestamp_us), para quantificar o efeito das rajadas do Wi-Fi;
  - atraso da GT (correlação da velocidade angular com a odometria no relógio do robô);
  - cobertura da GT, fps e perdas da câmera ao longo do tempo, buracos da telemetria.
Gera também a sessão convertida para o formato da dissertação, JÁ com os tempos corrigidos (odometria no relógio do
robô, GT deslocada pelo atraso estimado).

Uso:
    python3 testar_robo_percurso.py --voltas 3                  # 3 anti-horário + 3 horário, câmera gravando
    python3 testar_robo_percurso.py --voltas 1 --sem-camera     # só odometria x GT
    python3 testar_robo_percurso.py --sim --voltas 1            # simulador (valida a lógica)
Saída: resultados/robo_teste/percurso_<data_hora>/
"""
import argparse
import json
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from importar_slam_teaching import SLAM_TEACHING  # noqa: E402

RES = os.path.abspath(os.path.join(HERE, "..", "..", "resultados", "robo_teste"))
# obstáculos (x0, x1, y0, y1) no referencial da GT, medidos na imagem do teto (projeção conservadora no chão)
OBSTACULOS = {"mesa e gaveteiro": (1.84, 2.50, 0.14, 0.97), "pessoa (canto)": (1.67, 2.50, -0.60, 0.14),
              "pedestais": (1.10, 1.95, 1.93, 2.30), "placa": (2.19, 2.50, 1.38, 2.30)}
RETANGULO = [(0.25, 0.0), (1.30, 0.0), (1.30, 1.50), (0.25, 1.50)]       # anti-horário, ~5,1 m
GRAVATA = [(0.25, 0.0), (1.30, 1.50), (0.25, 1.50), (1.30, 0.0)]         # diagonais cruzando no centro
THETA_OFFSET = -0.079                          # rad, medido pelo testar_robo_slam_teaching.py em 2026-11-10
DIST_MESA_MIN, MARGEM_MIN, GT_MAX_IDADE = 0.20, 0.15, 0.5
TTL_MS = 400                                   # prazo de cada comando de roda (firmware com o patch)
WHEEL_BASE, V_RODA_MAX = 0.143, 0.25


def wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def dist_obst(x, y):
    """(distância ao obstáculo mais próximo, nome)."""
    return min((float(np.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1))), n)
               for n, (x0, x1, y0, y1) in OBSTACULOS.items())


class Abortar(Exception):
    pass


class Percurso:
    def __init__(self, a):
        self.a = a
        self.odom, self.rodas, self.gt_log, self.cam_log, self.voltas = [], [], [], [], []
        self.status_log = []
        self.out = os.path.join(RES, time.strftime("percurso_%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.out, exist_ok=True)
        self.linhas = []

    def log(self, txt):
        print(txt, flush=True)
        self.linhas.append(txt)

    # ------------------------------------------------------------------ conexão
    def conectar(self):
        sys.path.insert(0, self.a.repo)
        if self.a.sim:
            sys.path.insert(0, os.path.join(self.a.repo, "exemplos"))
            import simulador
            robo, gt = simulador.criar()
            robo._m.q[:] = (0.2, 0.25, np.radians(100))

            def amostrar():
                while True:
                    t = time.time()
                    self.odom.append((t, *robo.get_odometry(), t))
                    self.rodas.append((t, *robo.get_wheel_speeds()))
                    time.sleep(0.1)
            threading.Thread(target=amostrar, daemon=True).start()
        else:
            import robo
            import ground_truth as gt
            gt.THETA_OFFSET = THETA_OFFSET
            robo.connect(client_id=f"percurso-{os.getpid()}")
            if not robo._connected:
                raise Abortar("sem broker MQTT")
            original = robo._client.on_message

            def escuta(client, userdata, msg):
                t = time.time()
                try:
                    if msg.topic == robo.TOPIC_ODOM:
                        d = json.loads(msg.payload.decode())
                        x, y, th = (float(v) for v in d["odometria"].split(","))
                        self.odom.append((t, x, y, th, d["timestamp_us"] / 1e6))
                    elif msg.topic == "swarm/robot2/telemetry/reset":
                        self.log(f"  [último reset do robô: {msg.payload.decode(errors='replace')}"
                                 f"{' (retido)' if msg.retain else ''}]")
                    elif msg.topic == robo.TOPIC_STATUS and not msg.retain:
                        self.status_log.append((t, msg.payload.decode(errors="replace")))
                        print(f"  [status do robô: {msg.payload.decode(errors='replace')}]", flush=True)
                    elif msg.topic == robo.TOPIC_WHEELS:
                        d = json.loads(msg.payload.decode())
                        wl, wr = (float(v) for v in d["wheels"].split(","))
                        self.rodas.append((t, wl * robo.WHEEL_RADIUS, wr * robo.WHEEL_RADIUS))
                except Exception:                                          # noqa: BLE001
                    pass
                original(client, userdata, msg)
            robo._client.on_message = escuta
            robo._client.subscribe("swarm/robot2/telemetry/reset")    # motivo do reset (firmware com o patch)
        self.robo, self.gt = robo, gt
        robo.stop()

        def gt_cb(x, y, th):
            self.gt_log.append((time.time(), x, y, th))
            robo.log_ground_truth(x, y, th)
        if not gt.start(on_pose=gt_cb, timeout=20):
            raise Abortar("câmera do teto não abriu")
        t_lim = time.time() + 10
        while not self.gt_atual() and time.time() < t_lim:
            time.sleep(0.1)
        if not self.gt_atual():
            raise Abortar("GT não vê o robô")
        time.sleep(3.0)                                          # rajada do stream ao abrir
        import cv2
        self._area = gt.area_visivel().astype(np.float32).reshape(-1, 1, 2)
        self._cv2 = cv2

    def reinicio(self):
        """Reinício do firmware = odometria saltando para ~(0,0,0) longe de onde estava (o firmware só zera no boot).
        Status 'Offline' sozinho é queda do MQTT (o publish bloqueia com o Wi-Fi cheio), não reinício."""
        if len(self.odom) >= 2 and getattr(self, "t_ini", None):
            (_, x0, y0, *_), (_, x1, y1, th1, _) = self.odom[-2], self.odom[-1]
            if abs(x1) + abs(y1) + abs(th1) < 1e-3 and np.hypot(x0, y0) > 0.05:
                return True
        return False

    def gt_atual(self):
        ts = self.gt.get_timestamp()
        return ts is not None and time.time() - ts < GT_MAX_IDADE

    def margem(self, x, y):
        return float(self._cv2.pointPolygonTest(self._area, (float(x), float(y)), True))

    # ------------------------------------------------------------------ controle
    def ir_ate(self, ref, v_max, tol=0.08, timeout=40, margem_min=MARGEM_MIN):
        robo, gt = self.robo, self.gt
        t0, sem_gt = time.time(), None
        while time.time() - t0 < timeout:
            if self.reinicio():
                robo.send_wheel_speed(0, 0)
                raise Abortar("o ROBÔ REINICIOU (odometria voltou a zero / status Offline): queda de tensão? "
                              "bateria baixa? giro rápido com a câmera ligada?")
            if self.odom and time.time() - self.odom[-1][0] > 1.0:
                robo.send_wheel_speed(0, 0)                    # telemetria parada: os comandos também atrasam
                time.sleep(0.05)
                continue
            if not self.gt_atual():
                robo.send_wheel_speed(0, 0)
                sem_gt = sem_gt or time.time()
                if time.time() - sem_gt > 3.0:
                    raise Abortar("GT perdida por mais de 3 s")
                time.sleep(0.05)
                continue
            sem_gt = None
            x, y, th = gt.get_pose()
            do, no = dist_obst(x, y)
            if do < DIST_MESA_MIN:
                raise Abortar(f"perto de '{no}' ({do:.2f} m) em ({x:.2f}, {y:.2f})")
            if self.margem(x, y) < margem_min:
                raise Abortar(f"perto da borda ({self.margem(x, y):.2f} m) em ({x:.2f}, {y:.2f})")
            dx, dy = ref[0] - x, ref[1] - y
            d = float(np.hypot(dx, dy))
            if d < tol:
                return
            e = wrap(np.arctan2(dy, dx) - th)
            v = 0.0 if abs(e) > 0.5 else float(np.clip(0.8 * d * np.cos(e), 0.04, v_max))
            w = float(np.clip(1.6 * e, -self.a.w_max, self.a.w_max))
            vl, vr = v - w * WHEEL_BASE / 2, v + w * WHEEL_BASE / 2
            k = max(abs(vl), abs(vr)) / V_RODA_MAX
            if k > 1:
                vl, vr = vl / k, vr / k
            self.enviar(vl, vr)
            time.sleep(0.05)
        raise Abortar(f"não chegou a {ref} em {timeout} s")

    def enviar(self, vl, vr):
        """Comando das rodas com prazo (ttl_ms): com o firmware corrigido, o robô para sozinho se os comandos
        pararem de chegar (rede atrasada); o firmware antigo ignora o campo."""
        if self.a.sim:
            self.robo.send_wheel_speed(vl, vr)
            return
        R = self.robo.WHEEL_RADIUS
        self.robo._client.publish(self.robo.TOPIC_CMD_WHEELS,
                                  json.dumps({"l": vl / R, "r": vr / R, "ttl_ms": TTL_MS}))

    def amostrar_camera(self):
        while self._cam_rodando:
            s = self.robo.get_camera_stats()
            self.cam_log.append((time.time(), s["fps"], s["frames"], s["perdidos"], s["atraso"] or np.nan))
            time.sleep(1.0)

    # ------------------------------------------------------------------ execução
    def rodar(self):
        try:
            self.conectar()
            robo = self.robo
            x, y, _ = self.gt.get_pose()
            self.log(f"GT: ({x:.2f}, {y:.2f}); margem {self.margem(x, y):.2f} m; obstáculo mais perto: "
                     f"{dist_obst(x, y)[1]} a {dist_obst(x, y)[0]:.2f} m")
            inicio = min(RETANGULO, key=lambda p: np.hypot(p[0] - x, p[1] - y))
            self.log(f"indo ao canto de partida {inicio}")
            self.ir_ate(inicio, 0.10, margem_min=0.05)          # 1º trecho: pode partir perto da borda (vai p/ dentro)
            robo.send_wheel_speed(0, 0)
            time.sleep(1.0)
            if not self.a.sem_camera and not self.a.sim:
                robo.start_camera()
                time.sleep(2.0)
            self._cam_rodando = True
            threading.Thread(target=self.amostrar_camera, daemon=True).start()
            robo.set_odometry(*self.gt.get_pose())
            time.sleep(0.5)
            robo.record(True)
            self.t_ini = time.time()
            i0 = RETANGULO.index(inicio)
            j0 = GRAVATA.index(inicio)
            planos = [(nome, [RETANGULO[(i0 + sentido * k) % 4] for k in range(1, 5)], self.a.voltas)
                      for sentido, nome in ((1, "anti-horário"), (-1, "horário"))]
            planos.append(("gravata", [GRAVATA[(j0 + k) % 4] for k in range(1, 5)], self.a.gravatas))
            for nome, ordem, n in planos:
                for v in range(n):
                    t0 = time.time()
                    for p in ordem:
                        self.ir_ate(p, self.a.v)
                    self.voltas.append((nome, v + 1, t0, time.time()))
                    self.log(f"volta {v + 1} {nome}: {time.time() - t0:.0f} s")
            robo.send_wheel_speed(0, 0)
            time.sleep(2.0)
        except Abortar as e:
            self.log(f"INTERROMPIDO: {e}")
        except KeyboardInterrupt:
            self.log("Ctrl+C")
        finally:
            self._cam_rodando = False
            if getattr(self, "robo", None) is not None:
                self.robo.stop()
                self.robo.record(False)
                self.t_fim = time.time()
            if getattr(self, "gt", None) is not None:
                self.gt.stop()
            if getattr(self, "robo", None) is not None:
                self.robo.disconnect()
        if self.voltas:
            self.analisar()
        with open(os.path.join(self.out, "relatorio.txt"), "w") as f:
            f.write("\n".join(self.linhas) + "\n")

    # ------------------------------------------------------------------ análise
    def analisar(self):
        O = np.array(self.odom)                     # t_pc, x, y, th, t_robo
        G = np.array(self.gt_log)                   # t_pc, x, y, th
        sel = lambda A, t: A[(t >= self.t_ini - 1) & (t <= self.t_fim)]
        O, G = sel(O, O[:, 0]), sel(G, G[:, 0])
        zer = np.where((np.abs(O[:, 1]) + np.abs(O[:, 2]) + np.abs(O[:, 3]) < 1e-3)
                       & (np.r_[0, np.hypot(O[:-1, 1], O[:-1, 2])] > 0.05))[0]
        if len(zer):
            self.log(f"*** o robô REINICIOU {len(zer)} vez(es); análise só até o 1º reinício "
                     f"(t = {O[zer[0], 0] - self.t_ini:.0f} s) ***")
            t_corte = O[zer[0] - 1, 0]
            O, G = O[O[:, 0] <= t_corte], G[G[:, 0] <= t_corte]
            self.voltas = [v for v in self.voltas if v[3] <= t_corte] or self.voltas[:0]
            if not self.voltas:
                self.voltas = [("parcial", 1, self.t_ini, t_corte)]
        Go = G.copy()
        Go[:, 3] = np.unwrap(G[:, 3])
        th_o = np.unwrap(O[:, 3])
        dist_total = float(np.linalg.norm(np.diff(G[:, 1:3], axis=0), axis=1).sum())
        dur = self.t_fim - self.t_ini
        self.log(f"\n=== {len(self.voltas)} voltas, {dist_total:.1f} m (GT), {dur:.0f} s ===")

        # atraso da GT: correlação da velocidade angular (odometria no relógio do robô = sem rajadas)
        tt = np.arange(max(O[0, 4], G[0, 0]) + 1, min(O[-1, 4], G[-1, 0]) - 1, 0.01)
        wo = np.gradient(np.interp(tt, O[:, 4], th_o), tt)
        wg = np.gradient(np.interp(tt, G[:, 0], Go[:, 3]), tt)
        lags = np.arange(-100, 101)
        c = [np.dot(wo[max(0, -k):len(wo) - max(0, k)], wg[max(0, k):len(wg) - max(0, -k)]) for k in lags]
        lag = float(lags[int(np.argmax(c))] * 0.01)
        self.lag = lag
        self.log(f"atraso da GT (relógio do robô como referência): {1000 * lag:+.0f} ms")
        atr = O[:, 0] - O[:, 4]
        self.log(f"chegada da odometria - relógio do robô: mediana {1000 * np.median(atr):.0f} ms, "
                 f"p95 {1000 * np.percentile(atr, 95):.0f} ms, máx {1000 * atr.max():.0f} ms")
        d = np.diff(O[:, 0])
        self.log(f"buracos na chegada da odometria: máx {1000 * d.max():.0f} ms, {np.sum(d > 0.3)} acima de 300 ms "
                 f"(no relógio do robô: máx {1000 * np.diff(O[:, 4]).max():.0f} ms)")

        # erro da odometria nos instantes da GT (GT corrigida pelo atraso)
        tg = G[:, 0] - lag

        def erro(t_odo):
            ok = (tg >= t_odo[0]) & (tg <= t_odo[-1])
            P = np.c_[np.interp(tg[ok], t_odo, O[:, 1]), np.interp(tg[ok], t_odo, O[:, 2])]
            e = np.linalg.norm(P - G[ok, 1:3], axis=1)
            eth = np.degrees([wrap(a) for a in np.interp(tg[ok], t_odo, th_o) - Go[ok, 3]])
            return ok, e, eth
        ok_r, e_r, eth_r = erro(O[:, 4])
        ok_c, e_c, _ = erro(O[:, 0] - np.median(atr))       # chegada no PC (com o atraso médio descontado)
        rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
        self.log(f"erro da odometria x GT, relógio do robô:  RMS {100 * rms(e_r):.1f} cm, final {100 * e_r[-1]:.1f} cm "
                 f"({100 * e_r[-1] / dist_total:.2f}% de {dist_total:.1f} m), θ final {eth_r[-1]:+.1f}°")
        self.log(f"erro da odometria x GT, chegada no PC:    RMS {100 * rms(e_c):.1f} cm, máx {100 * e_c.max():.1f} cm "
                 f"(diferença = erro de SINCRONISMO causado pelas rajadas)")
        # por volta
        self.log("fim de cada volta (relógio do robô):")
        tgr = tg[ok_r]
        s0 = 0.0
        for nome, v, t0, t1 in self.voltas:
            k = int(np.argmin(np.abs(tgr - t1)))
            m = (G[:, 0] >= t0) & (G[:, 0] <= t1)
            L = float(np.linalg.norm(np.diff(G[m, 1:3], axis=0), axis=1).sum())
            s0 += L
            self.log(f"   {nome:13s} volta {v}: {t1 - t0:4.0f} s, {L:4.1f} m | erro acumulado {100 * e_r[k]:5.1f} cm "
                     f"({100 * e_r[k] / s0:.2f}% de {s0:.1f} m), θ {eth_r[k]:+5.1f}°")
        # GT
        dg = np.diff(G[:, 0])
        self.log(f"GT: {len(G) / dur:.1f} Hz; {np.sum(dg > 0.5)} buracos > 0,5 s (máx {1000 * dg.max():.0f} ms)")
        # câmera
        C = np.array(self.cam_log)
        if len(C) > 2 and C[-1, 2] > 0:
            dc = C[-1, 2] - C[0, 2]
            dp = C[-1, 3] - C[0, 3]
            self.log(f"câmera: {dc / (C[-1, 0] - C[0, 0]):.1f} fps médio (mín {np.nanmin(C[2:, 1]):.1f}, "
                     f"máx {np.nanmax(C[2:, 1]):.1f}); perdidos {dp:.0f} de {dc + dp:.0f} "
                     f"({100 * dp / max(dc + dp, 1):.0f}%); atraso mediano {1000 * np.nanmedian(C[:, 4]):.0f} ms")
        n_off = sum(1 for t, sx in self.status_log if sx == "Offline" and self.t_ini <= t <= self.t_fim)
        self.log(f"quedas do MQTT (status Offline): {n_off}")
        np.savez(os.path.join(self.out, "dados_brutos.npz"), odom=O, gt=G, cam=C, rodas=np.array(self.rodas),
                 voltas=np.array([(v, t0, t1) for _, v, t0, t1 in self.voltas]), lag=lag)
        with open(os.path.join(self.out, "odometria_relogio_robo.csv"), "w") as f:
            f.write("t_robo,t_pc,x,y,theta\n")
            for t, x, y, th, tr in O:
                f.write(f"{tr:.6f},{t:.6f},{x:.6f},{y:.6f},{th:.6f}\n")
        self.figura(O, G, th_o, Go, tg, ok_r, e_r, ok_c, e_c, C, dist_total)
        if not self.a.sim:
            self.converter(O, lag)

    def converter(self, O, lag):
        """Sessão no formato da dissertação, com a odometria no relógio do robô e a GT deslocada pelo atraso."""
        import importar_slam_teaching as ist
        dados = os.path.join(self.a.repo, "dados")
        s = ist.ler_sessoes(os.path.join(dados, "recording.csv"))[-1]
        if "camera" not in s:
            self.log("sessão sem câmera: não converto")
            return
        out = os.path.join(self.out, "sessao_convertida")
        n, r = ist.converter(s, ist.achar_pasta_camera(dados, s["camera"][0, 1]), out)
        th = np.unwrap(O[:, 3])
        with open(os.path.join(out, "data_sensor", "odometria_dataset.csv"), "w") as fh:
            fh.write("timestamp,x,y,theta,image\n")
            for (_, x, y, _, tr), t in zip(O, th):
                fh.write(f"{int(round(tr * 1e9))},{x:.6f},{y:.6f},{t:.6f},\n")
        f = os.path.join(out, "poses_gt_unix.csv")
        L = open(f).read().splitlines()
        with open(f, "w") as fh:
            fh.write(L[0] + "\n")
            for ln in L[1:]:
                p = ln.split(",")
                fh.write(f"{float(p[0]) - lag:.6f}," + ",".join(p[1:]) + "\n")
        with open(os.path.join(out, "LEIA-ME.txt"), "a") as fh:
            fh.write(f"\nTEMPOS CORRIGIDOS por testar_robo_percurso.py: odometria no relógio do robô (timestamp_us); "
                     f"GT deslocada em {-1000 * lag:+.0f} ms (atraso estimado).\n")
        self.log(f"sessão convertida: {n} imagens, {r['duracao_s']:.0f} s -> {out} (falta a calibração da câmera)")

    def figura(self, O, G, th_o, Go, tg, ok_r, e_r, ok_c, e_c, C, dist_total):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        AZ, LA, CI = "#1a73e8", "#e8710a", "0.55"
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))
        a = ax[0, 0]
        for k, (x0, x1, y0, y1) in enumerate(OBSTACULOS.values()):
            a.fill([x0, x1, x1, x0], [y0, y0, y1, y1], color="#c8a165", alpha=0.5, label="obstáculos" if k == 0 else None)
        ar = self.gt.area_visivel()
        a.plot(*np.vstack([ar, ar[:1]]).T, color=CI, ls="--", lw=1, label="área vista")
        a.plot(G[:, 1], G[:, 2], color=LA, lw=1.5, label="GT (teto)")
        a.plot(O[:, 1], O[:, 2], color=AZ, lw=1.2, label="odometria")
        a.set_aspect("equal")
        a.set_title(f"Trajetória: {len(self.voltas)} voltas, {dist_total:.1f} m")
        a.set_xlabel("x [m]")
        a.set_ylabel("y [m]")
        a.legend(fontsize=8)
        t0 = self.t_ini
        a = ax[0, 1]
        a.plot(tg[ok_r] - t0, 100 * e_r, color=AZ, label="odometria no relógio do robô")
        a.plot(tg[ok_c] - t0, 100 * e_c, color=LA, lw=0.8, alpha=0.8, label="odometria na chegada ao PC")
        for _, _, _, t1 in self.voltas:
            a.axvline(t1 - t0, color=CI, lw=0.6, ls=":")
        a.set_title("Erro de posição da odometria x GT (linhas: fim de cada volta)")
        a.set_xlabel("t [s]")
        a.set_ylabel("erro [cm]")
        a.legend(fontsize=8)
        a = ax[1, 0]
        a.plot(O[1:, 0] - t0, 1000 * np.diff(O[:, 0]), color=LA, lw=0.8, label="chegada no PC")
        a.plot(O[1:, 0] - t0, 1000 * np.diff(O[:, 4]), color=AZ, lw=0.8, label="relógio do robô")
        a.set_title("Intervalo entre mensagens de odometria")
        a.set_xlabel("t [s]")
        a.set_ylabel("ms")
        a.legend(fontsize=8)
        a = ax[1, 1]
        if len(C) > 2:
            a.plot(C[:, 0] - t0, C[:, 1], color=AZ, label="fps da câmera do robô")
            a.set_ylim(0, 12)
        a.set_title("Câmera do robô")
        a.set_xlabel("t [s]")
        a.set_ylabel("fps")
        for A in ax.flat:
            A.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out, "percurso.png"), dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=SLAM_TEACHING)
    ap.add_argument("--voltas", type=int, default=3, help="voltas em CADA sentido")
    ap.add_argument("--gravatas", type=int, default=2, help="voltas da gravata-borboleta (diagonais cruzadas)")
    ap.add_argument("--v", type=float, default=0.15, help="velocidade nas retas (m/s)")
    ap.add_argument("--w-max", type=float, default=0.6, help="velocidade angular máxima (rad/s) nos giros")
    ap.add_argument("--sem-camera", action="store_true", help="não liga o vídeo do robô")
    ap.add_argument("--sim", action="store_true")
    Percurso(ap.parse_args()).rodar()


if __name__ == "__main__":
    main()
