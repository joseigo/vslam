"""
Teste de bancada do robô do SLAM-Teaching antes de gravar conjuntos de dados para a dissertação.

Usa as bibliotecas do próprio SLAM-Teaching (utilitarios/robo.py e ground_truth.py) e confere, em etapas:

  1. comunicação e odometria  broker MQTT, status do robô, taxa e regularidade da odometria e das rodas,
                              relógio do robô (timestamp_us) x relógio do PC (SNTP), odometria parada não deriva
  2. câmera do robô           fps, frames perdidos, atraso captura->PC, resolução, imagem não escura/borrada
  3. ground truth (teto)      taxa, detecção, ruído com o robô parado (mm, graus), margem até a borda da área vista
  4. acionamento (--mover)    cada roda sozinha (sentido e lado certos), reta para frente e de ré, giro de 90° e
                              volta; compara rodas medidas x comandadas, odometria x GT (escala do encoder,
                              distância entre rodas efetiva), THETA_OFFSET do marcador e o atraso GT x odometria
  5. gravação + importação    (--gravar) grava a sessão do passo 4 com a câmera e converte com
                              importar_slam_teaching.py, como será feito com os conjuntos de verdade

SEGURANÇA no passo 4: só anda com a GT atual (< 0,5 s) e a pelo menos MARGEM_INICIO da borda da área vista; durante
o movimento para na hora se a GT sumir ou a margem cair abaixo de MARGEM_MIN. Movimentos curtos (<= 25 cm, 90°),
a 0,06-0,10 m/s. Ctrl+C para o robô. O robô para sempre no fim (finally).

Uso:
    python3 testar_robo_slam_teaching.py                       # passos 1-3 (o robô NÃO se move)
    python3 testar_robo_slam_teaching.py --mover               # + passo 4 (robô no chão, no centro da área)
    python3 testar_robo_slam_teaching.py --mover --gravar      # + passo 5
    python3 testar_robo_slam_teaching.py --sem-gt              # sem a câmera do teto (só 1-2)

Saída: resultados/robo_teste/<data_hora>/ com relatorio.txt, relatorio.json, teste_robo.png e quadro_camera.jpg.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from importar_slam_teaching import SLAM_TEACHING  # noqa: E402

RES = os.path.abspath(os.path.join(HERE, "..", "..", "resultados", "robo_teste"))
MARGEM_INICIO = 0.35       # m até a borda da área vista para começar a andar
MARGEM_MIN = 0.15          # m: abaixo disso, para
GT_MAX_IDADE = 0.5         # s
ASSENTAR = 2.0             # s parado após cada movimento: com o vídeo ligado a telemetria chega em rajadas de até ~2 s
V_TESTE = 0.10             # m/s (reta)
V_RODA = 0.06              # m/s (uma roda só)
W_TESTE = 1.0              # rad/s (giro)

OK, AVISO, FALHA, INFO = "OK", "AVISO", "FALHA", "info"


class Abortar(Exception):
    pass


class Teste:
    def __init__(self, args):
        self.a = args
        self.itens = []                          # (etapa, item, estado, valor, detalhe)
        self.msgs = {"odom": [], "rodas": [], "status": []}
        self.gt_log = []                         # (t_pc, x, y, th)
        self.cmd_log = []                        # (t_pc, vl, vr)
        self.segmentos = []                      # (nome, t0, t1, vl, vr)
        self.out = os.path.join(RES, time.strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.out, exist_ok=True)

    # ------------------------------------------------------------------ relatório
    def reg(self, etapa, item, estado, valor="", detalhe=""):
        self.itens.append(dict(etapa=etapa, item=item, estado=estado, valor=valor, detalhe=detalhe))
        marca = {OK: "  ok ", AVISO: "AVISO", FALHA: "FALHA", INFO: "     "}[estado]
        print(f"  [{marca}] {item}: {valor}" + (f"  ({detalhe})" if detalhe else ""), flush=True)

    # ------------------------------------------------------------------ captura das mensagens
    def _instalar_escuta(self):
        robo = self.robo
        original = robo._client.on_message

        def escuta(client, userdata, msg):
            t = time.time()
            try:
                if msg.topic == robo.TOPIC_ODOM:
                    d = json.loads(msg.payload.decode())
                    x, y, th = (float(v) for v in d["odometria"].split(","))
                    self.msgs["odom"].append((t, x, y, th, d.get("timestamp_us"), d.get("seq"), d.get("ticks")))
                elif msg.topic == robo.TOPIC_WHEELS:
                    d = json.loads(msg.payload.decode())
                    wl, wr = (float(v) for v in d["wheels"].split(","))
                    self.msgs["rodas"].append((t, wl * robo.WHEEL_RADIUS, wr * robo.WHEEL_RADIUS))
                elif msg.topic == robo.TOPIC_STATUS:
                    self.msgs["status"].append((t, msg.payload.decode(errors="replace")))
            except Exception as e:                                       # noqa: BLE001
                print(f"  (mensagem mal formada em {msg.topic}: {e})")
            original(client, userdata, msg)
        robo._client.on_message = escuta

    def _criar_sim(self):
        """Simulador do SLAM-Teaching (exemplos/simulador.py), robô perto da borda e marcador girado 40°, p/ validar a
        lógica do passo 4 sem hardware. As mensagens são amostradas a 10 Hz (não há MQTT)."""
        import threading
        sys.path.insert(0, os.path.join(self.a.repo, "exemplos"))
        import simulador
        robo, gt = simulador.criar()
        robo._m.q[:] = (0.05, 0.0, np.radians(200))           # perto da borda, apontando para fora
        off = np.radians(40)
        g_pose, m = gt.get_pose, robo._m

        class GTGirado:                                         # marcador colado 40° torto
            THETA_OFFSET = 0.0
            def start(s, on_pose=None, **k):
                m.on_pose = (lambda x, y, t: on_pose(x, y, float(np.arctan2(np.sin(t + off), np.cos(t + off))))) \
                    if on_pose else None
                return True
            def get_pose(s):
                x, y, t = g_pose()
                return x, y, float(np.arctan2(np.sin(t + off), np.cos(t + off)))
            def __getattr__(s, k):
                return getattr(gt, k)
        self._gt_sim = GTGirado()
        robo.TOPIC_ODOM = None

        def amostrar():
            while True:
                t = time.time()
                self.msgs["odom"].append((t, *robo.get_odometry(), int(t * 1e6)))
                self.msgs["rodas"].append((t, *robo.get_wheel_speeds()))
                time.sleep(0.1)
        threading.Thread(target=amostrar, daemon=True).start()
        return robo

    def _gt_cb(self, x, y, th):
        self.gt_log.append((time.time(), x, y, th))
        if self.robo is not None:
            self.robo.log_ground_truth(x, y, th)             # só escreve se estiver gravando

    # ------------------------------------------------------------------ 1. comunicação e odometria
    def etapa_comunicacao(self):
        print("\n1. Comunicação e odometria")
        sys.path.insert(0, self.a.repo)
        if self.a.sim:
            robo = self._criar_sim()
        else:
            import robo
        self.robo = robo
        robo.connect(client_id=f"teste-dissert-{os.getpid()}")
        if not getattr(robo, "_connected", True):
            self.reg("comunicação", "broker MQTT", FALHA, "sem conexão", "robô/broker ligados? mesma rede 192.168.0.x?")
            raise Abortar("sem broker")
        self.reg("comunicação", "broker MQTT", OK, "conectado")
        if not self.a.sim:
            self._instalar_escuta()
        robo.stop()                    # 1ª mensagem põe o firmware em RUNNING (e ele zera a odometria)
        t0 = time.time()
        time.sleep(self.a.janela)
        T = time.time() - t0
        st = robo.get_status()
        self.reg("comunicação", "status do robô", OK if st == "Online" else AVISO, st or "nenhum recebido",
                 "" if st == "Online" else "o status pode não ser retido no broker; vale se a odometria chegar")
        od = [m for m in self.msgs["odom"] if m[0] >= t0]
        rd = [m for m in self.msgs["rodas"] if m[0] >= t0]
        if not od:
            self.reg("comunicação", "odometria", FALHA, "nenhuma mensagem", "robô ligado e no Wi-Fi?")
            raise Abortar("sem odometria")
        f_od, f_rd = len(od) / T, len(rd) / T
        self.reg("odometria", "taxa da odometria", OK if f_od >= 8 else AVISO, f"{f_od:.1f} Hz", "esperado ~10 Hz")
        self.reg("odometria", "taxa das rodas", OK if f_rd >= 8 else AVISO, f"{f_rd:.1f} Hz", "esperado ~10 Hz")
        dt = np.diff([m[0] for m in od])
        if len(dt) > 2:
            self.reg("odometria", "maior intervalo entre mensagens (chegada no PC)",
                     OK if dt.max() < 0.3 else AVISO, f"{1000 * dt.max():.0f} ms",
                     f"mediana {1000 * np.median(dt):.0f} ms; buracos grandes = Wi-Fi")
        sq = np.array([m[5] for m in od if len(m) > 5 and m[5] is not None], float)
        if len(sq) > 2:                        # firmware com nº de sequência: perdas exatas
            ds = np.diff(sq)
            perd = int(np.sum(ds[ds > 1] - 1))
            self.reg("odometria", "mensagens perdidas (pelo seq)", OK if perd == 0 else AVISO,
                     f"{perd} de {int(sq[-1] - sq[0] + 1)}", "firmware novo")
            self.reg("odometria", "pulsos brutos dos encoders", OK if all(len(m) > 6 and m[6] for m in od) else AVISO,
                     "presentes" if all(len(m) > 6 and m[6] for m in od) else "ausentes")
        ts = np.array([m[4] for m in od if m[4] is not None], float)
        if len(ts) < 2:
            self.reg("tempo", "timestamp_us da odometria", AVISO, "ausente", "sem o relógio do robô")
        else:
            mono = np.all(np.diff(ts) > 0)
            self.reg("tempo", "timestamp_us crescente", OK if mono else FALHA, "sim" if mono else "NÃO")
            if ts[-1] < 1e15:                  # < 2001 em µs -> tempo desde o boot, não Unix
                self.reg("tempo", "relógio do robô", FALHA, f"{ts[-1] / 1e6:.0f} s (tempo desde o boot?)",
                         "SNTP não sincronizou: imagem e odometria ficariam em relógios diferentes")
            else:
                off = np.array([m[0] - m[4] / 1e6 for m in od if m[4] is not None])
                med = float(np.median(off))
                self.reg("tempo", "chegada no PC - relógio do robô (odometria)",
                         OK if 0 <= med < 0.3 else (AVISO if abs(med) < 1.0 else FALHA),
                         f"{1000 * med:.0f} ms (variação {1000 * np.ptp(off):.0f} ms)",
                         "atraso de rede + diferença de relógio; negativo ou > 1 s = relógios fora de sincronia "
                         "(rode 'timedatectl' no PC)")
                self.reg("tempo", "período no relógio do robô", INFO,
                         f"{np.median(np.diff(ts)) / 1e3:.0f} ms (máx {np.diff(ts).max() / 1e3:.0f} ms)")
        P = np.array([m[1:4] for m in od])
        mov = float(np.max(np.linalg.norm(P[:, :2] - P[0, :2], axis=1)))
        self.reg("odometria", "odometria com o robô parado", OK if mov < 0.005 else AVISO, f"variou {1000 * mov:.1f} mm",
                 "parado deveria ser 0")
        R = np.array([m[1:] for m in rd]) if rd else np.zeros((1, 2))
        self.reg("odometria", "rodas paradas", OK if np.abs(R).max() < 0.01 else AVISO,
                 f"máx {1000 * np.abs(R).max():.1f} mm/s")

    # ------------------------------------------------------------------ 2. câmera do robô
    def etapa_camera(self):
        print("\n2. Câmera do robô")
        robo = self.robo
        if not robo.start_camera():
            self.reg("câmera", "porta UDP 5001", FALHA, "não abriu", "outro programa recebendo o vídeo?")
            return
        atrasos, t0 = [], time.time()
        while time.time() - t0 < self.a.janela + 2:
            s = robo.get_camera_stats()
            if s["atraso"] is not None:
                atrasos.append(s["atraso"])
            time.sleep(0.1)
        s = robo.get_camera_stats()
        f = robo.get_camera_frame()
        if f is None:
            self.reg("câmera", "vídeo", FALHA, "nenhum frame", "firewall: sudo ufw allow 5001/udp")
            return
        import cv2
        cv2.imwrite(os.path.join(self.out, "quadro_camera.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        self.reg("câmera", "fps", OK if s["fps"] >= 8 else AVISO, f"{s['fps']:.1f}", "esperado ~10")
        tot = s["frames"] + s["perdidos"]
        pp = 100 * s["perdidos"] / tot if tot else 0
        self.reg("câmera", "frames perdidos", OK if pp < 5 else AVISO, f"{s['perdidos']} de {tot} ({pp:.1f}%)",
                 "o VO aguenta perdas isoladas; rajadas atrapalham")
        a = np.array(atrasos)
        if len(a):
            med = float(np.median(a))
            self.reg("tempo", "atraso captura -> PC (câmera)",
                     OK if 0 <= med < 0.3 else (AVISO if abs(med) < 1 else FALHA),
                     f"{1000 * med:.0f} ms [{1000 * a.min():.0f}-{1000 * a.max():.0f}]",
                     "negativo = relógios do robô e do PC fora de sincronia")
        h, w = f.shape[:2]
        self.reg("câmera", "resolução", OK if (w, h) == (640, 480) else AVISO, f"{w}x{h}")
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        brilho, nitidez = float(g.mean()), float(cv2.Laplacian(g, cv2.CV_64F).var())
        self.reg("câmera", "brilho médio", OK if 40 < brilho < 220 else AVISO, f"{brilho:.0f} de 255")
        self.reg("câmera", "nitidez (variância do Laplaciano)", OK if nitidez > 50 else AVISO, f"{nitidez:.0f}",
                 "baixa = lente suja/desfocada ou cena sem textura")
        n_orb = len(cv2.ORB_create(2000).detect(g, None))
        self.reg("câmera", "pontos ORB no quadro", OK if n_orb > 300 else AVISO, n_orb,
                 "o VO precisa de textura; poucos pontos = cena lisa à frente do robô")

    # ------------------------------------------------------------------ 3. ground truth
    def etapa_gt(self):
        print("\n3. Ground truth (câmera do teto)")
        if self.a.sim:
            gt = self._gt_sim
        else:
            import ground_truth as gt
        self.gt = gt
        if not gt.start(on_pose=self._gt_cb, timeout=20):
            self.reg("GT", "stream da câmera do teto", FALHA, "não abriu", "Raspberry ligada? config/env.txt?")
            self.gt = None
            return
        self.reg("GT", "stream da câmera do teto", OK, "conectado")
        t_lim = time.time() + 10
        while gt.get_timestamp() is None and time.time() < t_lim:
            time.sleep(0.1)
        if gt.get_timestamp() is None:
            self.reg("GT", "marcador do robô", FALHA, "não visto em 10 s",
                     f"robô dentro da área? ArUco ID {gt.ROBOT_ID} para cima?")
            return
        time.sleep(3.0)          # nos 1os ~2 s o stream esvazia quadros acumulados em rajada (~75 Hz): descarta
        t0 = time.time()
        time.sleep(self.a.janela)
        L = np.array([g for g in self.gt_log if g[0] >= t0])
        if len(L) < 5 * self.a.janela:                         # stream travou (Wi-Fi): religa uma vez
            print("  GT parou de chegar; religando o stream da câmera do teto")
            gt.stop()
            time.sleep(1)
            gt.start(on_pose=self._gt_cb, timeout=20)
            time.sleep(4.0)
            t0 = time.time()
            time.sleep(self.a.janela)
            L = np.array([g for g in self.gt_log if g[0] >= t0])
        f = len(L) / self.a.janela
        self.gt_ok = f >= 5
        self.reg("GT", "taxa de poses", OK if f >= 15 else (AVISO if f >= 5 else FALHA), f"{f:.1f} Hz", "câmera a 20 fps; menos = detecção falhando")
        if len(L) > 3:
            sx, sy = 1000 * L[:, 1].std(), 1000 * L[:, 2].std()
            sth = np.degrees(np.std(np.unwrap(L[:, 3])))
            self.reg("GT", "ruído parado (desvio padrão)", OK if max(sx, sy) < 3 and sth < 1 else AVISO,
                     f"x {sx:.1f} mm, y {sy:.1f} mm, θ {sth:.2f}°")
        self.reg("GT", "pose", INFO, "x={:.3f} y={:.3f} θ={:.1f}°".format(*gt.get_pose()[:2],
                                                                           np.degrees(gt.get_pose()[2])))
        self.reg("GT", "THETA_OFFSET configurado", INFO, f"{gt.THETA_OFFSET:.3f} rad",
                 "0 = provavelmente nunca calibrado; o passo 4 mede")
        m = self.margem()
        self.reg("GT", "distância até a borda da área vista", OK if m >= MARGEM_INICIO else AVISO, f"{m:.2f} m",
                 f"para o passo 4 precisa de >= {MARGEM_INICIO} m: coloque o robô perto do centro")

    def margem(self, p=None):
        import cv2
        if not hasattr(self, "_area"):
            self._area = self.gt.area_visivel().astype(np.float32)
        x, y = (p if p is not None else self.gt.get_pose()[:2])
        return float(cv2.pointPolygonTest(self._area.reshape(-1, 1, 2), (float(x), float(y)), True))

    def gt_atual(self):
        ts = self.gt.get_timestamp()
        return ts is not None and time.time() - ts < GT_MAX_IDADE

    # ------------------------------------------------------------------ 4. acionamento
    def mover(self, nome, vl, vr, dur):
        """Manda (vl, vr) por dur s a 20 Hz, parando se a GT sumir ou o robô chegar perto da borda."""
        robo = self.robo
        t0 = time.time()
        try:
            while time.time() - t0 < dur:
                if not self.gt_atual():
                    raise Abortar(f"{nome}: GT perdida (robô saiu da imagem ou o marcador sumiu)")
                if self.margem() < MARGEM_MIN:
                    raise Abortar(f"{nome}: perto da borda da área ({self.margem():.2f} m)")
                robo.send_wheel_speed(vl, vr)
                self.cmd_log.append((time.time(), vl, vr))
                time.sleep(0.05)
        finally:
            robo.send_wheel_speed(0, 0)
            self.cmd_log.append((time.time(), 0.0, 0.0))
        t1 = time.time()
        time.sleep(ASSENTAR)                 # assentar (a odometria e a GT alcançam o fim do movimento)
        self.segmentos.append((nome, t0, t1, vl, vr))
        return t0, t1

    def _pose_em(self, fonte, t):
        """Pose (x, y, θ desembrulhado) interpolada no instante t do PC."""
        A = np.array([m[:4] for m in self.msgs["odom"]]) if fonte == "odom" else np.array(self.gt_log)
        th = np.unwrap(A[:, 3])
        return np.array([np.interp(t, A[:, 0], A[:, k]) for k in (1, 2)] + [np.interp(t, A[:, 0], th)])

    def _delta(self, fonte, t0, t1):
        a, b = self._pose_em(fonte, t0 - 0.2), self._pose_em(fonte, t1 + ASSENTAR - 0.1)
        return b - a, a

    def _rodas(self, t0, t1):
        """Velocidade de cada roda no movimento: mediana das amostras não nulas que CHEGARAM entre o início e o fim do
        assentamento (a mensagem das rodas não tem carimbo do robô; com rajadas, a chegada não marca a medida).
        Roda sem amostra não nula -> 0."""
        R = np.array([m for m in self.msgs["rodas"] if t0 <= m[0] <= t1 + ASSENTAR - 0.1])
        if not len(R):
            return np.nan, np.nan
        med = lambda v: float(np.median(v[np.abs(v) > 0.005])) if np.any(np.abs(v) > 0.005) else 0.0
        return med(R[:, 1]), med(R[:, 2])

    def centralizar(self):
        """Afasta o robô da borda, em direção ao centro da área vista, só até MARGEM_INICIO (no máximo 40 cm: o centro
        pode ter obstáculo, como a mesa do laboratório). Não depende do THETA_OFFSET: um empurrão curto revela a
        direção real do robô; o giro usa só a VARIAÇÃO do θ da GT (que não depende do offset)."""
        robo, gt = self.robo, self.gt
        c = gt.area_visivel().mean(axis=0)
        print(f"  afastando da borda: margem {self.margem():.2f} m, em direção a ({c[0]:.2f}, {c[1]:.2f})")
        p0, th0 = np.array(gt.get_pose()[:2]), gt.get_pose()[2]
        for s in (1, -1):                                      # empurrão: frente; se não andou, ré
            t0 = time.time()
            while time.time() - t0 < 1.0:
                if not self.gt_atual() or self.margem() < 0.03:
                    break
                robo.send_wheel_speed(s * 0.05, s * 0.05)
                time.sleep(0.05)
            robo.send_wheel_speed(0, 0)
            time.sleep(0.8)
            d = (np.array(gt.get_pose()[:2]) - p0) * s
            if np.linalg.norm(d) > 0.02:
                break
        else:
            raise Abortar("centralizar: o robô não se moveu no empurrão (rodas? GT?)")
        rumo0 = np.arctan2(d[1], d[0])                                   # rumo real (empurrão reto) ~ no instante th0
        self.offset_empurrao = float(np.arctan2(np.sin(rumo0 - th0), np.cos(rumo0 - th0)))
        wrap = lambda a: float(np.arctan2(np.sin(a), np.cos(a)))
        rumo = lambda: rumo0 + wrap(gt.get_pose()[2] - th0)
        t_lim = time.time() + 40
        while time.time() < t_lim:
            if not self.gt_atual():
                raise Abortar("centralizar: GT perdida")
            p = np.array(gt.get_pose()[:2])
            dist = np.linalg.norm(c - p)
            if self.margem() >= MARGEM_INICIO + 0.03 or np.linalg.norm(p - p0) > 0.40 or dist < 0.05:
                break
            err = wrap(np.arctan2(c[1] - p[1], c[0] - p[0]) - rumo())
            w = float(np.clip(2.0 * err, -1.0, 1.0))
            v = 0.0 if abs(err) > 0.3 else float(np.clip(0.8 * dist, 0.03, 0.10))
            robo.send_wheel_speed(v - w * 0.143 / 2, v + w * 0.143 / 2)
            time.sleep(0.05)
        robo.send_wheel_speed(0, 0)
        time.sleep(0.8)
        self.reg("acionamento", "afastar da borda (GT)", OK if self.margem() >= MARGEM_INICIO else FALHA,
                 f"margem {self.margem():.2f} m",
                 f"empurrão: θ do marcador desviado {np.degrees(self.offset_empurrao):+.0f}° da frente do robô")

    def etapa_acionamento(self):
        print("\n4. Acionamento (o robô vai se mover)")
        if self.gt is None or not self.gt_atual() or not getattr(self, "gt_ok", False):
            self.reg("acionamento", "pré-condição", FALHA, "sem GT atual", "o teste de movimento exige a câmera do teto")
            return
        if self.margem() < MARGEM_INICIO and self.margem() > 0.03:
            self.centralizar()
        m = self.margem()
        if m < MARGEM_INICIO:
            self.reg("acionamento", "pré-condição", FALHA, f"margem {m:.2f} m < {MARGEM_INICIO} m",
                     "coloque o robô mais perto do centro da área vista e rode de novo")
            return
        robo = self.robo
        if not self.a.gravar and not self.a.sim:
            robo.stop_camera()                          # o vídeo satura o Wi-Fi do robô e atrasa a telemetria
            time.sleep(1.0)
        robo.set_odometry(*self.gt.get_pose())          # odometria no referencial da GT
        time.sleep(0.5)
        d = np.linalg.norm(np.array(robo.get_odometry()[:2]) - np.array(self.gt.get_pose()[:2]))
        self.reg("acionamento", "set_odometry na pose da GT", OK if d < 0.01 else FALHA, f"diferença {1000 * d:.0f} mm")
        if self.a.gravar:
            robo.record(True)
            self.t_gravacao = time.time()
            print("  (gravando em dados/recording.csv, com a câmera)")

        # 4a. cada roda sozinha: lado e sentido
        for lado, vl, vr, sinal in (("esquerda", V_RODA, 0.0, -1), ("direita", 0.0, V_RODA, +1)):
            t0, t1 = self.mover(f"roda {lado} p/ frente", vl, vr, 1.0)
            ml, mr = self._rodas(t0, t1)
            med, parada = (ml, mr) if lado == "esquerda" else (mr, ml)
            certo = med > 0.5 * V_RODA and abs(parada) < 0.3 * V_RODA
            self.reg("acionamento", f"roda {lado} sozinha (medida)", OK if certo else FALHA,
                     f"esq {1000 * ml:.0f} / dir {1000 * mr:.0f} mm/s (comando {1000 * V_RODA:.0f})",
                     "" if certo else "roda trocada, invertida ou sem encoder")
            dg, _ = self._delta("gt", t0, t1)
            do, _ = self._delta("odom", t0, t1)
            ok_g = np.sign(dg[2]) == sinal and abs(dg[2]) > np.radians(5)
            ok_o = np.sign(do[2]) == sinal
            self.reg("acionamento", f"roda {lado}: giro real (GT) e da odometria",
                     OK if ok_g and ok_o else FALHA, f"GT {np.degrees(dg[2]):+.1f}°, odometria {np.degrees(do[2]):+.1f}°",
                     f"esperado {'horário (−)' if sinal < 0 else 'anti-horário (+)'}")
            self.mover(f"roda {lado} volta", -vl, -vr, 1.0)

        # 4b. reta: para frente se couber, senão de ré primeiro
        x, y, th = self.gt.get_pose()
        L = V_TESTE * self.a.dur_reta
        frente = self.margem((x + L * np.cos(th), y + L * np.sin(th)))
        tras = self.margem((x - L * np.cos(th), y - L * np.sin(th)))
        ordem = [("frente", 1), ("ré", -1)] if frente >= tras else [("ré", -1), ("frente", 1)]
        if max(frente, tras) < MARGEM_MIN + 0.05:
            self.reg("acionamento", "reta", AVISO, "não coube", "gire o robô para apontar para o centro")
        else:
            for nome, s in ordem:
                t0, t1 = self.mover(f"reta {nome}", s * V_TESTE, s * V_TESTE, self.a.dur_reta)
                dg, pg = self._delta("gt", t0, t1)
                do, _ = self._delta("odom", t0, t1)
                ml, mr = self._rodas(t0, t1)
                dist_g, dist_o = np.linalg.norm(dg[:2]), np.linalg.norm(do[:2])
                esc = dist_o / dist_g if dist_g > 0.02 else np.nan
                self.reg("acionamento", f"reta {nome}: rodas medidas / comando",
                         OK if abs(ml / (s * V_TESTE) - 1) < 0.1 and abs(mr / (s * V_TESTE) - 1) < 0.1 else AVISO,
                         f"esq {ml / (s * V_TESTE):.2f}, dir {mr / (s * V_TESTE):.2f}", "PID das rodas")
                self.reg("acionamento", f"reta {nome}: distância odometria / GT",
                         OK if abs(esc - 1) < 0.03 else AVISO, f"{esc:.3f} ({1000 * dist_o:.0f} / {1000 * dist_g:.0f} mm)",
                         "≠ 1 = raio da roda errado no firmware (escala do encoder)")
                self.reg("acionamento", f"reta {nome}: desvio de rumo",
                         OK if abs(np.degrees(dg[2])) < 3 else AVISO,
                         f"GT {np.degrees(dg[2]):+.1f}°, odometria {np.degrees(do[2]):+.1f}°",
                         "rodas diferentes ou PID desbalanceado puxam para um lado")
                if nome == "frente" and dist_g > 0.05:
                    direcao = np.arctan2(dg[1], dg[0])          # corda de um arco: rumo médio = início + Δθ/2
                    rumo_medio = pg[2] + dg[2] / 2
                    off = float(np.arctan2(np.sin(direcao - rumo_medio), np.cos(direcao - rumo_medio)))
                    novo = float(np.arctan2(np.sin(self.gt.THETA_OFFSET + off), np.cos(self.gt.THETA_OFFSET + off)))
                    self.reg("GT", "θ do marcador x direção do movimento", OK if abs(np.degrees(off)) < 5 else FALHA,
                             f"{np.degrees(off):+.1f}°",
                             "" if abs(np.degrees(off)) < 5 else
                             f"ajuste THETA_OFFSET = {novo:.3f} no ground_truth.py (ou: python3 controle.py calibrar)")

        # 4c. giro de 90° e volta
        T = (np.pi / 2) / W_TESTE
        vr = W_TESTE * 0.143 / 2
        for nome, s in (("giro +90°", 1), ("giro −90° (volta)", -1)):
            t0, t1 = self.mover(nome, -s * vr, s * vr, T)
            dg, _ = self._delta("gt", t0, t1)
            do, _ = self._delta("odom", t0, t1)
            r = do[2] / dg[2] if abs(dg[2]) > 0.2 else np.nan
            self.reg("acionamento", f"{nome}: giro odometria / GT", OK if abs(r - 1) < 0.05 else AVISO,
                     f"{r:.3f} (odometria {np.degrees(do[2]):+.1f}°, GT {np.degrees(dg[2]):+.1f}°)",
                     f"≠ 1 = distância entre rodas efetiva {0.143 * r:.3f} m (firmware 0,143) ou derrapagem")
        if self.a.gravar:
            robo.record(False)
        T = np.array([m[0] for m in self.msgs["odom"] if self.segmentos[0][1] <= m[0] <= self.segmentos[-1][2]])
        if len(T) > 2:
            d = np.diff(T)
            self.reg("tempo", "odometria durante o movimento (chegada no PC)",
                     OK if d.max() < 0.3 else AVISO, f"maior buraco {1000 * d.max():.0f} ms, {np.sum(d > 0.3)} acima de 300 ms",
                     "vídeo " + ("LIGADO" if self.a.gravar else "desligado") + "; buracos = telemetria represada no Wi-Fi")

    # ------------------------------------------------------------------ atraso GT x odometria
    def atraso_gt_odom(self):
        if len(self.gt_log) < 20 or len(self.msgs["odom"]) < 20 or not self.segmentos:
            return
        t0, t1 = self.segmentos[0][1] - 1, self.segmentos[-1][2] + 1
        tt = np.arange(t0, t1, 0.01)

        def vel(fonte):
            P = np.array([self._pose_em(fonte, t) for t in tt])
            v = np.gradient(P[:, 2], tt)                 # usa a velocidade angular (os testes giram bastante)
            return v - v.mean()
        vo, vg = vel("odom"), vel("gt")
        lags = np.arange(-50, 51)                        # ±0,5 s
        c = [np.dot(vo[max(0, -k):len(vo) - max(0, k)], vg[max(0, k):len(vg) - max(0, -k)]) for k in lags]
        lag = lags[int(np.argmax(c))] * 0.01
        self.reg("tempo", "atraso da GT em relação à odometria (chegada no PC)",
                 OK if abs(lag) < 0.15 else AVISO, f"{1000 * lag:+.0f} ms",
                 "positivo = a GT chega depois; a GT carimba a chegada, não a captura")

    # ------------------------------------------------------------------ 5. importação
    def etapa_importacao(self):
        print("\n5. Gravação -> formato da dissertação")
        import importar_slam_teaching as ist
        dados = os.path.join(self.a.repo, "dados")
        s = ist.ler_sessoes(os.path.join(dados, "recording.csv"))[-1]
        r = ist.resumo(s)
        self.reg("importação", "sessão gravada", OK if r["odometria"] and r["camera"] else FALHA,
                 f"{r['duracao_s']:.0f} s, odometria {r['odometria']}, GT {r['gt']}, câmera {r['camera']} "
                 f"({r['fps']:.1f} fps)")
        if "camera" not in s:
            return
        pasta = ist.achar_pasta_camera(dados, s["camera"][0, 1])
        out = os.path.join(self.out, "sessao_convertida")
        n, _ = ist.converter(s, pasta, out, self.a.calib)
        self.reg("importação", "conversão", OK if n > 0 else FALHA, f"{n} imagens -> {out}",
                 "" if self.a.calib else "sem --calib: calibre a câmera (calibrar_camera.py) antes dos conjuntos")

    # ------------------------------------------------------------------ figura e arquivos
    def salvar(self):
        np.savez(os.path.join(self.out, "dados_brutos.npz"),
                 odom=np.array([m[:4] + ((m[4] or 0),) for m in self.msgs["odom"]], float).reshape(-1, 5),
                 rodas=np.array(self.msgs["rodas"], float).reshape(-1, 3),
                 gt=np.array(self.gt_log, float).reshape(-1, 4), cmd=np.array(self.cmd_log, float).reshape(-1, 3))
        with open(os.path.join(self.out, "relatorio.json"), "w") as f:
            json.dump(dict(itens=self.itens, segmentos=self.segmentos), f, indent=1, default=str)
        with open(os.path.join(self.out, "relatorio.txt"), "w") as f:
            for i in self.itens:
                f.write(f"[{i['estado']:5s}] {i['etapa']:12s} {i['item']}: {i['valor']}"
                        + (f"  ({i['detalhe']})" if i["detalhe"] else "") + "\n")
        if not self.segmentos:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        O = np.array([m[:4] for m in self.msgs["odom"]])
        G = np.array(self.gt_log)
        R = np.array(self.msgs["rodas"])
        C = np.array(self.cmd_log)
        t0 = self.segmentos[0][1] - 1
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        sel = lambda A: A[(A[:, 0] >= t0) & (A[:, 0] <= self.segmentos[-1][2] + 1.5)]
        Os, Gs = sel(O), sel(G)
        ax[0].plot(Gs[:, 1], Gs[:, 2], color="#e8710a", lw=2, label="GT (teto)")
        ax[0].plot(Os[:, 1], Os[:, 2], color="#1a73e8", lw=1.5, label="odometria")
        a = self.gt.area_visivel()
        ax[0].plot(*np.vstack([a, a[:1]]).T, color="0.6", ls="--", lw=1, label="área vista")
        ax[0].set_aspect("equal")
        ax[0].set_title("Trajetória")
        ax[0].set_xlabel("x [m]")
        ax[0].set_ylabel("y [m]")
        ax[0].legend(fontsize=8)
        ax[1].plot(Gs[:, 0] - t0, np.degrees(np.unwrap(Gs[:, 3])), color="#e8710a", label="GT")
        ax[1].plot(Os[:, 0] - t0, np.degrees(np.unwrap(Os[:, 3])), color="#1a73e8", label="odometria")
        ax[1].set_title("θ [graus]")
        ax[1].set_xlabel("t [s]")
        ax[1].legend(fontsize=8)
        Rs, Cs = sel(R), sel(C)
        ax[2].step(Cs[:, 0] - t0, 1000 * Cs[:, 1], where="post", color="#1a73e8", ls=":", label="esq. comando")
        ax[2].step(Cs[:, 0] - t0, 1000 * Cs[:, 2], where="post", color="#e8710a", ls=":", label="dir. comando")
        ax[2].plot(Rs[:, 0] - t0, 1000 * Rs[:, 1], color="#1a73e8", label="esq. medida")
        ax[2].plot(Rs[:, 0] - t0, 1000 * Rs[:, 2], color="#e8710a", label="dir. medida")
        ax[2].set_title("Rodas [mm/s]")
        ax[2].set_xlabel("t [s]")
        ax[2].legend(fontsize=8)
        for A in ax:
            A.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out, "teste_robo.png"), dpi=130)
        plt.close(fig)

    # ------------------------------------------------------------------ execução
    def rodar(self):
        self.robo, self.gt = None, None
        try:
            self.etapa_comunicacao()
            self.etapa_camera()
            if not self.a.sem_gt:
                self.etapa_gt()
            if self.a.mover:
                try:
                    self.etapa_acionamento()
                except Abortar as e:
                    self.reg("acionamento", "interrompido", FALHA, str(e))
                self.atraso_gt_odom()
                if self.a.gravar and hasattr(self, "t_gravacao"):       # só se esta execução gravou
                    self.etapa_importacao()
        except Abortar as e:
            print(f"\nInterrompido: {e}")
        except KeyboardInterrupt:
            print("\nCtrl+C")
        finally:
            if self.robo is not None and getattr(self.robo, "_connected", True):
                self.robo.stop()
                self.robo.record(False)
            if self.gt is not None:
                self.gt.stop()
            if self.robo is not None:
                self.robo.disconnect()
            self.salvar()
        n = {e: sum(i["estado"] == e for i in self.itens) for e in (OK, AVISO, FALHA)}
        print(f"\nResumo: {n[OK]} ok, {n[AVISO]} avisos, {n[FALHA]} falhas -> {self.out}")
        return 1 if n[FALHA] else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=SLAM_TEACHING, help="pasta utilitarios/ do SLAM-Teaching")
    ap.add_argument("--mover", action="store_true", help="faz o teste de acionamento (o robô se move)")
    ap.add_argument("--gravar", action="store_true", help="grava o passo 4 e converte com o importador")
    ap.add_argument("--sem-gt", action="store_true", help="não usa a câmera do teto")
    ap.add_argument("--calib", default=None, help="YAML da câmera do robô, p/ a conversão do passo 5")
    ap.add_argument("--sim", action="store_true", help="simulador do SLAM-Teaching (valida a lógica sem robô)")
    ap.add_argument("--janela", type=float, default=4.0, help="s de medição parada em cada etapa")
    ap.add_argument("--dur-reta", type=float, default=2.5, help="s de reta a V_TESTE (2,5 s = 25 cm)")
    a = ap.parse_args()
    if a.gravar and not a.mover:
        ap.error("--gravar precisa de --mover")
    if a.mover and a.sem_gt:
        ap.error("--mover precisa da GT (segurança)")
    sys.exit(Teste(a).rodar())


if __name__ == "__main__":
    main()
