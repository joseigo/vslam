"""
Bateria final de testes da dissertação: roda tudo, retomável, em ordem de prioridade, e gera a síntese.

Uso:
    python3 Codes/ferramentas/bateria.py                 # roda o que falta (2 execuções em paralelo) até as 08:00
    python3 Codes/ferramentas/bateria.py --ate 07:30     # outro horário-limite p/ INICIAR execuções
    python3 Codes/ferramentas/bateria.py --sem-limite    # sem horário-limite
    python3 Codes/ferramentas/bateria.py --estado        # só mostra o que está feito / faltando
    python3 Codes/ferramentas/bateria.py --sintese       # só refaz tabelas, figuras e o índice
    python3 Codes/ferramentas/bateria.py --seqs urban32 --configs completo_r1,gifs   # só uma parte

Agendado pelo timer systemd do usuário `bateria-dissert.timer` (todo dia às 21:00): cada noite continua de onde a
anterior parou; quando não falta nada, só refaz a síntese e sai.

Estrutura gerada em Dissert/resultados/:
    kaist/<seq>/<config>/        uma execução do pipeline completo (figuras .pdf, dados/, evo/; gifs/ no completo_r1)
    kaist/<seq>/<config>.log     saída da execução
    kaist/<seq>/completo_r1.rec  mensagens do grafo gravadas (p/ repetir só o grafo: lacos_falsos.py)
    inomi/<seq>/...              idem, robô da Inomi (sequências com GT)
    sintese/ablacao/             tabela de ablação + repetições (barras de erro), KAIST
    sintese/inomi/               idem, Inomi
    sintese/detector_lacos/      avaliação isolada do detector de laço (avaliar_lacos.py)
    sintese/lacos_falsos/        robustez a laços falsos injetados (lacos_falsos.py)
    bateria/                     log e estado da bateria
    testes_unitarios/            histórico dos testes isolados (Codes/tests/rodar_testes.py)
    README.md                    índice (gerado)
"""
import argparse
import csv
import datetime as dt
import glob
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
DISSERT = os.path.dirname(CODES)
RES = os.path.join(DISSERT, "resultados")
PIPE = os.path.join(CODES, "scripts", "rodar_sistema.py")
PY = sys.executable
DATASETS = "/home/joseigo/Documentos/TCC/Datasets"
INOMI = "/home/joseigo/Documentos/TCC/dataset_inomi"

# sequências (ordem: as mais longas primeiro, p/ equilibrar as 2 execuções em paralelo)
KAIST_AVAL = ["urban38", "urban39", "urban33", "urban30", "urban27", "urban26"]
KAIST_CEGO = ["urban28", "urban31", "urban32", "urban37"]       # fora do ajuste e do treino do vocabulário
INOMI_SEQS = ["dataset_16_16", "dataset_16_16_n2"]               # as únicas da Inomi com GT

CONFIGS = {
    "completo": [],
    "sem_laco": ["--no-loop"],
    "sem_realimentacao": ["--no-feedback"],
    "sem_chao": ["--vo", "use_ground_scale=False"],
    "ba_marg": ["--ba-mode", "marg"],
    "vo_marg": ["--no-feedback", "--ba-mode", "marg"],
    "sem_porta": ["--gpo", "loop_gate_min=1e9", "--gpo", "loop_gate_rot=1e9"],
    "sem_pcm": ["--gpo", "loop_pcm_chi2=inf", "--gpo", "loop_min_clique=1"],
    "sem_gnc": ["--gpo", "loop_gnc=False"],
    "sem_rejeicao": ["--gpo", "loop_gate_min=1e9", "--gpo", "loop_gate_rot=1e9", "--gpo", "loop_pcm_chi2=inf",
                     "--gpo", "loop_min_clique=1", "--gpo", "loop_gnc=False"],
}
N_REP = 3
FASES = [                                       # (nome, [configurações]) -- em ordem de prioridade
    ("sistema completo (r1)", ["completo_r1"]),
    ("sem laço / sem realimentação + GIFs", ["sem_laco", "sem_realimentacao", "gifs"]),
    ("repetições do sistema completo", [f"completo_r{k}" for k in range(2, N_REP + 1)]),
    ("componentes", ["sem_chao", "ba_marg", "vo_marg"]),
    ("camadas de rejeição de laço", ["sem_porta", "sem_pcm", "sem_gnc", "sem_rejeicao"]),
    ("detector de laço (isolado)", ["detector"]),
    ("laços falsos (replay do grafo)", ["lacos_falsos"]),
]
LOG_DIR = os.path.join(RES, "bateria")


# ---------------------------------------------------------------------------------------------- sequências
def images_dir(seq):
    if seq.startswith("dataset"):
        return os.path.join(INOMI, seq)
    best = None
    for dirpath, dirnames, _ in os.walk(os.path.join(DATASETS, seq)):
        if os.path.basename(dirpath) == "stereo_left" and glob.glob(os.path.join(dirpath, "*.png")):
            best = dirpath
            break
    return best


def group(seq):
    return "inomi" if seq.startswith("dataset") else "kaist"


def all_seqs():
    return KAIST_AVAL + KAIST_CEGO + INOMI_SEQS


# ---------------------------------------------------------------------------------------------- tarefas
class Job:
    def __init__(self, fase, seq, cfg):
        self.fase, self.seq, self.cfg = fase, seq, cfg
        self.base = os.path.join(RES, group(seq), seq)
        self.weight = 2 if cfg == "lacos_falsos" else 1        # lacos_falsos usa 2 processos

    @property
    def name(self):
        return f"{self.seq}/{self.cfg}"

    @property
    def out(self):
        if self.cfg == "gifs":
            return os.path.join(self.base, "completo_r1", "gifs")
        if self.cfg == "detector":
            return os.path.join(RES, "sintese", "detector_lacos", self.seq)
        if self.cfg == "lacos_falsos":
            return os.path.join(RES, "sintese", "lacos_falsos", self.seq)
        return os.path.join(self.base, self.cfg)

    def done(self):
        if self.cfg in ("gifs", "detector", "lacos_falsos"):
            return os.path.isfile(os.path.join(self.out, ".feito"))
        return os.path.isfile(os.path.join(self.out, "dados", "metricas.csv"))

    def ready(self):
        """Pré-requisitos (GIFs e laços falsos dependem do completo_r1)."""
        r1 = os.path.join(self.base, "completo_r1")
        if self.cfg == "gifs":
            return os.path.isfile(os.path.join(r1, "dados", "metricas.csv"))
        if self.cfg == "lacos_falsos":
            return os.path.isfile(os.path.join(self.base, "completo_r1.rec")) and \
                os.path.isfile(os.path.join(r1, "dados", "metricas.csv"))
        return True

    def command(self):
        img = images_dir(self.seq)
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg")
        if self.cfg == "gifs":
            return [PY, os.path.join(HERE, "gerar_gifs.py"), os.path.join(self.base, "completo_r1")], env
        if self.cfg == "detector":
            return [PY, os.path.join(HERE, "avaliar_lacos.py"), "--images", img, "--out", self.out], env
        if self.cfg == "lacos_falsos":
            return [PY, os.path.join(HERE, "lacos_falsos.py"), "--rec", os.path.join(self.base, "completo_r1.rec"),
                    "--images", img, "--dataset", self.seq, "--out", self.out, "--sementes", "2", "-j", "2"], env
        cfg = self.cfg.split("_r")[0] if self.cfg.startswith("completo_r") else self.cfg
        if self.cfg == "completo_r1":
            env["FUSION_RECORD"] = os.path.join(self.base, "completo_r1.rec")
        return [PY, PIPE, "--images", img, "--no-gui", "--out", self.out] + CONFIGS[cfg], env

    def prepare(self):
        os.makedirs(self.base, exist_ok=True)
        if self.cfg == "gifs":
            shutil.rmtree(self.out, ignore_errors=True)
        elif os.path.isdir(self.out):                          # execução incompleta de outra noite: refaz do zero
            shutil.rmtree(self.out)

    def finish(self, rc):
        if rc == 0 and self.cfg in ("gifs", "detector", "lacos_falsos") and os.path.isdir(self.out):
            open(os.path.join(self.out, ".feito"), "w").write(dt.datetime.now().isoformat())


FILTRO = dict(seqs=None, configs=None)


def jobs():
    out = []
    for fase, cfgs in FASES:
        for cfg in cfgs:
            for seq in all_seqs():
                if seq.startswith("dataset") and cfg in ("detector", "lacos_falsos"):
                    continue                                     # Inomi: 2,6 m, sem revisitas
                if FILTRO["seqs"] and seq not in FILTRO["seqs"] or FILTRO["configs"] and cfg not in FILTRO["configs"]:
                    continue
                out.append(Job(fase, seq, cfg))
    return out


# ---------------------------------------------------------------------------------------------- execução
def log(msg):
    os.makedirs(LOG_DIR, exist_ok=True)
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(os.path.join(LOG_DIR, "bateria.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def record(job, status, t0, t1):
    path = os.path.join(LOG_DIR, "estado.csv")
    new = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["sequencia", "config", "fase", "status", "inicio", "fim", "duracao_min"])
        w.writerow([job.seq, job.cfg, job.fase, status, f"{t0:%Y-%m-%d %H:%M}", f"{t1:%Y-%m-%d %H:%M}",
                    f"{(t1 - t0).total_seconds() / 60:.1f}"])


def deadline(hhmm):
    now = dt.datetime.now()
    h, m = map(int, hhmm.split(":"))
    d = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return d if d > now else d + dt.timedelta(days=1)


def run(args):
    import fcntl
    os.makedirs(LOG_DIR, exist_ok=True)
    lock = open(os.path.join(LOG_DIR, ".trava"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)        # uma bateria por vez (timer x execução manual)
    except OSError:
        print("outra bateria já está rodando (resultados/bateria/.trava); saindo")
        return
    limit = None if args.sem_limite else deadline(args.ate)
    pend = [j for j in jobs() if not j.done()]
    log(f"=== bateria: {len(pend)} tarefas pendentes; limite p/ iniciar: {limit or 'nenhum'}")
    running = []
    while True:
        for item in list(running):                             # terminadas
            p, j, t0, fh = item
            rc = p.poll()
            if rc is None:
                continue
            fh.close()
            running.remove(item)
            j.finish(rc)
            ok = rc == 0 and j.done()
            record(j, "ok" if ok else f"falhou (saída {rc})", t0, dt.datetime.now())
            log(f"{'ok    ' if ok else 'FALHOU'} {j.name} ({(dt.datetime.now() - t0).total_seconds() / 60:.0f} min)")
        if limit is not None and dt.datetime.now() >= limit and pend:
            log(f"horário-limite: {len(pend)} tarefas ficam p/ a próxima noite")
            pend = []
        used = sum(j.weight for _, j, _, _ in running)
        for j in list(pend):                                   # inicia as prontas, em ordem de prioridade
            if used >= args.paralelo:
                break
            if used + j.weight > args.paralelo and running:
                continue
            if not j.ready():
                continue
            pend.remove(j)
            cmd, env = j.command()
            if None in cmd:
                log(f"PULADA {j.name}: imagens não encontradas")
                continue
            j.prepare()
            log_path = os.path.join(j.base, f"{j.cfg}.log") if j.cfg not in ("detector", "lacos_falsos") \
                else os.path.join(os.path.dirname(j.out), f"{j.seq}.log")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            fh = open(log_path, "w")
            running.append((subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=DISSERT),
                            j, dt.datetime.now(), fh))
            log(f"início {j.name}  [{j.fase}]")
            used += j.weight
        if not running:
            if pend:                                           # sobrou só o que depende de algo que falhou
                log(f"{len(pend)} tarefas sem pré-requisito (ex.: completo_r1 falhou): " +
                    ", ".join(j.name for j in pend[:8]))
            break
        time.sleep(5)
    sintese()


# ---------------------------------------------------------------------------------------------- síntese
def _sh(cmd, log_name):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, log_name), "w") as fh:
        return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=DISSERT,
                               env=dict(os.environ, MPLBACKEND="Agg"))


def sintese():
    log("síntese: tabelas e figuras")
    S = os.path.join(RES, "sintese")
    if glob.glob(os.path.join(RES, "kaist", "*", "completo_r*", "dados", "metricas.csv")):
        _sh([PY, os.path.join(HERE, "tabela_ablacao.py"), os.path.join(RES, "kaist"), "--out",
             os.path.join(S, "ablacao"), "--cego", ",".join(KAIST_CEGO)], "sintese_ablacao.txt")
        shutil.copy(os.path.join(LOG_DIR, "sintese_ablacao.txt"), os.path.join(S, "ablacao", "tabela.txt"))
    if glob.glob(os.path.join(RES, "inomi", "*", "completo_r*", "dados", "metricas.csv")):
        _sh([PY, os.path.join(HERE, "tabela_ablacao.py"), os.path.join(RES, "inomi"), "--out",
             os.path.join(S, "inomi")], "sintese_inomi.txt")
        shutil.copy(os.path.join(LOG_DIR, "sintese_inomi.txt"), os.path.join(S, "inomi", "tabela.txt"))
    if glob.glob(os.path.join(S, "detector_lacos", "*", ".feito")):
        _sh([PY, os.path.join(HERE, "avaliar_lacos.py"), "--resumo", os.path.join(S, "detector_lacos")],
            "sintese_detector.txt")
    if glob.glob(os.path.join(S, "lacos_falsos", "*", ".feito")):
        _sh([PY, os.path.join(HERE, "lacos_falsos.py"), "--resumo", os.path.join(S, "lacos_falsos")],
            "sintese_lacos_falsos.txt")
        shutil.copy(os.path.join(LOG_DIR, "sintese_lacos_falsos.txt"), os.path.join(S, "lacos_falsos", "tabela.txt"))
    indice()
    log("síntese concluída")


def estado_tabela():
    js = jobs()
    seqs = all_seqs()
    cfgs = list(dict.fromkeys(j.cfg for j in js))
    st = {(j.seq, j.cfg): ("ok" if j.done() else "--") for j in js}
    return seqs, cfgs, st, sum(v == "ok" for v in st.values()), len(st)


def indice():
    seqs, cfgs, st, n_ok, n = estado_tabela()
    try:
        commit = subprocess.check_output(["git", "-C", DISSERT, "log", "-1", "--format=%h %cd", "--date=short"],
                                         text=True).strip()
        dirty = subprocess.call(["git", "-C", DISSERT, "diff", "--quiet", "--", "Codes"]) != 0
    except Exception:                                                      # noqa: BLE001
        commit, dirty = "?", False
    L = ["# Resultados da dissertação", "",
         f"Gerado por `Codes/ferramentas/bateria.py` em {dt.datetime.now():%Y-%m-%d %H:%M} "
         f"(código: commit {commit}{' + alterações não commitadas' if dirty else ''}). "
         f"Bateria: {n_ok}/{n} tarefas concluídas.", "",
         "## Onde está cada coisa", "",
         "| Pasta | Conteúdo |", "|---|---|",
         "| `kaist/<seq>/<config>/` | uma execução do pipeline completo: figuras (`fusao_*.pdf`), `dados/` (métricas, "
         "trajetórias, laços, tempos, mapas), `evo/` (APE/RPE com o evo); `completo_r1/gifs/` (GIFs) |",
         "| `kaist/<seq>/completo_r1.rec` | mensagens do grafo gravadas (repetir só o grafo: `lacos_falsos.py`) |",
         "| `inomi/<seq>/<config>/` | idem, robô da Inomi (as duas sequências com GT) |",
         "| `sintese/ablacao/` | **tabela de ablação** (`tabela.txt`, `ablacao_*.tex`, `ablacao.pdf`) e **barras de erro** "
         "(`repeticoes.*`): KAIST, avaliação + teste cego (†) |",
         "| `sintese/inomi/` | idem, Inomi |",
         "| `sintese/detector_lacos/` | avaliação isolada do detector de laço (recall@N, precisão, funil, cobertura) |",
         "| `sintese/lacos_falsos/` | robustez a laços falsos injetados (ATE x nº de laços falsos, por camada) |",
         "| `testes_unitarios/` | histórico dos testes isolados (`Codes/tests/rodar_testes.py`) |",
         "| `bateria/` | log (`bateria.log`) e estado (`estado.csv`) da bateria |",
         "| `_arquivo/` | resultados de versões anteriores do código (só p/ consulta; não usar no texto) |", "",
         "## Sequências", "",
         f"- KAIST, avaliação: {', '.join(KAIST_AVAL)}",
         f"- KAIST, teste cego (fora do ajuste e do treino do vocabulário, urban25/33/34): {', '.join(KAIST_CEGO)}",
         f"- Inomi (robô, com GT): {', '.join(INOMI_SEQS)}", "",
         "## Configurações", "",
         "`completo_r1..r3` = sistema completo (3 repetições: o pipeline completo não é determinístico); "
         "`sem_laco`, `sem_realimentacao` (também dá a linha \"só VO\"), `sem_chao`, `ba_marg`, `vo_marg` (só VO com "
         "BA marginalizado), `sem_porta`, `sem_pcm`, `sem_gnc`, `sem_rejeicao` (ver `CONFIGS` em bateria.py).", "",
         "## Estado", "",
         "| seq | " + " | ".join(c.replace("completo_", "") for c in cfgs) + " |",
         "|---|" + "---|" * len(cfgs)]
    for s in seqs:
        L.append(f"| {s} | " + " | ".join(st.get((s, c), "") for c in cfgs) + " |")
    with open(os.path.join(RES, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paralelo", type=int, default=2)
    ap.add_argument("--ate", default="08:00", help="horário-limite p/ INICIAR execuções (HH:MM)")
    ap.add_argument("--sem-limite", action="store_true")
    ap.add_argument("--estado", action="store_true")
    ap.add_argument("--sintese", action="store_true")
    ap.add_argument("--seqs", help="só estas sequências (vírgula)")
    ap.add_argument("--configs", help="só estas configurações/tarefas (vírgula; ex.: completo_r1,gifs)")
    a = ap.parse_args()
    FILTRO.update(seqs=a.seqs.split(",") if a.seqs else None, configs=a.configs.split(",") if a.configs else None)
    if a.estado:
        seqs, cfgs, st, n_ok, n = estado_tabela()
        print(f"{n_ok}/{n} concluídas")
        print(f"{'':18s}" + "".join(f"{c.replace('completo_', ''):>13s}" for c in cfgs))
        for s in seqs:
            print(f"{s:18s}" + "".join(f"{st.get((s, c), ''):>13s}" for c in cfgs))
    elif a.sintese:
        sintese()
    else:
        run(a)
