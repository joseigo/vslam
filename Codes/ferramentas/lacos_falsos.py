"""
Robustez do grafo a laços FALSOS: repete só o back-end (FusionBackend) a partir das mensagens gravadas de uma execução
real (FUSION_RECORD=arquivo.rec python3 rodar_sistema.py ...), injetando N laços errados, e mede o ATE com e sem
cada camada de rejeição (porta de deriva, PCM, GNC-TLS).

Uso:
    python3 lacos_falsos.py --rec gravacao/urban27.rec --images <pasta das imagens> --out resultados/lacos_falsos/urban27
    python3 lacos_falsos.py --resumo resultados/lacos_falsos      # junta as sequências -> figuras e tabela

Modelo do laço falso (aliasing perceptual: o detector "acha" que está num lugar onde já passou, mas não está):
    - nó b = nó mais novo do grafo no instante da injeção (como um laço real, detectado no KF atual);
    - nó a = nó antigo (>= MIN_DT s antes) que no GT está a > MIN_GT_DIST m de b -> laço certamente errado;
    - medida afirmada = pose típica de um laço verdadeiro: b 0-8 m à frente de a, ~1 m de lado, yaw ~ N(0, 3°);
      convertida p/ as unidades do VO pela escala estimada do nó a (o mesmo caminho dos laços verdadeiros).
    modo 'aleatorio': N laços independentes, espalhados uniformemente no percurso.
    modo 'grupo'    : N/5 grupos de 5 laços MUTUAMENTE CONSISTENTES (a_k = a0 + 4k, b_k = b0 + 4k; cada medida é a do
                      1º laço composta com a odometria estimada) -- uma rua inteira confundida com outra. É o caso
                      adversário do PCM (Sünderhauf & Protzel 2012; Mangelson et al. 2018).
    Os mesmos laços falsos (mesma semente) entram em todas as configurações.

Limitação: o replay não realimenta o VO (escala, mapa) com o grafo perturbado; mede só o efeito no grafo.

Saída em --out: execucoes.csv (uma linha por execução), lacos.csv (destino de cada laço injetado/verdadeiro).
--resumo: lacos_falsos.csv, lacos_falsos_ate.pdf/.png, lacos_falsos_destino.pdf/.png, tabela no terminal.
"""
import argparse
import csv
import dataclasses
import glob
import os
import pickle
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

CONFIGS = {
    "completo": {},
    "sem_porta": dict(loop_gate_min=1e9, loop_gate_rot=1e9),
    "sem_pcm": dict(loop_pcm_chi2=float("inf"), loop_min_clique=1),
    "sem_gnc": dict(loop_gnc=False),
    "sem_rejeicao": dict(loop_gate_min=1e9, loop_gate_rot=1e9, loop_pcm_chi2=float("inf"), loop_min_clique=1,
                         loop_gnc=False),
    # GNC como era até 2026-11-07: partia da estimativa online e considerava também laços isolados
    "gnc_antigo": dict(loop_gnc_init="online", loop_gnc_pcm=False),
    # sensibilidade da porta de deriva (padrão: 15 m + 10% do percurso)
    "porta_5pct": dict(loop_gate_frac=0.05),
    "porta_20pct": dict(loop_gate_frac=0.20),
    "porta_min5": dict(loop_gate_min=5.0),
    "porta_min30": dict(loop_gate_min=30.0),
}
CONFIGS_PADRAO = ["completo", "sem_porta", "sem_pcm", "sem_gnc", "sem_rejeicao", "gnc_antigo"]
LABELS = {"completo": "sistema completo", "sem_porta": "sem porta de deriva", "sem_pcm": "sem PCM",
          "sem_gnc": "sem GNC-TLS", "sem_rejeicao": "sem nenhuma rejeição",
          "gnc_antigo": "GNC antigo (da estimativa online, laços isolados)",
          "porta_5pct": "porta 15 m + 5% do percurso", "porta_20pct": "porta 15 m + 20% do percurso",
          "porta_min5": "porta 5 m + 10% do percurso", "porta_min30": "porta 30 m + 10% do percurso"}
MODES_SHORT = {"aleatorio": "laços falsos aleatórios", "grupo": "laços falsos em grupos de 5"}
MODES = {"aleatorio": "aleatórios (independentes)", "grupo": "em grupos de 5 (mutuamente consistentes)"}
MIN_DT = 60.0          # s entre a e b
MIN_GT_DIST = 50.0     # m: distância real mínima entre a e b (o laço afirma ~0-8 m)
START_FRAC = 0.05      # não injeta no começo (grafo ainda curto)
GROUP, GROUP_STEP = 5, 4
LOOP_GT_TOL = 5.0      # mesmo critério de rodar_sistema.py p/ laço verdadeiro correto


# ---------------------------------------------------------------------------------------------- replay
def load_rec(path):
    msgs = []
    with open(path, "rb") as f:
        head = pickle.load(f)
        while True:
            try:
                msgs.append(pickle.load(f))
            except (EOFError, pickle.UnpicklingError):      # fim (ou gravação ainda em andamento)
                break
    return head, msgs


def plan_injections(node_t, node_frame, gt_pos, n, mode, seed):
    """Escolhe os pares (a, b) e as medidas métricas afirmadas. Retorna {nó gatilho: [laços]}."""
    rng = np.random.default_rng(seed)
    N = len(node_t)
    plan = {}
    if n == 0:
        return plan
    n_ev = n if mode == "aleatorio" else max(n // GROUP, 1)
    span = (GROUP - 1) * GROUP_STEP if mode == "grupo" else 0
    lo = int(np.searchsorted(node_t, node_t[0] + max(MIN_DT, START_FRAC * (node_t[-1] - node_t[0]))))
    trig = np.linspace(lo, N - 1 - span, n_ev).astype(int)
    for b0 in trig:
        cand = np.where((node_t <= node_t[b0] - MIN_DT) & (np.arange(N) + span < b0))[0]
        far = [a for a in cand if all(np.linalg.norm(gt_pos[a + k * GROUP_STEP] - gt_pos[b0 + k * GROUP_STEP])
                                      > MIN_GT_DIST for k in range(GROUP if mode == "grupo" else 1))]
        if not far:
            continue
        a0 = int(rng.choice(far))
        z = dict(yaw=np.radians(rng.normal(0, 3.0)), t=np.array([rng.uniform(0, 8), rng.normal(0, 1.0),
                                                                 rng.normal(0, 0.2)]))
        if mode == "aleatorio":
            plan.setdefault(int(b0), []).append(dict(a=a0, b=int(b0), z=z, grp=-1))
        else:
            g = len(plan)
            plan.setdefault(int(b0 + span), []).extend(
                dict(a=a0 + k * GROUP_STEP, b=int(b0 + k * GROUP_STEP), z=z, grp=g, a0=a0, b0=int(b0))
                for k in range(GROUP))
    return plan


def _loop_msg(gpo, node_t, node_frame, L):
    """Mensagem de laço (como OnlineFusion._flush_loops) com a medida afirmada em unidades do VO."""
    from gtsam import Pose3, Rot3
    Zm = Pose3(Rot3.Yaw(L['z']['yaw']), L['z']['t'])
    if L['grp'] >= 0:                    # grupo consistente: compõe com a odometria ESTIMADA (fecha o ciclo do PCM)
        P = gpo.pose
        Zm = P(L['a0']).between(P(L['a'])).inverse().compose(Zm).compose(P(L['b0']).between(P(L['b'])))
    s = float(np.exp(gpo.current_log_scale(L['a'])))
    rel = Pose3(Zm.rotation(), np.asarray(Zm.translation()) / s).matrix()
    return dict(type='loop', frame_a=node_frame[L['a']], frame_b=node_frame[L['b']], rel=rel,
                t_a=node_t[L['a']], t_b=node_t[L['b']], n_pnp=-1, n_f=-1)


def replay(head, msgs, cfg_over, plan, node_t, node_frame):
    from Fusion import FusionBackend
    args, kwargs = head
    args = list(args)
    args[1] = dataclasses.replace(args[1], **cfg_over)
    be = FusionBackend(*args, **kwargs)
    online, trig = {}, sorted(plan)
    for m in msgs:
        out = be.add(m)
        for r in out:
            if m.get('type') != 'loop':
                online.setdefault(r['n'], r['pose'])
        n_now = len(be.gpo.t) - 1 if be.gpo is not None else -1
        while trig and trig[0] <= n_now:
            for L in plan[trig.pop(0)]:
                be.add(_loop_msg(be.gpo, node_t, node_frame, L))
    pre = [L.get('status') for L in be.loop_log]
    res = be.finalize()
    return res, online, pre


def run_one(job):
    rec, gt, ds, cfg, mode, n, seed, out = job
    from rodar_sistema import describe, _match_gt
    head, msgs = _CACHE.get(rec) or _CACHE.setdefault(rec, load_rec(rec))
    t_gt, T_gt = gt
    nodes = [m for m in msgs if m.get('type') != 'loop']
    node_t = np.array([m['t'] for m in nodes])
    node_frame = [m['frame'] for m in nodes]
    gi, _ = _match_gt(node_t, t_gt, tol=0.1)
    plan = plan_injections(node_t, node_frame, T_gt[gi, :3, 3], n, mode, seed)
    row = dict(dataset=ds, config=cfg, modo=mode, n=n, semente=seed, n_injetados=sum(len(v) for v in plan.values()))
    t0 = time.perf_counter()
    try:
        res, online, pre = replay(head, msgs, CONFIGS[cfg], plan, node_t, node_frame)
    except Exception as e:                                    # noqa: BLE001
        row.update(falhou=1, erro=f"{type(e).__name__}: {e}".replace("\n", " ")[:200],
                   tempo_s=time.perf_counter() - t0)
        traceback.print_exc()
        return row, []
    row['tempo_s'] = time.perf_counter() - t0
    tn = np.array(res['t_nodes'])
    origin = T_gt[_match_gt(tn[:1], t_gt)[0][0], :3, 3]
    suav = describe(tn, res['poses'], t_gt, T_gt, origin, with_scale=False)
    ko = sorted(online)
    onl = describe(tn[ko], np.array([online[k] for k in ko]), t_gt, T_gt, origin, with_scale=False)
    row.update(falhou=0, erro="", ate_suav=suav['ate'], deriva_suav=suav['drift'], erro_max_suav=float(suav['err'].max()),
               ate_online=onl['ate'], deriva_online=onl['drift'], percurso_m=suav['length'])
    # destino de cada laço
    lac = []
    for L, st0 in zip(res['loops'], pre):
        inj = L.get('n_pnp') == -1
        st = L.get('status', '')
        acc = st.startswith('aceito') or st == 'ativo'
        (ia, ib), _ = _match_gt(np.array([L['t_a'], L['t_b']]), t_gt, tol=0.1)
        gtd = float(np.linalg.norm(T_gt[ia, :3, 3] - T_gt[ib, :3, 3]))
        ok = (not inj) and 'z_dist' in L and abs(L['z_dist'] - gtd) < LOOP_GT_TOL
        dest = ("porta" if st0 == 'porta' else "aceito" if acc else
                "GNC" if 'GNC' in st else "PCM" if 'PCM' in str(st0) else "outro")
        lac.append(dict(dataset=ds, config=cfg, modo=mode, n=n, semente=seed, injetado=int(inj), correto=int(ok),
                        t_a=L['t_a'], t_b=L['t_b'], dist_gt=gtd, status_online=st0, status_final=st, destino=dest,
                        online_ativo=int(st0 == 'ativo')))
    I = [x for x in lac if x['injetado']]
    V = [x for x in lac if not x['injetado'] and x['destino'] != 'outro']
    row.update(inj_porta=sum(x['destino'] == 'porta' for x in I), inj_online_ativo=sum(x['online_ativo'] for x in I),
               inj_aceito=sum(x['destino'] == 'aceito' for x in I),
               verd_cand=len(V), verd_aceito_ok=sum(x['destino'] == 'aceito' and x['correto'] for x in V),
               verd_aceito_errado=sum(x['destino'] == 'aceito' and not x['correto'] for x in V))
    print(f"{ds} {cfg:13s} {mode:9s} n={n:3d} s={seed} | ATE suav {suav['ate']:8.2f} m online {onl['ate']:8.2f} m | "
          f"falsos aceitos {row['inj_aceito']:3d}/{row['n_injetados']:3d} | {row['tempo_s']:.0f} s", flush=True)
    return row, lac


_CACHE = {}


def _write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def main_run(args):
    from rodar_sistema import find_dataset_files
    from fusao_offline import load_gt
    gt = load_gt(args.gt if args.gt else find_dataset_files(args.images)[1]["gt"])
    ds = args.dataset or os.path.basename(os.path.splitext(args.rec)[0])
    os.makedirs(args.out, exist_ok=True)
    jobs = [(args.rec, gt, ds, c, "aleatorio", 0, 0, args.out) for c in args.configs]
    jobs += [(args.rec, gt, ds, c, m, n, s, args.out) for n in args.n if n > 0 for m in args.modos
             for s in range(args.sementes) for c in args.configs]
    print(f"{len(jobs)} execuções do grafo ({args.jobs} em paralelo)")
    import multiprocessing as mp
    with mp.get_context("fork").Pool(args.jobs) as pool:
        res = pool.map(run_one, jobs, chunksize=1)
    _write(os.path.join(args.out, "execucoes.csv"), [r for r, _ in res])
    _write(os.path.join(args.out, "lacos.csv"), [x for _, l in res for x in l])
    print(f"-> {args.out}/execucoes.csv, lacos.csv")


# ---------------------------------------------------------------------------------------------- resumo
def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main_resumo(pasta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for p in sorted(glob.glob(os.path.join(pasta, "*", "execucoes.csv"))) for r in _read(p)]
    lac = [r for p in sorted(glob.glob(os.path.join(pasta, "*", "lacos.csv"))) for r in _read(p)]
    if not rows:
        raise SystemExit(f"nada em {pasta}/*/execucoes.csv")
    datasets = sorted({r['dataset'] for r in rows})
    cfgs = [c for c in CONFIGS if any(r['config'] == c for r in rows)]
    ns = sorted({int(r['n']) for r in rows})
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else np.nan

    def sel(ds, c, m, n, k):
        v = [f(r, k) if r['falhou'] == "0" else np.inf for r in rows if r['dataset'] == ds and r['config'] == c
             and int(r['n']) == n and (n == 0 or r['modo'] == m)]
        return np.array(v)

    # ---- tabela (mediana entre sementes) e CSV
    out_rows = []
    for m in MODES:
        print(f"\nATE da fusão suavizada [m] com laços falsos {MODES[m]} (mediana das sementes; 'falhou' = erro do solver)")
        for ds in datasets:
            print(f"\n  {ds}: " + "".join(f"{'n=' + str(n):>10s}" for n in ns))
            for c in cfgs:
                vals = []
                for n in ns:
                    v = sel(ds, c, m, n, 'ate_suav')
                    med = float(np.median(v)) if len(v) else np.nan
                    acc = sel(ds, c, m, n, 'inj_aceito')
                    out_rows.append(dict(dataset=ds, modo=m, config=c, n=n, ate_suav_mediana=med,
                                         ate_suav_min=float(v.min()) if len(v) else np.nan,
                                         ate_suav_max=float(v.max()) if len(v) else np.nan,
                                         ate_online_mediana=float(np.median(sel(ds, c, m, n, 'ate_online'))) if len(v) else np.nan,
                                         falsos_aceitos_mediana=float(np.median(acc)) if len(acc) else np.nan,
                                         falhas=int(np.isinf(v).sum())))
                    vals.append("falhou" if np.isinf(med) else f"{med:.1f}")
                print(f"    {LABELS[c]:24s}" + "".join(f"{s:>10s}" for s in vals))
    _write(os.path.join(pasta, "lacos_falsos.csv"), out_rows)

    # ---- figura 1: ATE x nº de laços falsos (small multiples: modo x sequência)
    TEXT, GRID = "#52514e", "#e4e3df"
    COL = dict(zip(CONFIGS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6",
                             "#0f8a8a", "#b8860b", "#8c564b", "#7f7f7f"]))
    MARK = dict(zip(CONFIGS, "osD^vPXhp*"))
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8.5, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "axes.edgecolor": "#8a8984", "axes.labelcolor": TEXT, "xtick.color": TEXT,
        "ytick.color": TEXT, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True, "legend.frameon": False,
        "pdf.fonttype": 42})
    for key, name, fname in (("ate_suav", "ATE da fusão suavizada [m]", "lacos_falsos_ate"),
                             ("ate_online", "ATE da fusão online (causal) [m]", "lacos_falsos_ate_online")):
        fig, axs = plt.subplots(len(MODES), len(datasets), figsize=(3.6 * len(datasets) + 1.4, 6.4),
                                squeeze=False, sharex=True)
        x = np.arange(len(ns))
        for i, m in enumerate(MODES):
            for j, ds in enumerate(datasets):
                ax = axs[i, j]
                base = np.median(sel(ds, "completo", m, 0, key))
                ymax = 1.0
                for k, c in enumerate(cfgs):
                    med, lo, hi, fail = [], [], [], []
                    for n in ns:
                        v = sel(ds, c, m, n, key)
                        fin = v[np.isfinite(v)]
                        med.append(np.median(fin) if len(fin) else np.nan)
                        lo.append(fin.min() if len(fin) else np.nan)
                        hi.append(fin.max() if len(fin) else np.nan)
                        fail.append(len(v) - len(fin))
                    med, lo, hi = (np.array(a) / base for a in (med, lo, hi))
                    xo = x + (k - (len(cfgs) - 1) / 2) * 0.07
                    ax.vlines(xo, lo, hi, color=COL[c], lw=1.4, alpha=0.6)
                    ax.plot(xo, med, color=COL[c], lw=1.8, marker=MARK[c], ms=5, mec="white", mew=0.6,
                            label=LABELS[c] if (i, j) == (0, 0) else None)
                    for xx, fl in zip(xo, fail):
                        if fl:
                            ax.annotate("✕", (xx, 1.0), color=COL[c], ha="center", va="bottom",
                                        xycoords=("data", "axes fraction"), fontsize=8)
                    ymax = max(ymax, np.nanmax(hi))
                ax.axhline(1, color=TEXT, lw=0.9, ls="--")
                ax.set_yscale("log")
                ax.set_xticks(x, [str(n) for n in ns])
                if i == 0:
                    ax.set_title(f"{ds}  (sem laços falsos: {base:.1f} m)")
                if i == len(MODES) - 1:
                    ax.set_xlabel("laços falsos injetados")
                if j == 0:
                    ax.set_ylabel(f"{MODES_SHORT[m]}\nATE / ATE sem laços falsos")
                ax.grid(axis="x", visible=False)
        fig.suptitle(f"{name} com laços falsos injetados", y=0.995, fontsize=10.5, color=TEXT)
        fig.text(0.5, 0.955, f"linha: mediana de {_n_seeds(rows)} sementes · barra: mín–máx · ✕: solver falhou · "
                 "escala log · tracejado: sem laços falsos", ha="center", fontsize=8.5, color=TEXT)
        fig.legend(loc="upper center", ncol=min(len(cfgs), 4), bbox_to_anchor=(0.5, 0.94))
        fig.tight_layout(rect=(0, 0, 1, 0.90 if len(cfgs) > 4 else 0.91))
        fig.savefig(os.path.join(pasta, fname + ".pdf"))
        fig.savefig(os.path.join(pasta, fname + ".png"), dpi=150)
        plt.close(fig)

    # ---- figura 2: destino dos laços falsos (todas as sequências e sementes, n máximo)
    nmax = max(ns)
    STAGE = [("porta", "barrado na porta de deriva", "#2a78d6"),
             ("PCM", "fora do clique do PCM (e sem GNC)", "#1baf7a"),
             ("GNC", "rejeitado pelo GNC-TLS (lote final)", "#6250d6"),
             ("aceito", "ACEITO no grafo final", "#d03b3b")]
    fig, axs = plt.subplots(1, len(MODES), figsize=(11, 0.45 * len(cfgs) + 1.6), sharey=True, squeeze=False)
    for ax, m in zip(axs[0], MODES):
        y = np.arange(len(cfgs))[::-1]
        for yi, c in zip(y, cfgs):
            L = [r for r in lac if r['config'] == c and r['modo'] == m and int(r['n']) == nmax and r['injetado'] == "1"]
            tot = max(len(L), 1)
            left = 0.0
            for key, lab, col in STAGE:
                fr = sum(r['destino'] == key for r in L) / tot
                if fr > 0:
                    ax.barh(yi, fr, left=left, color=col, height=0.62, edgecolor="white", linewidth=1,
                            label=lab if yi == y[0] or True else None)
                    if fr >= 0.06:
                        ax.text(left + fr / 2, yi, f"{100 * fr:.0f}%", ha="center", va="center", color="white",
                                fontsize=8)
                left += fr
        ax.set_yticks(y, [LABELS[c] for c in cfgs])
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(f"laços falsos {MODES[m]}, n = {nmax}")
        ax.grid(axis="y", visible=False)
    h, l = axs[0][0].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    fig.legend([uniq[lab] for _, lab, _ in STAGE if lab in uniq], [lab for _, lab, _ in STAGE if lab in uniq],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Destino dos laços falsos injetados ({', '.join(datasets)}; todas as sementes)", color=TEXT)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(pasta, "lacos_falsos_destino.pdf"))
    fig.savefig(os.path.join(pasta, "lacos_falsos_destino.png"), dpi=150)
    print(f"\n-> {pasta}/lacos_falsos.csv, lacos_falsos_ate(.pdf/.png), lacos_falsos_ate_online, lacos_falsos_destino")


def _n_seeds(rows):
    return len({r['semente'] for r in rows if int(r['n']) > 0}) or 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec")
    ap.add_argument("--images", help="pasta das imagens da sequência (p/ achar o GT)")
    ap.add_argument("--dataset")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, nargs="+", default=[0, 10, 50, 100])
    ap.add_argument("--modos", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--configs", nargs="+", default=CONFIGS_PADRAO, choices=list(CONFIGS))
    ap.add_argument("--sementes", type=int, default=3)
    ap.add_argument("--gt", help="global_pose.csv (alternativa a --images)")
    ap.add_argument("-j", "--jobs", type=int, default=2)
    ap.add_argument("--resumo", metavar="PASTA")
    a = ap.parse_args()
    if a.resumo:
        main_resumo(a.resumo)
    else:
        main_run(a)
