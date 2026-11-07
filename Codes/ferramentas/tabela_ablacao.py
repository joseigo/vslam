"""
Tabelas e figuras de síntese da bateria (ablação + repetições) do sistema completo (câmera monocular + roda).

Uso:
    python3 tabela_ablacao.py resultados/kaist --out resultados/sintese/ablacao --cego urban28,urban31,urban32,urban37

Entrada: <pasta>/<dataset>/<configuração>/dados/{metricas,lacos}.csv gerados por test_vo_fusion_rt.py. Uma
configuração pode ter repetições (<configuração>_r1, _r2, ...): cada métrica entra pela MEDIANA das repetições.
Linhas da tabela:
    - referências: só odometria de rodas, só VO (BA fixo e marginalizado), fusão online do sistema completo;
    - fusão suavizada (após a otimização final em lote) de cada configuração.
--cego: sequências do teste cego (fora do ajuste e do treino do vocabulário) -> colunas à direita e média própria.

Saída em --out (padrão: <pasta>):
    ablacao.csv              dataset, linha, percurso e métricas (medianas)
    ablacao_<métrica>.tex    tabela LaTeX (booktabs): configurações x sequências + média ponderada pelo percurso
    ablacao.pdf/.png         ATE e mediana do RPE de cada configuração relativos ao sistema completo
    repeticoes.csv/.tex      sistema completo: mediana, mín. e máx. das repetições por sequência (barras de erro)
    repeticoes.pdf/.png      idem, em figura
"""
import argparse
import csv
import glob
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (chave, pasta da execução, prefixos da linha em metricas.csv, rótulo)
ROWS = [
    ("roda", "completo", ("odometria de rodas", "só roda"), "Só odometria de rodas"),
    ("vo", "sem_realimentacao", ("VO monocular",), "Só VO monocular (Sim(3)), BA fixo"),
    ("vo_marg", "vo_marg", ("VO monocular",), "Só VO monocular (Sim(3)), BA marginalizado"),
    ("online", "completo", ("fusão online",), "Fusão online (causal), sistema completo"),
    ("completo", "completo", ("fusão suavizada",), "Sistema completo"),
    ("sem_laco", "sem_laco", ("fusão suavizada",), "sem fechamento de laço"),
    ("sem_realimentacao", "sem_realimentacao", ("fusão suavizada",), "sem realimentação da escala no VO"),
    ("sem_chao", "sem_chao", ("fusão suavizada",), "sem âncora de escala pelo chão"),
    ("ba_marg", "ba_marg", ("fusão suavizada",), "BA local com marginalização"),
    ("sem_porta", "sem_porta", ("fusão suavizada",), "sem porta de deriva"),
    ("sem_pcm", "sem_pcm", ("fusão suavizada",), "sem PCM"),
    ("sem_gnc", "sem_gnc", ("fusão suavizada",), "sem GNC-TLS no lote final"),
    ("sem_rejeicao", "sem_rejeicao", ("fusão suavizada",), "sem nenhuma rejeição de laço"),
]
REF = ("roda", "vo", "vo_marg", "online")
METRICS = [("ate", "ATE [m]", 1), ("drift", "deriva final [%]", 2), ("rpe_t", "RPE transl. mediana [%]", 2),
           ("rpe_r", "RPE rot. mediana [°/100 m]", 3), ("loops", "laços aceitos", 0), ("loops_bad", "laços errados", 0)]
TEXT, GRID = "#52514e", "#e4e3df"
GROUP_COL = {"ref": "#8a8984", "comp": "#2a78d6", "loop": "#eb6834"}


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _runs(ds_dir, run):
    """Pastas da configuração: <run> e/ou <run>_r<k>, só as concluídas (com metricas.csv)."""
    cand = [os.path.join(ds_dir, run)] + sorted(glob.glob(os.path.join(ds_dir, run + "_r*")))
    return [d for d in cand if re.fullmatch(re.escape(run) + r"(_r\d+)?", os.path.basename(d))
            and os.path.isfile(os.path.join(d, "dados", "metricas.csv"))]


def _one(d, prefixes):
    row = next((r for r in _read(os.path.join(d, "dados", "metricas.csv")) if r["metodo"].startswith(prefixes)), None)
    if row is None:
        return None
    f = lambda k: float(row.get(k) or "nan")                                    # noqa: E731
    e = dict(ate=f("ate_m"), drift=f("deriva_pct"), length=f("percurso_m"), rpe_t=f("rpe_transl_mediana_pct"),
             rpe_r=f("rpe_rot_mediana_deg_100m"), loops=np.nan, loops_bad=np.nan)
    if prefixes == ("fusão suavizada",):
        lc = os.path.join(d, "dados", "lacos.csv")
        acc = [r for r in _read(lc) if r["kind"] == "aceito"] if os.path.isfile(lc) else []
        e.update(loops=float(len(acc)), loops_bad=float(sum(r["correct"] != "True" for r in acc)))
    return e


def load(pasta):
    data, reps = {}, {}                             # data[ds][linha] = {métrica: mediana}; reps[ds][linha] = [dict]
    for ds_dir in sorted(glob.glob(os.path.join(pasta, "*/"))):
        ds = os.path.basename(os.path.dirname(ds_dir))
        for key, run, prefixes, _ in ROWS:
            es = [e for e in (_one(d, prefixes) for d in _runs(ds_dir, run)) if e is not None]
            if not es:
                continue
            data.setdefault(ds, {})[key] = {m: float(np.nanmedian([e[m] for e in es])) if not all(
                np.isnan(e[m]) for e in es) else np.nan for m in es[0]}
            reps.setdefault(ds, {})[key] = es
    return data, reps


def weighted(data, datasets, key, m):
    v = [(data[ds][key][m], data[ds][key]['length']) for ds in datasets if key in data[ds]]
    if not datasets or len(v) < len(datasets):
        return np.nan
    if m in ("loops", "loops_bad"):
        return float(sum(x for x, _ in v))
    return float(sum(x * w for x, w in v) / sum(w for _, w in v))


def _fmt(v, d):
    return f"{v:.{d}f}" if np.isfinite(v) else "--"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cego", default="", help="sequências do teste cego, separadas por vírgula")
    args = ap.parse_args()
    out_dir = args.out or args.pasta
    os.makedirs(out_dir, exist_ok=True)
    data, reps = load(args.pasta)
    cego = [s for s in args.cego.split(",") if s]
    all_ds = [ds for ds in data if "completo" in data[ds]]
    groups = [("avaliação", [d for d in all_ds if d not in cego]), ("cego", [d for d in all_ds if d in cego])]
    groups = [(g, l) for g, l in groups if l]
    datasets = [d for _, l in groups for d in l]
    if not datasets:
        raise SystemExit(f"nenhuma execução 'completo' em {args.pasta}")
    labels = {k: lab for k, _, _, lab in ROWS}
    keys = [k for k, *_ in ROWS if any(k in data[ds] for ds in datasets)]
    mark = lambda ds: ds + ("†" if ds in cego else "")                         # noqa: E731

    with open(os.path.join(out_dir, "ablacao.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "grupo", "linha", "rotulo", "n_execucoes", "percurso_m"] + [m for m, _, _ in METRICS])
        for ds in datasets:
            for k in keys:
                if k in data[ds]:
                    e = data[ds][k]
                    w.writerow([ds, "cego" if ds in cego else "avaliação", k, labels[k], len(reps[ds][k]),
                                f"{e['length']:.0f}"] + [f"{e[m]:.4f}" for m, _, _ in METRICS])

    for m, name, d in METRICS:
        print(f"\n{name}  (mediana das repetições; média ponderada pelo percurso; laços: soma; † = teste cego)")
        print(f"{'':46s}" + "".join(f"{mark(ds):>10s}" for ds in datasets)
              + "".join(f"{'média ' + g:>16s}" for g, _ in groups))
        for k in keys:
            vals = [data[ds][k][m] if k in data[ds] else np.nan for ds in datasets]
            if all(np.isnan(vals)):
                continue
            print(f"{labels[k]:46s}" + "".join(f"{_fmt(v, d):>10s}" for v in vals)
                  + "".join(f"{_fmt(weighted(data, l, k, m), d):>16s}" for _, l in groups))
        with open(os.path.join(out_dir, f"ablacao_{m}.tex"), "w", encoding="utf-8") as f:
            f.write("% gerado por Codes/ferramentas/tabela_ablacao.py (mediana das repetições; † = teste cego)\n")
            f.write("\\begin{tabular}{l" + "r" * (len(datasets) + len(groups)) + "}\n\\toprule\n")
            f.write("Configuração & " + " & ".join(mark(ds) for ds in datasets) + " & "
                    + " & ".join(f"Média {g}" for g, _ in groups) + " \\\\\n")
            f.write(" & " + " & ".join(f"{data[ds]['completo']['length'] / 1000:.1f} km" for ds in datasets) + " & "
                    + " & ".join(f"{sum(data[ds]['completo']['length'] for ds in l) / 1000:.1f} km"
                                 for _, l in groups) + " \\\\\n\\midrule\n")
            for k in keys:
                vals = [data[ds][k][m] if k in data[ds] else np.nan for ds in datasets]
                if all(np.isnan(vals)):
                    continue
                if k == "completo":
                    f.write("\\midrule\n")
                cells = [_fmt(v, d) for v in vals] + [_fmt(weighted(data, l, k, m), d) for _, l in groups]
                lab = labels[k] if k in REF + ("completo",) else "\\quad " + labels[k]
                if k == "completo":
                    cells = [f"\\textbf{{{c}}}" for c in cells]
                    lab = f"\\textbf{{{lab}}}"
                f.write(f"{lab} & " + " & ".join(cells) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")

    _style()
    _fig_ablacao(data, datasets, keys, labels, cego, out_dir)
    _repeticoes(reps, datasets, cego, out_dir)
    print(f"\n-> {out_dir}/ablacao.csv, ablacao_*.tex, ablacao.pdf/.png, repeticoes.csv/.tex/.pdf/.png")


def _style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8.5, "axes.edgecolor": "#8a8984", "axes.labelcolor": TEXT, "xtick.color": TEXT,
        "ytick.color": TEXT, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True, "legend.frameon": False,
        "pdf.fonttype": 42})


def _fig_ablacao(data, datasets, keys, labels, cego, out_dir):
    """Razão em relação ao sistema completo (log), um ponto por sequência (vazado = teste cego)."""
    comp = [k for k in keys if k != "completo"]
    group = lambda k: "ref" if k in REF else "loop" if k in (                  # noqa: E731
        "sem_porta", "sem_pcm", "sem_gnc", "sem_rejeicao") else "comp"
    y = np.arange(len(comp))[::-1]
    markers = "osD^vP*Xhp<>"
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.42 * len(comp) + 1.8), sharey=True)
    for ax, m, title in ((axs[0], "ate", "ATE"), (axs[1], "rpe_t", "mediana do RPE de translação")):
        for yi, k in zip(y, comp):
            r = np.array([data[ds][k][m] / data[ds]["completo"][m] if k in data[ds] else np.nan for ds in datasets])
            ok = np.isfinite(r) & (r > 0)
            if not ok.any():
                continue
            col = GROUP_COL[group(k)]
            ax.plot([r[ok].min(), r[ok].max()], [yi, yi], color=col, lw=1.2, alpha=0.5)
            for j, ds in enumerate(datasets):
                if ok[j]:
                    blind = ds in cego
                    ax.scatter(r[j], yi, s=28, marker=markers[j % len(markers)], zorder=3,
                               color="white" if blind else col, edgecolor=col if blind else "white",
                               linewidth=1.0 if blind else 0.6,
                               label=(ds + (" (cego)" if blind else "")) if yi == y[0] else None)
            g = np.exp(np.mean(np.log(r[ok])))
            ax.scatter(g, yi, s=70, marker="|", color=TEXT, zorder=4, linewidth=2,
                       label="média geométrica" if yi == y[0] else None)
        ax.axvline(1, color=TEXT, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set(xlabel="razão em relação ao sistema completo (log; > 1 = pior)", title=title)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y, [labels[k] for k in comp])
    for yt, k in zip(axs[0].get_yticklabels(), comp):
        yt.set_color(GROUP_COL[group(k)] if group(k) != "comp" else TEXT)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(len(lab), 6), fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Ablação: cada configuração comparada ao sistema completo (fusão suavizada; mediana das repetições)",
                 color=TEXT)
    fig.tight_layout(rect=(0, 0.07 if len(datasets) > 6 else 0.05, 1, 1))
    fig.savefig(os.path.join(out_dir, "ablacao.pdf"))
    fig.savefig(os.path.join(out_dir, "ablacao.png"), dpi=150)
    plt.close(fig)


def _repeticoes(reps, datasets, cego, out_dir):
    """Barras de erro do sistema completo: dispersão entre repetições (o pipeline completo não é determinístico:
    back-end assíncrono + realimentação)."""
    ds_r = [ds for ds in datasets if "completo" in reps[ds]]
    MET = [("ate", "ATE [m]", 2), ("drift", "deriva final [%]", 3), ("rpe_t", "RPE transl. mediana [%]", 3),
           ("rpe_r", "RPE rot. mediana [°/100 m]", 3)]
    rows = []
    for ds in ds_r:
        for key, lab in (("completo", "fusão suavizada"), ("online", "fusão online")):
            es = reps[ds].get(key, [])
            if not es:
                continue
            for m, _, _ in MET:
                v = np.array([e[m] for e in es])
                rows.append(dict(dataset=ds, grupo="cego" if ds in cego else "avaliação", linha=lab, metrica=m,
                                 n=len(v), mediana=np.median(v), minimo=v.min(), maximo=v.max(),
                                 dispersao_pct=100 * (v.max() - v.min()) / np.median(v) if np.median(v) else np.nan,
                                 valores=" ".join(f"{x:.4f}" for x in v)))
    if not rows:
        return
    with open(os.path.join(out_dir, "repeticoes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    get = {(r['dataset'], r['linha'], r['metrica']): r for r in rows}
    with open(os.path.join(out_dir, "repeticoes.tex"), "w", encoding="utf-8") as f:
        f.write("% gerado por Codes/ferramentas/tabela_ablacao.py: sistema completo (fusão suavizada), mediana "
                "[mín.--máx.] das repetições\n")
        f.write("\\begin{tabular}{lr" + "c" * len(MET) + "}\n\\toprule\n")
        f.write("Sequência & $n$ & " + " & ".join(n for _, n, _ in MET) + " \\\\\n\\midrule\n")
        for ds in ds_r:
            cells = []
            for m, _, d in MET:
                r = get.get((ds, "fusão suavizada", m))
                cells.append(f"{r['mediana']:.{d}f} [{r['minimo']:.{d}f}--{r['maximo']:.{d}f}]" if r else "--")
            n = get.get((ds, "fusão suavizada", "ate"), {}).get("n", 0)
            f.write(f"{ds}{'†' if ds in cego else ''} & {n} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    fig, axs = plt.subplots(1, len(MET), figsize=(3.0 * len(MET) + 1, 0.32 * len(ds_r) + 1.6), sharey=True)
    y = np.arange(len(ds_r))[::-1]
    for ax, (m, name, _) in zip(axs, MET):
        for yi, ds in zip(y, ds_r):
            for key, col, off in (("completo", "#1baf7a", 0.12), ("online", "#eb6834", -0.12)):
                es = reps[ds].get(key, [])
                if not es:
                    continue
                v = np.array([e[m] for e in es])
                ax.plot([v.min(), v.max()], [yi + off] * 2, color=col, lw=2, alpha=0.6, solid_capstyle="round")
                ax.scatter(v, [yi + off] * len(v), s=12, color=col, zorder=3, edgecolor="white", linewidth=0.4)
                ax.scatter(np.median(v), yi + off, s=60, marker="|", color=TEXT, zorder=4, linewidth=1.8)
        ax.set_xscale("log")
        ax.set_title(name)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y, [ds + (" (cego)" if ds in cego else "") for ds in ds_r])
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color="#1baf7a", lw=2, marker="o"), Line2D([], [], color="#eb6834", lw=2, marker="o"),
                Line2D([], [], color=TEXT, lw=0, marker="|", ms=10, mew=1.8)],
               ["fusão suavizada (sistema completo)", "fusão online (causal)", "mediana"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    n = max(len(reps[ds]["completo"]) for ds in ds_r)
    fig.suptitle(f"Variação entre execuções do sistema completo (até {n} repetições por sequência; pontos = "
                 "execuções, barra = mín.–máx.; escala log)", color=TEXT)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(out_dir, "repeticoes.pdf"))
    fig.savefig(os.path.join(out_dir, "repeticoes.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
