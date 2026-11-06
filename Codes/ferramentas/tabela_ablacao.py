"""
Tabela de ablação do sistema completo (câmera monocular + odometria de rodas).

Uso:
    python3 tabela_ablacao.py resultados/ablacao

Entrada: <pasta>/<dataset>/<configuração>/dados/{metricas,lacos}.csv gerados por test_vo_fusion_rt.py (uma pasta por
configuração; ver CONFIGS). Linhas da tabela:
    - referências: só odometria de rodas e só VO (tirados das execuções 'completo' e 'sem_realimentacao');
    - fusão online (causal) do sistema completo;
    - fusão suavizada (após a otimização final em lote) de cada configuração.

Saída em <pasta>/:
    ablacao.csv           formato longo: dataset, linha, métrica, valor
    ablacao_<métrica>.tex tabela LaTeX (booktabs) configurações x sequências + média ponderada pelo percurso
    ablacao.pdf/.png      ATE e mediana do RPE de cada configuração relativos ao sistema completo, por sequência
"""
import argparse
import csv
import glob
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (chave, pasta da execução, prefixo da linha em metricas.csv, rótulo)
ROWS = [
    ("roda", "completo", ("odometria de rodas", "só roda"), "Só odometria de rodas"),
    ("vo", "sem_realimentacao", ("VO monocular",), "Só VO monocular (alinhado em Sim(3))"),
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
METRICS = [("ate", "ATE [m]", 1), ("drift", "deriva final [%]", 2), ("rpe_t", "RPE transl. mediana [%]", 2),
           ("rpe_r", "RPE rot. mediana [°/100 m]", 3), ("loops", "laços aceitos", 0), ("loops_bad", "laços errados", 0)]
TEXT, GRID = "#52514e", "#e4e3df"
GROUP_COL = {"ref": "#8a8984", "comp": "#2a78d6", "loop": "#eb6834"}


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load(pasta):
    data = {}                                       # data[ds][linha] = {métrica: valor}
    for ds_dir in sorted(glob.glob(os.path.join(pasta, "*/"))):
        ds = os.path.basename(os.path.dirname(ds_dir))
        for key, run, prefixes, _ in ROWS:
            met = os.path.join(ds_dir, run, "dados", "metricas.csv")
            if not os.path.isfile(met):
                continue
            row = next((r for r in _read(met) if r["metodo"].startswith(prefixes)), None)
            if row is None:
                continue
            f = lambda k: float(row.get(k) or "nan")
            e = dict(ate=f("ate_m"), drift=f("deriva_pct"), length=f("percurso_m"),
                     rpe_t=f("rpe_transl_mediana_pct"), rpe_r=f("rpe_rot_mediana_deg_100m"),
                     loops=np.nan, loops_bad=np.nan)
            lc = os.path.join(ds_dir, run, "dados", "lacos.csv")
            if prefixes == ("fusão suavizada",) and os.path.isfile(lc):
                acc = [r for r in _read(lc) if r["kind"] == "aceito"]
                e.update(loops=float(len(acc)), loops_bad=float(sum(r["correct"] != "True" for r in acc)))
            elif prefixes == ("fusão suavizada",):
                e.update(loops=0.0, loops_bad=0.0)
            data.setdefault(ds, {})[key] = e
    return data


def weighted(data, datasets, key, m):
    v = [(data[ds][key][m], data[ds][key]['length']) for ds in datasets if key in data[ds]]
    if len(v) < len(datasets):
        return np.nan
    if m in ("loops", "loops_bad"):
        return float(sum(x for x, _ in v))
    return float(sum(x * w for x, w in v) / sum(w for _, w in v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta")
    args = ap.parse_args()
    data = load(args.pasta)
    datasets = [ds for ds in data if "completo" in data[ds]]
    if not datasets:
        raise SystemExit(f"nenhuma execução 'completo' em {args.pasta}")
    labels = {k: lab for k, _, _, lab in ROWS}
    keys = [k for k, *_ in ROWS if any(k in data[ds] for ds in datasets)]

    with open(os.path.join(args.pasta, "ablacao.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "linha", "rotulo", "percurso_m"] + [m for m, _, _ in METRICS])
        for ds in datasets:
            for k in keys:
                if k in data[ds]:
                    e = data[ds][k]
                    w.writerow([ds, k, labels[k], f"{e['length']:.0f}"] + [f"{e[m]:.4f}" for m, _, _ in METRICS])

    tot = sum(data[ds]["completo"]["length"] for ds in datasets) / 1000
    for m, name, d in METRICS:
        print(f"\n{name}  (média ponderada pelo percurso; laços: soma)")
        print(f"{'':44s}" + "".join(f"{ds:>9s}" for ds in datasets) + f"{'média':>9s}")
        for k in keys:
            vals = [data[ds][k][m] if k in data[ds] else np.nan for ds in datasets]
            if all(np.isnan(vals)):
                continue
            print(f"{labels[k]:44s}" + "".join(f"{v:9.{d}f}" if np.isfinite(v) else f"{'-':>9s}" for v in vals)
                  + f"{weighted(data, datasets, k, m):9.{d}f}")
        # LaTeX
        with open(os.path.join(args.pasta, f"ablacao_{m}.tex"), "w", encoding="utf-8") as f:
            f.write("% gerado por Codes/ferramentas/tabela_ablacao.py\n")
            f.write("\\begin{tabular}{l" + "r" * (len(datasets) + 1) + "}\n\\toprule\n")
            f.write("Configuração & " + " & ".join(datasets) + " & Média \\\\\n")
            f.write(" & " + " & ".join(f"{data[ds]['completo']['length'] / 1000:.1f} km" for ds in datasets)
                    + f" & {tot:.1f} km \\\\\n\\midrule\n")
            for k in keys:
                vals = [data[ds][k][m] if k in data[ds] else np.nan for ds in datasets]
                if all(np.isnan(vals)):
                    continue
                if k == "completo":
                    f.write("\\midrule\n")
                cells = [f"{v:.{d}f}" if np.isfinite(v) else "--" for v in vals]
                cells.append(f"{weighted(data, datasets, k, m):.{d}f}")
                lab = labels[k] if k in ("roda", "vo", "online", "completo") else "\\quad " + labels[k]
                if k == "completo":
                    cells = [f"\\textbf{{{c}}}" for c in cells]
                    lab = f"\\textbf{{{lab}}}"
                f.write(f"{lab} & " + " & ".join(cells) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")

    # ---- figura: razão em relação ao sistema completo (log), um ponto por sequência
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8.5, "axes.edgecolor": "#8a8984", "axes.labelcolor": TEXT, "xtick.color": TEXT,
        "ytick.color": TEXT, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True, "legend.frameon": False,
        "pdf.fonttype": 42})
    comp = [k for k in keys if k != "completo"]
    group = lambda k: "ref" if k in ("roda", "vo", "online") else "loop" if k in (
        "sem_porta", "sem_pcm", "sem_gnc", "sem_rejeicao") else "comp"
    y = np.arange(len(comp))[::-1]
    markers = "osD^vP*X"
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.42 * len(comp) + 1.6), sharey=True)
    for ax, m, title in ((axs[0], "ate", "ATE"), (axs[1], "rpe_t", "mediana do RPE de translação")):
        for yi, k in zip(y, comp):
            r = np.array([data[ds][k][m] / data[ds]["completo"][m] if k in data[ds] else np.nan for ds in datasets])
            ok = np.isfinite(r) & (r > 0)
            if not ok.any():
                continue
            ax.plot([r[ok].min(), r[ok].max()], [yi, yi], color=GROUP_COL[group(k)], lw=1.2, alpha=0.5)
            for j, ds in enumerate(datasets):
                if ok[j]:
                    ax.scatter(r[j], yi, s=26, marker=markers[j % len(markers)], color=GROUP_COL[group(k)],
                               edgecolor="white", linewidth=0.6, zorder=3, label=ds if yi == y[0] else None)
            g = np.exp(np.mean(np.log(r[ok])))
            ax.scatter(g, yi, s=70, marker="|", color=TEXT, zorder=4, linewidth=2,
                       label="média geométrica" if yi == y[0] else None)
        ax.axvline(1, color=TEXT, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set(xlabel=f"{title} / {title} do sistema completo (escala log; > 1 = pior)", title=title)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y, [labels[k] for k in comp])
    for yt, k in zip(axs[0].get_yticklabels(), comp):
        yt.set_color(GROUP_COL[group(k)] if group(k) != "comp" else TEXT)
    axs[1].legend(loc="lower right", ncol=2, fontsize=7.5)
    fig.suptitle("Ablação: cada configuração comparada ao sistema completo (fusão suavizada), por sequência", y=1.0)
    fig.tight_layout()
    out = os.path.join(args.pasta, "ablacao")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=150)
    print(f"\n{args.pasta}/ablacao.csv, ablacao_*.tex, ablacao.pdf/.png")


if __name__ == "__main__":
    main()
