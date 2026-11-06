"""
Compara o gauge do BA local no VO puro: KFs fixos ('fix') x marginalização ('marg'), com repetições.

Uso:
    python3 comparar_ba.py resultados/comparacao_ba

Entrada: <pasta>/<dataset>_<modo>_r<k>/ (ou <dataset>_<modo>/, tratado como r1) gerados por
    test_vo_fusion_rt.py --no-loop --no-feedback --ba-mode <modo> --out <pasta>/<dataset>_<modo>_r<k>
(só a linha do VO monocular é usada: sem realimentação, ela não depende da fusão). O VO tem BA assíncrono e não
é determinístico, por isso cada configuração é repetida e a variação entre repetições é reportada.

Saída em <pasta>/:
    comparacao_ba_execucoes.csv  uma linha por execução: RPE (KITTI 100-800 m, média e mediana dos segmentos), ATE
                                 Sim(3), deriva, tempo do BA, reinícios, falhas da marginalização
    comparacao_ba.csv            por dataset e modo: média, mínimo e máximo entre repetições
    comparacao_ba_rpe.pdf        mediana do RPE por comprimento de segmento (linha: média das repetições; finas:
                                 cada repetição), um painel por dataset
    comparacao_ba_resumo.pdf     mediana do RPE de cada execução, por dataset
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

MODES = {"fix": ("KFs fixos", "#8a8984", "--"), "marg": ("marginalização", "#2a78d6", "-")}
TEXT, GRID = "#52514e", "#e4e3df"
METRICS = [("rpe_t_med", "RPE transl. mediana [%]"), ("rpe_r_med", "RPE rot. mediana [°/100 m]"),
           ("rpe_t", "RPE transl. média [%]"), ("rpe_r", "RPE rot. média [°/100 m]"),
           ("ate", "ATE Sim(3) [m]"), ("drift", "deriva final [%]"), ("ba_ms", "BA [ms]"),
           ("resets", "reinícios do mapa"), ("marg_fail", "falhas da marginalização")]


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _vo(rows):
    return [r for r in rows if r["metodo"].startswith("VO")]


def load(run):
    d = os.path.join(run, "dados")
    met = _vo(_rows(os.path.join(d, "metricas.csv")))[0]
    rpe = _vo(_rows(os.path.join(d, "rpe.csv")))
    ba = np.loadtxt(os.path.join(d, "tempos_ba.csv"), delimiter=",", skiprows=1, ndmin=2)
    log = open(run + ".log", encoding="utf-8", errors="replace").read() if os.path.isfile(run + ".log") else ""
    col = lambda rows, k: np.array([float(r.get(k) or "nan") for r in rows])
    return dict(
        ate=float(met["ate_m"]), drift=float(met["deriva_pct"]), length=float(met["percurso_m"]),
        rpe_t=float(met["rpe_transl_pct"]), rpe_r=float(met["rpe_rot_deg_100m"]),
        rpe_t_med=float(met.get("rpe_transl_mediana_pct") or "nan"),
        rpe_r_med=float(met.get("rpe_rot_mediana_deg_100m") or "nan"),
        L=col(rpe, "L_m"), t_med=col(rpe, "transl_mediana_pct"), r_med=col(rpe, "rot_mediana_deg_100m"),
        ba_ms=float(np.mean(ba[:, 1])) if len(ba) else np.nan,
        resets=len(re.findall(r"\[RESET\]", log)), marg_fail=len(re.findall(r"Marginalização falhou", log)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta")
    args = ap.parse_args()
    runs = {}                                            # runs[ds][modo] = [execução, ...]
    for p in sorted(glob.glob(os.path.join(args.pasta, "*_*/dados/tempos_ba.csv"))):   # escrito por último
        run = os.path.dirname(os.path.dirname(p))
        m = re.fullmatch(r"(.+)_(fix|marg)(?:_r(\d+))?", os.path.basename(run))
        if m:
            runs.setdefault(m[1], {}).setdefault(m[2], []).append(load(run))
    datasets = [ds for ds in runs if set(runs[ds]) == set(MODES)]
    if not datasets:
        raise SystemExit(f"nenhum dataset com os dois modos em {args.pasta}")
    arr = lambda ds, mode, k: np.array([e[k] for e in runs[ds][mode]])

    with open(os.path.join(args.pasta, "comparacao_ba_execucoes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "modo", "repeticao", "percurso_m"] + [k for k, _ in METRICS])
        for ds in datasets:
            for mode in MODES:
                for i, e in enumerate(runs[ds][mode], 1):
                    w.writerow([ds, mode, i, f"{e['length']:.0f}"] + [f"{e[k]:.4f}" for k, _ in METRICS])
    with open(os.path.join(args.pasta, "comparacao_ba.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "modo", "n_execucoes", "percurso_m"]
                   + [f"{k}_{s}" for k, _ in METRICS for s in ("media", "min", "max")])
        for ds in datasets:
            for mode in MODES:
                w.writerow([ds, mode, len(runs[ds][mode]), f"{runs[ds][mode][0]['length']:.0f}"]
                           + [f"{g(arr(ds, mode, k)):.4f}" for k, _ in METRICS for g in (np.mean, np.min, np.max)])

    # ---- tabela no terminal: média entre repetições [mín-máx]
    fmt = lambda a, d: f"{np.mean(a):{5 + d}.{d}f} [{np.min(a):.{d}f}-{np.max(a):.{d}f}]"
    for key, name, d in (("rpe_t_med", "RPE de translação, mediana dos segmentos [%]", 2),
                         ("rpe_r_med", "RPE de rotação, mediana dos segmentos [°/100 m]", 3),
                         ("rpe_t", "RPE de translação, média dos segmentos [%]", 2),
                         ("ate", "ATE Sim(3) [m]", 1)):
        print(f"\n{name}: média entre repetições [mín-máx]")
        print(f"{'dataset':9s} {'n':>3s}  {'fix':>22s}  {'marg':>22s}  {'marg/fix':>8s}  {'marg melhor':>11s}")
        for ds in datasets:
            a, b = arr(ds, "fix", key), arr(ds, "marg", key)
            wins = sum(x < y for x in b for y in a)            # pares (marg, fix) em que marg ganhou
            print(f"{ds:9s} {len(a)}/{len(b)}  {fmt(a, d):>22s}  {fmt(b, d):>22s}  {np.mean(b) / np.mean(a):8.2f}  "
                  f"{wins:>4d}/{len(a) * len(b):<6d}")
        for mode in MODES:
            Lt = sum(runs[ds][mode][0]['length'] for ds in datasets)
            wt = sum(np.mean(arr(ds, mode, key)) * runs[ds][mode][0]['length'] for ds in datasets) / Lt
            print(f"  média ponderada pelo percurso ({mode}): {wt:.{d}f}")
    print("\nfalhas da marginalização por execução (marg): "
          + ", ".join(f"{ds} {list(arr(ds, 'marg', 'marg_fail').astype(int))}" for ds in datasets))
    print("reinícios do mapa (fix | marg): "
          + ", ".join(f"{ds} {int(arr(ds, 'fix', 'resets').sum())}|{int(arr(ds, 'marg', 'resets').sum())}"
                      for ds in datasets))
    print("BA [ms] (fix | marg): " + ", ".join(
        f"{ds} {np.mean(arr(ds, 'fix', 'ba_ms')):.0f}|{np.mean(arr(ds, 'marg', 'ba_ms')):.0f}" for ds in datasets))

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": "#8a8984", "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
        "legend.frameon": False, "pdf.fonttype": 42})

    # ---- figura 1: mediana do RPE por comprimento de segmento
    n = len(datasets)
    fig, axs = plt.subplots(2, n, figsize=(2.6 * n, 5.4), squeeze=False, sharex=True)
    for j, ds in enumerate(datasets):
        for i, key in enumerate(("t_med", "r_med")):
            ax = axs[i, j]
            for mode, (lab, col, ls) in MODES.items():
                es = runs[ds][mode]
                for e in es:
                    ax.plot(e["L"], e[key], color=col, ls=ls, lw=0.8, alpha=0.45)
                ax.plot(es[0]["L"], np.mean([e[key] for e in es], axis=0), color=col, ls=ls, lw=2, marker="o",
                        ms=4, label=lab)
            ax.set_ylim(bottom=0)
            if i == 0:
                ax.set_title(f"{ds} ({runs[ds]['fix'][0]['length'] / 1000:.1f} km)")
            else:
                ax.set_xlabel("comprimento do segmento [m]")
    axs[0, 0].set_ylabel("erro de translação, mediana [%]")
    axs[1, 0].set_ylabel("erro de rotação, mediana [°/100 m]")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    nrep = min(len(runs[ds][m]) for ds in datasets for m in MODES)
    fig.suptitle(f"RPE do VO monocular (Sim(3)) por comprimento de segmento: KFs fixos x marginalização "
                 f"(linha grossa: média de {nrep} execuções; finas: cada execução)", y=0.94, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out1 = os.path.join(args.pasta, "comparacao_ba_rpe.pdf")
    fig.savefig(out1)
    fig.savefig(out1[:-4] + ".png", dpi=150)

    # ---- figura 2: mediana do RPE de cada execução, por dataset
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 0.45 * n + 1.4), sharey=True)
    y = np.arange(n)[::-1]
    for ax, key, xl in ((axs[0], "rpe_t_med", "erro de translação, mediana [%]"),
                        (axs[1], "rpe_r_med", "erro de rotação, mediana [°/100 m]")):
        for k, (mode, (lab, col, _)) in enumerate(MODES.items()):
            off = 0.12 if k == 0 else -0.12
            for yi, ds in zip(y, datasets):
                v = arr(ds, mode, key)
                ax.plot([v.min(), v.max()], [yi + off] * 2, color=col, lw=1.2, alpha=0.6)
                ax.scatter(v, [yi + off] * len(v), s=22, color=col, edgecolor="white", linewidth=0.8, zorder=3,
                           label=lab if yi == y[0] else None)
        ax.set_xlabel(xl)
        ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y, datasets)
    axs[1].legend(loc="upper left")
    fig.suptitle("Mediana do RPE (segmentos de 100-800 m) de cada execução do VO monocular", fontsize=9.5)
    fig.tight_layout()
    out2 = os.path.join(args.pasta, "comparacao_ba_resumo.pdf")
    fig.savefig(out2)
    fig.savefig(out2[:-4] + ".png", dpi=150)
    print(f"\n{args.pasta}/comparacao_ba.csv, comparacao_ba_execucoes.csv\n{out1}\n{out2}")


if __name__ == "__main__":
    main()
