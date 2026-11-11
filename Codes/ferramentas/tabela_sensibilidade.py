"""
Sensibilidade a erros de calibração dos sensores: ATE do sistema completo com cada perturbação (bateria.py, configs
sens_*) comparado com a faixa [mín, máx] das repetições sem perturbação (completo_r*). Como o pipeline completo não é
determinístico, uma diferença só conta como efeito da perturbação se cair FORA dessa faixa.

Uso:
    python3 tabela_sensibilidade.py resultados/kaist --out resultados/sintese/sensibilidade

Saída: sensibilidade.csv, tabela.txt, sensibilidade.tex.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

ROTULOS = {
    "sens_enc_mais2": "diâmetro das rodas +2%",
    "sens_enc_menos2": "diâmetro das rodas -2%",
    "sens_altura_mais5": "altura da câmera +5 cm",
    "sens_altura_menos5": "altura da câmera -5 cm",
    "sens_atraso_mais50": "atraso roda-câmera +50 ms",
    "sens_atraso_menos50": "atraso roda-câmera -50 ms",
}


def ler(run, metodo):
    f = os.path.join(run, "dados", "metricas.csv")
    if not os.path.isfile(f):
        return np.nan
    m = pd.read_csv(f).set_index("metodo")
    return float(m.loc[metodo, "ate_m"]) if metodo in m.index else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raiz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    linhas = []
    for seq_dir in sorted(glob.glob(os.path.join(a.raiz, "*"))):
        seq = os.path.basename(seq_dir)
        if not any(os.path.isdir(os.path.join(seq_dir, c)) for c in ROTULOS):
            continue
        base = [ler(r, "fusão suavizada") for r in sorted(glob.glob(os.path.join(seq_dir, "completo_r*")))]
        base_roda = [ler(r, "odometria de rodas") for r in sorted(glob.glob(os.path.join(seq_dir, "completo_r*")))]
        base = np.array([b for b in base if np.isfinite(b)])
        if not len(base):
            continue
        med, lo, hi = np.median(base), base.min(), base.max()
        linhas.append(dict(seq=seq, config="sem perturbação", rotulo="sem perturbação (mediana [mín-máx])",
                           ate=med, ate_min=lo, ate_max=hi, ate_roda=np.nanmedian(base_roda),
                           var_pct=0.0, fora_da_faixa=False))
        for c, rot in ROTULOS.items():
            runs = [os.path.join(seq_dir, c)] + sorted(glob.glob(os.path.join(seq_dir, c + "_r[0-9]")))
            v = np.array([ler(r, "fusão suavizada") for r in runs])
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            ate = float(np.median(v))                    # com repetições: mediana; "fora" = mediana fora da faixa
            linhas.append(dict(seq=seq, config=c, rotulo=rot, ate=ate, ate_min=v.min(), ate_max=v.max(), n_exec=len(v),
                               ate_roda=ler(runs[0], "odometria de rodas"), var_pct=100 * (ate / med - 1),
                               fora_da_faixa=bool(ate < lo or ate > hi)))
    if not linhas:
        print("nenhuma execução sens_* encontrada")
        return
    os.makedirs(a.out, exist_ok=True)
    D = pd.DataFrame(linhas)
    D.to_csv(os.path.join(a.out, "sensibilidade.csv"), index=False)

    seqs = list(dict.fromkeys(D.seq))
    ordem = ["sem perturbação"] + list(ROTULOS)

    def cel(r, roda=False):
        if roda:
            return f"{r.ate_roda:.1f}"
        if r.config == "sem perturbação":
            return f"{r.ate:.1f} [{r.ate_min:.1f}-{r.ate_max:.1f}]"
        faixa = f" [{r.ate_min:.1f}-{r.ate_max:.1f}]" if r.get("n_exec", 1) > 1 else ""
        return f"{r.ate:.1f}{'*' if r.fora_da_faixa else ''}{faixa}"

    txt = ["ATE da fusão suavizada [m]: mediana [mín-máx] das execuções (* = mediana fora da faixa [mín-máx] das 3 execuções sem perturbação)", ""]
    W = 22
    txt.append(f"{'perturbação':28s}" + "".join(f"{s:>{W}s}" for s in seqs) + f"{'var. mediana':>14s}")
    tex = ["% gerado por tabela_sensibilidade.py", "\\begin{tabular}{l" + "r" * (len(seqs) + 1) + "}", "\\toprule",
           "Perturbação & " + " & ".join(seqs) + " & Var. mediana \\\\", "\\midrule"]
    for c in ordem:
        g = D[D.config == c].set_index("seq")
        if g.empty:
            continue
        rot = g.rotulo.iloc[0] if c != "sem perturbação" else "sem perturbação"
        vals = [cel(g.loc[s]) if s in g.index else "--" for s in seqs]
        var = "" if c == "sem perturbação" else f"{np.median(g.var_pct):+.0f}%"
        txt.append(f"{rot:28s}" + "".join(f"{v:>{W}s}" for v in vals) + f"{var:>14s}")
        tex.append(rot.replace("%", chr(92) + "%") + " & " + " & ".join(v.replace("*", "$^*$") for v in vals) + f" & {var.replace('%', chr(92) + '%')} \\\\")
    txt += ["", "ATE só da roda com a mesma perturbação [m] (referência: quanto a perturbação estraga a roda sozinha)", ""]
    for c in ordem:
        g = D[D.config == c].set_index("seq")
        if g.empty:
            continue
        rot = g.rotulo.iloc[0] if c != "sem perturbação" else "sem perturbação"
        txt.append(f"{rot:28s}" + "".join(f"{(cel(g.loc[s], True) if s in g.index else '--'):>{W}s}" for s in seqs))
    tex += ["\\bottomrule", "\\end{tabular}"]
    s = "\n".join(txt)
    open(os.path.join(a.out, "tabela.txt"), "w").write(s + "\n")
    open(os.path.join(a.out, "sensibilidade.tex"), "w").write("\n".join(tex) + "\n")
    print(s)


if __name__ == "__main__":
    main()
