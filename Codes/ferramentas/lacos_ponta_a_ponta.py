"""
Laços de ponta a ponta: dos laços que o DETECTOR entregou ao grafo (já verificados geometricamente), quantos tinham
pose ruim e em que camada do grafo cada um parou (porta de deriva, PCM/GNC) ou se foi aceito.

Uso:
    python3 lacos_ponta_a_ponta.py resultados/kaist --out resultados/sintese/lacos_ponta_a_ponta [--configs completo]

Entrada: <raiz>/<seq>/<config>_r<k>/dados/lacos.csv (ou <config>/) de rodar_sistema.py.
  correct = |distância medida - distância no GT| < LOOP_GT_TOL (5 m); "grosseiro" = erro > 50 m.
Saída: lacos.csv (todos os laços, com seq/execução), resumo.csv (por sequência), tabela.txt e tabela.tex.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

GROSSEIRO_M = 50.0


def carregar(raiz, config):
    fs = sorted(glob.glob(os.path.join(raiz, "*", f"{config}_r*", "dados", "lacos.csv")) +
                glob.glob(os.path.join(raiz, "*", config, "dados", "lacos.csv")))
    out = []
    for f in fs:
        L = pd.read_csv(f)
        if L.empty:
            continue
        p = f.split(os.sep)
        L.insert(0, "execucao", p[-3])
        L.insert(0, "seq", p[-4])
        out.append(L)
    if not out:
        return pd.DataFrame()
    A = pd.concat(out, ignore_index=True)
    A["correct"] = A["correct"].astype(str) == "True"
    A["camada"] = np.select([A["kind"] == "aceito", A.status == "porta",
                             A.status.str.contains("PCM"), A.status.str.contains("GNC")],
                            ["aceito", "porta de deriva", "PCM", "GNC-TLS"], "outro")
    A["grosseiro"] = A["meas_err"] > GROSSEIRO_M
    return A


def resumir(A):
    def linha(g):
        n_exec = g["execucao"].nunique()
        ruins = ~g.correct
        return pd.Series(dict(
            execucoes=n_exec,            # (TOTAL: nº de execuções distintas por sequência, não a soma)
            entregues=len(g), ruins=int(ruins.sum()), grosseiros=int(g.grosseiro.sum()),
            barrados_porta=int((g.camada == "porta de deriva").sum()),
            barrados_pcm=int((g.camada == "PCM").sum()),
            barrados_gnc=int((g.camada == "GNC-TLS").sum()),
            ruins_barrados=int((ruins & (g.camada != "aceito")).sum()),
            grosseiros_barrados=int((g.grosseiro & (g.camada != "aceito")).sum()),
            aceitos=int((g.camada == "aceito").sum()),
            aceitos_ruins=int((ruins & (g.camada == "aceito")).sum()),
            bons_barrados=int((g.correct & (g.camada != "aceito")).sum()),
            erro_max_aceito_ruim_m=float(g.loc[ruins & (g.camada == "aceito"), "meas_err"].max())
            if (ruins & (g.camada == "aceito")).any() else np.nan))
    R = pd.DataFrame({k: linha(g) for k, g in A.groupby("seq", sort=False)}).T
    R.loc["TOTAL"] = R.sum(numeric_only=True)
    R.loc["TOTAL", "execucoes"] = A["execucao"].nunique()
    R.loc["TOTAL", "erro_max_aceito_ruim_m"] = R["erro_max_aceito_ruim_m"].iloc[:-1].max()
    return R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raiz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", default="completo", help="configuração(ões), separadas por vírgula")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    partes = []
    for cfg in a.configs.split(","):
        A = carregar(a.raiz, cfg)
        if not A.empty:
            A.insert(0, "config", cfg)
            partes.append(A)
    if not partes:
        print("nenhum lacos.csv encontrado")
        return
    A = pd.concat(partes, ignore_index=True)
    A.to_csv(os.path.join(a.out, "lacos.csv"), index=False)
    txt = []
    for cfg, g in A.groupby("config", sort=False):
        R = resumir(g)
        R.to_csv(os.path.join(a.out, f"resumo_{cfg}.csv"))
        txt.append(f"Laços entregues pelo detector ao grafo -- config '{cfg}' (soma das execuções; ruim = erro da "
                   f"distância >= 5 m; grosseiro = > {GROSSEIRO_M:.0f} m)\n")
        txt.append(R.drop(columns="erro_max_aceito_ruim_m").astype(int).to_string())
        txt.append(f"\nmaior erro de um laço ruim aceito: {R.loc['TOTAL', 'erro_max_aceito_ruim_m']:.1f} m\n")
        cols = ["entregues", "ruins", "grosseiros", "barrados_porta", "barrados_gnc", "aceitos", "aceitos_ruins",
                "bons_barrados"]
        tex = R[cols].astype(int)
        with open(os.path.join(a.out, f"tabela_{cfg}.tex"), "w") as f:
            f.write("% gerado por lacos_ponta_a_ponta.py\n\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n")
            f.write("Sequência & Entregues & Ruins & Grosseiros & Porta & PCM/GNC & Aceitos & Aceitos ruins & "
                    "Bons barrados \\\\\n\\midrule\n")
            for k, r in tex.iterrows():
                if k == "TOTAL":
                    f.write("\\midrule\n")
                f.write(f"{k} & " + " & ".join(str(v) for v in r.values) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
    s = "\n".join(txt)
    open(os.path.join(a.out, "tabela.txt"), "w").write(s)
    print(s)


if __name__ == "__main__":
    main()
