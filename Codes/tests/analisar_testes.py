"""
Análise do histórico dos testes isolados (resultados/testes/historico.csv, gravado pelo rodar_testes.py).

Uso:
    python3 analisar_testes.py                        # execuções + comparação da última com a anterior
    python3 analisar_testes.py --a -3 --b -1          # compara duas execuções (índice, id ou rótulo)
    python3 analisar_testes.py --a "antes" --b "fix KLT"
    python3 analisar_testes.py --metrica klt ate      # histórico das métricas cujo nome contém algum filtro
    python3 analisar_testes.py --metrica klt --fig    # ... e salva a figura (resultados/testes/analise_*.pdf)
    python3 analisar_testes.py --latex                # tabela LaTeX das métricas da última execução
    python3 analisar_testes.py --figs                 # figuras da análise em resultados/testes/<execucao B>/analise/
    python3 analisar_testes.py --tol 5                # variação relativa (%) a partir da qual uma métrica é destacada

Execução: índice (-1 = última, 0 = primeira), id (2026-10-29_14-03-12, ou o começo dele) ou rótulo.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "resultados", "testes")
CSV_PATH = os.path.join(RESULTS_DIR, "historico.csv")


def load(path=CSV_PATH) -> pd.DataFrame:
    if not os.path.isfile(path):
        sys.exit(f"{path} não existe: rode antes  python3 {os.path.join(HERE, 'rodar_testes.py')}")
    df = pd.read_csv(path, dtype=dict(execucao=str, commit=str, rotulo=str, metrica=str, unidade=str,
                                      mensagem=str), keep_default_na=False)
    df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
    df['tempo_s'] = pd.to_numeric(df['tempo_s'], errors='coerce')
    return df


def runs_table(df) -> pd.DataFrame:
    """Uma linha por execução: commit, rótulo, nº de testes ok/falhou/pulado, tempo total."""
    per_test = df.drop_duplicates(['execucao', 'teste'])
    g = per_test.groupby('execucao', sort=True)
    t = pd.DataFrame(dict(
        data_hora=g['data_hora'].first(), commit=g['commit'].first(), rotulo=g['rotulo'].first(),
        testes=g.size(), ok=g['status'].apply(lambda s: int((s == 'ok').sum())),
        falhou=g['status'].apply(lambda s: int((s == 'falhou').sum())),
        pulado=g['status'].apply(lambda s: int((s == 'pulado').sum())), tempo_s=g['tempo_s'].sum().round(1)))
    return t.reset_index()


def resolve(runs, key) -> str:
    """Execução por índice (-1), id (ou prefixo do id) ou rótulo (a última com esse rótulo)."""
    ids = list(runs['execucao'])
    try:
        return ids[int(key)]
    except (ValueError, IndexError):
        pass
    by_label = runs[runs['rotulo'] == key]
    if len(by_label):
        return by_label['execucao'].iloc[-1]
    pref = [i for i in ids if i.startswith(key)]
    if len(pref) == 1:
        return pref[0]
    sys.exit(f"execução '{key}' não encontrada (ou ambígua). Execuções: {', '.join(ids)}")


def _fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    a = abs(v)
    if a == 0:
        return "0"
    if a < 1e-3 or a >= 1e5:
        return f"{v:.3e}"
    if float(v).is_integer():
        return f"{int(v)}"
    return f"{v:.4g}"


def _label(runs, ex):
    r = runs[runs['execucao'] == ex].iloc[0]
    return f"{ex} [{r['commit'] or '?'}]" + (f" '{r['rotulo']}'" if r['rotulo'] else "")


def compare(df, runs, a, b, tol_pct, atol=1e-12):
    """Mudanças de status dos testes e variação das métricas entre as execuções a e b."""
    print(f"\nComparação  A = {_label(runs, a)}\n            B = {_label(runs, b)}")
    st = df.drop_duplicates(['execucao', 'teste']).pivot(index='teste', columns='execucao', values='status')
    sa, sb = st.get(a), st.get(b)
    changed = [(t, sa.get(t, ''), sb.get(t, '')) for t in st.index
               if (sa.get(t, '') if sa is not None else '') != (sb.get(t, '') if sb is not None else '')]
    print("\nStatus que mudaram:" if changed else "\nNenhum teste mudou de status.")
    for t, x, y in changed:
        x, y = x if isinstance(x, str) and x else "(ausente)", y if isinstance(y, str) and y else "(ausente)"
        tag = "  <-- QUEBROU" if y == 'falhou' else ("  <-- consertado" if x == 'falhou' and y == 'ok' else "")
        print(f"  {t:70s} {x:>9s} -> {y:<9s}{tag}")
    falhando = [t for t in st.index if sb is not None and sb.get(t) == 'falhou']
    if falhando:
        print(f"\nFalhando em B ({len(falhando)}):")
        for t in falhando:
            m = df[(df['execucao'] == b) & (df['teste'] == t)]['mensagem'].iloc[0]
            print(f"  {t}  -- {m}")

    m = df[df['metrica'] != ''].copy()
    m['chave'] = m['teste'].str.replace('test_', '', regex=False) + " :: " + m['metrica']
    pv = m.pivot_table(index='chave', columns='execucao', values='valor', aggfunc='first')
    if a not in pv or b not in pv:
        print("\n(sem métricas em uma das execuções)")
        return
    unit = m.drop_duplicates('chave').set_index('chave')['unidade']
    va, vb = pv[a], pv[b]
    rows = []
    for k in pv.index:
        x, y = va.get(k), vb.get(k)
        if pd.isna(x) and pd.isna(y):
            continue
        if pd.isna(x) or pd.isna(y):
            rows.append((k, x, y, None, "nova" if pd.isna(x) else "sumiu"))
            continue
        d = y - x
        rel = d / abs(x) * 100 if abs(x) > atol else (0.0 if abs(d) <= atol else np.inf)
        if abs(d) > atol and abs(rel) >= tol_pct:
            rows.append((k, x, y, rel, ""))
    n_tot = int((~va.isna() & ~vb.isna()).sum())
    print(f"\nMétricas com variação >= {tol_pct:g} % ({len(rows)} de {n_tot} comparáveis):")
    if not rows:
        return
    w = max(len(r[0]) for r in rows)
    print(f"  {'métrica':{w}s} {'A':>11s} {'B':>11s} {'var.':>9s}  unid.")
    for k, x, y, rel, note in sorted(rows, key=lambda r: -abs(r[3]) if r[3] is not None and np.isfinite(r[3])
                                      else -np.inf):
        rs = note if note else ("  novo≠0" if not np.isfinite(rel) else f"{rel:+8.1f}%")
        print(f"  {k:{w}s} {_fmt(None if pd.isna(x) else x):>11s} {_fmt(None if pd.isna(y) else y):>11s} "
              f"{rs:>9s}  {unit.get(k, '')}")


def history(df, runs, filtros, fig=False):
    m = df[(df['metrica'] != '')].copy()
    m['chave'] = m['teste'].str.replace('test_', '', regex=False) + " :: " + m['metrica']
    sel = m[m['chave'].apply(lambda c: any(f in c for f in filtros))]
    if sel.empty:
        sys.exit(f"nenhuma métrica com {filtros}. Disponíveis:\n  " + "\n  ".join(sorted(m['chave'].unique())))
    pv = sel.pivot_table(index='chave', columns='execucao', values='valor', aggfunc='first')
    pv = pv[[e for e in runs['execucao'] if e in pv.columns]]
    unit = sel.drop_duplicates('chave').set_index('chave')['unidade']
    short = {e: f"#{i}" for i, e in enumerate(runs['execucao'])}
    print("\nExecuções:", "  ".join(f"{short[e]} = {_label(runs, e)}" for e in pv.columns))
    w = max(len(k) for k in pv.index)
    print(f"\n  {'métrica':{w}s} " + " ".join(f"{short[e]:>10s}" for e in pv.columns) + "  unid.")
    for k, row in pv.iterrows():
        print(f"  {k:{w}s} " + " ".join(f"{_fmt(v) if pd.notna(v) else '-':>10s}" for v in row)
              + f"  {unit.get(k, '')}")
    if fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(pv)
        cols = min(3, n)
        rws = int(np.ceil(n / cols))
        f, axs = plt.subplots(rws, cols, figsize=(4.6 * cols, 2.8 * rws), squeeze=False)
        x = np.arange(pv.shape[1])
        for ax, (k, row) in zip(axs.ravel(), pv.iterrows()):
            ax.plot(x, row.values, "o-", color="#2a78d6", lw=1.4, ms=4)
            ax.set_title(k.split(" :: ")[-1] + (f" [{unit.get(k)}]" if unit.get(k) else ""), fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels([short[e] for e in pv.columns], fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        for ax in axs.ravel()[n:]:
            ax.axis("off")
        f.tight_layout()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        nome = re.sub(r"[^A-Za-z0-9_.-]+", "-", "_".join(filtros))[:60]
        out = os.path.join(RESULTS_DIR, f"analise_{nome}.pdf")
        f.savefig(out, bbox_inches="tight")
        print(f"\nfigura: {out}")


# ---------------------------------------------------------------------- figuras da análise
def _plt():
    sys.path.insert(0, HERE)
    from _comum import nova_figura, COR
    import matplotlib.pyplot as plt
    return nova_figura, COR, plt


def _save(fig, d, nome):
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, nome)
    fig.savefig(p + ".pdf", bbox_inches="tight")
    fig.savefig(p + ".png", bbox_inches="tight", dpi=130)
    print(f"figura: {p}.pdf")


def fig_execucao(df, runs, b, out_dir):
    """Tempo de cada teste, com o status (ok / falhou / pulado) na cor e no rótulo."""
    nova_figura, COR, plt = _plt()
    t = df[df['execucao'] == b].drop_duplicates('teste')
    cor = {'ok': COR['vo'], 'falhou': COR['rejeitado'], 'pulado': COR['neutro']}
    fig, ax = nova_figura(w=7.5, h=0.22 * len(t) + 1.0)
    y = np.arange(len(t))[::-1]
    ax.barh(y, t['tempo_s'], height=0.7, color=[cor.get(x, COR['neutro']) for x in t['status']])
    for yi, (_, r) in zip(y, t.iterrows()):
        tag = {'ok': '', 'falhou': '  FALHOU', 'pulado': '  pulado'}.get(r['status'], r['status'])
        ax.text(r['tempo_s'], yi, f" {r['tempo_s']:.1f} s{tag}", va="center", fontsize=7, color=COR['texto'])
    ax.set_yticks(y)
    ax.set_yticklabels([x.replace("test_", "") for x in t['teste']], fontsize=7)
    ax.set_xlabel("tempo [s]")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, t['tempo_s'].max() * 1.35)
    n = t['status'].value_counts()
    ax.set_title(f"{_label(runs, b)}: {n.get('ok', 0)} ok, {n.get('falhou', 0)} falharam, "
                 f"{n.get('pulado', 0)} pulados", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "execucao_testes")
    plt.close(fig)


def fig_comparacao(df, runs, a, b, tol_pct, out_dir, clip=200.0):
    """Variação relativa (B vs A) das métricas que mudaram >= tol; tempos ficam de fora (variam à toa)."""
    nova_figura, COR, plt = _plt()
    m = df[(df['metrica'] != '') & (df['unidade'] != 's')].copy()
    m['chave'] = m['teste'].str.replace('test_', '', regex=False) + " :: " + m['metrica']
    pv = m.pivot_table(index='chave', columns='execucao', values='valor', aggfunc='first')
    if a not in pv or b not in pv:
        return
    d = pv[[a, b]].dropna()
    rel = (d[b] - d[a]) / d[a].abs().where(d[a].abs() > 1e-12) * 100
    rel = rel[(d[b] - d[a]).abs() > 1e-12].fillna(np.inf)
    rel = rel[rel.abs() >= tol_pct].sort_values()
    st = df.drop_duplicates(['execucao', 'teste']).pivot(index='teste', columns='execucao', values='status')
    mud = [f"{t.replace('test_', '')}: {st.at[t, a]} -> {st.at[t, b]}" for t in st.index
           if a in st and b in st and st.at[t, a] != st.at[t, b]]
    fig, ax = nova_figura(w=8.0, h=0.28 * max(len(rel), 1) + 1.4 + 0.15 * len(mud))
    if len(rel):
        y = np.arange(len(rel))
        v = np.clip(rel.values, -clip, clip)
        ax.barh(y, v, height=0.7, color=[COR['fusao'] if x > 0 else COR['vo'] for x in v])
        for yi, (k, x) in zip(y, rel.items()):
            lab = f"{x:+.0f} %" if np.isfinite(x) else "de 0"
            ax.text(np.clip(x, -clip, clip), yi, f" {lab} " , va="center", fontsize=7, color=COR['texto'],
                    ha="left" if x > 0 else "right")
        ax.set_yticks(y)
        ax.set_yticklabels(rel.index, fontsize=7)
        ax.axvline(0, color=COR['neutro'], lw=0.8)
        lim = min(clip, np.nanmax(np.abs(v[np.isfinite(v)])) if np.isfinite(v).any() else clip) * 1.35
        ax.set_xlim(-lim, lim)
        ax.grid(axis="y", visible=False)
    else:
        ax.text(0.5, 0.5, f"nenhuma métrica variou >= {tol_pct:g} %", ha="center", va="center",
                transform=ax.transAxes, color=COR['texto'])
        ax.set_yticks([])
    ax.set_xlabel(f"variação de B em relação a A [%]  (barras cortadas em ±{clip:g} %)")
    tit = f"A = {_label(runs, a)}\nB = {_label(runs, b)}"
    if mud:
        tit += "".join(f"\nstatus  {x}" for x in mud)
    ax.set_title(tit, fontsize=8.5, loc="left")
    fig.tight_layout()
    _save(fig, out_dir, "comparacao_A_B")
    plt.close(fig)


def fig_historico(df, runs, out_dir):
    """Nº de testes ok / falhou / pulado em cada execução."""
    nova_figura, COR, plt = _plt()
    fig, ax = nova_figura(w=max(4.5, 0.55 * len(runs) + 2), h=3.0)
    x = np.arange(len(runs))
    base = np.zeros(len(runs))
    for col, cor, nome in (("ok", COR['vo'], "ok"), ("falhou", COR['rejeitado'], "falhou"),
                           ("pulado", COR['neutro'], "pulado")):
        v = runs[col].values.astype(float)
        ax.bar(x, v, 0.65, bottom=base, color=cor, label=nome, edgecolor="white", linewidth=1.0)
        base += v
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{i}" + (f"\n{r}" if r else "") for i, r in enumerate(runs['rotulo'])], fontsize=7)
    ax.set_ylabel("nº de testes")
    ax.set_title("Testes por execução")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    _save(fig, out_dir, "historico_execucoes")
    plt.close(fig)


def latex(df, ex):
    """Tabela LaTeX (booktabs) das métricas da execução ex, agrupadas por arquivo de teste."""
    m = df[(df['execucao'] == ex) & (df['metrica'] != '')]
    tex = ["\\begin{tabular}{llr l}", "\\toprule", "Teste & Métrica & Valor & Unid. \\\\", "\\midrule"]
    last_file = None
    for _, r in m.iterrows():
        arq, nome = r['teste'].split("::")
        arq = arq.replace("test_", "").replace("_", "\\_")
        if arq != last_file:
            if last_file is not None:
                tex.append("\\midrule")
            last_file = arq
        nome = nome.replace("test_", "").replace("_", "\\_")
        tex.append(f"{arq} / {nome} & {r['metrica'].replace('_', chr(92) + '_')} & {_fmt(r['valor'])} & "
                   f"{r['unidade']} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    print("\n".join(tex))


def main():
    ap = argparse.ArgumentParser(description="Análise do histórico dos testes isolados")
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--a", help="execução A (padrão: penúltima)")
    ap.add_argument("--b", help="execução B (padrão: última)")
    ap.add_argument("--tol", type=float, default=10.0, help="variação relativa mínima (%%) p/ listar a métrica")
    ap.add_argument("--metrica", nargs="+", help="histórico das métricas cujo nome contém algum destes textos")
    ap.add_argument("--fig", action="store_true", help="com --metrica: salva a figura do histórico")
    ap.add_argument("--latex", action="store_true", help="tabela LaTeX das métricas da execução B (padrão: última)")
    ap.add_argument("--figs", action="store_true", help="figuras da análise em resultados/testes/<B>/analise/")
    ap.add_argument("--ultimas", type=int, default=15, help="quantas execuções listar")
    args = ap.parse_args()

    df = load(args.csv)
    runs = runs_table(df)
    b = resolve(runs, args.b) if args.b else runs['execucao'].iloc[-1]
    if args.latex:
        latex(df, b)
        return
    print(f"Histórico: {args.csv} ({len(runs)} execuções)\n")
    print(runs.tail(args.ultimas).to_string(index=True))
    if args.metrica:
        history(df, runs, args.metrica, fig=args.fig)
        return
    if args.a:
        a = resolve(runs, args.a)
    elif len(runs) >= 2:
        a = runs['execucao'].iloc[list(runs['execucao']).index(b) - 1] if b != runs['execucao'].iloc[0] else None
    else:
        a = None
    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.csv)), b, "analise")
    if args.figs:
        print()
        fig_execucao(df, runs, b, out_dir)
        if len(runs) >= 2:
            fig_historico(df, runs, out_dir)
    if a is None:
        print("\n(só uma execução: nada a comparar; rode os testes de novo depois de mudar o código)")
        return
    compare(df, runs, a, b, args.tol)
    if args.figs:
        fig_comparacao(df, runs, a, b, args.tol, out_dir)


if __name__ == "__main__":
    main()
