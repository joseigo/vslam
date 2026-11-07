"""
Executa os testes isolados SEM depender do pytest (com pytest também funciona: `python3 -m pytest Codes/tests`).

Uso:
    python3 rodar_testes.py                       # todos
    python3 rodar_testes.py gpo laco              # só os arquivos/testes cujo nome contém algum dos filtros
    python3 rodar_testes.py -v                    # imprime a saída de cada teste (tabelas de validação)
    python3 rodar_testes.py --rotulo "fix KLT"    # rótulo da execução no histórico (p/ comparar antes/depois)
    python3 rodar_testes.py --sem-salvar          # não grava nada em resultados/testes_unitarios/
    python3 rodar_testes.py --sem-figuras         # grava o CSV, mas não gera as figuras (mais rápido)

Resultados (Dissert/resultados/testes_unitarios/):
    historico.csv              uma linha por (execução, teste, métrica); acumula todas as execuções
    <execucao>/saida.txt       saída completa da execução (inclusive tracebacks)
    <execucao>/figuras/        figuras dos dados dos testes (.pdf p/ a dissertação + .png)
Análise: python3 analisar_testes.py
"""
import argparse
import csv
import datetime
import glob
import importlib
import io
import os
import subprocess
import sys
import time
import traceback
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _comum                                                     # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(_comum.CODES_DIR), "resultados", "testes_unitarios")
CSV_PATH = os.path.join(RESULTS_DIR, "historico.csv")
CSV_COLS = ["execucao", "data_hora", "commit", "rotulo", "teste", "status", "tempo_s", "metrica", "valor",
            "unidade", "mensagem"]


def git_commit():
    """Hash curto do HEAD (+ '-sujo' se há mudanças não commitadas em Codes/src)."""
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_comum.CODES_DIR, capture_output=True,
                           text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=_comum.CODES_DIR,
                               capture_output=True, text=True).stdout.strip()
        return h + ("-sujo" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return ""


def main():
    ap = argparse.ArgumentParser(description="Testes isolados (dados sintéticos)")
    ap.add_argument("filtros", nargs="*", help="só testes cujo 'arquivo::nome' contém algum destes textos")
    ap.add_argument("-v", "--verbose", action="store_true", help="imprime a saída de cada teste")
    ap.add_argument("--rotulo", default="", help="rótulo da execução no histórico")
    ap.add_argument("--sem-salvar", action="store_true", help="não grava em resultados/testes_unitarios/")
    ap.add_argument("--sem-figuras", action="store_true", help="não gera as figuras dos testes")
    args = ap.parse_args()

    now = datetime.datetime.now()
    execucao = now.strftime("%Y-%m-%d_%H-%M-%S")
    meta = dict(execucao=execucao, data_hora=now.isoformat(timespec="seconds"), commit=git_commit(),
                rotulo=args.rotulo)
    log = io.StringIO()
    run_dir = os.path.join(RESULTS_DIR, execucao)
    _comum.FIG_DIR = None if (args.sem_salvar or args.sem_figuras) else os.path.join(run_dir, "figuras")

    def out(s=""):
        print(s)
        log.write(s + "\n")

    files = sorted(glob.glob(os.path.join(HERE, "test_*.py")))
    n_ok = n_fail = n_skip = 0
    falhas, rows = [], []
    t_all = time.time()
    out(f"execução {execucao}  commit {meta['commit'] or '?'}" + (f"  rótulo '{args.rotulo}'" if args.rotulo else ""))
    for f in files:
        mod_name = os.path.splitext(os.path.basename(f))[0]
        mod = importlib.import_module(mod_name)
        tests = [(n, getattr(mod, n)) for n in sorted(vars(mod)) if n.startswith("test_") and callable(getattr(mod, n))]
        tests.sort(key=lambda nt: nt[1].__code__.co_firstlineno)
        for name, fn in tests:
            full = f"{mod_name}::{name}"
            if args.filtros and not any(a in full for a in args.filtros):
                continue
            buf = io.StringIO()
            _comum.METRICAS.clear()
            msg = ""
            t0 = time.time()
            try:
                with redirect_stdout(buf):
                    fn()
                status, n_ok = "ok", n_ok + 1
            except unittest.SkipTest as e:
                status, n_skip, msg = "pulado", n_skip + 1, str(e)
            except Exception as e:
                status, n_fail = "falhou", n_fail + 1
                msg = f"{type(e).__name__}: {e}".strip().rstrip(":")
                falhas.append((full, traceback.format_exc(), buf.getvalue()))
            dt = time.time() - t0
            linha = f"{status.upper() if status == 'falhou' else status:>8s}  {dt:6.1f} s  {full}"
            out(linha + (f"  -- {msg}" if status == "pulado" else ""))
            texto = buf.getvalue().strip()
            if texto:
                indent = "          " + texto.replace("\n", "\n          ")
                if args.verbose:
                    print(indent)
                log.write(indent + "\n")             # a saída dos testes vai sempre p/ o saida.txt
            base = dict(meta, teste=full, status=status, tempo_s=f"{dt:.3f}", mensagem=msg)
            mets = list(_comum.METRICAS) or [("", None, "")]
            for nome, valor, unidade in mets:      # métricas registradas até o ponto da falha também entram
                rows.append(dict(base, metrica=nome, valor="" if valor is None else repr(valor), unidade=unidade))
    for full, tb, txt in falhas:
        out(f"\n{'=' * 100}\n{full}\n{'-' * 100}")
        if txt.strip():
            out(txt.rstrip())
        out(tb.rstrip())
    out(f"\n{n_ok} ok, {n_fail} falharam, {n_skip} pulados ({time.time() - t_all:.0f} s)")

    if not args.sem_salvar and rows:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "saida.txt"), "w") as fh:
            fh.write(log.getvalue())
        new = not os.path.isfile(CSV_PATH)
        with open(CSV_PATH, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLS)
            if new:
                w.writeheader()
            w.writerows(rows)
        n_fig = len(glob.glob(os.path.join(run_dir, "figuras", "*.pdf")))
        print(f"resultados: {CSV_PATH} (+{len(rows)} linhas), {run_dir}/saida.txt"
              + (f", {n_fig} figuras em {run_dir}/figuras/" if n_fig else ""))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
