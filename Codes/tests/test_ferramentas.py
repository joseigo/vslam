"""
Ferramentas de avaliação: plano de injeção de laços falsos (Codes/ferramentas/lacos_falsos.py).

Os laços falsos precisam ser de fato falsos (lugar real a > MIN_GT_DIST m do afirmado, >= MIN_DT s de distância) e,
no modo 'grupo', formar blocos de GROUP laços em nós consecutivos (a_k = a0 + k*GROUP_STEP, b_k = b0 + k*GROUP_STEP).
"""
import os
import sys

import numpy as np

from _comum import metrica, CODES_DIR

sys.path.insert(0, os.path.join(CODES_DIR, "ferramentas"))
import lacos_falsos as lf  # noqa: E402


def _trajetoria(n=2000, dt=0.5):
    """Dois quarteirões (retângulo 200 x 100 m) percorridos 2 vezes a 5 m/s: há revisitas verdadeiras."""
    s = 5.0 * dt * np.arange(n)
    per = 600.0
    u = s % per
    x = np.where(u < 200, u, np.where(u < 300, 200, np.where(u < 500, 500 - u, 0)))
    y = np.where(u < 200, 0, np.where(u < 300, u - 200, np.where(u < 500, 100, 600 - u)))
    return dt * np.arange(n), np.c_[x, y, np.zeros(n)]


def test_plano_lacos_falsos_aleatorios_sao_falsos():
    t, P = _trajetoria()
    plan = lf.plan_injections(t, list(range(len(t))), P, 50, "aleatorio", seed=0)
    L = [x for v in plan.values() for x in v]
    d = np.array([np.linalg.norm(P[x['a']] - P[x['b']]) for x in L])
    dt = np.array([t[x['b']] - t[x['a']] for x in L])
    print(f"{len(L)} laços falsos; distância real mín. {d.min():.0f} m; intervalo mín. {dt.min():.0f} s")
    metrica("falsos_planejados", len(L))
    metrica("dist_real_min", d.min(), "m")
    assert len(L) >= 45
    assert d.min() > lf.MIN_GT_DIST
    assert dt.min() >= lf.MIN_DT
    assert all(k == x['b'] for k, v in plan.items() for x in v)          # injetado no nó atual (b)


def test_plano_lacos_falsos_em_grupo():
    t, P = _trajetoria()
    plan = lf.plan_injections(t, list(range(len(t))), P, 50, "grupo", seed=1)
    grupos = {}
    for k, v in plan.items():
        for x in v:
            grupos.setdefault(x['grp'], []).append(x)
            assert k >= x['b']                                          # gatilho depois do último b do grupo
            assert np.linalg.norm(P[x['a']] - P[x['b']]) > lf.MIN_GT_DIST
    print(f"{len(grupos)} grupos: tamanhos {sorted({len(g) for g in grupos.values()})}")
    assert len(grupos) >= 8
    for g in grupos.values():
        assert len(g) == lf.GROUP
        assert np.all(np.diff([x['a'] for x in g]) == lf.GROUP_STEP)
        assert np.all(np.diff([x['b'] for x in g]) == lf.GROUP_STEP)


def test_lacos_ponta_a_ponta_classifica_camadas(tmp_path=None):
    """Cada laço entregue ao grafo cai em exatamente uma camada; 'ruim' = correct False; 'grosseiro' = erro > 50 m."""
    import tempfile
    import pandas as pd
    import lacos_ponta_a_ponta as lp
    raiz = tempfile.mkdtemp()
    linhas = [("aceito (GNC)", "aceito", True, 1.0), ("aceito (GNC)", "aceito", False, 8.0),
              ("porta", "porta", False, 300.0), ("fora do clique (PCM)", "rejeitado", False, 60.0),
              ("rejeitado (GNC)", "rejeitado", True, 2.0), ("ativo", "aceito", True, 0.5)]
    d = os.path.join(raiz, "urbanX", "completo_r1", "dados")
    os.makedirs(d)
    pd.DataFrame([dict(t_a=0, t_b=100, status=s, kind=k, correct=c, stop=False, meas_err=e) for s, k, c, e in linhas]
                 ).to_csv(os.path.join(d, "lacos.csv"), index=False)
    R = lp.resumir(lp.carregar(raiz, "completo")).loc["TOTAL"]
    print(R.to_dict())
    metrica("lacos_classificados", R.entregues)
    assert R.entregues == 6 and R.ruins == 3 and R.grosseiros == 2
    assert R.barrados_porta == 1 and R.barrados_pcm == 1 and R.barrados_gnc == 1
    assert R.aceitos == 3 and R.aceitos_ruins == 1 and R.bons_barrados == 1
    assert R.barrados_porta + R.barrados_pcm + R.barrados_gnc + R.aceitos == R.entregues


def test_sensibilidade_marca_fora_da_faixa():
    """Perturbação só conta como efeito se o ATE sair da faixa [mín, máx] das repetições sem perturbação."""
    import tempfile
    import subprocess
    import pandas as pd
    raiz = tempfile.mkdtemp()
    cols = "metodo,ate_m\n"
    for nome, ate in (("completo_r1", 10.0), ("completo_r2", 12.0), ("completo_r3", 11.0),
                      ("sens_enc_mais2", 11.5), ("sens_altura_mais5", 20.0)):
        d = os.path.join(raiz, "urbanX", nome, "dados")
        os.makedirs(d)
        open(os.path.join(d, "metricas.csv"), "w").write(cols + f"fusão suavizada,{ate}\nodometria de rodas,50\n")
    out = os.path.join(raiz, "out")
    subprocess.check_call([sys.executable, os.path.join(CODES_DIR, "ferramentas", "tabela_sensibilidade.py"), raiz,
                           "--out", out], stdout=subprocess.DEVNULL)
    D = pd.read_csv(os.path.join(out, "sensibilidade.csv")).set_index("config")
    print(D[["ate", "var_pct", "fora_da_faixa"]])
    metrica("var_altura_pct", D.loc["sens_altura_mais5", "var_pct"], "%")
    assert not D.loc["sens_enc_mais2", "fora_da_faixa"]
    assert D.loc["sens_altura_mais5", "fora_da_faixa"]
    assert abs(D.loc["sens_altura_mais5", "var_pct"] - 100 * (20 / 11 - 1)) < 1e-6
