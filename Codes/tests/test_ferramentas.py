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
