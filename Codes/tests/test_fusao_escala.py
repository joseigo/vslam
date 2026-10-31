"""
Fusion.py / GPO.py: escala do VO como estado do grafo, com dados sintéticos.

- Escala que deriva (senoide) é recuperada pela roda; trajetória fundida melhor que só roda.
- Perda do VO e reinício com OUTRA escala (x10): a escala é re-semeada (n_reseed) e reconvergida.
- BackendProcess: inline (mesma thread) e em outro processo dão o mesmo resultado.
"""
import numpy as np

from _comum import metrica, figuras_ativas, nova_figura, salvar_figura, COR, planar_trajectory, relative, noisy_rel, ate
from Fusion import FusionBackend, BackendProcess

DT = 0.5


def _messages(T, scale_of_t, rng, lost=(), vo_rot=0.002, vo_frac=0.01, wh_rot=0.001, wh_frac=0.01,
              wh_yaw_bias=0.002):
    """Uma mensagem por nó (formato do FusionBackend). VO no ref. do veículo, translação em unidades do VO
    (= metros / escala(t_i)). lost: nós em que o VO não tem medida."""
    msgs = [dict(t=0.0, frame=0)]
    for k in range(1, len(T)):
        rel = relative(T[k - 1], T[k])
        vo = noisy_rel(rel, rng, vo_rot, vo_frac)
        vo[:3, 3] /= scale_of_t((k - 1) * DT)            # o grafo multiplica a aresta i->j por exp(S_i)
        wh = noisy_rel(rel, rng, wh_rot, wh_frac, yaw_bias=wh_yaw_bias)
        wh[2, 3] = 0.0
        msgs.append(dict(t=k * DT, frame=10 * k, vo_rel=None if k in lost else vo,
                         wheel=(wh, float(np.linalg.norm(wh[:3, 3])), float(np.arctan2(wh[1, 0], wh[0, 0])))))
    return msgs


def _run(msgs, sensors=("vo", "wheel"), batch=True):
    be = FusionBackend(list(sensors))
    online = {}
    for m in msgs:
        for r in be.add(m):
            online[r['n']] = r['pose']
    return be, online, be.finalize(batch=batch)


def _dead_reckoning(msgs):
    P = [np.eye(4)]
    for m in msgs[1:]:
        P.append(P[-1] @ m['wheel'][0])
    return np.array(P)


def _scale_err(fin, scale, k0=20):
    s_est = np.exp(fin['log_scales'])
    s_true = np.array([scale(t) for t in fin['t_nodes']])
    return np.abs(np.log(s_est[k0:] / s_true[k0:]))       # k0: depois da inicialização


def _fig_escala(res):
    fig, axs = nova_figura(ncols=len(res), w=3.6, h=2.9, sharey=True)
    for ax, (nome, scale, fin, err, *_r) in zip(axs, res):
        t = np.asarray(fin['t_nodes'])
        ax.plot(t, [scale(x) for x in t], color=COR['gt'], lw=1.4, label="verdadeira")
        ax.plot(t, np.exp(fin['log_scales']), color=COR['fusao'], label="estimada (grafo)")
        ax.set_title(f"{nome}: erro mediano {100 * np.median(err):.1f} %")
        ax.set_xlabel("t [s]")
    axs[0].set_ylabel("escala do VO [m / unid.]")
    axs[0].legend(loc="lower left")
    fig.tight_layout()
    salvar_figura(fig, "fusao_escala_deriva")


def _fig_reinicio(fin, scale, lost):
    fig, ax = nova_figura(w=6.0, h=3.0)
    t = np.asarray(fin['t_nodes'])
    lo, hi = min(lost) * DT, max(lost) * DT
    ax.axvspan(lo, hi, color=COR['grade'], lw=0, label="VO perdido")
    ax.plot(t, [scale(x) for x in t], color=COR['gt'], lw=1.4, label="verdadeira")
    ax.plot(t, np.exp(fin['log_scales']), color=COR['fusao'], label="estimada (grafo)")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.set_yticklabels(["1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlabel("t [s]")
    ax.set_ylabel("escala do VO [m / unid.]")
    ax.set_title(f"Reinício do VO com escala 10x: re-semeaduras = {fin['n_reseed']}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    salvar_figura(fig, "fusao_reinicio_vo")


def test_escala_que_deriva_e_recuperada():
    """Escala constante e deriva lenta (+-15% em 120 s) recuperadas pela roda; fusão melhor que só a roda.
    Deriva rápida (+-25% em 60 s) só é reportada: o passeio aleatório da escala (scale_rw_sigma) contra o sigma de
    translação do VO (vo_trans_frac) define a 'banda' do estimador -- escala que varia mais rápido fica suavizada."""
    n = 240                                              # 2 min, 600 m
    T = planar_trajectory(n)
    casos = [("constante", lambda t: 2.0, 0.01, 0.02),
             ("deriva lenta", lambda t: 2.0 * (1 + 0.15 * np.sin(2 * np.pi * t / 120.0)), 0.015, 0.04),
             ("deriva rápida", lambda t: 2.0 * (1 + 0.25 * np.sin(2 * np.pi * t / 60.0)), None, None)]
    res = []
    for nome, scale, tol_med, tol_max in casos:
        msgs = _messages(T, scale, np.random.default_rng(10))
        be, _, fin = _run(msgs)
        err = _scale_err(fin, scale)
        poses = fin['poses']
        e_fus = ate(poses[:, :3, 3], T[:len(poses), :3, 3])
        e_wh = ate(_dead_reckoning(msgs)[:, :3, 3], T[:, :3, 3])
        res.append((nome, scale, fin, err, poses, e_fus, e_wh, tol_med, tol_max))
        print(f"{nome:14s} erro da escala: mediano {100 * np.median(err):5.2f} %, máx {100 * err.max():5.2f} %  |  "
              f"ATE fusão {e_fus:5.2f} m, só roda {e_wh:5.2f} m")
        slug = nome.replace(" ", "_").replace("á", "a")
        metrica(f"{slug}_escala_erro_mediano", np.median(err))
        metrica(f"{slug}_escala_erro_max", err.max())
        metrica(f"{slug}_ate_fusao", e_fus, "m")
        metrica(f"{slug}_ate_so_roda", e_wh, "m")
    if figuras_ativas():
        _fig_escala(res)
    for nome, scale, fin, err, poses, e_fus, e_wh, tol_med, tol_max in res:
        assert len(poses) == n
        if tol_med is not None:
            assert np.median(err) < tol_med and err.max() < tol_max
            assert e_fus < 0.8 * e_wh


def test_reinicio_do_vo_re_semeia_escala():
    """VO perde o rastreamento (nós 100-104) e volta com escala 10x maior (como no urban38)."""
    rng = np.random.default_rng(11)
    n = 200
    T = planar_trajectory(n)

    def scale(t):
        return 1.5 if t < 100 * DT else 15.0
    lost = set(range(100, 105))
    msgs = _messages(T, scale, rng, lost=lost)
    be, online, fin = _run(msgs)
    s_est = np.exp(fin['log_scales'])
    after = s_est[110:]
    before = s_est[20:100]
    on = np.array([online[k][:3, 3] for k in sorted(online)])
    jump = np.linalg.norm(np.diff(on, axis=0), axis=1).max()
    print(f"re-semeaduras: {fin['n_reseed']}; escala antes {np.median(before):.2f} (1.5), depois "
          f"{np.median(after):.2f} (15.0); maior passo online {jump:.1f} m (passo real 2.5 m)")
    metrica("n_reseed", fin['n_reseed'])
    if figuras_ativas():
        _fig_reinicio(fin, scale, lost)
    metrica("escala_antes", np.median(before))
    metrica("escala_depois", np.median(after))
    metrica("maior_passo_online", jump, "m")
    assert fin['n_reseed'] == 1
    assert abs(np.median(before) / 1.5 - 1) < 0.03
    assert abs(np.median(after) / 15.0 - 1) < 0.03
    assert jump < 10.0                                   # sem salto da pose online no reinício


def test_backend_inline_igual_processo():
    rng = np.random.default_rng(12)
    T = planar_trajectory(80)
    msgs = _messages(T, lambda t: 3.0, rng)
    res = {}
    for inline in (True, False):
        bp = BackendProcess(["vo", "wheel"], inline=inline)
        for m in msgs:
            bp.send(m)
        fin = bp.finalize(batch=True, timeout=120)
        res[inline] = fin
    d = np.abs(res[True]['poses'] - res[False]['poses']).max()
    print(f"diferença máx. inline x processo: {d:.2e}")
    metrica("diff_inline_processo", d, "m")
    assert res[True]['poses'].shape == res[False]['poses'].shape
    assert d < 1e-9
