"""
GPO.py: rejeição robusta de laços (porta de deriva -> PCM online -> GNC-TLS no lote), com laços falsos injetados.

Cenário: circuito circular (R = 50 m) percorrido 2 vezes; odometria (VO + roda) com deriva de yaw. Laços verdadeiros
entre a 2ª e a 1ª volta + laços falsos ("aliasing": a medida diz que b está no lugar de a, mas está a 6-12 m), cada
um com um deslocamento diferente (não consistentes entre si). Um laço grosseiro (100 m) testa a porta.

Validação para a dissertação: com -v, imprime a tabela de cada camada (verdadeiros aceitos / falsos aceitos).
"""
import numpy as np
from gtsam import Pose3

from _comum import metrica, figuras_ativas, nova_figura, salvar_figura, COR, relative, noisy_rel, ate, make_T, yaw_R
from Fusion import FusionBackend

DT, SPEED, RADIUS = 0.5, 5.0, 50.0


def _circuit(n_laps=2.0):
    w = SPEED / RADIUS
    n = int(n_laps * 2 * np.pi / (w * DT)) + 1
    T = np.tile(np.eye(4), (n, 1, 1))
    for k in range(n):
        th = w * k * DT
        T[k, :3, :3] = yaw_R(th)
        T[k, :3, 3] = [RADIUS * np.sin(th), RADIUS * (1 - np.cos(th)), 0.0]
    return T, int(round(2 * np.pi / (w * DT)))


def _scenario(seed, n_true=10, n_false=6, gross=True):
    rng = np.random.default_rng(seed)
    T, lap = _circuit()
    msgs = [dict(t=0.0, frame=0)]
    for k in range(1, len(T)):
        rel = relative(T[k - 1], T[k])
        vo = noisy_rel(rel, rng, 0.002, 0.01, yaw_bias=0.0015)
        wh = noisy_rel(rel, rng, 0.001, 0.01, yaw_bias=0.0015)
        wh[2, 3] = 0.0
        msgs.append(dict(t=k * DT, frame=k, vo_rel=vo,
                         wheel=(wh, float(np.linalg.norm(wh[:3, 3])), float(np.arctan2(wh[1, 0], wh[0, 0])))))
    loops = []                                           # (mensagem, verdadeiro?)
    bs = np.linspace(lap + 30, len(T) - 5, n_true + n_false).astype(int)
    false_idx = set(rng.choice(np.arange(2, len(bs)), n_false, replace=False).tolist())
    for q, b in enumerate(bs):
        a = b - lap + int(rng.integers(-2, 3))
        rel = noisy_rel(relative(T[a], T[b]), rng, 0.002, 0.0, trans_min=0.1)
        ok = q not in false_idx
        if not ok:                                       # falso: deslocamento lateral/longitudinal e yaw aleatórios
            off = rng.uniform(6, 12) * np.array([np.cos(ang := rng.uniform(0, 2 * np.pi)), np.sin(ang), 0])
            rel = rel @ make_T(yaw_R(np.radians(rng.uniform(-8, 8))), off)
        loops.append((b, dict(type='loop', frame_a=a, frame_b=b, rel=rel, t_a=a * DT, t_b=b * DT, n_pnp=50,
                              n_f=100), ok))
    if gross:
        b = len(T) - 3
        rel = relative(T[b - lap], T[b]) @ make_T(t=[100.0, 0, 0])
        loops.append((b, dict(type='loop', frame_a=b - lap, frame_b=b, rel=rel, t_a=0, t_b=0, n_pnp=50, n_f=100),
                      'grosseiro'))
    return T, msgs, loops


def _run(T, msgs, loops, use_loops=True):
    be = FusionBackend(["vo", "wheel"])
    by_b = {}
    for b, m, ok in loops:
        by_b.setdefault(b, []).append((m, ok))
    status = []
    for m in msgs:
        be.add(m)
        for lm, ok in by_b.get(m['frame'], []) if use_loops else []:
            be.add(lm)
            status.append(ok)
    fin = be.finalize(batch=True)
    return fin, status


def _tabela(loops_out, truth):
    t_ok = [L for L, ok in zip(loops_out, truth) if ok is True]
    f_ok = [L for L, ok in zip(loops_out, truth) if ok is False]
    g_ok = [L for L, ok in zip(loops_out, truth) if ok == 'grosseiro']
    return t_ok, f_ok, g_ok


def _fig_trajetoria(T, fin0, fin, loops, truth):
    """Trajetórias (GT, sem laços, com laços) e um marcador por laço na posição real (GT) do nó b:
    verdadeiro aceito, falso rejeitado, grosseiro barrado na porta (e, se houver, os erros do GNC)."""
    fig, ax = nova_figura(w=5.6, h=5.4)
    P0, P1 = fin0['poses'][:, :3, 3], fin['poses'][:, :3, 3]
    ax.plot(T[:, 0, 3], T[:, 1, 3], color=COR['gt'], lw=1.2, label="GT")
    ax.plot(P0[:, 0], P0[:, 1], color=COR['vo'], ls="--", lw=1.4, label="fusão sem laços")
    ax.plot(P1[:, 0], P1[:, 1], color=COR['fusao'], lw=1.6, label="fusão com laços (lote + GNC)")
    tipos = {}
    for (b, m, ok), L in zip(loops, fin['loops']):
        aceito = L['status'].startswith('aceito')
        if ok == 'grosseiro':
            key, cor, mk = "laço grosseiro (barrado na porta)", COR['neutro'], "^"
        elif ok:
            key, cor, mk = ("laço verdadeiro aceito", COR['aceito'], "o") if aceito else \
                ("laço verdadeiro REJEITADO", COR['rejeitado'], "o")
        else:
            key, cor, mk = ("laço falso rejeitado", COR['rejeitado'], "x") if not aceito else \
                ("laço falso ACEITO", COR['aceito'], "x")
        x, y = T[m['frame_b'], 0, 3], T[m['frame_b'], 1, 3]
        r = np.hypot(x, y - 50.0)                             # empurra o marcador p/ fora do círculo
        f = (r + 7.0) / max(r, 1e-9)
        ax.plot([x, x * f], [y, 50.0 + (y - 50.0) * f], color=cor, lw=0.8)
        ax.plot(x * f, 50.0 + (y - 50.0) * f, mk, color=cor, ms=7, mew=1.6,
                mfc="none" if mk == "o" else cor, label=None if key in tipos else key)
        tipos[key] = 1
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"ATE: sem laços {ate(P0, T[:, :3, 3]):.1f} m, com laços {ate(P1, T[:, :3, 3]):.1f} m")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    salvar_figura(fig, "lacos_trajetoria")


def _fig_pcm(nodes, n_act, n_false, is_true):
    fig, ax = nova_figura(w=6.0, h=2.8)
    ax.step(nodes, n_act, where="post", color=COR['vo'], label="laços ativos (clique do PCM)")
    ax.step(nodes, n_false, where="post", color=COR['rejeitado'], label="falsos ativos")
    for x, ok in zip(nodes, is_true):
        ax.plot(x, -0.6, marker="|" if ok else "x", ms=8, color=COR['aceito'] if ok else COR['rejeitado'], mew=1.5)
    ax.set_ylim(-1.2, max(n_act) + 1)
    ax.set_xlabel("nó do grafo na chegada do laço")
    ax.set_ylabel("nº de laços")
    ax.set_title("PCM online: laços ativos no ISAM2 ao longo da execução")
    ax.plot([], [], "|", color=COR['aceito'], ms=8, mew=1.5, label="chegada de laço verdadeiro")
    ax.plot([], [], "x", color=COR['rejeitado'], ms=7, mew=1.5, label="chegada de laço falso")
    ax.legend(loc="upper left")
    fig.tight_layout()
    salvar_figura(fig, "lacos_pcm_online")


def test_porta_pcm_gnc_rejeitam_lacos_falsos():
    T, msgs, loops = _scenario(seed=20)
    fin0, _ = _run(T, msgs, loops, use_loops=False)
    fin, truth = _run(T, msgs, loops)
    t_ok, f_ok, g_ok = _tabela(fin['loops'], truth)

    def n_status(Ls, pref):
        return sum(L['status'].startswith(pref) for L in Ls)
    porta = lambda Ls: sum(L['status'] == 'porta' for L in Ls)     # noqa: E731
    acc = lambda Ls: n_status(Ls, 'aceito')                         # noqa: E731
    e0 = ate(fin0['poses'][:, :3, 3], T[:, :3, 3])
    e1 = ate(fin['poses'][:, :3, 3], T[:, :3, 3])
    print(f"{'':12s} {'total':>5s} {'porta':>6s} {'aceitos (GNC)':>14s}")
    for nome, Ls in (("verdadeiros", t_ok), ("falsos", f_ok), ("grosseiro", g_ok)):
        print(f"{nome:12s} {len(Ls):5d} {porta(Ls):6d} {acc(Ls):14d}")
    print(f"ATE sem laços {e0:.2f} m -> com laços {e1:.2f} m")
    metrica("verdadeiros_aceitos", acc(t_ok))
    metrica("verdadeiros_total", len(t_ok))
    metrica("falsos_aceitos", acc(f_ok))
    metrica("falsos_total", len(f_ok))
    metrica("grosseiro_barrado_porta", porta(g_ok))
    metrica("ate_sem_lacos", e0, "m")
    metrica("ate_com_lacos", e1, "m")
    if figuras_ativas():
        _fig_trajetoria(T, fin0, fin, loops, truth)
    assert porta(g_ok) == len(g_ok) == 1                 # porta de deriva barra o grosseiro
    assert porta(t_ok) == 0                              # e não barra os verdadeiros
    assert acc(f_ok) == 0                                # nenhum falso aceito no final
    assert acc(t_ok) >= len(t_ok) - 1                    # no máx. 1 verdadeiro perdido
    assert e1 < 0.5 * e0


def test_pcm_online_clique_dominado_por_verdadeiros():
    """Online (antes do lote), os laços ativos no ISAM2 são o clique máximo do PCM.

    Limitação observada (e documentada aqui): o PCM tolera deriva relativa da odometria de loop_pcm_frac (5%) x a
    distância entre os laços; com laços a ~60 m um do outro isso dá ~3 m, e um falso com 6-12 m de deslocamento
    pode ser consistente com a maioria dos verdadeiros e entrar no clique. Ele fica minoria e o GNC-TLS do lote o
    remove (test_porta_pcm_gnc_rejeitam_lacos_falsos), mas a estimativa ONLINE pode carregar um laço falso."""
    T, msgs, loops = _scenario(seed=21, gross=False)
    be = FusionBackend(["vo", "wheel"])
    by_b = {}
    for b, m, ok in loops:
        by_b.setdefault(b, []).append((m, ok))
    truth_by_ab = {(m['frame_a'], m['frame_b']): ok for _, m, ok in loops}
    print(f"{'nó':>4s} {'laço':>6s} {'status':>22s} {'ativos':>6s} {'falsos ativos':>13s}")
    hist, nodes, kinds = [], [], []
    for m in msgs:
        be.add(m)
        for lm, ok in by_b.get(m['frame'], []):
            r = be.add(lm)[0]
            frames = [(be.frames[a], be.frames[b]) for a, b in r['loops_active']]
            n_f = sum(truth_by_ab[f] is False for f in frames)
            hist.append((len(frames), n_f))
            nodes.append(m['frame'])
            kinds.append(ok)
            print(f"{m['frame']:4d} {'verd.' if ok else 'FALSO':>6s} {r['loop']['status']:>22s} {len(frames):6d} "
                  f"{n_f:13d}")
    n_act, n_false = np.array(hist).T
    metrica("ativos_fim", n_act[-1])
    metrica("falsos_ativos_max", n_false.max())
    metrica("falsos_ativos_fim", n_false[-1])
    if figuras_ativas():
        _fig_pcm(nodes, n_act, n_false, kinds)
    assert n_act[-1] >= 6
    assert n_false.max() <= 1                            # nunca mais de um falso ao mesmo tempo
    assert np.all(n_false[n_act >= 4] <= 0.25 * n_act[n_act >= 4])   # falsos sempre minoria no clique


def test_gnc_sozinho_rejeita_falsos():
    """GNC-TLS no lote final parte de TODOS os candidatos (inclusive os fora do clique): sozinho já rejeita."""
    T, msgs, loops = _scenario(seed=22, gross=False)
    fin, truth = _run(T, msgs, loops)
    t_ok, f_ok, _ = _tabela(fin['loops'], truth)
    w_true = np.array([L['w'] for L in t_ok])
    w_false = np.array([L['w'] for L in f_ok])
    print(f"pesos GNC: verdadeiros {np.round(w_true, 2).tolist()}, falsos {np.round(w_false, 2).tolist()}")
    metrica("gnc_frac_verdadeiros_aceitos", np.mean(w_true > 0.5))
    metrica("gnc_peso_max_falsos", w_false.max())
    assert np.all(w_false < 0.5)
    assert np.mean(w_true > 0.5) >= 0.9
