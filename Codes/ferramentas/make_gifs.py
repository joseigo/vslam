"""
GIFs para apresentação a partir de uma pasta de resultados do test_vo_fusion_rt.py (não roda o VO de novo).

Uso:
    python3 Codes/ferramentas/make_gifs.py resultados/urban38              # todos, em resultados/urban38/gifs/
    python3 Codes/ferramentas/make_gifs.py resultados/urban38 --only 1,3   # só alguns

GIFs (cada um só é gerado se os dados existirem em <pasta>/dados/):
    1_trajetoria_ao_vivo.gif     fusão online x só roda x GT ao longo do tempo, laços acontecendo e o erro caindo
    2_laco_antes_depois.gif      o grafo "encaixando" nos laços que mais corrigiram (grafo_lacos.npz)
    3_reconhecimento_de_lugar.gif pares imagem atual x imagem antiga com os pontos casados (matriz F) + mini-mapa
    4_otimizacao_global.gif      trajetória online (causal) virando a suavizada (otimização em lote no fim)
    5_mapa_fundido.gif           mapa do VO no mundo fundido sendo construído, visto de cima (mapa_fundido.npz)
    6_laco_completo.gif          2 + 3 juntos: imagens casando -> grafo encaixando, etapa por etapa

Referencial de todos: o dos CSVs (origem no GT em t0, alinhado no início), em metros.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

COL = dict(gt="#0b0b0b", online="#eb6834", suav="#1baf7a", roda="#eda100", antes="#e34948", depois="#1baf7a",
           laco="#6250d6", q="#2a78d6", c="#e34948")
DPI = 80


# ------------------------------------------------------------------ dados
def load(res_dir):
    D = os.path.join(res_dir, "dados")
    p = lambda n: os.path.join(D, n)
    rd = lambda n: np.loadtxt(p(n), delimiter=",", skiprows=1, ndmin=2) if os.path.isfile(p(n)) else None
    d = dict(dir=res_dir, D=D, online=rd("traj_fusao_online.csv"), suav=rd("traj_fusao_suavizada.csv"),
             roda=rd("traj_so_roda.csv"))
    vo = glob.glob(p("traj_vo*.csv"))
    d["vo"] = np.loadtxt(vo[0], delimiter=",", skiprows=1, ndmin=2) if vo else None
    d["loops"] = list(csv.DictReader(open(p("lacos.csv")))) if os.path.isfile(p("lacos.csv")) else []
    d["exec"] = {}
    if os.path.isfile(p("execucao.txt")):
        for ln in open(p("execucao.txt")):
            if ":" in ln:
                k, v = ln.split(":", 1)
                d["exec"].setdefault(k.strip(), v.strip())
    d["metricas"] = {r["metodo"]: r for r in csv.DictReader(open(p("metricas.csv")))} \
        if os.path.isfile(p("metricas.csv")) else {}
    if d["online"] is None or d["suav"] is None:
        sys.exit(f"Faltam traj_fusao_online.csv / traj_fusao_suavizada.csv em {D}")
    _frame(d)
    return d


def _frame(d):
    """t0 absoluto e a transformação rígida mundo do grafo -> referencial dos CSVs (a partir das poses suavizadas,
    que existem nos dois: suavizado_tum.txt no mundo do grafo e traj_fusao_suavizada.csv já alinhada)."""
    tum = np.loadtxt(os.path.join(d["D"], "suavizado_tum.txt"), ndmin=2)
    tt = tum[:, 0] * (1e-9 if tum[0, 0] > 1e12 else 1.0)
    ts = d["suav"][:, 0]
    best, t0 = -1, tt[0]
    for k in range(min(300, len(tt))):
        j = np.clip(np.searchsorted(tt, ts[:200] + tt[k]), 0, len(tt) - 1)
        n = int(np.sum(np.abs(tt[j] - ts[:200] - tt[k]) < 1e-3))
        if n > best:
            best, t0 = n, tt[k]
    j = np.clip(np.searchsorted(tt, ts + t0 - 1e-4), 0, len(tt) - 1)
    ok = np.abs(tt[j] - ts - t0) < 1e-3
    A, B = tum[j[ok], 1:4], d["suav"][ok, 1:4]
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((B - mb).T @ (A - ma))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    d["t0"], d["R"], d["tv"] = t0, R, mb - R @ ma
    d["to_csv"] = lambda P: np.asarray(P) @ R.T + d["tv"]
    res = np.linalg.norm(d["to_csv"](A) - B, axis=1)
    if np.median(res) > 0.5:
        print(f"[aviso] alinhamento mundo do grafo -> CSV com resíduo mediano {np.median(res):.2f} m")
    o = d["online"]
    d["gt_t"], d["gt"] = o[:, 0], o[:, 4:6]


def gt_at(d, t):
    return np.column_stack([np.interp(t, d["gt_t"], d["gt"][:, i]) for i in range(2)])


def limits(ax, *P, margin=0.06):
    P = np.vstack([np.asarray(q)[:, :2] for q in P if q is not None and len(q)])
    lo, hi = P.min(0), P.max(0)
    c, h = (lo + hi) / 2, (hi - lo).max() / 2 * (1 + margin) + 1.0
    ax.set_xlim(c[0] - h, c[0] + h)
    ax.set_ylim(c[1] - h, c[1] + h)
    ax.set_aspect("equal")


def save(anim, path, fps):
    anim.save(path, writer=PillowWriter(fps=fps), dpi=DPI)
    plt.close(anim._fig)
    print(f"  {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def accepted_loops(d):
    return [L for L in d["loops"] if L.get("kind") == "aceito" and L.get("stop", "False") != "True"]


# ------------------------------------------------------------------ 1: trajetória ao vivo
def gif_trajetoria(d, out, n_frames=180, fps=12, morph_frames=8, min_move=5.0):
    """O carro andando. A trajetória desenhada é a estimativa ATUAL do grafo p/ todo o caminho já percorrido
    (dados/grafo_lacos.npz): quando um laço entra, o trecho passado se ajusta (transição de morph_frames quadros).
    Sem grafo_lacos.npz, usa a trajetória online (causal, com os saltos)."""
    o, roda = d["online"], d["roda"]
    t = o[:, 0]
    G_all = o[:, 4:6]
    path = os.path.join(d["D"], "grafo_lacos.npz")
    snaps = []
    if os.path.isfile(path):
        z = np.load(path)
        nt = z["node_t"] - d["t0"]
        for k in range(len(z["t_b"])):
            snaps.append(dict(tb=float(z["t_b"][k] - d["t0"]), ta=float(z["t_a"][k] - d["t0"]),
                              b=d["to_csv"](z[f"antes_{k}"])[:, :2], a=d["to_csv"](z[f"depois_{k}"])[:, :2]))
        snaps.sort(key=lambda x: x["tb"])
    elif accepted_loops(d):
        print("  (1) sem dados/grafo_lacos.npz: trajetória online causal (com saltos nos laços)")

    def estimate(tn, snap=None, w=1.0):
        """Estimativa do grafo no instante tn: último snapshot (ou a transição dele) + pose online depois dele."""
        if snap is None:
            prev = [x for x in snaps if x["tb"] <= tn]
            snap = prev[-1] if prev else None
        i = np.searchsorted(t, tn, side="right")
        if snap is None:
            return o[:i, 1:3], t[:i]
        P = (1 - w) * snap["b"] + w * snap["a"]
        tp = nt[:len(P)]
        j0 = np.searchsorted(t, tp[-1], side="right")
        return np.vstack([P, o[j0:i, 1:3]]), np.r_[tp, t[j0:i]]

    # linha do tempo: quadros regulares + transição nos laços que movem o grafo
    ts = np.linspace(t[0], t[-1], n_frames)
    timeline = []
    for k, tn in enumerate(ts):
        t_prev = ts[k - 1] if k else -np.inf
        for x in snaps:
            if t_prev < x["tb"] <= tn and np.linalg.norm(x["a"] - x["b"], axis=1).max() >= min_move:
                timeline += [(x["tb"], x, ease(q / (morph_frames - 1))) for q in range(morph_frames)]
        timeline.append((tn, None, 1.0))

    L = sorted(accepted_loops(d), key=lambda L: float(L["t_b"]))
    lt_b = np.array([float(x["t_b"]) for x in L])
    lt_a = np.array([float(x["t_a"]) for x in L])
    regions, last = [], -1e9                                   # 1 marcador por região de laço (>= 20 s)
    for q in range(len(L)):
        if lt_b[q] - last >= 20:
            regions.append(q)
            last = lt_b[q]

    LOOP = "#4b5563"
    fig, (ax, ae) = plt.subplots(1, 2, figsize=(13, 6.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.subplots_adjust(bottom=0.2)
    ax.plot(G_all[:, 0], G_all[:, 1], color="0.88", lw=4, zorder=0)
    gt_l, = ax.plot([], [], color=COL["gt"], lw=1.6, label="GT (referência)", zorder=2)
    rd_l, = ax.plot([], [], color=COL["roda"], lw=1.3, ls=":", label="odometria de rodas (dead reckoning)", zorder=1)
    est_l, = ax.plot([], [], color=COL["suav"], lw=2.4, zorder=3,
                     label="estimativa do grafo (VO + odometria de rodas + laços)" if snaps else "fusão online")
    head, = ax.plot([], [], "o", color=COL["suav"], ms=9, mec="white", mew=1.5, zorder=5)
    lm, = ax.plot([], [], "o", ms=7, mfc="white", mec=LOOP, mew=1.5, ls="none", zorder=1, label="laços aceitos")
    limits(ax, G_all, o[:, 1:3])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ttl = ax.set_title("")
    flash = ax.text(0.5, 0.05, "", transform=ax.transAxes, ha="center", fontsize=12, weight="bold", color="black",
                    bbox=dict(fc="#fff3b0", ec="#c9a400", boxstyle="round,pad=0.4"), zorder=10)
    flash.set_visible(False)

    # painel de erro
    e_on, = ae.plot([], [], color=COL["online"], lw=1.4, label="erro da pose atual (online)")
    e_tr, = ae.plot([], [], color=COL["suav"], lw=2.2, label="erro médio acumulado da trajetória estimada")
    e_rd, = ae.plot([], [], color=COL["roda"], lw=1.6, label="odometria de rodas")
    ymax = 1.5 * max(np.max(o[:, 7]), roda[:, 7].max() if roda is not None else 0.0, 10.0)
    ae.set_yscale("log")                     # roda (centenas de m) e fusão (poucos m) visíveis juntas
    ae.set(xlim=(t[0], t[-1]), ylim=(1.0, ymax), xlabel="t [s]", ylabel="erro de posição vs GT [m] (escala log)")
    ae.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ae.set_title("Erro de posição ao longo do tempo (marcas na base: laços aceitos)", fontsize=11)
    ae.grid(True, alpha=0.3)
    ae.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8.5, frameon=False)
    rug, = ae.plot([], [], "|", color=LOOP, ms=14, mew=1.5, clip_on=False)
    rug_y = 1.0
    hist = dict(t=[], e=[])
    state = dict(flash=-1, shown=0)

    def upd(f):
        tn, snap, w = timeline[f]
        P, tp = estimate(tn, snap, w)
        i = np.searchsorted(t, tn, side="right")
        gt_l.set_data(G_all[:i, 0], G_all[:i, 1])
        est_l.set_data(P[:, 0], P[:, 1])
        head.set_data([P[-1, 0]], [P[-1, 1]])
        if roda is not None:
            j = np.searchsorted(roda[:, 0], tn, side="right")
            rd_l.set_data(roda[:j, 1], roda[:j, 2])
            e_rd.set_data(roda[:j, 0], np.maximum(roda[:j, 7], 1.0))
        e_on.set_data(t[:i], np.maximum(o[:i, 7], 1.0))
        if snap is None:                     # erro médio da trajetória estimada até agora (só em quadros regulares)
            g = gt_at(d, tp)
            hist["t"].append(tn)
            hist["e"].append(float(np.linalg.norm(P - g, axis=1).mean()) if len(P) else 0.0)
            e_tr.set_data(hist["t"], np.maximum(hist["e"], 1.0))
        reg = [q for q in regions if lt_b[q] <= tn]
        if reg:
            gb = gt_at(d, lt_b[reg])
            lm.set_data(gb[:, 0], gb[:, 1])
        n_new = int(np.sum(lt_b <= tn))
        rug.set_data(lt_b[:n_new], np.full(n_new, rug_y))
        if snap is not None:
            mv = float(np.linalg.norm(snap["a"] - snap["b"], axis=1).max())
            flash.set_text(f"FECHAMENTO DE LAÇO: local de t={snap['ta']:.0f} s reconhecido\ncorreção máxima da trajetória: {mv:.0f} m")
            flash.set_visible(True)
            state["flash"] = f + 6
        elif n_new > state["shown"] and not snaps:
            q = n_new - 1
            flash.set_text(f"FECHAMENTO DE LAÇO: local de t={lt_a[q]:.0f} s reconhecido")
            flash.set_visible(True)
            state["flash"] = f + 8
        state["shown"] = n_new
        if f > state["flash"]:
            flash.set_visible(False)
        ttl.set_text(f"t = {tn:5.0f} s | {o[max(i - 1, 0), 8]:5.0f} m percorridos | erro da pose atual "
                     f"{o[max(i - 1, 0), 7]:5.1f} m")
        return []
    save(FuncAnimation(fig, upd, frames=len(timeline)), os.path.join(out, "1_trajetoria_ao_vivo.gif"), fps)


# ------------------------------------------------------------------ laços: grafo antes/depois + imagens
def loop_events(d, n_events, tag):
    """Os n_events laços que mais moveram o grafo (dados/grafo_lacos.npz), em ordem de tempo. Posições no
    referencial dos CSVs; eb/ea = erro médio vs GT antes/depois."""
    path = os.path.join(d["D"], "grafo_lacos.npz")
    if not os.path.isfile(path):
        print(f"  ({tag}) " + ("nenhum laço aceito nesta execução" if not accepted_loops(d) else
                               "sem dados/grafo_lacos.npz: rode o test_vo_fusion_rt.py de novo p/ gerar"))
        return [], None
    z = np.load(path)
    nt = z["node_t"] - d["t0"]
    ev = []
    for k in range(len(z["t_a"])):
        b, a = d["to_csv"](z[f"antes_{k}"]), d["to_csv"](z[f"depois_{k}"])
        g = gt_at(d, nt[:len(b)])
        ta, tb = float(z["t_a"][k] - d["t0"]), float(z["t_b"][k] - d["t0"])
        ev.append(dict(k=k, b=b, a=a, g=g, move=float(np.linalg.norm(a - b, axis=1).max()),
                       eb=float(np.linalg.norm(b[:, :2] - g, axis=1).mean()),
                       ea=float(np.linalg.norm(a[:, :2] - g, axis=1).mean()), ta=ta, tb=tb,
                       ia=int(np.argmin(np.abs(nt[:len(b)] - ta))), ib=int(np.argmin(np.abs(nt[:len(b)] - tb)))))
    ev = sorted(ev, key=lambda e: -e["move"])[:n_events]
    return sorted(ev, key=lambda e: e["tb"]), nt


def image_matcher(d, tag):
    """match(t_atual, t_antiga) -> imagens + casamentos ORB que passam na matriz F (mesmo critério do detector)."""
    import cv2
    from LoopClosure import make_orb, _match, LoopConfig
    img_dir = d["exec"].get("imagens")
    if not img_dir or not os.path.isdir(img_dir):
        print(f"  ({tag}) pasta de imagens não encontrada (execucao.txt)")
        return None
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))
    from GPO import image_stamp_ns
    ft = np.array([image_stamp_ns(f) * 1e-9 for f in files])
    cfg = LoopConfig()
    orb = make_orb(cfg.n_features)

    def img_at(t_rel):
        i = int(np.argmin(np.abs(ft - (t_rel + d["t0"]))))
        g = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
        return g if g.ndim == 2 else cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)

    def match(tq, tc):
        gq, gc = img_at(tq), img_at(tc)
        kq, dq = orb.detectAndCompute(gq, None)
        kc, dc = orb.detectAndCompute(gc, None)
        qi, ci = _match(dq, dc, cfg.ratio, cfg.max_hamming)
        pq = np.float32([kq[i].pt for i in qi]).reshape(-1, 2)
        pc = np.float32([kc[i].pt for i in ci]).reshape(-1, 2)
        inl = np.zeros(len(qi), bool)
        if len(qi) >= 8:
            _, m = cv2.findFundamentalMat(pq, pc, cv2.FM_RANSAC, cfg.f_ransac_px, 0.99)
            if m is not None:
                inl = m.ravel().astype(bool)
        return dict(tq=tq, tc=tc, gq=gq, gc=gc, pq=pq[inl], pc=pc[inl], n_match=len(qi), n_inl=int(inl.sum()))
    return match


def draw_pair(ax, fr, n_lines=70, frac=1.0):
    """Imagem atual | imagem antiga, com uma fração (frac) das linhas de casamento."""
    ax.cla()
    h = min(fr["gq"].shape[0], fr["gc"].shape[0])
    W = fr["gq"].shape[1]
    ax.imshow(np.hstack([fr["gq"][:h], fr["gc"][:h]]), cmap="gray")
    sub = np.random.default_rng(0).permutation(len(fr["pq"]))[:n_lines]
    sub = sub[:int(round(frac * len(sub)))]
    for i in sub:
        ax.plot([fr["pq"][i, 0], fr["pc"][i, 0] + W], [fr["pq"][i, 1], fr["pc"][i, 1]], "-", color="#1baf7a",
                lw=0.8, alpha=0.8)
    if len(sub):
        ax.plot(fr["pq"][sub, 0], fr["pq"][sub, 1], "o", color=COL["q"], ms=3)
        ax.plot(fr["pc"][sub, 0] + W, fr["pc"][sub, 1], "o", color=COL["c"], ms=3)
    ax.set_axis_off()


def draw_graph(ax, e, w, legend=True):
    """Estimativa do grafo interpolada entre antes (w=0, vermelho) e depois (w=1, verde) do laço e."""
    P = (1 - w) * e["b"] + w * e["a"]
    ax.cla()
    ax.plot(e["g"][:, 0], e["g"][:, 1], color="0.6", lw=4, alpha=0.6, label="GT")
    col = np.array(matplotlib.colors.to_rgb(COL["antes"])) * (1 - w) + \
        np.array(matplotlib.colors.to_rgb(COL["depois"])) * w
    ax.plot(P[:, 0], P[:, 1], color=col, lw=2.2, label="estimativa do grafo")
    ia, ib = e["ia"], e["ib"]
    ax.plot([P[ia, 0], P[ib, 0]], [P[ia, 1], P[ib, 1]], "--", color=COL["laco"], lw=2, label="restrição do laço")
    ax.plot(P[[ia, ib], 0], P[[ia, ib], 1], "*", color=COL["laco"], ms=16, mec="white")
    ax.plot(P[-1, 0], P[-1, 1], "o", color=col, ms=10, mec="white", mew=1.5)
    limits(ax, e["g"], e["b"], e["a"])
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(loc="upper left", fontsize=9)


def ease(w):
    w = min(max(w, 0.0), 1.0)
    return w * w * (3 - 2 * w)


# ------------------------------------------------------------------ 2: grafo antes x depois do laço
def gif_laco_antes_depois(d, out, n_events=3, fps=12):
    ev, nt = loop_events(d, n_events, "2")
    if not ev:
        return
    hold0, morph, hold1 = 12, 20, 16
    per = hold0 + morph + hold1
    fig, ax = plt.subplots(figsize=(10, 8))

    def upd(f):
        e = ev[f // per]
        s = f % per
        w = ease((s - hold0) / morph)
        draw_graph(ax, e, w)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fase = "ANTES do laço" if w == 0 else "DEPOIS do laço" if w == 1 else "otimizando o grafo..."
        ax.set_title(f"Laço {ev.index(e) + 1}/{len(ev)}: em t={e['tb']:.0f} s o carro reconhece o lugar de "
                     f"t={e['ta']:.0f} s\n{fase} | erro médio vs GT: {e['eb']:.1f} m -> {e['ea']:.1f} m",
                     fontsize=12)
        return []
    save(FuncAnimation(fig, upd, frames=per * len(ev)), os.path.join(out, "2_laco_antes_depois.gif"), fps)


# ------------------------------------------------------------------ 3: reconhecimento de lugar (imagens)
def gif_reconhecimento(d, out, n_loops=6, sec_per_loop=3.0):
    match = image_matcher(d, "3")
    if match is None:
        return
    L = [L for L in accepted_loops(d) if L.get("correct") == "True"]
    L = sorted(L, key=lambda L: float(L["t_b"]))
    sel, last = [], -1e9                                     # espalhados no tempo (>= 60 s entre si)
    for x in L:
        if float(x["t_b"]) - last >= 60:
            sel.append(x)
            last = float(x["t_b"])
    if len(sel) > n_loops:
        sel = [sel[i] for i in np.linspace(0, len(sel) - 1, n_loops).astype(int)]
    if not sel:
        print("  (3) nenhum laço aceito correto")
        return
    frames = []
    for x in sel:
        fr = match(float(x["t_b"]), float(x["t_a"]))
        fr.update(meas=float(x.get("z_dist") or np.nan), gt=float(x.get("gt_dist") or np.nan))
        frames.append(fr)
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[1, 1.4])
    ai, am, at = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    o = d["online"]

    def upd(k):
        fr = frames[k]
        am.cla(); at.cla()
        draw_pair(ai, fr)
        ai.set_title(f"Quadro de consulta (t = {fr['tq']:.0f} s)            x            quadro candidato (t = {fr['tc']:.0f} s, "
                     f"Δt = {fr['tq'] - fr['tc']:.0f} s)", fontsize=12)
        am.plot(o[:, 4], o[:, 5], color="0.75", lw=2)
        g = gt_at(d, [fr["tq"], fr["tc"]])
        am.plot(*g[0], "o", color=COL["q"], ms=11, mec="white", label="posição atual")
        am.plot(*g[1], "s", color=COL["c"], ms=9, mec="white", label="passagem anterior")
        am.set_aspect("equal")
        am.set_xticks([]); am.set_yticks([])
        am.legend(fontsize=9, loc="best")
        am.set_title("Posição do veículo (trajetória de referência, GT)", fontsize=11)
        at.set_axis_off()
        txt = (f"Laço {k + 1}/{len(frames)}\n\n"
               f"1. BoW: similaridade de aparência entre os quadros\n"
               f"2. {fr['n_match']} correspondências ORB  ->  {fr['n_inl']} inliers\n"
               f"    da geometria epipolar (matriz F, linhas verdes)\n"
               f"3. PnP com pontos 3D -> pose relativa entre os quadros\n"
               f"4. restrição inserida como fator no grafo -> correção da deriva\n\n"
               f"distância medida pelo laço: {fr['meas']:.1f} m (GT: {fr['gt']:.1f} m)")
        at.text(0.02, 0.95, txt, va="top", fontsize=12, family="monospace", transform=at.transAxes)
        return []
    save(FuncAnimation(fig, upd, frames=len(frames)), os.path.join(out, "3_reconhecimento_de_lugar.gif"),
         1.0 / sec_per_loop)


# ------------------------------------------------------------------ 4: online -> suavizada
def gif_otimizacao_global(d, out, fps=12):
    o, s = d["online"], d["suav"]
    t = s[:, 0]
    P0 = np.column_stack([np.interp(t, o[:, 0], o[:, i]) for i in (1, 2)])
    P1 = s[:, 1:3]
    G = s[:, 4:6]
    m = d["metricas"]
    a0 = float(m.get("fusão online", {}).get("ate_m", np.nan))
    a1 = float(m.get("fusão suavizada", {}).get("ate_m", np.nan))
    hold0, morph, hold1 = 15, 30, 20
    fig, ax = plt.subplots(figsize=(10, 8))
    limits(ax, G, P0, P1)

    def upd(f):
        w = 0.0 if f < hold0 else 1.0 if f >= hold0 + morph else (f - hold0) / morph
        w = w * w * (3 - 2 * w)
        P = (1 - w) * P0 + w * P1
        xl, yl = ax.get_xlim(), ax.get_ylim()
        ax.cla()
        ax.plot(G[:, 0], G[:, 1], color="0.6", lw=4, alpha=0.6, label="GT")
        if w > 0:
            ax.plot(P0[:, 0], P0[:, 1], color=COL["online"], lw=1, alpha=0.3)
        col = np.array(matplotlib.colors.to_rgb(COL["online"])) * (1 - w) + np.array(matplotlib.colors.to_rgb(COL["suav"])) * w
        ax.plot(P[:, 0], P[:, 1], color=col, lw=2.2, label="estimativa")
        ax.set_xlim(xl); ax.set_ylim(yl); ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=10)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        if w == 0:
            ttl = f"Fusão online (estimativa causal)  -  ATE {a0:.1f} m"
        elif w == 1:
            ttl = f"Fusão suavizada (otimização em lote com todos os laços)  -  ATE {a1:.1f} m"
        else:
            ttl = "Otimização em lote do grafo completo..."
        ax.set_title(ttl, fontsize=12)
        return []
    save(FuncAnimation(fig, upd, frames=hold0 + morph + hold1), os.path.join(out, "4_otimizacao_global.gif"), fps)


# ------------------------------------------------------------------ 5: mapa fundido sendo construído
def gif_mapa(d, out, n_frames=110, fps=11, max_pts=45000):
    path = os.path.join(d["D"], "mapa_fundido.npz")
    if not os.path.isfile(path):
        print("  (5) sem dados/mapa_fundido.npz: rode o test_vo_fusion_rt.py de novo p/ gerar")
        return
    from scipy.spatial import cKDTree
    z = np.load(path)
    P, C = d["to_csv"](z["points"]), z["colors"] / 255.0
    T = d["to_csv"](z["poses"][:, :3, 3])
    ts_raw = np.asarray(z["timestamps"], float)
    ts_raw = ts_raw * (1e-9 if len(ts_raw) and ts_raw[0] > 1e12 else 1.0)      # VO guarda em ns
    tt = ts_raw - d["t0"] if len(ts_raw) == len(T) else np.linspace(0, d["online"][-1, 0], len(T))
    keep = np.linalg.norm(P[:, :2] - np.median(P[:, :2], 0), axis=1)
    keep = keep <= np.percentile(keep, 98)
    P, C = P[keep], C[keep]
    if len(P) > max_pts:
        idx = np.random.default_rng(0).choice(len(P), max_pts, replace=False)
        P, C = P[idx], C[idx]
    _, near = cKDTree(T[:, :2]).query(P[:, :2])
    pt_t = tt[near]
    order = np.argsort(pt_t)
    P, C, pt_t = P[order], C[order], pt_t[order]
    ts = np.linspace(tt[0], tt[-1], n_frames)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.plot(d["online"][:, 4], d["online"][:, 5], color="0.85", lw=3, zorder=0)
    sc = ax.scatter([], [], s=0.6, c=[], rasterized=True)
    tr, = ax.plot([], [], color="magenta", lw=1.5)
    hd, = ax.plot([], [], "o", color="magenta", ms=8, mec="white")
    limits(ax, T[:, :2], P[:, :2])
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ttl = ax.set_title("")

    def upd(k):
        n = np.searchsorted(pt_t, ts[k], side="right")
        i = np.searchsorted(tt, ts[k], side="right")
        sc.set_offsets(P[:n, :2])
        sc.set_color(C[:n])
        tr.set_data(T[:i, 0], T[:i, 1])
        if i:
            hd.set_data([T[i - 1, 0]], [T[i - 1, 1]])
        ttl.set_text(f"Mapa 3D do VO no referencial da fusão (vista superior)  |  t = {ts[k]:.0f} s  |  {n} landmarks")
        return []
    save(FuncAnimation(fig, upd, frames=n_frames), os.path.join(out, "5_mapa_fundido.gif"), fps)


# ------------------------------------------------------------------ 6: laço completo (imagens -> grafo)
def gif_laco_completo(d, out, n_events=3, fps=8):
    """Para cada um dos laços que mais corrigiram: (1) imagem atual x antiga, (2) casamentos aparecendo,
    (3) o grafo encaixando, (4) resultado -- com um painel que destaca a etapa."""
    ev, _ = loop_events(d, n_events, "6")
    match = image_matcher(d, "6") if ev else None
    if not ev or match is None:
        return
    for e in ev:
        e["fr"] = match(e["tb"], e["ta"])
    A, B, C, Dh = 6, 8, 14, 10                   # quadros: reconhece | casa pontos | otimiza | resultado
    per = A + B + C + Dh
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.35], width_ratios=[1.5, 1])
    ai, ag, at = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    steps = ["1. reconhecimento de local (BoW):\n   similaridade de aparência com quadro anterior",
             "2. verificação geométrica:\n   inliers da geometria epipolar (matriz F)\n   + PnP -> pose relativa entre os quadros",
             "3. inserção do fator de laço no grafo:\n   reotimização incremental (iSAM2)",
             "4. correção da deriva acumulada"]

    def upd(f):
        n, s = divmod(f, per)
        e, fr = ev[n], ev[n]["fr"]
        stage = 0 if s < A else 1 if s < A + B else 2 if s < A + B + C else 3
        draw_pair(ai, fr, frac=0.0 if stage == 0 else ease((s - A) / B) if stage == 1 else 1.0)
        ai.set_title(f"Laço {n + 1}/{len(ev)}   |   quadro de consulta (t = {e['tb']:.0f} s)        x        quadro candidato "
                     f"(t = {e['ta']:.0f} s, Δt = {e['tb'] - e['ta']:.0f} s)", fontsize=12)
        w = ease((s - A - B) / C) if stage >= 2 else 0.0
        draw_graph(ag, e, w)
        ag.set_xlabel("x [m]")
        ag.set_ylabel("y [m]")
        ag.set_title("grafo antes do fechamento de laço" if w == 0 else "grafo após o fechamento de laço" if w == 1 else "otimização em andamento...",
                     fontsize=12, color=COL["antes"] if w == 0 else COL["depois"] if w == 1 else "0.2")
        at.cla()
        at.set_axis_off()
        y = 0.97
        for k, txt in enumerate(steps):
            on = k == stage
            at.text(0.0, y, txt, va="top", fontsize=12.5 if on else 11, family="monospace", transform=at.transAxes,
                    color="black" if on else "0.65", weight="bold" if on else "normal",
                    bbox=dict(fc="#eef6ff", ec=COL["q"], boxstyle="round,pad=0.4") if on else None)
            y -= 0.2
        info = f"{fr['n_match']} correspondências ORB -> {fr['n_inl']} inliers"
        if stage == 3:
            info += f"\nerro médio vs GT: {e['eb']:.1f} m -> {e['ea']:.1f} m"
        at.text(0.0, 0.08, info, va="bottom", fontsize=12, family="monospace", transform=at.transAxes,
                color=COL["laco"], weight="bold")
        return []
    save(FuncAnimation(fig, upd, frames=per * len(ev)), os.path.join(out, "6_laco_completo.gif"), fps)


def main():
    ap = argparse.ArgumentParser(description="GIFs p/ apresentação a partir de resultados/<dataset>")
    ap.add_argument("res_dir", help="pasta de resultados (ex.: resultados/urban38)")
    ap.add_argument("--only", default="1,2,3,4,5,6", help="quais GIFs (ex.: 1,3)")
    args = ap.parse_args()
    d = load(args.res_dir)
    out = os.path.join(args.res_dir, "gifs")
    os.makedirs(out, exist_ok=True)
    which = {int(x) for x in args.only.split(",") if x.strip()}
    print(f"GIFs em {out}/ (t0 = {d['t0']:.3f})")
    for k, fn in ((1, gif_trajetoria), (2, gif_laco_antes_depois), (3, gif_reconhecimento),
                  (4, gif_otimizacao_global), (5, gif_mapa), (6, gif_laco_completo)):
        if k in which:
            print(f"[{k}] {fn.__name__}")
            fn(d, out)


if __name__ == "__main__":
    main()
