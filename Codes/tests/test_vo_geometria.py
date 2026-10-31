"""
VisualOdometry.py: peças geométricas isoladas, com cena sintética (pontos 3D e poses conhecidos).

- _check_triangulation: aceita pontos bons; rejeita atrás da câmera, longe demais (paralaxe), com reprojeção ruim.
- _pnp: pose correta com ruído de 0.5 px e 30% de outliers, partindo de uma predição ruim (RANSAC) e de uma boa.
- _initialize_map: convenção das poses (T_wc) e do mapa na inicialização por matriz essencial (escala = init_scale).
- _prep: a remoção de distorção leva um ponto distorcido ao pixel ideal (mesmo K/distorção).
- _klt: deslocamento sub-pixel; deslocamento grande com o fluxo previsto (init).
- frame_units: unidades acumuladas pelas escalas externas (unit_events).
"""
import numpy as np
import cv2

from _comum import metrica, figuras_ativas, nova_figura, salvar_figura, COR, K_URBAN, DIST_URBAN, IMG_W, IMG_H, LK_PARAMS, make_T, cam_yaw_R, rot_err_deg, textured_image
from VisualOdometry import VisualOdometry


def _vo(**kp):
    return VisualOdometry(K_URBAN, dict(nfeatures=1000), LK_PARAMS, kp, dist_coeffs=None, enable_plot=False,
                          verbose=False, async_ba=False)


def _project(K, T_wc, Xw):
    T_cw = np.linalg.inv(T_wc)
    Xc = Xw @ T_cw[:3, :3].T + T_cw[:3, 3]
    uv = Xc @ K.T
    return uv[:, :2] / uv[:, 2:3], Xc[:, 2]


def _scene(rng, n=300, zmin=6, zmax=40):
    return np.column_stack([rng.uniform(-12, 12, n), rng.uniform(-3, 2, n), rng.uniform(zmin, zmax, n)])


def test_check_triangulation():
    rng = np.random.default_rng(30)
    vo = _vo()
    X = _scene(rng, 200, 5, 20)
    T1 = make_T(t=[1.0, 0, 0.5])                          # câmera 1 (T_wc)
    R1, t1 = np.linalg.inv(T1)[:3, :3], np.linalg.inv(T1)[:3, 3]
    uv0, _ = _project(K_URBAN, np.eye(4), X)
    uv1, _ = _project(K_URBAN, T1, X)
    good, ang = vo._check_triangulation(X, np.eye(3), np.zeros(3), R1, t1, uv0, uv1)
    # sem ruído, só a paralaxe decide: pontos perto do epipolo (direção do movimento) têm paralaxe baixa
    a, b = X, X - T1[:3, 3]
    ang_true = np.degrees(np.arccos(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))))
    clear = np.abs(ang_true - vo.min_tri_angle) > 0.05
    assert np.array_equal(good[clear], ang_true[clear] > vo.min_tri_angle)
    assert good.mean() > 0.8
    # atrás da câmera, longe (paralaxe < min_tri_angle), reprojeção ruim
    Xb = np.array([[0, 0, -5.0], [0.5, 0, 400.0], [1, 0, 10.0]])
    uv0b, _ = _project(K_URBAN, np.eye(4), Xb)
    uv1b, _ = _project(K_URBAN, T1, Xb)
    uv1b[2] += 8.0
    gb, angb = vo._check_triangulation(Xb, np.eye(3), np.zeros(3), R1, t1, uv0b, uv1b)
    metrica("triangulacao_frac_bons", good.mean())
    print(f"bons: {good.mean():.0%} (o resto tem paralaxe < {vo.min_tri_angle}°); paralaxe do ponto a 400 m: {angb[1]:.2f}° (mín {vo.min_tri_angle}°)")
    assert not gb.any()


def _pnp_case(rng, outlier_frac, T_pred_far):
    vo = _vo()
    X = _scene(rng, 250)
    T_true = make_T(cam_yaw_R(np.radians(3.0)), [0.4, -0.05, 2.0])
    uv, z = _project(K_URBAN, T_true, X)
    uv += rng.normal(0, 0.5, uv.shape)
    n_out = int(outlier_frac * len(X))
    out = rng.choice(len(X), n_out, replace=False)
    uv[out] += rng.uniform(-60, 60, (n_out, 2))
    T_pred = np.eye(4) if T_pred_far else T_true @ make_T(cam_yaw_R(np.radians(0.2)), [0.02, 0, 0.03])
    ok, T_est, inl = vo._pnp(X, uv, T_pred)
    assert ok
    e_t = np.linalg.norm(T_est[:3, 3] - T_true[:3, 3])
    e_r = rot_err_deg(T_est[:3, :3], T_true[:3, :3])
    out_mask = np.zeros(len(X), bool)
    out_mask[out] = True
    return e_t, e_r, inl, out_mask


def test_pnp_robusto():
    rng = np.random.default_rng(31)
    for far in (True, False):
        e_t, e_r, inl, out = _pnp_case(rng, 0.3, far)
        tag = "predição ruim (RANSAC)" if far else "predição boa"
        print(f"{tag:24s} erro: {100 * e_t:.1f} cm, {e_r:.3f}°; outliers aceitos {int((inl & out).sum())}/"
              f"{int(out.sum())}, inliers perdidos {int((~inl & ~out).sum())}")
        caso = "ransac" if far else "predicao"
        metrica(f"pnp_{caso}_erro_t", e_t, "m")
        metrica(f"pnp_{caso}_erro_r", e_r, "deg")
        metrica(f"pnp_{caso}_outliers_aceitos", (inl & out).sum())
        assert e_t < 0.05 and e_r < 0.1
        assert (inl & out).sum() <= 0.05 * out.sum()


def test_inicializacao_do_mapa_convencoes():
    """Par inicial sintético: a pose do KF1 (T_wc) aponta na direção do movimento real, |t| = init_scale, e o mapa
    triangulado é a cena real escalada por init_scale / |t_real|."""
    rng = np.random.default_rng(32)
    vo = _vo(init_scale=1.0, min_init_landmarks=40)
    X = _scene(rng, 400, 5, 30)
    T1 = make_T(cam_yaw_R(np.radians(2.0)), [0.3, 0.0, 1.5])
    uv0, _ = _project(K_URBAN, np.eye(4), X)
    uv1, _ = _project(K_URBAN, T1, X)
    inside = (uv1[:, 0] > 10) & (uv1[:, 0] < IMG_W - 10) & (uv1[:, 1] > 10) & (uv1[:, 1] < IMG_H - 10) & \
             (uv0[:, 0] > 10) & (uv0[:, 0] < IMG_W - 10) & (uv0[:, 1] > 10) & (uv0[:, 1] < IMG_H - 10)
    X, uv0, uv1 = X[inside], uv0[inside], uv1[inside] + rng.normal(0, 0.3, (inside.sum(), 2))
    vo.init_ref = dict(pts0=uv0.astype(np.float32), pts=uv1.astype(np.float32), frame_idx=0)
    color = cv2.cvtColor(textured_image(rng, IMG_H, IMG_W), cv2.COLOR_GRAY2BGR)
    assert vo._initialize_map(color, timestamp=1.0, frame_idx=5)
    T_kf1 = vo.keyframes_dict[1].pose
    d_est = T_kf1[:3, 3] / np.linalg.norm(T_kf1[:3, 3])
    d_true = T1[:3, 3] / np.linalg.norm(T1[:3, 3])
    ang = np.degrees(np.arccos(np.clip(d_est @ d_true, -1, 1)))
    e_r = rot_err_deg(T_kf1[:3, :3], T1[:3, :3])
    s = 1.0 / np.linalg.norm(T1[:3, 3])
    P = np.array([p.position for p in vo.Map.points.values()])
    # cada landmark guarda a 1ª observação (KF0): casa com o ponto real pela projeção
    obs0 = np.array([p.observations[0][1] for p in vo.Map.points.values()])
    idx = np.argmin(np.linalg.norm(obs0[:, None] - uv0[None], axis=2), axis=1)
    rel = np.linalg.norm(P - s * X[idx], axis=1) / np.linalg.norm(s * X[idx], axis=1)
    print(f"landmarks: {len(P)}; direção da translação: {ang:.2f}°; rotação: {e_r:.3f}°; |t| = "
          f"{np.linalg.norm(T_kf1[:3, 3]):.3f}; erro relativo mediano do mapa: {100 * np.median(rel):.2f} %")
    metrica("init_direcao_t", ang, "deg")
    metrica("init_rotacao", e_r, "deg")
    metrica("init_mapa_erro_rel_mediano", np.median(rel))
    metrica("init_landmarks", len(P))
    assert abs(np.linalg.norm(T_kf1[:3, 3]) - 1.0) < 1e-6
    assert ang < 2.0 and e_r < 0.2
    assert np.median(rel) < 0.02
    assert np.allclose(vo.keyframes_dict[0].pose, np.eye(4))
    assert len(vo.poses_est) == 1 and np.allclose(vo.poses_est[0], T_kf1)


def test_remocao_de_distorcao():
    """Desenha um ponto na posição DISTORCIDA de pixels ideais conhecidos; depois do _prep ele deve cair no ideal."""
    vo = VisualOdometry(K_URBAN, dict(nfeatures=500), LK_PARAMS, {}, dist_coeffs=DIST_URBAN, enable_plot=False,
                        verbose=False, async_ba=False)
    ideal = np.array([[100.0, 60.0], [1180.0, 500.0], [608.5, 263.5], [1150.0, 80.0]])
    rays = np.column_stack([(ideal - K_URBAN[:2, 2]) / np.diag(K_URBAN)[:2], np.ones(len(ideal))])
    dist_px, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K_URBAN, DIST_URBAN)
    dist_px = dist_px.reshape(-1, 2)
    worst, max_shift = 0.0, 0.0
    for p_ideal, p_dist in zip(ideal, dist_px):
        img = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        cv2.circle(img, (int(round(p_dist[0] * 16)), int(round(p_dist[1] * 16))), 3 * 16, (255, 255, 255), -1,
                   lineType=cv2.LINE_AA, shift=4)
        _, gray = vo._prep(img)
        ys, xs = np.nonzero(gray > 20)
        w = gray[ys, xs].astype(float)
        c = np.array([(xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()])
        worst = max(worst, float(np.linalg.norm(c - p_ideal)))
        max_shift = max(max_shift, float(np.linalg.norm(p_dist - p_ideal)))
    print(f"distorção desloca até {max_shift:.1f} px; após o _prep, erro máx. {worst:.2f} px")
    metrica("distorcao_deslocamento_max", max_shift, "px")
    metrica("distorcao_erro_max", worst, "px")
    assert max_shift > 2.0                               # o teste só vale se a distorção for relevante
    assert worst < 0.5


def _shifted(g0, shift):
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    return cv2.warpAffine(g0, M, (IMG_W, IMG_H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def _inside(pts, shift, m=25):
    q = pts + np.float32(shift)
    return (q[:, 0] > m) & (q[:, 0] < IMG_W - m) & (q[:, 1] > m) & (q[:, 1] < IMG_H - m)


def test_klt_subpixel():
    rng = np.random.default_rng(33)
    vo = _vo()
    g0 = textured_image(rng, IMG_H, IMG_W, n_shapes=800)
    pts = vo._detect(g0, None, 400)
    assert len(pts) > 300
    for shift in ((3.3, -1.7), (25.0, 9.0)):
        p1, ok = vo._klt(g0, _shifted(g0, shift), pts)
        okd = ok & _inside(pts, shift)
        err = np.linalg.norm(p1[okd] - pts[okd] - np.float32(shift), axis=1)
        print(f"deslocamento {shift}: rastreados {okd.sum()}/{_inside(pts, shift).sum()}, "
              f"erro mediano {np.median(err):.3f} px")
        metrica(f"klt_{shift[0]:g}px_taxa", okd.sum() / _inside(pts, shift).sum())
        metrica(f"klt_{shift[0]:g}px_erro_mediano", np.median(err), "px")
        assert okd.sum() >= 0.95 * _inside(pts, shift).sum()
        assert np.median(err) < 0.1


def _taxa_volta_com_chute(vo, g0, g1, pts, init):
    """Mesma checagem forward-backward, mas com p0 como chute na VOLTA (a correção proposta), p/ a figura."""
    p0 = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, np.asarray(init, np.float32).reshape(-1, 1, 2),
                                         flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **vo.lk_params)
    pb, stb, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, p0.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **vo.lk_params)
    fb = np.linalg.norm(p0 - pb, axis=2).ravel()
    return (st.ravel() == 1) & (stb.ravel() == 1) & (fb < vo.fb_thr)


def _fig_klt(dxs, atual, proposta):
    fig, ax = nova_figura(w=5.2, h=3.2)
    ax.plot(dxs, 100 * np.asarray(atual), "o-", color=COR['rejeitado'], ms=5, label="atual (volta sem chute)")
    ax.plot(dxs, 100 * np.asarray(proposta), "s-", color=COR['vo'], ms=5, label="volta com chute p0 (proposta)")
    ax.set_ylim(0, 105)
    ax.set_xlabel("deslocamento horizontal da imagem [px] (com fluxo previsto)")
    ax.set_ylabel("tracks mantidos [%]")
    ax.set_title("KLT + checagem forward-backward")
    ax.legend(loc="lower left")
    fig.tight_layout()
    salvar_figura(fig, "vo_klt_deslocamento_grande")


def test_klt_fluxo_previsto_deslocamento_grande():
    """Curvas: o fluxo previsto (init) deve permitir rastrear deslocamentos de 70-150 px.

    PROBLEMA ENCONTRADO: em _klt_core a volta da checagem forward-backward roda SEM chute inicial; a ida acerta
    100% com o init, mas a volta não acha o ponto e a checagem descarta o track (urban: 39% perdidos a 70 px, 94% a
    150 px). Correção: na volta, usar p0 como chute (flags=cv2.OPTFLOW_USE_INITIAL_FLOW, nextPts=p0.copy())."""
    rng = np.random.default_rng(33)
    vo = _vo()
    g0 = textured_image(rng, IMG_H, IMG_W, n_shapes=800)
    pts = vo._detect(g0, None, 400)
    taxas, proposta = {}, []
    dxs = (20.0, 40.0, 70.0, 100.0, 130.0, 150.0)
    for dx in dxs:
        shift = (dx, 0.35 * dx)
        p1, ok = vo._klt(g0, _shifted(g0, shift), pts, pts + np.float32(shift))
        dentro = _inside(pts, shift)
        okd = ok & dentro
        taxas[dx] = okd.sum() / dentro.sum()
        err = np.linalg.norm(p1[okd] - pts[okd] - np.float32(shift), axis=1)
        print(f"deslocamento {dx:5.0f} px: rastreados {100 * taxas[dx]:5.1f} %, erro mediano "
              f"{np.median(err) if len(err) else np.nan:.3f} px")
        metrica(f"klt_init_{dx:g}px_taxa", taxas[dx])
        if figuras_ativas():
            g1 = _shifted(g0, shift)
            proposta.append((_taxa_volta_com_chute(vo, g0, g1, pts, pts + np.float32(shift)) & dentro).sum()
                            / dentro.sum())
    if figuras_ativas():
        _fig_klt(dxs, [taxas[d] for d in dxs], proposta)
    assert min(taxas.values()) >= 0.9, "checagem forward-backward descarta tracks bons com fluxo previsto grande"


def test_frame_units():
    vo = _vo()
    vo.unit_events = [(10, np.log(2.0), 0), (20, np.log(1.5), 0)]
    assert abs(vo.frame_units(5) - 1.0) < 1e-12
    assert abs(vo.frame_units(10) - 2.0) < 1e-12
    assert abs(vo.frame_units(19) - 2.0) < 1e-12
    assert abs(vo.frame_units(25) - 3.0) < 1e-12
    vo.unit_events.append((30, np.log(0.5), 1))           # o cache (unit_arrays) tem que ver o evento novo
    assert abs(vo.frame_units(31) - 1.5) < 1e-12
