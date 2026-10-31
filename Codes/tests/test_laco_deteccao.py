"""
LoopClosure.py: reconhecimento de lugar e verificação geométrica isolados.

- hamming: igual à contagem de bits bruta.
- bow_score / BowDatabase: mesma imagem = 1, disjuntos = 0, índice invertido = score direto, respeita n_max.
- OrbVocabulary: treino pequeno determinístico; save/load preserva as palavras.
- Cena sintética RENDERIZADA (corredor de 4 paredes texturizadas, poses conhecidas):
    * _verify: pose relativa T_cq do laço confere com a verdade (F + triangulação com vizinhos + PnP);
      um corredor diferente (outra textura) é rejeitado;
    * fluxo completo (add_keyframe -> ilhas -> consistência temporal -> verificação) dispara o laço só na
      k-ésima consulta consistente, com a pose certa.
"""
import os
import tempfile

import numpy as np
import cv2

from _comum import metrica, figuras_ativas, nova_figura, salvar_figura, COR, K_URBAN, IMG_W, IMG_H, VOCAB_DIR, make_T, cam_yaw_R, rot_err_deg, textured_image, skip
from LoopClosure import _match, hamming, bow_score, BowDatabase, OrbVocabulary, LoopDetector, LoopConfig, make_orb

VOCAB = os.path.join(VOCAB_DIR, "orb_vocab_k10L5_indep.npz")


def test_hamming():
    rng = np.random.default_rng(50)
    A = rng.integers(0, 256, (37, 32), dtype=np.uint8)
    B = rng.integers(0, 256, (23, 32), dtype=np.uint8)
    ref = np.unpackbits(A[:, None, :] ^ B[None, :, :], axis=2).sum(2)
    assert np.array_equal(hamming(A, B, chunk=10), ref)
    assert np.all(np.diag(hamming(A, A)) == 0)


def _bow(words, w):
    w = np.asarray(w, np.float32)
    o = np.argsort(words)
    return np.asarray(words)[o], (w / w.sum())[o]


def test_bow_score_e_banco():
    rng = np.random.default_rng(51)
    a = _bow([1, 5, 9, 20], [1, 2, 3, 4])
    assert abs(bow_score(a, a) - 1.0) < 1e-6
    assert bow_score(a, _bow([2, 3], [1, 1])) == 0.0
    b = _bow([5, 9, 33], [1, 1, 2])
    assert abs(bow_score(a, b) - bow_score(b, a)) < 1e-7
    db = BowDatabase()
    bows = [_bow(rng.choice(200, 30, replace=False), rng.random(30) + 0.1) for _ in range(20)]
    for v in bows:
        db.add(v)
    q = bows[7]
    sc = db.query(q, 20)
    ref = np.array([bow_score(q, v) for v in bows])
    assert np.allclose(sc, ref, atol=1e-6)
    assert int(np.argmax(sc)) == 7
    assert len(db.query(q, 5)) == 5 and np.allclose(db.query(q, 5), ref[:5], atol=1e-6)


def test_vocabulario_treino_save_load():
    rng = np.random.default_rng(52)
    centers = rng.integers(0, 256, (6, 32), dtype=np.uint8)
    D = []                                               # 6 grupos de descritores (centro + poucos bits trocados)
    for c in centers:
        bits = np.unpackbits(np.tile(c, (80, 1)), axis=1)
        flip = rng.random(bits.shape) < 0.03
        D.append(np.packbits(bits ^ flip, axis=1))
    voc = OrbVocabulary.train(D, k=6, L=2, verbose=False)
    voc2 = OrbVocabulary.train(D, k=6, L=2, verbose=False)
    Dall = np.vstack(D)
    w = voc.words(Dall)
    assert np.array_equal(w, voc2.words(Dall))           # determinístico (semente)
    # descritores do mesmo grupo caem (quase todos) no mesmo ramo do 1º nível
    top = w // voc.k
    pure = np.mean([np.bincount(top[i * 80:(i + 1) * 80]).max() / 80 for i in range(6)])
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "v.npz")
        voc.save(p)
        v3 = OrbVocabulary.load(p)
    assert np.array_equal(v3.words(Dall), w)
    a = voc.transform(D[0])
    assert abs(a[1].sum() - 1.0) < 1e-5 and np.all(np.diff(a[0]) > 0)
    print(f"pureza dos grupos no 1º nível: {100 * pure:.0f} %; palavras: {voc.n_words}")
    metrica("pureza_grupos", pure)
    assert pure > 0.95


# ---------------------------------------------------------------------- cena renderizada
def _planes(seed):
    """Corredor (câmera: x dir., y baixo, z frente): paredes x = +-3, chão y = 1.5, teto y = -2, de z = 4 a 60."""
    rng = np.random.default_rng(seed)
    z0, z1 = 4.0, 60.0
    P = []
    for O, a, b, size in (([-3, -2, z0], [0, 0, z1 - z0], [0, 3.5, 0], (2400, 400)),     # parede esq.
                          ([3, -2, z0], [0, 0, z1 - z0], [0, 3.5, 0], (2400, 400)),      # parede dir.
                          ([-3, 1.5, z0], [0, 0, z1 - z0], [6, 0, 0], (2400, 500)),      # chão
                          ([-3, -2, z0], [0, 0, z1 - z0], [6, 0, 0], (2400, 500))):      # teto
        w, h = size
        tex = textured_image(rng, h, w, n_shapes=3000, blur=0.8)
        P.append((np.array(O, float), np.array(a, float) / w, np.array(b, float) / h, tex))
    return P


def _render(planes, T_wc):
    T_cw = np.linalg.inv(T_wc)
    R, t = T_cw[:3, :3], T_cw[:3, 3]
    img = np.zeros((IMG_H, IMG_W), np.uint8)
    for O, a, b, tex in planes:
        H = K_URBAN @ np.column_stack([R @ a, R @ b, R @ O + t])
        warped = cv2.warpPerspective(tex, H, (IMG_W, IMG_H), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full(tex.shape, 255, np.uint8), H, (IMG_W, IMG_H), flags=cv2.INTER_NEAREST)
        img[mask > 0] = warped[mask > 0]
    return img


def _cand_pose(i):
    return make_T(t=[0.0, 0.0, 0.5 * (i - 6)])           # KF 6 na origem, vizinhos a +-1.5 e +-3 m


Q_POSE = make_T(cam_yaw_R(np.radians(4.0)), [0.3, 0.05, 0.8])


def _detector(cfg=None):
    if not os.path.isfile(VOCAB):
        skip(f"vocabulário {VOCAB} não encontrado")
    return LoopDetector(OrbVocabulary.load(VOCAB), K_URBAN, cfg or LoopConfig())


def _entry(det, kf, t, gray):
    kps, desc = det.orb.detectAndCompute(gray, None)
    return dict(kf=kf, t=t, frame=kf, kp=np.array([k.pt for k in kps], np.float32), desc=desc,
                bow=det.voc.transform(desc), T_wc=None, X3=None)


def _add_candidates(det, planes):
    for i in range(13):
        det.add_keyframe(i, float(i), i, _render(planes, _cand_pose(i)))
    for i in range(13):
        det.set_pose(i, _cand_pose(i))


def _pair_panel(ax, img_c, img_q, c, q, n_lines, rng):
    """Candidato em cima, consulta embaixo; linhas entre os casamentos (c, q: (n, 2) px)."""
    ax.imshow(np.vstack([img_c, img_q]), cmap="gray", vmin=0, vmax=255)
    sel = rng.choice(len(c), min(n_lines, len(c)), replace=False) if len(c) else []
    return [(c[i], q[i] + [0, img_c.shape[0]]) for i in sel]


def _fig_corredor(det, planes, q, other_planes, other):
    rng = np.random.default_rng(0)
    cand = det.kfs[6]
    img_c = _render(planes, _cand_pose(6))
    fig, axs = nova_figura(ncols=2, w=5.0, h=4.6)
    for ax, ent, img_q, titulo, cor in ((axs[0], q, _render(planes, Q_POSE), "mesmo lugar: inliers da matriz F",
                                         COR['aceito']),
                                        (axs[1], other, _render(other_planes, Q_POSE),
                                         "outro corredor: casamentos (rejeitado na F)", COR['rejeitado'])):
        qi, ci = _match(ent['desc'], cand['desc'], det.cfg.ratio, det.cfg.max_hamming)
        c, qq = cand['kp'][ci], ent['kp'][qi]
        n_f = 0
        if len(qi) >= 8:
            _, mask = cv2.findFundamentalMat(qq, c, cv2.FM_RANSAC, det.cfg.f_ransac_px, 0.99)
            if mask is not None and ent is q:
                keep = mask.ravel().astype(bool)
                c, qq = c[keep], qq[keep]
            n_f = int(mask.sum()) if mask is not None else 0
        for a, b in _pair_panel(ax, img_c, img_q, c, qq, 60, rng):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=cor, lw=0.7, alpha=0.85)
        ax.set_title(f"{titulo}\n{len(qi)} casamentos ORB, {n_f} inliers da F", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.text(8, 30, "candidato", color="w", fontsize=7.5)
        ax.text(8, img_c.shape[0] + 30, "consulta", color="w", fontsize=7.5)
    fig.tight_layout()
    salvar_figura(fig, "laco_corredor_casamentos")


def test_verificacao_geometrica_pose_do_laco():
    det = _detector()
    planes = _planes(60)
    _add_candidates(det, planes)
    q = _entry(det, 100, 1000.0, _render(planes, Q_POSE))
    r = det._verify(q, 6)
    assert r is not None, det._rec
    T_true = np.linalg.inv(_cand_pose(6)) @ Q_POSE        # pose da consulta no ref. do candidato
    e_t = np.linalg.norm(r['T_cq'][:3, 3] - T_true[:3, 3])
    e_r = rot_err_deg(r['T_cq'][:3, :3], T_true[:3, :3])
    print(f"F inliers {r['n_f']}, pontos 3D {r['n_3d']}, PnP inliers {r['n_pnp']}; erro da pose do laço "
          f"{100 * e_t:.1f} cm / {e_r:.3f}° (deslocamento real {np.linalg.norm(T_true[:3, 3]):.2f} m, 4°)")
    metrica("laco_erro_t", e_t, "m")
    metrica("laco_erro_r", e_r, "deg")
    metrica("f_inliers", r['n_f'])
    metrica("pnp_inliers", r['n_pnp'])
    other_planes = _planes(61)
    other = _entry(det, 101, 1001.0, _render(other_planes, Q_POSE))
    if figuras_ativas():
        _fig_corredor(det, planes, q, other_planes, other)
    assert e_t < 0.08 and e_r < 0.3
    # outro corredor (mesma geometria, outra textura): não pode virar laço
    assert det._verify(other, 6) is None
    print(f"corredor diferente rejeitado em '{det._rec['stage']}' (casamentos {det._rec['n_match']}, "
          f"F inliers {det._rec['n_f']})")
    metrica("outro_corredor_f_inliers", det._rec['n_f'])


def test_fluxo_completo_consistencia_temporal():
    """3 consultas seguidas no mesmo lugar (temporal_k = 3): só a 3ª dispara o laço; o candidato é o KF certo."""
    cfg = LoopConfig(min_kf_gap=0)                       # só 13 KFs de candidato (no carro: >= 100 KFs)
    det = _detector(cfg)
    planes = _planes(62)
    _add_candidates(det, planes)
    out = []
    for k, dz in enumerate((0.4, 0.6, 0.8)):
        T = make_T(cam_yaw_R(np.radians(3.0)), [0.2, 0.0, dz])
        out.append(det.add_keyframe(100 + k, 1000.0 + k, 100 + k, _render(planes, T)))
    stages = [rec['stage'] for rec in det.stats['log'][-3:]]
    print(f"etapas das 3 consultas: {stages}; candidato do laço: KF {out[-1]['c_kf'] if out[-1] else None}")
    assert out[0] is None and out[1] is None and out[2] is not None
    assert stages[:2] == ['inconsistente', 'inconsistente'] and stages[2] == 'verificado'
    assert abs(out[2]['c_kf'] - 7.6) <= 2                # KF mais próximo de z = 0.8 m é o 7.6 (6 + 0.8/0.5)


def test_sem_candidato_antes_do_intervalo_minimo():
    """Revisitar logo em seguida (< min_time_gap) não é laço (carro parado / andando devagar)."""
    det = _detector()
    planes = _planes(63)
    _add_candidates(det, planes)
    r = det.add_keyframe(50, 13.0, 50, _render(planes, _cand_pose(6)))
    assert r is None and det.stats['log'][-1]['stage'] == 'sem_elegivel'
