"""
Detecção de fechamento de laço (loop closure) para o VO monocular + grafo de poses (GPO / Fusion).

Pipeline (técnicas da literatura):
    1. Reconhecimento de lugar: bag of binary words, DBoW2
           Gálvez-López & Tardós, "Bags of Binary Words for Fast Place Recognition in Image Sequences", T-RO 2012.
       - vocabulário em árvore (k-majority + k-means++ hierárquico) sobre descritores ORB, pesos tf-idf, score L1;
       - o BoW só RANQUEIA (limiar baixo, como no VINS-Mono); quem decide é a geometria. Medido no urban32 em
         revisitas genuínas (>200 m rodados entre as passagens): o 1º do ranking é o lugar certo em 87% das consultas,
         mas o score bruto (0.045-0.11) se sobrepõe ao do melhor falso (p99 0.077), e o score normalizado do DBoW2,
         eta = s(q, c) / s(q, q-1), também não separa (s(q, q-1) é baixo com o carro rápido). Já a matriz fundamental
         separa bem: 52-285 inliers nos verdadeiros x <= 25 nos falsos;
       - candidatos só entre KFs antigos em tempo E em nº de KFs (carro parado no semáforo não gera "laço");
       - "ilhas" (candidatos próximos no tempo são agrupados) e consistência temporal (k consultas seguidas).
    2. Verificação geométrica (como no VINS-Mono: Qin, Li & Shen, T-RO 2018):
       - 2D-2D: matriz fundamental com RANSAC (descarta casamentos de aparência sem geometria);
       - 3D-2D: os ORB do candidato são triangulados com um KF vizinho (poses do VO), como os map points do ORB-SLAM,
         e o PnP RANSAC usa os casamentos que passaram na F -> pose relativa T_cq (unidades do VO na região do
         candidato). (Descritores calculados nos cantos rastreados pelo KLT não servem: só no nível 0 da pirâmide,
         mudam demais com o avanço do carro -- Hamming mediano 68-121 p/ o MESMO ponto a 3-8 KFs.)
    A pose relativa vai para o grafo (GPO.add_loop): fator 4-DoF com escala do VO; porta de deriva + PCM online e
    GNC-TLS no lote final.

Roda em OUTRO processo (não trava o VO). O processo principal só manda, por keyframe, a imagem cinza e, alguns KFs
depois (já refinada pelo BA), a pose do KF.

Vocabulário: treinar uma vez, de preferência em OUTRAS sequências (não na de teste):
    python3 LoopClosure.py train --images .../urban26/stereo_left .../urban27/stereo_left --out orb_vocab_k10L5.npz
"""
import argparse
import glob
import multiprocessing as mp
import os
import queue
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import cv2

_POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)     # popcount de 1 byte


def hamming(A: np.ndarray, B: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """Distâncias de Hamming entre descritores binários (N,32) x (M,32) uint8 -> (N,M)."""
    out = np.empty((len(A), len(B)), np.int32)
    for s in range(0, len(A), chunk):
        out[s:s + chunk] = _POP[A[s:s + chunk, None, :] ^ B[None, :, :]].sum(-1, dtype=np.int32)
    return out


def make_orb(n_features=1000):
    """Mesmos parâmetros do ORB-SLAM / DBoW2 (8 níveis, fator 1.2)."""
    return cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8, edgeThreshold=31, fastThreshold=20)


# ====================================================================== vocabulário (DBoW2)
def _kmajority(D, k, iters, rng, fit_max):
    """k-means++ (semente) + k-majority (centróide binário = voto bit a bit) em distância de Hamming."""
    n = len(D)
    if n <= k:
        return D.copy()
    F = D if n <= fit_max else D[rng.choice(n, fit_max, replace=False)]
    first = int(rng.integers(len(F)))
    C = [F[first]]
    dmin = hamming(F, F[first][None])[:, 0].astype(np.float64)
    for _ in range(1, k):
        p = dmin ** 2
        tot = p.sum()
        c = int(rng.integers(len(F))) if tot <= 0 else int(rng.choice(len(F), p=p / tot))
        C.append(F[c])
        dmin = np.minimum(dmin, hamming(F, F[c][None])[:, 0])
    C = np.array(C)
    bits = np.unpackbits(F, axis=1)
    prev = None
    for _ in range(iters):
        lab = hamming(F, C).argmin(1)
        if prev is not None and np.array_equal(lab, prev):
            break
        prev = lab
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = np.packbits(bits[m].mean(0) >= 0.5)
    return C


class OrbVocabulary:
    """Árvore de vocabulário k^L (DBoW2). Nível l tem k^(l+1) nós; filhos do nó n são n*k + (0..k-1)."""

    def __init__(self, k, L, cent, valid, idf):
        self.k, self.L, self.cent, self.valid, self.idf = k, L, cent, valid, idf

    @property
    def n_words(self):
        return int(self.valid[-1].sum())

    @classmethod
    def train(cls, image_desc: List[np.ndarray], k=10, L=5, iters=10, fit_max=200000, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        D = np.vstack(image_desc)
        cent = [np.zeros((k ** (l + 1), 32), np.uint8) for l in range(L)]
        valid = [np.zeros(k ** (l + 1), bool) for l in range(L)]
        frontier = [(0, np.arange(len(D)))]
        for l in range(L):
            t0 = time.time()
            nxt = []
            for node, idx in frontier:
                C = _kmajority(D[idx], k, iters, rng, fit_max)
                lab = hamming(D[idx], C).argmin(1)
                for j in range(len(C)):
                    sub = idx[lab == j]
                    if len(sub) == 0:
                        continue
                    cid = node * k + j
                    cent[l][cid], valid[l][cid] = C[j], True
                    nxt.append((cid, sub))
            frontier = nxt
            if verbose:
                print(f"  nível {l + 1}/{L}: {len(frontier)} nós ({time.time() - t0:.0f} s)")
        voc = cls(k, L, cent, valid, np.zeros(k ** L, np.float32))
        n_img = np.zeros(k ** L, np.int64)            # nº de imagens em que cada palavra aparece -> idf
        for d in image_desc:
            n_img[np.unique(voc.words(d))] += 1
        N = len(image_desc)
        voc.idf = np.where(n_img > 0, np.log(N / np.maximum(n_img, 1)), np.log(N)).astype(np.float32)
        return voc

    def save(self, path):
        np.savez_compressed(path, k=self.k, L=self.L, idf=self.idf,
                            **{f"cent{l}": c for l, c in enumerate(self.cent)},
                            **{f"valid{l}": v for l, v in enumerate(self.valid)})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        k, L = int(z["k"]), int(z["L"])
        return cls(k, L, [z[f"cent{l}"] for l in range(L)], [z[f"valid{l}"] for l in range(L)], z["idf"])

    def words(self, D: np.ndarray) -> np.ndarray:
        """Desce a árvore (vetorizado): em cada nível vai p/ o filho de menor Hamming."""
        node = np.zeros(len(D), np.int64)
        ar = np.arange(self.k)
        for l in range(self.L):
            ch = node[:, None] * self.k + ar
            d = _POP[self.cent[l][ch] ^ D[:, None, :]].sum(-1, dtype=np.int32)
            d[~self.valid[l][ch]] = 1 << 20
            node = ch[np.arange(len(D)), d.argmin(1)]
        return node

    def transform(self, D: np.ndarray):
        """Vetor BoW esparso tf-idf normalizado em L1: (palavras ordenadas, pesos)."""
        u, c = np.unique(self.words(D), return_counts=True)
        v = (c / len(D)) * self.idf[u]
        s = v.sum()
        return u, (v / s if s > 0 else v).astype(np.float32)


def bow_score(a, b) -> float:
    """Score L1 do DBoW2 p/ vetores L1-normalizados não negativos: 1 - |a-b|_1 / 2 = soma de min(a_w, b_w)."""
    _, ia, ib = np.intersect1d(a[0], b[0], assume_unique=True, return_indices=True)
    return float(np.minimum(a[1][ia], b[1][ib]).sum())


class BowDatabase:
    """Índice invertido (palavra -> [(entrada, peso)])."""

    def __init__(self):
        self.ids, self.ws = {}, {}
        self.n = 0

    def add(self, bow) -> int:
        e = self.n
        for w, v in zip(bow[0].tolist(), bow[1].tolist()):
            self.ids.setdefault(w, []).append(e)
            self.ws.setdefault(w, []).append(v)
        self.n += 1
        return e

    def query(self, bow, n_max: int) -> np.ndarray:
        """Scores de todas as entradas < n_max."""
        sc = np.zeros(max(n_max, 0), np.float32)
        if n_max <= 0:
            return sc
        for w, v in zip(bow[0].tolist(), bow[1].tolist()):
            ids = self.ids.get(w)
            if ids is None:
                continue
            ids, ws = np.asarray(ids), np.asarray(self.ws[w], np.float32)
            m = ids < n_max
            np.add.at(sc, ids[m], np.minimum(v, ws[m]))
        return sc


# ====================================================================== detector (roda no processo do laço)
@dataclass
class LoopConfig:
    n_features: int = 1000          # ORB por KF (BoW, casamento e pontos 3D)
    min_time_gap: float = 30.0      # s: candidatos só entre KFs mais antigos que isso
    min_kf_gap: int = 100           # ... e com >= N KFs de distância (KFs só nascem com movimento: exclui paradas)
    min_score: float = 0.04         # limiar do score BoW bruto: só descarta o que nem parece (o ranking é o que vale)
    alpha: float = 0.0              # limiar do score normalizado eta do DBoW2 (0 = desligado; ver docstring)
    min_prev_score: float = 0.005   # s(q, q-1) muito baixo = imagem pouco informativa: não consulta
    island_dt: float = 2.0          # s: candidatos mais próximos que isso formam uma ilha
    temporal_k: int = 3             # nº de consultas seguidas com ilhas consistentes (DBoW2)
    temporal_dt: float = 3.0        # s: tolerância de sobreposição entre ilhas de consultas seguidas
    ratio: float = 0.8              # teste da razão de Lowe
    max_hamming: int = 64
    f_ransac_px: float = 2.0
    min_f_inliers: int = 40         # inliers da matriz fundamental (2D-2D); falsos no urban32: <= 25
    tri_neighbors: tuple = (3, -3, 6, -6)   # KFs vizinhos (c + N) usados p/ triangular os ORB do candidato
    tri_max_px: float = 2.0         # erro de reprojeção máx. na triangulação
    tri_min_parallax_deg: float = 0.2   # carro andando p/ frente: paralaxe mediana entre KFs vizinhos ~0.2-0.7°
    pnp_px: float = 3.0
    min_pnp_inliers: int = 20       # inliers do PnP (3D-2D)
    min_loop_gap: float = 2.0       # s entre laços aceitos (evita uma enxurrada de fatores redundantes)
    pose_delay_kf: int = 10         # pose do KF é enviada N KFs depois (já refinada pelo BA)


def _match(qd, cd, ratio, max_ham):
    if qd is None or cd is None or len(qd) < 2 or len(cd) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(qd, cd, k=2)
    qi, ci = [], []
    for m in knn:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance and m[0].distance <= max_ham:
            qi.append(m[0].queryIdx)
            ci.append(m[0].trainIdx)
    return np.array(qi, int), np.array(ci, int)


class LoopDetector:
    def __init__(self, vocab: OrbVocabulary, K: np.ndarray, cfg: LoopConfig = LoopConfig()):
        self.voc, self.K, self.cfg = vocab, np.asarray(K, np.float64), cfg
        self.orb = make_orb(cfg.n_features)
        self.kfs: List[dict] = []
        self.by_id = {}
        self.times: List[float] = []
        self.db = BowDatabase()
        self.consistent = 0
        self.last_island = None
        self.t_last_loop = -np.inf
        self.stats = dict(kfs=0, queries=0, candidates=0, consistent=0, fail_f=0, fail_3d=0, fail_pnp=0, loops=0)
        self.stats['log'] = []          # 1 registro por consulta: onde parou no funil (avaliação com GT depois)
        self._rec = {}

    def add_keyframe(self, kf_id, t, frame, gray) -> Optional[dict]:
        t0 = time.perf_counter()
        kps, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(kps) < 50:
            return None
        e = dict(kf=kf_id, t=t, frame=frame, kp=np.array([k.pt for k in kps], np.float32), desc=desc,
                 bow=self.voc.transform(desc), T_wc=None, X3=None)
        self.stats['kfs'] += 1
        t1 = time.perf_counter()
        n_log = len(self.stats['log'])
        res = self._query(e)
        if len(self.stats['log']) > n_log:          # tempos p/ as métricas de processamento
            self.stats['log'][-1].update(dt_orb=t1 - t0, dt_query=time.perf_counter() - t1, n_kfs=len(self.kfs))
        self.by_id[kf_id] = len(self.kfs)
        self.kfs.append(e)
        self.times.append(t)
        self.db.add(e['bow'])
        return res

    def set_pose(self, kf_id, T_wc):
        i = self.by_id.get(kf_id)
        if i is not None:
            self.kfs[i]['T_wc'] = np.asarray(T_wc, np.float64)

    # ---------------------------------------------------------- DBoW2: consulta + ilhas + consistência temporal
    def _reset(self):
        self.consistent = 0
        self.last_island = None

    def _query(self, q) -> Optional[dict]:
        c = self.cfg
        if not self.kfs:
            return None
        self.stats['queries'] += 1
        self._rec = rec = dict(q_t=q['t'], c_t=np.nan, stage='sem_elegivel', score=np.nan, n_match=0, n_f=0, n_3d=0,
                               n_pnp=0)
        self.stats['log'].append(rec)
        s_prev = bow_score(q['bow'], self.kfs[-1]['bow'])
        n_elig = min(int(np.searchsorted(self.times, q['t'] - c.min_time_gap, side='right')),
                     len(self.kfs) - c.min_kf_gap)
        if n_elig <= 0 or s_prev < c.min_prev_score:
            rec['stage'] = 'sem_elegivel' if n_elig <= 0 else 'pouco_informativa'
            self._reset()
            return None
        sc = self.db.query(q['bow'], n_elig)
        eta = sc / s_prev
        cand = np.where((sc >= c.min_score) & (eta >= c.alpha))[0]
        if not len(cand):
            rec['stage'] = 'sem_candidato'
            self._reset()
            return None
        self.stats['candidates'] += 1
        tc = np.asarray(self.times)[cand]
        islands = np.split(cand, np.where(np.diff(tc) > c.island_dt)[0] + 1)
        best = max(islands, key=lambda isl: sc[isl].sum())
        isl = (self.times[best[0]], self.times[best[-1]])
        if self.last_island is not None and isl[0] <= self.last_island[1] + c.temporal_dt and \
                isl[1] >= self.last_island[0] - c.temporal_dt:
            self.consistent += 1
        else:
            self.consistent = 1
        self.last_island = isl
        cidx = int(best[np.argmax(sc[best])])
        rec.update(c_t=self.times[cidx], score=float(sc[cidx]))
        if self.consistent < c.temporal_k or q['t'] - self.t_last_loop < c.min_loop_gap:
            rec['stage'] = 'inconsistente' if self.consistent < c.temporal_k else 'intervalo_laco'
            return None
        self.stats['consistent'] += 1
        r = self._verify(q, cidx)
        if r is None:
            return None
        r['eta'], r['score'] = float(eta[cidx]), float(sc[cidx])
        rec['stage'] = 'verificado'
        self.t_last_loop = q['t']
        self.stats['loops'] += 1
        return r

    # ---------------------------------------------------------- pontos 3D do candidato (como os map points do ORB-SLAM)
    def _points3d(self, i) -> Optional[np.ndarray]:
        """ORB do KF i triangulados com os KFs vizinhos i +- N (poses do VO). Retorna (n_kp, 3) no referencial da
        câmera i, NaN onde não triangulou. Cacheado."""
        e = self.kfs[i]
        if e['X3'] is not None:
            return e['X3']
        if e['T_wc'] is None:
            return None
        c = self.cfg
        X3 = np.full((len(e['kp']), 3), np.nan)
        for j in (i + d for d in c.tri_neighbors):
            if 0 <= j < len(self.kfs) and self.kfs[j]['T_wc'] is not None:
                self._triangulate(e, self.kfs[j], X3)
        e['X3'] = X3
        return X3

    def _triangulate(self, e, n, X3):
        """Triangula os casamentos ORB e <-> n; preenche X3 (ref. da câmera de e) onde ainda é NaN."""
        c = self.cfg
        a, b = _match(e['desc'], n['desc'], c.ratio, c.max_hamming)
        if len(a) < 8:
            return
        T_ji = np.linalg.inv(n['T_wc']) @ e['T_wc']                           # câmera e -> câmera n
        Xh = cv2.triangulatePoints(self.K @ np.eye(3, 4), self.K @ T_ji[:3],
                                   e['kp'][a].T.astype(np.float64), n['kp'][b].T.astype(np.float64))
        w = np.where(np.abs(Xh[3]) > 1e-12, Xh[3], np.nan)
        X = (Xh[:3] / w).T
        Xj = (T_ji[:3, :3] @ X.T).T + T_ji[:3, 3]

        def reproj(Xc, uv):
            p = (self.K @ Xc.T).T
            return np.linalg.norm(p[:, :2] / p[:, 2:] - uv, axis=1)
        cj = -T_ji[:3, :3].T @ T_ji[:3, 3]                                    # centro da câmera n no ref. de e
        cosp = np.einsum('ij,ij->i', X, X - cj) / (np.linalg.norm(X, axis=1) * np.linalg.norm(X - cj, axis=1) + 1e-12)
        with np.errstate(invalid='ignore'):
            ok = (X[:, 2] > 0) & (Xj[:, 2] > 0)
            ok &= (reproj(X, e['kp'][a]) < c.tri_max_px) & (reproj(Xj, n['kp'][b]) < c.tri_max_px)
            ok &= cosp < np.cos(np.radians(c.tri_min_parallax_deg))
        ok &= np.isnan(X3[a, 0])
        X3[a[ok]] = X[ok]

    # ---------------------------------------------------------- verificação geométrica (VINS-Mono / ORB-SLAM)
    def _verify(self, q, cidx) -> Optional[dict]:
        c = self.cfg
        cnd = self.kfs[cidx]
        qi, ci = _match(q['desc'], cnd['desc'], c.ratio, c.max_hamming)
        rec = self._rec
        rec.update(stage='falha_f', n_match=int(len(qi)))
        if len(qi) < 8:
            self.stats['fail_f'] += 1
            return None
        # F roda mesmo c/ < min_f_inliers casamentos (não muda o resultado: n_f <= casamentos) p/ o diagnóstico
        _, mask = cv2.findFundamentalMat(q['kp'][qi], cnd['kp'][ci], cv2.FM_RANSAC, c.f_ransac_px, 0.99)
        n_f = int(mask.sum()) if mask is not None else 0
        rec['n_f'] = n_f
        if n_f < c.min_f_inliers:
            self.stats['fail_f'] += 1
            return None
        inl_f = mask.ravel().astype(bool)
        qi, ci = qi[inl_f], ci[inl_f]
        X3 = self._points3d(cidx)
        rec['stage'] = 'falha_3d'
        if X3 is None:
            self.stats['fail_3d'] += 1
            return None
        has = np.isfinite(X3[ci, 0])
        rec['n_3d'] = int(has.sum())
        if has.sum() < c.min_pnp_inliers:
            self.stats['fail_3d'] += 1
            return None
        obj, img = X3[ci[has]], q['kp'][qi[has]].astype(np.float64)
        ok, rv, tv, inl = cv2.solvePnPRansac(obj, img, self.K, None, iterationsCount=300, reprojectionError=c.pnp_px,
                                             confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        rec.update(stage='falha_pnp', n_pnp=int(len(inl)) if ok and inl is not None else 0)
        if not ok or inl is None or len(inl) < c.min_pnp_inliers:
            self.stats['fail_pnp'] += 1
            return None
        inl = inl[:, 0]
        rv, tv = cv2.solvePnPRefineLM(obj[inl], img[inl], self.K, None, rv, tv)
        T_qc = np.eye(4)
        T_qc[:3, :3] = cv2.Rodrigues(rv)[0]
        T_qc[:3, 3] = tv[:, 0]
        return dict(q_kf=q['kf'], q_t=q['t'], q_frame=q['frame'], c_kf=cnd['kf'], c_t=cnd['t'],
                    c_frame=cnd['frame'], T_cq=np.linalg.inv(T_qc), n_f=n_f, n_pnp=int(len(inl)), n_3d=int(has.sum()))


# ====================================================================== processo + interface p/ o loop principal
def _loop_proc(q_in, q_out, vocab_path, K, cfg):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        os.nice(5)
    except OSError:
        pass
    cv2.setNumThreads(1)
    det = LoopDetector(OrbVocabulary.load(vocab_path), K, cfg)
    while True:
        msg = q_in.get()
        if msg is None:
            q_out.put(('stats', det.stats))
            break
        if msg[0] == 'kf':
            r = det.add_keyframe(*msg[1:])
            if r is not None:
                q_out.put(('loop', r))
        elif msg[0] == 'pose':
            det.set_pose(*msg[1:])


class LoopClosure:
    """Uso (a cada frame, depois de vo.process_frame_pnp):
        lc.on_frame(vo, gray)
        for loop in lc.poll(): fusion.add_loop(vo, loop)"""

    def __init__(self, K, vocab_path, cfg: LoopConfig = LoopConfig(), process=True):
        self.cfg, self.process = cfg, process
        self._last_kid = None
        self.pending = deque()
        self.stats = {}
        self._units = {}        # frame do KF -> vo.frame_units no envio da pose (escala externa muda as unidades)
        if process:
            ctx = mp.get_context("spawn")
            self.q_in, self.q_out = ctx.Queue(), ctx.Queue()
            self.proc = ctx.Process(target=_loop_proc, args=(self.q_in, self.q_out, vocab_path, np.asarray(K), cfg),
                                    daemon=True)
            self.proc.start()
        else:
            self.det = LoopDetector(OrbVocabulary.load(vocab_path), K, cfg)
            self._out = []

    def _send(self, msg):
        if self.process:
            self.q_in.put(msg)
        elif msg[0] == 'kf':
            r = self.det.add_keyframe(*msg[1:])
            if r is not None:
                self._out.append(r)
        else:
            self.det.set_pose(*msg[1:])

    def on_frame(self, vo, gray):
        kid = vo.cur_kf_id
        if kid is not None and kid != self._last_kid:
            self._last_kid = kid
            kf = vo.keyframes_dict[kid]
            if vo.initialized and vo.lost == 0 and kf.frame_idx == len(vo.poses_est) - 1:
                self._send(('kf', kid, float(kf.timestamp) * 1e-9, kf.frame_idx, gray))
                self.pending.append(kid)
        # pose do KF, N KFs depois (já refinada pelo BA): usada p/ triangular os ORB dele se virar candidato
        while self.pending and vo.kf_counter - 1 - self.pending[0] >= self.cfg.pose_delay_kf:
            self._send_pose(vo, self.pending.popleft())

    def _send_pose(self, vo, kid):
        kf = vo.keyframes_dict.get(kid)
        if kf is not None:
            self._units[kf.frame_idx] = vo.frame_units(kf.frame_idx) if hasattr(vo, 'frame_units') else 1.0
            self._send(('pose', kid, kf.pose.copy()))

    def _tag(self, r):
        r['c_units'] = self._units.get(r['c_frame'], 1.0)      # T_cq está nas unidades do mapa do candidato
        return r

    def poll(self) -> List[dict]:
        if not self.process:
            out, self._out = self._out, []
            return [self._tag(r) for r in out]
        out = []
        while True:
            try:
                kind, r = self.q_out.get_nowait()
            except queue.Empty:
                return out
            if kind == 'loop':
                out.append(self._tag(r))
            else:
                self.stats = r

    def finish(self, vo, timeout=900.0) -> List[dict]:
        """Manda as poses pendentes, espera o processo esvaziar a fila e devolve os laços restantes."""
        while self.pending:
            self._send_pose(vo, self.pending.popleft())
        if not self.process:
            self.stats = self.det.stats
            return self.poll()
        self.q_in.put(None)
        out = []
        t_end = time.time() + timeout
        while time.time() < t_end:
            try:
                kind, r = self.q_out.get(timeout=max(t_end - time.time(), 0.01))
            except queue.Empty:
                break
            if kind == 'loop':
                out.append(self._tag(r))
            else:
                self.stats = r
                break
        self.proc.join(5)
        return out


# ====================================================================== treino do vocabulário (CLI)
def main():
    ap = argparse.ArgumentParser(description="Treina o vocabulário ORB (DBoW2) a partir de pastas de imagens")
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--every", type=int, default=10, help="usa 1 imagem a cada N")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--features", type=int, default=LoopConfig.n_features)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "vocabularios", "orb_vocab_k10L5.npz"))
    args = ap.parse_args()
    orb = make_orb(args.features)
    descs = []
    t0 = time.time()
    for d in args.images:
        paths = sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))[::args.every]
        for p in paths:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            _, desc = orb.detectAndCompute(g, None)
            if desc is not None and len(desc) >= 50:
                descs.append(desc)
        print(f"{d}: {len(paths)} imagens")
    print(f"{len(descs)} imagens, {sum(map(len, descs))} descritores ({time.time() - t0:.0f} s). Treinando "
          f"k={args.k}, L={args.L}...")
    voc = OrbVocabulary.train(descs, k=args.k, L=args.L)
    voc.save(args.out)
    print(f"Vocabulário salvo em {args.out}: {voc.n_words} palavras ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
