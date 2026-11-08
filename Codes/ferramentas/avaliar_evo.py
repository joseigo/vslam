"""
Avaliação com o evo (https://github.com/MichaelGrupp/evo): APE (= ATE) e RPE, como nos artigos.

Uso:
    python3 avaliar_evo.py resultados/urban38           # -> resultados/urban38/evo/
    python3 avaliar_evo.py resultados/urban38 --delta 50 # RPE em segmentos de 50 m

Entrada: resultados/<dataset>/evo/*.txt em TUM padrão (tempo em s): gt.txt + um arquivo por método, todos no
referencial do VEÍCULO (gravados pelo rodar_sistema.py). Se a pasta não existir (execuções antigas), monta a
partir de dados/{online,suavizado,vo}_tum.txt e do GT do dataset.

Saída em resultados/<dataset>/evo/:
    resumo_evo.csv            APE (rmse, média, mediana, desvio, máx) e RPE (trans. %, rot. °/100 m) por método
    ape_<método>.zip          resultado do evo (abre com: evo_res ape_*.zip)
    rpe_<método>.zip
    evo_trajetorias.pdf       trajetórias alinhadas ao GT (vista de cima)
    evo_ape_<método>.pdf      erro APE colorido sobre a trajetória + APE no tempo
Alinhamento: SE(3) (Umeyama sem escala) p/ os métodos métricos; Sim(3) p/ o VO (escala própria).
"""
import argparse
import copy
import csv
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {"fusao_suavizada": "fusão suavizada", "fusao_online": "fusão online", "so_roda": "odometria de rodas"}
COLORS = {"fusao_suavizada": "#1baf7a", "fusao_online": "#eb6834", "so_roda": "#eda100"}


def label(name):
    return LABELS.get(name, "VO (Sim3)" if name.startswith("vo") else name)


def color(name):
    return COLORS.get(name, "#2a78d6" if name.startswith("vo") else None)


def write_tum(path, t_s, T):
    """TUM padrão: tempo [s] tx ty tz qx qy qz qw."""
    from scipy.spatial.transform import Rotation
    T = np.asarray(T)
    q = Rotation.from_matrix(T[:, :3, :3]).as_quat()
    np.savetxt(path, np.column_stack([np.asarray(t_s, float), T[:, :3, 3], q]),
               fmt=["%.9f"] + ["%.6f"] * 7)


def read_tum(path):
    from evo.core.trajectory import PoseTrajectory3D
    d = np.loadtxt(path, ndmin=2)
    t = d[:, 0] * (1e-9 if d[0, 0] > 1e12 else 1.0)          # aceita também o TUM em ns de dados/
    return PoseTrajectory3D(positions_xyz=d[:, 1:4], orientations_quat_wxyz=d[:, [7, 4, 5, 6]], timestamps=t)


def _fallback(res_dir, evo_dir):
    """Execução antiga (sem evo/): converte dados/*_tum.txt e o GT do dataset p/ TUM em segundos."""
    D = os.path.join(res_dir, "dados")
    src = {"fusao_online": "online_tum.txt", "fusao_suavizada": "suavizado_tum.txt", "vo": "vo_tum.txt"}
    if not any(os.path.isfile(os.path.join(D, f)) for f in src.values()):
        sys.exit(f"Nem {evo_dir}/ nem trajetórias em {D}/")
    imgs = None
    for ln in open(os.path.join(D, "execucao.txt")):
        if ln.startswith("imagens:"):
            imgs = ln.split(":", 1)[1].strip()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "scripts"))
    sys.path.insert(0, os.path.join(here, "..", "src"))
    from fusao_offline import load_gt
    gt = None
    d = os.path.abspath(imgs)
    for _ in range(5):
        for p in ("poses/global_pose.csv", "global_pose.csv", "poses_gt_unix.csv", "data_sensor/poses_gt_unix.csv"):
            if gt is None and os.path.isfile(os.path.join(d, p)):
                gt = os.path.join(d, p)
        d = os.path.dirname(d)
    if gt is None:
        sys.exit("GT do dataset não encontrado")
    os.makedirs(evo_dir, exist_ok=True)
    t_gt, T_gt = load_gt(gt)
    if np.median(np.diff(t_gt)) > 0.05:            # GT esparso (Inomi ~2.7 Hz): interpola nos instantes estimados
        from scipy.spatial.transform import Rotation, Slerp
        ts = np.unique(np.concatenate([read_tum(os.path.join(D, f)).timestamps for f in src.values()
                                       if os.path.isfile(os.path.join(D, f))]))
        ts = ts[(ts >= t_gt[0]) & (ts <= t_gt[-1])]
        T = np.tile(np.eye(4), (len(ts), 1, 1))
        T[:, :3, :3] = Slerp(t_gt, Rotation.from_matrix(T_gt[:, :3, :3]))(ts).as_matrix()
        T[:, :3, 3] = np.column_stack([np.interp(ts, t_gt, T_gt[:, i, 3]) for i in range(3)])
        t_gt, T_gt = ts, T
        if "inomi" in gt.lower():                  # GT da Inomi: eixo 'frente' do corpo = y (ver rodar_sistema)
            T_gt[:, :3, :3] = T_gt[:, :3, :3] @ Rotation.from_euler('z', 90, degrees=True).as_matrix()
    write_tum(os.path.join(evo_dir, "gt.txt"), t_gt, T_gt)
    from GPO import R_VC_NOMINAL
    T_vc = np.eye(4)
    T_vc[:3, :3] = R_VC_NOMINAL
    t_start = read_tum(os.path.join(D, src["fusao_online"])).timestamps[0] \
        if os.path.isfile(os.path.join(D, src["fusao_online"])) else -np.inf
    for name, f in src.items():
        p = os.path.join(D, f)
        if not os.path.isfile(p):
            continue
        tr = read_tum(p)
        T = np.array(tr.poses_se3)
        m = tr.timestamps >= t_start - 1e-6                    # só a partir do início da fusão (VO já iniciado)
        if name == "vo":                                       # câmera (z frente) -> veículo (x frente)
            T = T_vc @ T @ np.linalg.inv(T_vc)
        write_tum(os.path.join(evo_dir, f"{name}.txt"), tr.timestamps[m], T[m])
    print(f"(execução antiga: TUM montados em {evo_dir}/ a partir de dados/ e de {gt})")


def avaliar(res_dir, delta_m=100.0, max_dt=0.02, verbose=True):
    from evo.core import metrics, sync
    from evo.core.metrics import PoseRelation, Unit
    from evo.tools.settings import SETTINGS
    SETTINGS.plot_backend = "Agg"                 # só nesta execução (o plot do evo forçaria o Qt da config global)
    from evo.tools import file_interface, plot
    evo_dir = os.path.join(res_dir, "evo")
    if not os.path.isfile(os.path.join(evo_dir, "gt.txt")):
        _fallback(res_dir, evo_dir)
    ref_all = read_tum(os.path.join(evo_dir, "gt.txt"))
    length = float(np.linalg.norm(np.diff(ref_all.positions_xyz, axis=0), axis=1).sum())
    if length < 3 * delta_m:                       # trajetória curta (robô): segmento do RPE ~ 1/5 do percurso
        delta_m = max(round(length / 5, 1), 0.1)
        if verbose:
            print(f"(percurso de {length:.1f} m: RPE em segmentos de {delta_m} m)")
    names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(evo_dir, "*.txt"))
                   if os.path.basename(p) != "gt.txt")
    names.sort(key=lambda n: ["fusao_suavizada", "fusao_online"].index(n) if n in ("fusao_suavizada", "fusao_online")
               else 2 + n.startswith("so"))
    rows, aligned = [], {}
    for name in names:
        est = read_tum(os.path.join(evo_dir, f"{name}.txt"))
        ref, est = sync.associate_trajectories(copy.deepcopy(ref_all), est, max_diff=max_dt)
        sim3 = name.startswith("vo")
        est.align(ref, correct_scale=sim3)
        ape = metrics.APE(PoseRelation.translation_part)
        ape.process_data((ref, est))
        s = ape.get_all_statistics()
        file_interface.save_res_file(os.path.join(evo_dir, f"ape_{name}.zip"),
                                     ape.get_result(ref_name="gt", est_name=name), confirm_overwrite=False)
        r = dict(metodo=label(name), arquivo=name, alinhamento="Sim3" if sim3 else "SE3", poses=ref.num_poses,
                 rpe_segmento_m=delta_m,
                 ape_rmse=s["rmse"], ape_media=s["mean"], ape_mediana=s["median"], ape_desvio=s["std"],
                 ape_max=s["max"])
        try:                                        # RPE em segmentos de delta_m metros (trajetória curta: pula)
            rt = metrics.RPE(PoseRelation.translation_part, delta=delta_m, delta_unit=Unit.meters, all_pairs=True)
            rt.process_data((ref, est))
            rr = metrics.RPE(PoseRelation.rotation_angle_deg, delta=delta_m, delta_unit=Unit.meters, all_pairs=True)
            rr.process_data((ref, est))
            file_interface.save_res_file(os.path.join(evo_dir, f"rpe_{name}.zip"),
                                         rt.get_result(ref_name="gt", est_name=name), confirm_overwrite=False)
            mt, mr = rt.get_statistic(metrics.StatisticsType.mean), rr.get_statistic(metrics.StatisticsType.mean)
            norm = delta_m >= 10                    # % e °/100 m só fazem sentido p/ segmentos longos
            r.update(rpe_trans_m=mt, rpe_rot_deg=mr, rpe_trans_pct=100 * mt / delta_m if norm else np.nan,
                     rpe_rot_deg_100m=100 * mr / delta_m if norm else np.nan)
        except Exception as e:
            r.update(rpe_trans_m=np.nan, rpe_rot_deg=np.nan, rpe_trans_pct=np.nan, rpe_rot_deg_100m=np.nan)
            if verbose:
                print(f"  RPE de {name} não calculado ({type(e).__name__}: trajetória curta p/ {delta_m:g} m?)")
        rows.append(r)
        aligned[name] = (ref, est, ape)

    keys = list(rows[0])
    with open(os.path.join(evo_dir, "resumo_evo.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    if verbose:
        print(f"\nAvaliação com o evo (APE de translação; RPE em segmentos de {delta_m:g} m, todos os pares):")
        print(f"{'':16s} {'alinh.':>6s} {'APE rmse':>9s} {'média':>7s} {'mediana':>8s} {'máx':>7s} "
              f"{'RPE [m]':>8s} {'RPE [°]':>8s} {'RPE %':>7s} {'°/100m':>7s}")
        for r in rows:
            print(f"{r['metodo']:16s} {r['alinhamento']:>6s} {r['ape_rmse']:9.2f} {r['ape_media']:7.2f} "
                  f"{r['ape_mediana']:8.2f} {r['ape_max']:7.1f} {r['rpe_trans_m']:8.3f} {r['rpe_rot_deg']:8.3f} "
                  f"{r['rpe_trans_pct']:7.2f} {r['rpe_rot_deg_100m']:7.3f}")

    # figuras
    fig = plt.figure(figsize=(9, 8))
    ax = plot.prepare_axis(fig, plot.PlotMode.xy)
    ref0 = next(iter(aligned.values()))[0]
    plot.traj(ax, plot.PlotMode.xy, ref0, style="-", color="0.55", label="GT", alpha=0.8)
    for name, (ref, est, _) in aligned.items():
        plot.traj(ax, plot.PlotMode.xy, est, style="-", color=color(name), label=label(name), alpha=0.9)
    ax.legend(fontsize=9)
    ax.set_title("Trajetórias alinhadas ao GT (evo)")
    fig.savefig(os.path.join(evo_dir, "evo_trajetorias.pdf"), bbox_inches="tight")
    plt.close(fig)
    for name, (ref, est, ape) in aligned.items():
        fig = plt.figure(figsize=(14, 6))
        a1 = plot.prepare_axis(fig, plot.PlotMode.xy, subplot_arg=121)
        plot.traj(a1, plot.PlotMode.xy, ref, style="--", color="0.6", label="GT")
        plot.traj_colormap(a1, est, ape.error, plot.PlotMode.xy, min_map=float(np.min(ape.error)),
                           max_map=float(np.max(ape.error)), title=f"APE de {label(name)} [m]")
        a2 = fig.add_subplot(122)
        tt = est.timestamps - est.timestamps[0]
        a2.plot(tt, ape.error, color=color(name) or "k", lw=1.2)
        a2.axhline(ape.get_statistic(metrics.StatisticsType.rmse), color="0.3", ls="--", lw=1,
                   label=f"rmse {ape.get_statistic(metrics.StatisticsType.rmse):.2f} m")
        a2.set(xlabel="t [s]", ylabel="APE [m]", title=f"APE no tempo ({label(name)})")
        a2.grid(True, alpha=0.3)
        a2.legend()
        fig.savefig(os.path.join(evo_dir, f"evo_ape_{name}.pdf"), bbox_inches="tight")
        plt.close(fig)
    if verbose:
        print(f"evo: {evo_dir}/ (resumo_evo.csv, ape_*.zip, rpe_*.zip, evo_*.pdf)")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Avaliação com o evo (APE/RPE) de resultados/<dataset>")
    ap.add_argument("res_dir")
    ap.add_argument("--delta", type=float, default=100.0, help="m dos segmentos do RPE")
    args = ap.parse_args()
    avaliar(args.res_dir, delta_m=args.delta)


if __name__ == "__main__":
    main()
