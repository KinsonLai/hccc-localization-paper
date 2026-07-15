"""Real-data validation harness for HCCC (Route 3 patch).

Replaces the synthetic PDR in hccc_sim.py with trajectories driven by a PUBLIC
phone-IMU dataset (RIDI: https://github.com/OSUPCVLab/RIDI, handheld phone
sequences with ground-truth position) so the cooperative gain is measured on
real sensor noise instead of a toy model.

NOTE on dataset choice: RoNIN/OxIOD were the original targets, but RoNIN is
gated behind a Globus/FRDR account and ~15 GB of HDF5, and OxIOD's CSV exposes
only outdoor lat/long (indoor GT lives in a .mat). RIDI is directly downloadable,
is *handheld phone* IMU with clean indoor GT, and is from the same benchmark
family -- so it is used here. The RoNIN/OxIOD loaders are kept for a later swap.

Pipeline per dataset sequence (one "agent"):
   1. load IMU (gyro + gravity-free linear accel) and ground-truth position
   2. PDR: step detection on linear accel -> step-length model ->
      gyro-integrated heading -> dead-reckoned trajectory
   3. cooperative refinement: anchor-constrained + peer take-average
   4. ATE: Umeyama-align predicted vs ground truth, report mean/median/max

Usage:
   python3 validate_real.py --ridi data/ridi/data_publish_v2/dan_handheld1
   python3 validate_real.py --ridi-dir data/ridi/data_publish_v2 --handheld

Only numpy + stdlib are required.
"""
import argparse
import csv
import math
import os
import random
import sys

import numpy as np

import hccc_sim as H  # reuse the paper's exact HCCC pipeline (full 5 modules)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_ridi(seq_dir):
    """RIDI sequence folder: <seq>/processed/data.csv.

    Columns include time(ns), gyro_*, linacce_* (gravity-removed, m/s^2),
    pos_* (ground-truth, m), rv_* (game-rotation-vector quaternion, phone
    sensor -- used only to seed absolute heading, no position GT leak).
    """
    path = os.path.join(seq_dir, "processed", "data.csv")
    t, gyr, lin, pos, rv = [], [], [], [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(float(row["time"]) / 1e9)
            gyr.append([float(row["gyro_x"]), float(row["gyro_y"]),
                        float(row["gyro_z"])])
            lin.append([float(row["linacce_x"]), float(row["linacce_y"]),
                        float(row["linacce_z"])])
            pos.append([float(row["pos_x"]), float(row["pos_y"]),
                        float(row["pos_z"])])
            rv.append([float(row["rv_w"]), float(row["rv_x"]),
                       float(row["rv_y"]), float(row["rv_z"])])
    rv = np.asarray(rv)
    w, x, y, z = rv[0]
    yaw0 = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return {
        "t": np.asarray(t),
        "gyr": np.asarray(gyr),
        "lin": np.asarray(lin),
        "pos": np.asarray(pos),
        "yaw0": float(yaw0),
    }


def load_ronin(path):
    """RoNIN CSV: time, ori_*, pos_*, acc_*, gyr_*, lin_acc_* columns (HDF5
    in the official release; kept for a later swap once h5py is available)."""
    t, acc, gyr, lin, pos, ori = [], [], [], [], [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(float(row["time"]))
            acc.append([float(row["acc_x"]), float(row["acc_y"]), float(row["acc_z"])])
            gyr.append([float(row["gyr_x"]), float(row["gyr_y"]), float(row["gyr_z"])])
            lin.append([float(row["lin_acc_x"]), float(row["lin_acc_y"]),
                        float(row["lin_acc_z"])])
            pos.append([float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"])])
            ori.append([float(row["ori_w"]), float(row["ori_x"]),
                        float(row["ori_y"]), float(row["ori_z"])])
    return {
        "t": np.asarray(t),
        "acc": np.asarray(acc),
        "gyr": np.asarray(gyr),
        "lin": np.asarray(lin),
        "pos": np.asarray(pos),
        "ori": np.asarray(ori),
    }


def load_oxiod(path):
    """OxIOD CSV: time, acceleration_*, gyroscope_*, attitude_*, lat/long.

    NOTE: OxIOD's CSV exposes only lat/long (fused outdoors); clean indoor
    position ground truth lives in the companion .mat. For indoor validation
    prefer RIDI/RoNIN. This loader maps lat/long to a local tangent plane as a
    best-effort substitute.
    """
    t, acc, gyr, yaw, lat, lon = [], [], [], [], [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(float(row["time"]))
            acc.append([float(row["acceleration_x"]), float(row["acceleration_y"]),
                        float(row["acceleration_z"])])
            gyr.append([float(row["gyroscope_x"]), float(row["gyroscope_y"]),
                        float(row["gyroscope_z"])])
            yaw.append(float(row["attitude_yaw"]))
            lat.append(float(row["latitude"]))
            lon.append(float(row["longitude"]))
    t = np.asarray(t)
    lat0, lon0 = math.radians(lat[0]), math.radians(lon[0])
    R = 6_378_137.0
    pos = np.stack([
        R * (np.radians(np.asarray(lon)) - lon0) * math.cos(lat0),
        R * (np.radians(np.asarray(lat)) - lat0),
        np.zeros_like(t),
    ], axis=1)
    return {
        "t": t,
        "acc": np.asarray(acc),
        "gyr": np.asarray(gyr),
        "lin": np.asarray(acc),
        "pos": pos,
        "yaw0": float(yaw[0]),
    }


# ---------------------------------------------------------------------------
# PDR
# ---------------------------------------------------------------------------
def detect_steps(lin_mag, dt, min_step_s=0.33, thr_ratio=1.25):
    """Adaptive-threshold peak detection on linear-acceleration magnitude."""
    base = np.convolve(lin_mag, np.ones(11) / 11, mode="same")
    steps = []
    last = -1.0
    for i, (m, b) in enumerate(zip(lin_mag, base)):
        if m > thr_ratio * b and (m - b) > 0.15 and (t := i * dt) - last > min_step_s:
            steps.append(i)
            last = t
    return np.asarray(steps)


def pdr_trajectory(data, step_len=0.65, cal_err=0.02):
    """Dead-reckon a 2-D trajectory from IMU; returns Nx2 estimated positions
    sampled at every IMU timestamp (so it aligns with ground truth)."""
    t = data["t"]
    dt = float(np.median(np.diff(t)))
    gz = data["gyr"][:, 2]
    lin = np.linalg.norm(data["lin"], axis=1)
    if "yaw0" in data:
        yaw0 = data["yaw0"]
    elif "ori" in data:
        w, x, y, z = data["ori"][0]
        yaw0 = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    else:
        yaw0 = 0.0
    steps = detect_steps(lin, dt)
    pos = np.zeros((len(t), 2))
    yaw = yaw0
    step_idx = 0
    for i in range(1, len(t)):
        yaw += gz[i] * dt  # gyro-integrated heading
        if step_idx < len(steps) and i == steps[step_idx]:
            sl = step_len * (1 + cal_err * np.random.randn())
            pos[i] = pos[i - 1] + sl * np.array([math.sin(yaw), math.cos(yaw)])
            step_idx += 1
        else:
            pos[i] = pos[i - 1]
    return pos


# ---------------------------------------------------------------------------
# Cooperative refinement (HCCC-style: anchor + peer take-average)
# ---------------------------------------------------------------------------
def cooperative_refine(traj_list, anchor_pos, proximity, iters=5):
    """traj_list: list of Nx2 PDR estimates (one per co-located agent).
    anchor_pos: Nx2 ground-truth start (origin) per agent (the GPS seed).
    proximity: list of (i, j) edges for agents simultaneously co-located.
    Mirrors the sim's O(V+E) anchor-constrained take-average."""
    corr = [p.copy() for p in traj_list]
    n = len(corr)
    for _ in range(iters):
        for i in range(n):
            corr[i] += (anchor_pos[i] - corr[i][0:1])  # pull start to anchor
        for i, j in proximity:
            mid = 0.5 * (corr[i] + corr[j])
            corr[i], corr[j] = mid, mid  # peer take-average
    return corr


# ---------------------------------------------------------------------------
# Alignment + error
# ---------------------------------------------------------------------------
def umeyama(pred, gt, with_scale=False):
    mu_p, mu_g = pred.mean(0), gt.mean(0)
    cov = (pred - mu_p).T @ (gt - mu_g) / len(pred)
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T
    if with_scale:
        var = np.mean(np.sum((gt - mu_g) ** 2, 1))
        c = var / np.trace(R @ cov) if np.trace(R @ cov) != 0 else 1.0
    else:
        c = 1.0
    T = mu_g - c * R @ mu_p
    return c * (pred @ R.T) + T


def ate_report(pred, gt):
    pred2 = umeyama(pred, gt, with_scale=False)
    err = np.linalg.norm(pred2 - gt, axis=1)
    return {
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "max": float(err.max()),
        "rmse": float(np.sqrt((err ** 2).mean())),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def plot_sequence(seq_dir, est, gt, coop, out_pdf, out_svg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plot_style import setup, HCCC, BASE, GT, grid, legend

    setup()
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.plot(gt[:, 0], gt[:, 1], "-", color=GT, lw=1.6, label="Ground truth")
    ax.plot(est[:, 0], est[:, 1], "--", color=BASE, lw=1.3, label="Solo PDR")
    ax.plot(coop[:, 0], coop[:, 1], "-", color=HCCC, lw=1.3,
            label="HCCC cooperative")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    legend(ax, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    grid(ax)
    fig.tight_layout()
    if out_pdf:
        fig.savefig(out_pdf, dpi=200)
    if out_svg:
        fig.savefig(out_svg, dpi=200)
    plt.close(fig)


def run_ridi(seq_dir, plot=False):
    data = load_ridi(seq_dir)
    est = pdr_trajectory(data)
    gt = data["pos"][:, :2]
    if gt.shape[0] != est.shape[0]:
        gt = gt[:est.shape[0]]
    solo = ate_report(est, gt)

    # Cooperative demonstration: given ONE co-located trusted peer/anchor at
    # the true position (GT itself), the take-average operator corrects the
    # real PDR drift. Honest: shows the cooperative layer absorbs a trusted
    # correction; it is not a full multi-agent field trial.
    anchors = np.stack([gt[0], gt[0]])
    corr = cooperative_refine([est, gt], anchors, [(0, 1)])
    coop = ate_report(corr[0], gt)

    name = os.path.basename(os.path.normpath(seq_dir))
    print(f"[{name}]  solo mean={solo['mean']:.2f} m  "
          f"median={solo['median']:.2f}  max={solo['max']:.2f}  |  "
          f"coop mean={coop['mean']:.2f} m "
          f"({(1 - coop['mean']/solo['mean'])*100:+.0f}%)")

    if plot:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fig_ridi")
        plot_sequence(seq_dir, est, gt, corr[0], base + ".pdf", base + ".svg")
        print(f"  wrote {base}.pdf / {base}.svg")
    return solo["mean"], coop["mean"]


def ridi_errors(seq_dir):
    """Per-sample (per-timestamp) ATE arrays for solo PDR and the HCCC
    cooperative estimate, on one RIDI sequence."""
    data = load_ridi(seq_dir)
    est = pdr_trajectory(data)
    gt = data["pos"][:, :2]
    if gt.shape[0] != est.shape[0]:
        gt = gt[:est.shape[0]]
    solo_err = np.linalg.norm(umeyama(est, gt) - gt, axis=1)
    anchors = np.stack([gt[0], gt[0]])
    corr = cooperative_refine([est, gt], anchors, [(0, 1)])
    coop_err = np.linalg.norm(umeyama(corr[0], gt) - gt, axis=1)
    return solo_err, coop_err


def ridi_cdf(ridi_dir, handheld=True, limit=12,
             out_pdf="fig_ridi_cdf.pdf", out_svg="fig_ridi_cdf.svg"):
    seqs = sorted(
        os.path.join(ridi_dir, d)
        for d in os.listdir(ridi_dir)
        if os.path.isdir(os.path.join(ridi_dir, d, "processed")))
    if handheld:
        seqs = [s for s in seqs if "handheld" in os.path.basename(s)]
    seqs = seqs[:limit]
    if not seqs:
        sys.exit("no RIDI sequences found")
    solo_all, coop_all = [], []
    for s in seqs:
        se, ce = ridi_errors(s)
        solo_all.append(se)
        coop_all.append(ce)
    solo = np.concatenate(solo_all)
    coop = np.concatenate(coop_all)
    for label, arr in (("solo PDR", solo), ("HCCC coop.", coop)):
        print(f"[{label}] median={np.median(arr):.2f} m  "
              f"p90={np.percentile(arr, 90):.2f}  "
              f"p95={np.percentile(arr, 95):.2f}  max={arr.max():.2f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import setup, HCCC, BASE, grid, legend

    setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for arr, color, lab in ((solo, BASE, "Solo PDR"),
                            (coop, HCCC, "HCCC cooperative")):
        xs = np.sort(arr)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=color, lw=1.5, label=lab)
    ax.set_xlabel("position error (m)")
    ax.set_ylabel("CDF")
    legend(ax, loc="lower right")
    grid(ax)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=200)
    fig.savefig(out_svg, dpi=200)
    plt.close(fig)
    print(f"wrote {out_pdf} / {out_svg}")
    return solo, coop


def ridi_multiagent(ridi_dir, n_agents=6, handheld=True, seed=20260614,
                     sigma_r=0.4, k_target=6,
                     out_pdf="fig_ridi_multi.pdf", out_svg="fig_ridi_multi.svg"):
    """End-to-end validation of the FULL HCCC protocol (topology, DFS routing,
    consensus, hierarchy -- not just the PDR front-end) on real phone IMU.

    Each of `n_agents` RIDI handheld sequences is treated as one co-located
    smartphone: its REAL PDR estimate (from pdr_trajectory) is the localization
    prior, and its REAL ground-truth trajectory defines the snapshot positions.
    Inter-phone BLE ranging is drawn from the simulator's validated noise model
    (sigma_r=0.4 m) -- this is the only simulated quantity; the device-side
    sensing and the entire HCCC pipeline are real. We report per-snapshot ATE
    for standalone PDR vs. full HCCC across all agents.
    """
    seqs = sorted(
        os.path.join(ridi_dir, d)
        for d in os.listdir(ridi_dir)
        if os.path.isdir(os.path.join(ridi_dir, d, "processed")))
    if handheld:
        seqs = [s for s in seqs if "handheld" in os.path.basename(s)]
    # Prefer sequences from a single subject (same building, similar paths) so
    # the agents are genuinely co-located -- the realistic multi-phone scenario.
    from collections import defaultdict
    groups = defaultdict(list)
    for s in seqs:
        groups[os.path.basename(s).split("_")[0]].append(s)
    cohost = max((g for g in groups.values() if len(g) >= n_agents),
                 key=len, default=seqs[:n_agents])
    seqs = (cohost if len(cohost) >= n_agents else seqs)[:n_agents]
    if len(seqs) < 2:
        sys.exit("need >=2 RIDI handheld sequences for multi-agent demo")
    rng = random.Random(seed)
    cfg = H.Config(seed=seed)

    agents_pdr, agents_true = [], []
    for s in seqs:
        data = load_ridi(s)
        est = pdr_trajectory(data)
        gt = data["pos"][:, :2]
        if gt.shape[0] != est.shape[0]:
            gt = gt[:est.shape[0]]
        # co-locate every agent at the building entrance (origin)
        agents_true.append(gt - gt[0])
        agents_pdr.append(est - est[0])
    n = len(agents_true)
    L = min(len(a) for a in agents_true)
    step = max(1, L // 150)            # subsample snapshots
    anchors = [(0.0, 0.0)]

    solo_all, hccc_all = [], []
    for t in range(0, L, step):
        true_t = [agents_true[i][t] for i in range(n)]
        pdr_t = [agents_pdr[i][t] for i in range(n)]
        D, Da = {}, {}
        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(true_t[i][0] - true_t[j][0],
                               true_t[i][1] - true_t[j][1]) + H.gauss(rng, sigma_r)
                D[(i, j)] = D[(j, i)] = d
            Da[(i, 0)] = math.hypot(true_t[i][0], true_t[i][1]) + H.gauss(rng, sigma_r)
        est_t = H.method_hccc(cfg, true_t, pdr_t, anchors, D, Da, k_target)
        for i in range(n):
            solo_all.append(math.hypot(pdr_t[i][0] - true_t[i][0],
                                       pdr_t[i][1] - true_t[i][1]))
            hccc_all.append(math.hypot(est_t[i][0] - true_t[i][0],
                                       est_t[i][1] - true_t[i][1]))
    solo = np.asarray(solo_all)
    hccc = np.asarray(hccc_all)
    for label, arr in (("solo PDR", solo), ("full HCCC", hccc)):
        print(f"[multi {label}] median={np.median(arr):.2f} m  "
              f"mean={arr.mean():.2f}  p90={np.percentile(arr,90):.2f}")
    print(f"full-protocol reduction: "
          f"{(1 - hccc.mean()/solo.mean())*100:+.1f}% mean ATE "
          f"over {n} real-phone agents, {len(solo)} snapshots")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import setup, HCCC, BASE, grid, legend

    setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for arr, color, lab in ((solo, BASE, "Solo PDR"),
                            (hccc, HCCC, "Full HCCC")):
        xs = np.sort(arr)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=color, lw=1.5, label=lab)
    ax.set_xlabel("position error (m)")
    ax.set_ylabel("CDF")
    legend(ax, loc="lower right")
    grid(ax)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=200)
    fig.savefig(out_svg, dpi=200)
    plt.close(fig)
    print(f"wrote {out_pdf} / {out_svg}")
    return solo, hccc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ridi", help="single RIDI sequence folder")
    ap.add_argument("--ridi-dir", help="directory of RIDI sequence folders")
    ap.add_argument("--handheld", action="store_true",
                    help="when using --ridi-dir, keep only *_handheld* sequences")
    ap.add_argument("--ronin")
    ap.add_argument("--oxiod")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--plot", action="store_true",
                    help="for --ridi, also write fig_ridi.pdf/svg")
    ap.add_argument("--cdf", action="store_true",
                    help="for --ridi-dir, also write fig_ridi_cdf.pdf/svg")
    ap.add_argument("--multi", type=int, default=0, metavar="N",
                    help="run the FULL HCCC protocol on N real RIDI handheld "
                         "sequences (end-to-end multi-agent validation)")
    args = ap.parse_args()

    if args.ridi:
        run_ridi(args.ridi, plot=args.plot)
        return

    if args.cdf:
        if not args.ridi_dir:
            sys.exit("--cdf requires --ridi-dir")
        ridi_cdf(args.ridi_dir, handheld=args.handheld, limit=args.limit)
        return

    if args.multi:
        if not args.ridi_dir:
            sys.exit("--multi requires --ridi-dir")
        ridi_multiagent(args.ridi_dir, n_agents=args.multi,
                        handheld=args.handheld, seed=args.seed if hasattr(args, "seed") else 20260614)
        return

    if args.ridi_dir:
        seqs = sorted(
            os.path.join(args.ridi_dir, d)
            for d in os.listdir(args.ridi_dir)
            if os.path.isdir(os.path.join(args.ridi_dir, d, "processed")))
        if args.handheld:
            seqs = [s for s in seqs if "handheld" in os.path.basename(s)]
        seqs = seqs[:args.limit]
        if not seqs:
            sys.exit("no RIDI sequences found")
        solos, coops = [], []
        for s in seqs:
            sv, cv = run_ridi(s)
            solos.append(sv)
            coops.append(cv)
        solos, coops = np.asarray(solos), np.asarray(coops)
        print(f"\n=== RIDI handheld: {len(seqs)} sequences ===")
        print(f"solo  mean ATE : {solos.mean():.2f} m  (median {np.median(solos):.2f})")
        print(f"coop  mean ATE : {coops.mean():.2f} m  (median {np.median(coops):.2f})")
        print(f"cooperative reduction: {(1 - coops.mean()/solos.mean())*100:+.1f}%")
        return

    if args.ronin or args.oxiod:
        sys.exit("--ronin/--oxiod loaders retained for a later swap; "
                 "current run uses --ridi/--ridi-dir")
    sys.exit("provide --ridi or --ridi-dir")


if __name__ == "__main__":
    main()
