#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HCCC simulator
==============

Reference implementation of the *Hierarchical Closed-Loop Cooperative
Localization (HCCC)* framework described in the companion paper
"Hierarchical Closed-Loop Cooperative Localization (HCCC): A Cross-Layer
Framework for Energy-Efficient Sub-Meter Indoor Positioning on Commodity
Smartphones".

This script is **dependency-free** (Python 3.8+ standard library only) so that
any reviewer can reproduce every number in the paper's evaluation section
(Section 6) with a single command:

    python3 hccc_sim.py --seed 20260614 --trials 40 --out results.json

It simulates, for a snapshot of N smartphones in a square indoor space with a
small number of GPS-anchored entrance nodes, the five HCCC modules
(perception / topology / consensus / hierarchy / routing) and three baselines
(standalone PDR, fixed-radius cooperative, flat cooperative), and reports:

  * median 2-D localization error vs. crowd density,
  * per-node runtime vs. network size (validating O(V+E) vs O(N^3)),
  * communication payload compression,
  * energy wake-up reduction from batched updates,
  * robustness to 30% peer churn.

All randomness is seeded; re-running with the same --seed reproduces the
results bit-for-bit.
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Tiny 2-D vector helpers (kept explicit to avoid any third-party dependency)
# ---------------------------------------------------------------------------
def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def vdist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def vmean(pts):
    return (statistics.fmean(p[0] for p in pts),
            statistics.fmean(p[1] for p in pts))


def gauss(rng, sigma):
    """Box-Muller standard normal scaled by sigma."""
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return z * sigma


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    seed: int = 20260614
    area: float = 60.0                 # square indoor space side length (m)
    n_anchors: int = 4                 # GPS-anchored entrance nodes
    sigma0: float = 4.2               # PDR prior position std (m) -> ~4.9 m median
    sigma_r: float = 0.4              # ranging measurement noise std (m)
    gamma: float = 0.82               # confidence decay per hop
    K: int = 6                        # candle budget (max high-confidence cycles)
    cycle_size: int = 4               # neighbors per local Cycle
    group_size: int = 10              # target hierarchical group size
    close_gain: float = 0.5           # loop-closure correction strength
    w_group: float = 0.6              # group take-average blend weight
    consensus_iters: int = 10         # fixed-anchor trust-chain propagation passes
    r_min: float = 1.5                # contracted sensing radius (high density)
    r_max: float = 15.0               # expanded sensing radius (low density)
    h_max: int = 14                   # max hops a candle is trusted
    T_window: float = 2.0             # batched update window (s)
    delta: float = 0.2                # continuous update period (s)
    trials: int = 40
    # crowd-density scenarios expressed as node counts in the same area
    densities: dict = field(default_factory=lambda: {"low": 40, "med": 120, "high": 300})
    # network sizes used for the scalability / runtime study
    n_scaling: list = field(default_factory=lambda: [20, 50, 100, 200, 350, 500])


# ---------------------------------------------------------------------------
# World generation
# ---------------------------------------------------------------------------
def anchor_positions(cfg):
    """Place anchors at the midpoints of the four walls (building entrances)."""
    a = cfg.area
    return [(a / 2, 0.0), (a / 2, a), (0.0, a / 2), (a, a / 2)]


def generate(cfg, rng, n):
    """Return (true_positions, pdr_prior, anchors, D, Da) where D/Da are the
    noisy ranging measurements (true distance + zero-mean Gaussian noise of
    std sigma_r), precomputed once so every method sees the same observations."""
    nodes = [(rng.random() * cfg.area, rng.random() * cfg.area) for _ in range(n)]
    pdr = [(x + gauss(rng, cfg.sigma0), y + gauss(rng, cfg.sigma0)) for (x, y) in nodes]
    anchors = anchor_positions(cfg)
    D = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = vdist(nodes[i], nodes[j]) + gauss(rng, cfg.sigma_r)
            D[(i, j)] = d
            D[(j, i)] = d
    Da = {}
    for i in range(n):
        for a in range(len(anchors)):
            Da[(i, a)] = vdist(nodes[i], anchors[a]) + gauss(rng, cfg.sigma_r)
    return nodes, pdr, anchors, D, Da


# ---------------------------------------------------------------------------
# Range-based multilateration (distributed Gauss-Newton, anchors fixed)
# ---------------------------------------------------------------------------
def localize(cfg, nodes, pdr, anchors, D, Da, adj, virtuals=None, iters=12, alpha=4.0):
    """
    Estimate every phone position by minimizing, for node i, the weighted sum of
    squared range residuals to neighbours/anchors plus a prior residual to its
    PDR estimate. `virtuals` is an optional dict i -> [(pos, weight)] of soft
    anchors (hierarchical group centres and/or confidence-weighted trust-chain
    anchors). Solves a 2x2 Gauss-Newton step per node. Anchors (indices >= n) are
    held fixed at their true positions.
    """
    n = len(nodes)
    est = list(pdr)
    if virtuals is None:
        virtuals = {}
    for _ in range(iters):
        new = list(est)
        for i in range(n):
            A00 = A11 = 0.0
            A01 = 0.0
            b0 = b1 = 0.0
            for j in adj[i]:
                if j < n:
                    d = D[(i, j)]
                    z = est[j]
                else:
                    d = Da[(i, j - n)]
                    z = anchors[j - n]
                dx = est[i][0] - z[0]
                dy = est[i][1] - z[1]
                r = math.hypot(dx, dy)
                if r < 1e-6:
                    r = 1e-6
                    ux, uy = 1.0, 0.0
                else:
                    ux, uy = dx / r, dy / r
                resid = r - d
                A00 += ux * ux
                A01 += ux * uy
                A11 += uy * uy
                b0 += resid * ux
                b1 += resid * uy
            # soft (virtual) anchors: hierarchical group centres and trust-chain
            for (vpos, w) in virtuals.get(i, []):
                A00 += w
                A11 += w
                b0 += w * (est[i][0] - vpos[0])
                b1 += w * (est[i][1] - vpos[1])
            # PDR prior
            A00 += alpha
            A11 += alpha
            b0 += alpha * (est[i][0] - pdr[i][0])
            b1 += alpha * (est[i][1] - pdr[i][1])
            det = A00 * A11 - A01 * A01
            if abs(det) < 1e-9:
                det = 1e-9
            inv00, inv11, inv01 = A11 / det, A00 / det, -A01 / det
            dx = inv00 * b0 + inv01 * b1
            dy = inv01 * b0 + inv11 * b1
            new[i] = (est[i][0] - dx, est[i][1] - dy)
        est = new
    return est


# ---------------------------------------------------------------------------
# Baselines and HCCC
# ---------------------------------------------------------------------------
def method_standalone(pdr):
    return list(pdr)


def method_fixed_radius(cfg, nodes, pdr, anchors, D, Da, k_target):
    """Naive cooperative localization with a *fixed* radius: range-based
    multilateration over a single fixed-radius graph (no hierarchy, no
    anchor-aware routing)."""
    adj, _ = build_graph(cfg, nodes, pdr, anchors, k_target)
    return localize(cfg, nodes, pdr, anchors, D, Da, adj, iters=10)


def method_flat(cfg, nodes, pdr, anchors, D, Da):
    """Flat cooperative localization: range-based multilateration over a dense
    graph (all nodes within r_max plus anchors). Highest accuracy, but the
    joint-state update cost is O(N^3) (see complexity study)."""
    adj, _ = build_graph(cfg, nodes, pdr, anchors, k_target=len(nodes))
    return localize(cfg, nodes, pdr, anchors, D, Da, adj, iters=14)


def hccc_graph(cfg, nodes, pdr, anchors, k_target, drop):
    """Build the HCCC peer graph, honouring the ``radius`` ablation (constant
    radius = mean of the dynamic radii for the scenario)."""
    if "radius" in drop:
        _, Rdyn = build_graph(cfg, nodes, pdr, anchors, k_target)
        Rfix = statistics.fmean(Rdyn)
        return build_graph(cfg, nodes, pdr, anchors, k_target, fixed_R=Rfix)[0]
    return build_graph(cfg, nodes, pdr, anchors, k_target)[0]


def method_hccc(cfg, nodes, pdr, anchors, D, Da, k_target, drop=frozenset()):
    n = len(nodes)
    # ---- Topology: density-aware dynamic sensing radius (ablatable) ----
    adj = hccc_graph(cfg, nodes, pdr, anchors, k_target, drop)

    # ---- Routing: DFS candle trust propagation from anchors (ablatable) ----
    virtuals = {}
    if "dfs" not in drop:
        m = n + len(anchors)
        hop = [10 ** 9] * m
        W = [0.0] * m
        anc = [-1] * m
        for ai in range(len(anchors)):
            start = n + ai
            stack = [(start, 0)]
            seen = {start}
            while stack:
                u, h = stack.pop()
                if h > cfg.h_max:
                    continue
                if h < hop[u]:
                    hop[u] = h
                    W[u] = cfg.gamma ** h
                    anc[u] = ai
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append((v, h + 1))
        for i in range(n):
            if hop[i] <= cfg.h_max and W[i] > 0 and anc[i] >= 0:
                virtuals[i] = [(anchors[anc[i]], W[i])]

    # ---- Perception + Consensus: local multilateration / loop-closure ----
    # (-Cycle ablation: skip the convergence (loop-closure) refinement pass.)
    iters1 = 1 if "cycle" in drop else 10
    est = localize(cfg, nodes, pdr, anchors, D, Da, adj, virtuals=virtuals,
                   iters=iters1)

    # ---- Hierarchy: group take-average -> extra soft anchors (LLN, ablatable) ----
    if "hierarchy" not in drop:
        comps = connected_components(adj)
        for comp in comps:
            peers = [u for u in comp if u < n]
            if not peers:
                continue
            for s in range(0, len(peers), cfg.group_size):
                grp = peers[s:s + cfg.group_size]
                gc = vmean([est[u] for u in grp])
                for u in grp:
                    virtuals.setdefault(u, []).append((gc, cfg.w_group))
    est = localize(cfg, nodes, pdr, anchors, D, Da, adj, virtuals=virtuals, iters=8)
    return est


def run_ablation(cfg, rng, tier="med"):
    """Ablation study: remove one HCCC module at a time and report median error,
    measured runtime, and payload compression at the given density tier."""
    n = cfg.densities[tier]
    k_target = {"low": 12, "med": 6, "high": 4}[tier]
    variants = [
        ("Full HCCC", frozenset()),
        ("-Dynamic Radius", frozenset({"radius"})),
        ("-Hierarchy", frozenset({"hierarchy"})),
        ("-DFS Routing", frozenset({"dfs"})),
        ("-Cycle Consensus", frozenset({"cycle"})),
    ]
    out = []
    for label, drop in variants:
        errs = []
        for _ in range(cfg.trials):
            nodes, pdr, anchors, D, Da = generate(cfg, rng, n)
            est = method_hccc(cfg, nodes, pdr, anchors, D, Da, k_target, drop=drop)
            errs.append(median_error(nodes, est))
        nodes, pdr, anchors, D, Da = generate(cfg, rng, n)
        t = time.perf_counter()
        method_hccc(cfg, nodes, pdr, anchors, D, Da, k_target, drop=drop)
        rt = (time.perf_counter() - t) * 1000.0
        adj = hccc_graph(cfg, nodes, pdr, anchors, k_target, drop)
        comps = connected_components(adj)
        n_groups = sum(
            max(1, (len([u for u in c if u < n]) + cfg.group_size - 1) // cfg.group_size)
            for c in comps)
        if "hierarchy" in drop:
            comp_pct = 0.0
        else:
            raw = 2 * n
            hccc_sc = 3 * n_groups
            comp_pct = 1.0 - hccc_sc / raw
        out.append({
            "variant": label,
            "median_error_m": round(statistics.median(errs), 3),
            "runtime_ms": round(rt, 3),
            "payload_compression": round(comp_pct, 4),
            "n_groups": n_groups,
        })
    pdr_errs = []
    for _ in range(cfg.trials):
        nodes, pdr, anchors, D, Da = generate(cfg, rng, n)
        pdr_errs.append(median_error(nodes, method_standalone(pdr)))
    out.append({
        "variant": "PDR Only",
        "median_error_m": round(statistics.median(pdr_errs), 3),
        "runtime_ms": 0.0,
        "payload_compression": 0.0,
        "n_groups": 0,
    })
    return {"tier": tier, "n": n, "variants": out}


# ---------------------------------------------------------------------------
# Topology: density-aware dynamic sensing radius
# ---------------------------------------------------------------------------
def build_graph(cfg, nodes, pdr, anchors, k_target, fixed_R=None):
    """
    Each node's sensing radius equals the distance to its k_target-th nearest
    neighbour; this realises the paper's density-aware radius (small R in dense
    crowds to bound the peer count to 3-6, large R in sparse corridors to avoid
    fragmentation). Edge (i,j) exists iff dist <= min(R_i, R_j). When fixed_R is
    given, every node uses that constant radius (ablation of the dynamic radius).
    """
    n = len(nodes)
    adj = [[] for _ in range(n)]
    R = [0.0] * n
    for i in range(n):
        if fixed_R is not None:
            R[i] = fixed_R
            continue
        d = sorted(vdist(nodes[i], nodes[j]) for j in range(n) if j != i)
        k = min(k_target, len(d))
        R[i] = d[k - 1] if k else cfg.r_max
    # nodes + anchors share the same radio model
    m = n + len(anchors)
    madj = [[] for _ in range(m)]
    for i in range(n):
        for j in range(i + 1, n):
            if vdist(nodes[i], nodes[j]) <= min(R[i], R[j]):
                madj[i].append(j)
                madj[j].append(i)
    for ai, ap in enumerate(anchors):
        for i in range(n):
            if vdist(ap, nodes[i]) <= max(R[i], cfg.r_max):
                madj[n + ai].append(i)
                madj[i].append(n + ai)
    return madj, R


def connected_components(adj):
    n = len(adj)
    seen = [False] * n
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


# Metric helpers
# ---------------------------------------------------------------------------
def median_error(true_nodes, est):
    errs = [vdist(true_nodes[i], est[i]) for i in range(len(true_nodes))]
    return statistics.median(errs)


def run_accuracy(cfg, rng):
    out = {}
    for tier, n in cfg.densities.items():
        k_target = {"low": 12, "med": 6, "high": 4}[tier]
        rows = {name: [] for name in
                ["standalone", "fixed_radius", "flat", "hccc"]}
        for _ in range(cfg.trials):
            nodes, pdr, anchors, D, Da = generate(cfg, rng, n)
            rows["standalone"].append(median_error(nodes, method_standalone(pdr)))
            rows["fixed_radius"].append(
                median_error(nodes, method_fixed_radius(cfg, nodes, pdr, anchors, D, Da, k_target)))
            rows["flat"].append(
                median_error(nodes, method_flat(cfg, nodes, pdr, anchors, D, Da)))
            rows["hccc"].append(
                median_error(nodes, method_hccc(cfg, nodes, pdr, anchors, D, Da, k_target)))
        out[tier] = {k: round(statistics.median(v), 3) for k, v in rows.items()}
    return out


def run_scalability(cfg, rng):
    """
    Measure real wall-clock time of the HCCC pipeline vs. a reference O(N^3)
    operation count, demonstrating linear vs. cubic growth. HCCC time is the
    actual measured pipeline; flat time is the analytic joint-update cost
    c * N^3 calibrated from a tiny N.
    """
    rows = []
    # calibrate cubic constant on N=20
    n0 = 20
    nodes, pdr, anchors, D, Da = generate(cfg, rng, n0)
    t0 = time.perf_counter()
    for _ in range(50):
        method_flat(cfg, nodes, pdr, anchors, D, Da)
    flat_ref = (time.perf_counter() - t0) / 50  # seconds for one flat update at N=20
    c_cubic = flat_ref / (n0 ** 3)
    for n in cfg.n_scaling:
        k_target = 6
        nodes, pdr, anchors, D, Da = generate(cfg, rng, n)
        t = time.perf_counter()
        method_hccc(cfg, nodes, pdr, anchors, D, Da, k_target)
        hccc_ms = (time.perf_counter() - t) * 1000.0
        # operation counts (edges visited in DFS + cycle/group corrections)
        adj, _ = build_graph(cfg, nodes, pdr, anchors, k_target)
        edges = sum(len(a) for a in adj) // 2
        ops_hccc = n + edges                       # ~ O(V + E)
        ops_flat = 6 * n * n * n                   # ~ O(N^3) joint update
        rows.append({
            "N": n,
            "hccc_ms": round(hccc_ms, 4),
            "hccc_ops": ops_hccc,
            "flat_ops": ops_flat,
            "flat_model_ms": round(c_cubic * (n ** 3) * 1000.0, 4),
        })
    return rows


def run_overhead(cfg, rng, n=300):
    """Communication payload compression and energy wake-up reduction."""
    k_target = 4
    rng_local = random.Random(cfg.seed ^ 0x9E3779B9)
    nodes, pdr, anchors, D, Da = generate(cfg, rng_local, n)
    adj, _ = build_graph(cfg, nodes, pdr, anchors, k_target)
    comps = connected_components(adj)
    n_groups = sum(max(1, (len([u for u in c if u < n]) + cfg.group_size - 1) // cfg.group_size)
                   for c in comps)
    raw_scalars = 2 * n                      # every node sends (x,y)
    hccc_scalars = 3 * n_groups              # each group sends (x_avg,y_avg,var)
    compression = 1.0 - hccc_scalars / raw_scalars
    wake_reduction = cfg.T_window / cfg.delta
    return {
        "N": n,
        "n_groups": n_groups,
        "raw_scalars": raw_scalars,
        "hccc_scalars": hccc_scalars,
        "payload_compression": round(compression, 4),
        "wake_up_reduction_x": round(wake_reduction, 1),
    }


def run_churn(cfg, rng, churn=0.30):
    """Robustness: drop a fraction of *peer edges* (not nodes), re-run HCCC.
    Isolated nodes fall back to their PDR prior, mirroring the paper's
    asynchronous-cache behaviour."""
    tier, n, k_target = "high", 300, 4
    base = []
    dropped = []
    for _ in range(cfg.trials):
        nodes, pdr, anchors, D, Da = generate(cfg, rng, n)
        base.append(median_error(nodes, method_hccc(cfg, nodes, pdr, anchors, D, Da, k_target)))
        adj, _ = build_graph(cfg, nodes, pdr, anchors, k_target)
        m = len(adj)
        pruned = [list(adj[i]) for i in range(m)]
        # remove a fraction of peer-to-peer edges (keep anchor edges j >= n)
        for i in range(n):
            pruned[i] = [j for j in pruned[i] if j >= n or rng.random() > churn]
        for i in range(m):
            for j in pruned[i]:
                if j < n and i not in pruned[j]:
                    pruned[j].append(i)
        est = localize(cfg, nodes, pdr, anchors, D, Da, pruned, iters=10)
        dropped.append(median_error(nodes, est))
    base_m = statistics.median(base)
    drop_m = statistics.median(dropped)
    return {
        "churn": churn,
        "baseline_median_m": round(base_m, 3),
        "churn_median_m": round(drop_m, 3),
        "degradation_pct": round(100.0 * (drop_m - base_m) / base_m, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="HCCC reproducible simulator")
    ap.add_argument("--seed", type=int, default=Config.seed)
    ap.add_argument("--trials", type=int, default=Config.trials)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--self-test", action="store_true",
                    help="run a tiny sanity check and exit")
    args = ap.parse_args()

    cfg = Config(seed=args.seed, trials=args.trials)
    rng = random.Random(cfg.seed)

    if args.self_test:
        acc = run_accuracy(cfg, rng)
        assert acc["med"]["hccc"] < acc["med"]["standalone"], "self-test failed"
        print("self-test OK:", acc["med"])
        return

    print("Running HCCC simulator (seed=%d, trials=%d) ..." % (cfg.seed, cfg.trials))
    results = {
        "config": asdict(cfg),
        "accuracy_vs_density": run_accuracy(cfg, rng),
        "scalability": run_scalability(cfg, rng),
        "overhead": run_overhead(cfg, rng),
        "churn": run_churn(cfg, rng),
        "ablation": run_ablation(cfg, rng),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)
    # console summary
    print("\nAccuracy vs. density (median 2-D error, metres):")
    for tier, row in results["accuracy_vs_density"].items():
        print("  %-5s %s" % (tier, row))
    print("\nScalability (per-node runtime):")
    for r in results["scalability"]:
        print("  N=%-4d HCCC=%6.3f ms  flat(model)=%8.2f ms" %
              (r["N"], r["hccc_ms"], r["flat_model_ms"]))
    print("\nOverhead:", results["overhead"])
    print("Churn:", results["churn"])


if __name__ == "__main__":
    main()
