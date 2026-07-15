import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

d = json.load(open("results.json"))
rows = d["scalability"]
N = np.array([r["N"] for r in rows], dtype=float)
hccc = np.array([r["hccc_ms"] for r in rows], dtype=float)
flat = np.array([r["flat_model_ms"] for r in rows], dtype=float)

# Fit log-log slopes (used only to draw the trend lines through the data)
h_slope, h_int = np.polyfit(np.log10(N), np.log10(hccc), 1)
f_slope, f_int = np.polyfit(np.log10(N), np.log10(flat), 1)
print(f"HCCC slope = {h_slope:.3f}, Flat slope = {f_slope:.3f}")

from plot_style import setup, HCCC as COL_H, BASE as COL_R

setup()

fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.set_xscale("log")
ax.set_yscale("log")

# Data markers
ax.loglog(N, flat, "o", color="#c0392b", markersize=5, markeredgewidth=0.6,
          label=r"Flat cooperative ($\mathcal{O}(N^3)$)", zorder=3)
ax.loglog(N, hccc, "s", color="#1e7d34", markersize=5, markeredgewidth=0.6,
          label=r"HCCC ($\mathcal{O}(V+E)$)", zorder=3)

# Fitted trend lines
xs = np.logspace(np.log10(N.min()), np.log10(N.max()), 50)
ax.loglog(xs, 10 ** (f_int + f_slope * np.log10(xs)), "-", color="#c0392b",
          linewidth=1.4, zorder=2)
ax.loglog(xs, 10 ** (h_int + h_slope * np.log10(xs)), "-", color="#1e7d34",
          linewidth=1.4, zorder=2)

# Axis ticks with real values (minor ticks kept as marks but unlabeled, so the
# log-scale "2x10^1, 3x10^1, ..." labels do not collide)
ax.xaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 350, 500]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100, 1000, 10000]))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_minor_formatter(NullFormatter())

ax.set_xlabel(r"$\log_{10} N$  (network size)")
ax.set_ylabel(r"$\log_{10}$ (per-node runtime [ms])")
ax.set_xlim(15, 620)
ax.set_ylim(0.5, 60000)
ax.grid(True, which="major", ls=":", lw=0.5, color="#bbbbbb")
ax.legend(loc="upper left", fontsize=7.5, framealpha=0.95, edgecolor="#888888")

# Slope annotations
ax.annotate(r"slope $\approx 3$ (cubic)", xy=(120, 120), fontsize=7,
            color="#c0392b")
ax.annotate(r"slope $\approx 1.4$ (near-linear)", xy=(70, 1.4), fontsize=7,
            color="#1e7d34")

fig.tight_layout()
fig.savefig("fig_runtime.svg", format="svg", bbox_inches="tight")
fig.savefig("fig_runtime.pdf", format="pdf", bbox_inches="tight")
print("wrote fig_runtime.svg and fig_runtime.pdf")
