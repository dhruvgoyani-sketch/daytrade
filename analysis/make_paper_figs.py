#!/usr/bin/env python3
"""
make_paper_figs.py -- every statistical figure in the paper, from the CSVs
written by analysis/paper_stats.py. Run paper_stats.py first.

Figures (written to figs/ as vector PDF + 300-dpi PNG):
  fig_signal_returns   mean forward return by signal x horizon, day-clustered 95% CI
  fig_inference        naive per-poll |t| vs day-clustered |t| (the central figure)
  fig_daily_edge       distribution of the 150 daily mean signed edges (15 min)
  fig_overlap_acf      autocorrelation of the signed-edge series vs wall-clock lag
  fig_robustness       day-clustered t across the robustness battery (15 min)

Style follows Rougier, Droettboom & Bourne (2014), "Ten Simple Rules for
Better Figures": muted colour-blind-safe palette, no chart junk, figures drawn
at final column width so fonts are never scaled.
"""
import math
import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STATS = os.path.join(HERE, "paper_stats")
FIGS = os.path.join(HERE, "..", "figs")
os.makedirs(FIGS, exist_ok=True)

# ---------------------------------------------------------------- house style
# Publication-quality restyle: one colour-blind-safe blue/orange/grey palette
# shared across every figure, readable serif type at final print size, clean
# spines, and light gridlines only where they aid reading.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.0,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#4a4f54",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.color": "#4a4f54",
    "ytick.color": "#4a4f54",
    "xtick.labelcolor": "#222426",
    "ytick.labelcolor": "#222426",
    "text.color": "#222426",
    "axes.labelcolor": "#222426",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.grid": False,
})
# ---- shared semantic palette (colour-blind safe: blue / orange / grey) -------
# BLUE   -> the valid, dependence-aware effect (day-clustered, "long", ACF)
# GRAY   -> the naive / baseline / "flat" reference concept
# ORANGE -> "short" side and every reference/threshold line (|t|=1.96, means)
BLUE = "#2c6d9e"       # primary: day-clustered inference / long / valid effect
GRAY = "#7a828b"       # naive per-poll / flat / neutral baseline
ORANGE = "#e08214"     # short side + reference lines (|t|=1.96, mean edge)
FILL = "#c3d6e6"       # light-blue fill (histogram bars)
BAND = "#eef1f4"       # soft shaded acceptance / context bands
INK = "#222426"        # zero lines and dark text
GRID = "#e6e8ea"       # light gridlines
# legacy aliases kept so downstream data logic is untouched
NAVY = BLUE
BRICK = ORANGE
STONE = GRAY
GOLD = ORANGE
COL_W = 3.35           # \columnwidth in inches for this layout
FULL_W = 7.0           # full text width (figure*) in inches


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name):
    fig.savefig(os.path.join(FIGS, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGS, f"{name}.png"), dpi=300)
    plt.close(fig)
    print("wrote", name)


Z95 = 1.959963984540054

# ============================================================ Figure 2
# mean forward return by signal and horizon -- dots with clustered 95% CI
sr = pd.read_csv(os.path.join(STATS, "signal_returns.csv"))
fig, ax = plt.subplots(figsize=(COL_W, 2.5))
H = [5, 15, 30]
x = np.arange(len(H), dtype=float)
spec = {"long": (-0.22, NAVY, "o", "long"),
        "flat": (0.0, STONE, "D", "flat"),
        "short": (0.22, BRICK, "s", "short")}
for sig, (dx, c, mk, lab) in spec.items():
    m = [sr[(sr.signal == sig) & (sr.horizon == h)]["cl_mean_daily"].iloc[0] for h in H]
    e = [Z95 * sr[(sr.signal == sig) & (sr.horizon == h)]["cl_se"].iloc[0] for h in H]
    ax.errorbar(x + dx, m, yerr=e, fmt=mk, ms=3.6, color=c, ecolor=c,
                elinewidth=0.9, capsize=2.2, capthick=0.9, lw=0, label=lab,
                mfc=c if sig != "flat" else "white", mec=c, mew=0.9)
ax.axhline(0, color=INK, lw=0.6, zorder=0)
ax.set_xticks(x)
ax.set_xticklabels([f"{h} min" for h in H])
ax.set_xlabel("forward-return horizon")
ax.set_ylabel("mean forward return (bps)")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14),
          handletextpad=0.15, columnspacing=1.0)
ax.grid(axis="y", color=GRID, lw=0.5)
ax.set_axisbelow(True)
despine(ax)
save(fig, "fig_signal_returns")

# ============================================================ Figure 3
# the central figure: naive per-poll |t| vs day-clustered |t|
ed = pd.read_csv(os.path.join(STATS, "directional_edge.csv"))
fig, ax = plt.subplots(figsize=(FULL_W, 2.9))
nt = ed["naive_t"].abs().to_numpy()
ct = ed["cl_t"].abs().to_numpy()
for i in range(len(H)):
    ax.plot([x[i], x[i]], [nt[i], ct[i]], color="#c9cdd1", lw=1.0, zorder=1)
    ax.annotate("", xy=(x[i], ct[i] + 0.28), xytext=(x[i], nt[i] - 0.28),
                arrowprops=dict(arrowstyle="-|>", color="#b7bcc1", lw=0.9), zorder=1)
ax.scatter(x, nt, s=26, facecolor="white", edgecolor=STONE, lw=1.1, zorder=3,
           label="naive per-poll (invalid: overlapping obs.)")
ax.scatter(x, ct, s=26, color=NAVY, zorder=3,
           label="day-clustered (primary inference)")
for i in range(len(H)):
    ax.text(x[i] + 0.06, nt[i], f"{nt[i]:.1f}", fontsize=8, va="center", color="#6b7076")
    ax.text(x[i] + 0.06, ct[i] + 0.05, f"{ct[i]:.2f}", fontsize=8, va="bottom", color=NAVY)
ax.axhline(Z95, color=ORANGE, lw=1.1, ls=(0, (4, 3)))
ax.text(2.50, Z95 + 0.15, "$|t|=1.96$", color=ORANGE, fontsize=8.5, ha="right")
ax.set_xticks(x)
ax.set_xticklabels([f"{h} min" for h in H])
ax.set_xlim(-0.45, 2.55)
ax.set_ylim(0, 10.2)
ax.set_xlabel("forward-return horizon")
ax.set_ylabel(r"$|t|$ of signed directional edge")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))
ax.grid(axis="y", color=GRID, lw=0.5)
ax.set_axisbelow(True)
despine(ax)
save(fig, "fig_inference")

# ============================================================ Figure 4
# distribution of daily mean signed edge, 15-minute horizon
de = pd.read_csv(os.path.join(STATS, "daily_edge_15m.csv"))
v = de["edge_bps"].to_numpy()
fig, ax = plt.subplots(figsize=(COL_W, 2.4))
lim = math.ceil(np.percentile(np.abs(v), 99.5))
bins = np.arange(-lim, lim + 0.75, 0.75)
ax.hist(v, bins=bins, color=FILL, edgecolor="white", lw=0.4, zorder=2)
ax.axvline(0, color=INK, lw=0.8, zorder=3)
ax.axvline(v.mean(), color=ORANGE, lw=1.3, ls=(0, (4, 2)), zorder=3)
ax.text(v.mean() + 1.1, ax.get_ylim()[1] * 0.99,
        f"mean $= {v.mean():+.2f}$ bps", color=ORANGE, fontsize=8, va="top")
ax.text(0.985, 0.80, f"{(v > 0).mean() * 100:.0f}\\% of days $> 0$\n"
        f"{(v < 0).mean() * 100:.0f}\\% of days $< 0$".replace("\\%", "%"),
        transform=ax.transAxes, fontsize=8, ha="right", va="top", color=INK)
ax.set_xlabel("daily mean signed edge, 15-min horizon (bps)")
ax.set_ylabel("number of days")
despine(ax)
save(fig, "fig_daily_edge")

# ============================================================ Figure 5
# overlap diagnostic: ACF of the signed-edge series vs wall-clock lag
acf = pd.read_csv(os.path.join(STATS, "acf_signed_15m.csv"))
acf = acf[acf.lag_med_min <= 25]
fig, ax = plt.subplots(figsize=(COL_W, 2.4))
ax.axvspan(0, 15, color=BAND, zorder=0)
ax.plot(acf.lag_med_min, acf.acf, color=BLUE, lw=1.6, zorder=3)
ax.axhline(0, color=INK, lw=0.6)
ax.axvline(15, color=ORANGE, lw=1.1, ls=(0, (4, 3)))
ax.text(15.4, 0.86, "horizon\n$H=15$ min", color=ORANGE, fontsize=8, va="top")
ax.text(7.0, 0.92, "overlapping\nforward windows", fontsize=8, color="#5a6066",
        ha="center", va="center")
ax.annotate(r"$\rho \approx 0.87$ at one poll ($\approx$15 s)",
            xy=(acf.lag_med_min.iloc[0], acf.acf.iloc[0]),
            xytext=(6.2, 0.68), fontsize=8, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=0.7))
ax.set_xlim(0, 25)
ax.set_ylim(-0.05, 1.0)
ax.set_xlabel("lag between polls (minutes)")
ax.set_ylabel("autocorrelation of signed edge")
despine(ax)
save(fig, "fig_overlap_acf")

# ============================================================ Figure 6
# robustness battery at the 15-minute horizon: day-clustered t
ed15 = ed[ed.horizon == 15].iloc[0]
sk = pd.read_csv(os.path.join(STATS, "skip_one_poll.csv"))
no = pd.read_csv(os.path.join(STATS, "non_overlapping.csv"))
ls = pd.read_csv(os.path.join(STATS, "long_short_split.csv"))
ox = pd.read_csv(os.path.join(STATS, "ex_opex.csv"))
td = pd.read_csv(os.path.join(STATS, "time_of_day.csv"))
rows = [
    ("baseline (day-clustered)", ed15.cl_t),
    ("skip-one-poll entry", sk[sk.horizon == 15].cl_t.iloc[0]),
    ("non-overlapping polls", no[no.horizon == 15].cl_t.iloc[0]),
    ("Newey--West (per poll)", ed15.hac_t),
    ("excl. monthly OPEX days", ox[ox.horizon == 15].cl_t.iloc[0]),
    ("long signals only", ls[(ls.horizon == 15) & (ls.side == "long")].cl_t.iloc[0]),
    ("short signals only", ls[(ls.horizon == 15) & (ls.side == "short")].cl_t.iloc[0]),
    ("open (9:30--11:00)", td[(td.horizon == 15) & (td.bucket.str.startswith("open"))].cl_t.iloc[0]),
    ("midday (11:00--14:00)", td[(td.horizon == 15) & (td.bucket.str.startswith("mid"))].cl_t.iloc[0]),
    ("close (14:00--16:00)", td[(td.horizon == 15) & (td.bucket.str.startswith("close"))].cl_t.iloc[0]),
]
labels = [r[0].replace("--", "–") for r in rows]
tv = [r[1] for r in rows]
fig, ax = plt.subplots(figsize=(COL_W, 2.9))
y = np.arange(len(rows))[::-1]
ax.axvspan(-Z95, Z95, color=BAND, zorder=0)
ax.axvline(0, color=INK, lw=0.6)
for s in (-Z95, Z95):
    ax.axvline(s, color=ORANGE, lw=1.0, ls=(0, (4, 3)))
ax.scatter(tv, y, s=26, color=BLUE, zorder=3)
for yi, ti in zip(y, tv):
    ax.plot([0, ti], [yi, yi], color="#c1c6cb", lw=1.0, zorder=1)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlim(-3.2, 3.2)
ax.set_xlabel(r"$t$-statistic, 15-min signed edge (dependence-aware)")
ax.text(Z95 + 0.10, y[0] + 0.35, "$|t|=1.96$", color=ORANGE, fontsize=8, ha="left")
despine(ax)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
save(fig, "fig_robustness")

print("all figures ->", os.path.abspath(FIGS))
