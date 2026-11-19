"""Sketch: why the stimulus artifact was so large on 29.10 (hypothesis), and a better layout.
Potential map = toy model of point current sources in a uniform bath (V ~ I / r), not a measurement.
Saves plots/artifact_sketch_29102026.svg and .png."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon

C_SENSOR = "#4a4a4a"
C_SLICE_EDGE = "#c98e78"
C_GLASS = "#e9f4f7"
C_GLASS_EDGE = "#6b8e99"
C_TUNGSTEN = "#2b2e33"
C_AGCL = "#9a9a9a"
C_RED = "#c0392b"
R_CHAMBER = 3.0
EPS = 0.08                     # avoids the 1/r singularity at the electrode tip

plt.rcParams.update({"font.size": 9, "font.family": "sans-serif"})


def potential(x, y, sources):
    """Sum of I / r for point sources [(x, y, I)], I = +1 (stimulus tip) or -1 (return)."""
    return sum(i / np.sqrt((x - sx) ** 2 + (y - sy) ** 2 + EPS ** 2) for sx, sy, i in sources)


def draw_chamber(ax, sources, title, rec_tip, pellet, stim_line, ret_marker):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-3.6, 3.4)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")

    g = np.linspace(-R_CHAMBER, R_CHAMBER, 500)
    X, Y = np.meshgrid(g, g)
    V = np.abs(potential(X, Y, sources))
    V[X ** 2 + Y ** 2 > R_CHAMBER ** 2] = np.nan
    levels = np.logspace(-1, 1.1, 12)
    cf = ax.contourf(X, Y, np.clip(V, levels[0], levels[-1]), levels=levels, cmap="Reds",
                     norm=plt.matplotlib.colors.LogNorm(), alpha=0.85)
    ax.contour(X, Y, V, levels=levels[::2], colors="white", linewidths=0.4)
    ax.add_patch(Circle((0, 0), R_CHAMBER, fc="none", ec="#7fb3d9", lw=2))

    ax.add_patch(Rectangle((-0.45, -0.45), 0.9, 0.9, fc=C_SENSOR, ec="black", lw=0.8, alpha=0.45))
    ax.text(0, 0.05, "GMR", color="white", ha="center", va="center", fontsize=8, fontweight="bold")
    slice_pts = [(-1.6, -0.7), (-1.1, 0.95), (0.2, 1.25), (1.5, 0.8), (1.7, -0.5), (0.6, -1.15), (-0.9, -1.1)]
    ax.add_patch(Polygon(slice_pts, closed=True, fc="none", ec=C_SLICE_EDGE, lw=1.2, ls="--"))

    tx, ty = rec_tip
    ax.add_patch(Polygon([(-3.3, ty + 0.15), (-3.3, ty + 0.35), (tx, ty + 0.02), (tx + 0.05, ty)],
                         closed=True, fc=C_GLASS, ec=C_GLASS_EDGE, lw=1))
    ax.plot(*rec_tip, "o", ms=5, mfc="white", mec="black", zorder=5)
    ax.add_patch(Circle(pellet, 0.2, fc=C_AGCL, ec="black", lw=0.8, zorder=5))
    stim_line(ax)
    ret_marker(ax)
    small = dict(fontsize=7.5, color="0.2", ha="center")
    ax.text(pellet[0], pellet[1] - 0.42, "Ag/AgCl pellet\n(rec. reference)", va="top", **small)
    ax.text(-2.5, ty - 0.05, "recording pipette", va="top", **small)
    return cf


fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 7.2))

# ---------- A: 29.10 layout (hypothesis) ----------
stim_tip = (0.45, -0.18)
ret_wire = (2.35, 1.85)
rec_tip = (-0.5, 0.27)
pellet = (-2.35, -1.85)
srcA = [(*stim_tip, 1), (*ret_wire, -1)]


def stimA(ax):
    ax.plot([3.3, stim_tip[0]], [-0.33, stim_tip[1]], color=C_TUNGSTEN, lw=2.2)


def retA(ax):
    ax.plot([2.2, 2.5], [1.75, 1.95], color=C_AGCL, lw=3)
    ax.text(2.75, 1.45, "Ag/AgCl\nreturn wire", fontsize=7.5, color="0.2", ha="center", va="top")


cf = draw_chamber(axA, srcA, "A   29.10 (hypothesis): monopolar stimulation\n     right next to the recording site and the sensor",
                  rec_tip, pellet, stimA, retA)
vA = potential(*rec_tip, srcA) - potential(*pellet, srcA)

kw = dict(fontsize=8.5, ha="left", va="center",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.95))
axA.annotate("① Monopolar tip: current spreads\n    through the whole bath\n    → strong field around the tip",
             xy=stim_tip, xytext=(0.9, -2.3), arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8), **kw)
axA.annotate("② Recording tip sits in the strong field,\n    pellet far away in a weak field:\n    V_FP = V(tip) − V(pellet) → large",
             xy=rec_tip, xytext=(-3.35, 2.4), arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8), **kw)
axA.annotate("③ GMR chip right under the stimulus tip:\n    bath potential couples into the sensor\n    (thin passivation / capacitive coupling)",
             xy=(0, -0.45), xytext=(-3.35, -3.15), arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8), **kw)
axA.annotate("Return wire on the far side:\ncurrent crosses the slice,\nchip and recording site",
             xy=ret_wire, xytext=(1.0, 3.05), arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8), **kw)
axA.annotate("", xy=(2.1, 1.6), xytext=(0.7, 0.05),
             arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=1.6, mutation_scale=12, connectionstyle="arc3,rad=-0.25"))
axA.annotate("", xy=(-0.35, 0.2), xytext=(0.3, -0.05),
             arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=1.3, ls="--", mutation_scale=10))
axA.annotate("", xy=(0.1, -0.35), xytext=(0.4, -0.25),
             arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=1.3, ls="--", mutation_scale=10))

# ---------- B: better layout ----------
stim_tipB = (1.6, -0.25)
ret_B = (1.75, -0.25)          # concentric/bipolar: return a few hundred µm from the tip
pelletB = (-2.35, 1.7)
srcB = [(*stim_tipB, 1), (*ret_B, -1)]


def stimB(ax):
    ax.plot([3.3, stim_tipB[0]], [-0.45, stim_tipB[1]], color=C_TUNGSTEN, lw=3.2)
    ax.plot([3.3, ret_B[0]], [-0.45, ret_B[1]], color="#888888", lw=1.0)


def retB(ax):
    pass


draw_chamber(axB, srcB, "B   Better: bipolar/concentric stimulation,\n     further from the recording site and the sensor",
             rec_tip, pelletB, stimB, retB)
vB = potential(*rec_tip, srcB) - potential(*pelletB, srcB)

axB.annotate("① Bipolar/concentric electrode:\n    current returns locally (tip ↔ ring)\n    → field falls off fast (~1/r²)",
             xy=stim_tipB, xytext=(0.6, -2.4), arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8), **kw)
axB.annotate("② ~300–500 µm between stimulation\n    and recording along SR:\n    recording tip outside the strong field",
             xy=rec_tip, xytext=(-3.35, 2.6), arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8), **kw)
axB.annotate("③ Sensor not directly under the\n    stimulation tip; one good bath ground",
             xy=(0, -0.45), xytext=(-3.35, -3.1), arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8), **kw)

cbar = fig.colorbar(cf, ax=[axA, axB], orientation="horizontal", fraction=0.035, pad=0.02, aspect=50)
cbar.set_label("|potential from the stimulus current| (relative, log scale; toy model V ~ I/r, not measured)")
cbar.set_ticks([])

print(f"toy model, V(rec tip) - V(pellet): A = {vA:.2f}, B = {vB:.3f}, ratio = {abs(vA / vB):.0f}x")

out = Path("plots")
out.mkdir(exist_ok=True)
for ext in ("svg", "png"):
    fig.savefig(out / f"artifact_sketch_29102026.{ext}", dpi=200, bbox_inches="tight")
print("Saved plots/artifact_sketch_29102026.svg / .png")
