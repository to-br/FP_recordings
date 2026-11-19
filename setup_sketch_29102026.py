"""Schematic of the GMR sensor test setup (29.10.2026): top view and side view.
Saves plots/setup_29102026.svg and .png. Not to scale."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch, Ellipse

SHOW_CURRENT_PATH = False   # True: draw the stimulus current path (red arrows) for the artifact discussion

C_SALINE = "#dceefb"
C_SALINE_EDGE = "#7fb3d9"
C_SENSOR = "#4a4a4a"
C_GOLD = "#c9a227"
C_SLICE = "#f2c9b8"
C_SLICE_EDGE = "#c98e78"
C_GLASS = "#e9f4f7"
C_GLASS_EDGE = "#6b8e99"
C_TUNGSTEN = "#55595e"
C_AGCL = "#9a9a9a"
C_WIRE = "#333333"
C_BOX = "#f5f5f5"
C_RED = "#c0392b"

plt.rcParams.update({"font.size": 9, "font.family": "sans-serif"})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.15, 1]})
for ax in (axA, axB):
    ax.set_aspect("equal")
    ax.axis("off")


def label(ax, text, xy, xytext, ha="center"):
    ax.annotate(text, xy=xy, xytext=xytext, ha=ha, va="center", fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color="0.35", lw=0.7, shrinkA=2, shrinkB=1))


def box(ax, x, y, w, h, text):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=C_BOX, ec="0.3", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


def wire(ax, pts, color=C_WIRE, ls="-"):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=0.9, ls=ls, solid_capstyle="round")


# ---------------- A: top view ----------------
ax = axA
ax.set_xlim(-6.2, 6.2)
ax.set_ylim(-5.0, 4.6)
ax.set_title("A   Top view (not to scale)", loc="left", fontsize=11, fontweight="bold")

ax.add_patch(Circle((0, 0), 3.0, fc=C_SALINE, ec=C_SALINE_EDGE, lw=2))          # chamber with ACSF
ax.text(0, 2.55, "ACSF bath", ha="center", fontsize=8.5, color="#3b6f94")

for dx in (-0.25, -0.08, 0.08, 0.25):                                             # contact lines to chamber edge
    wire(ax, [(dx, -0.45), (dx * 1.6, -2.95)], color=C_GOLD)
ax.add_patch(Rectangle((-0.45, -0.45), 0.9, 0.9, fc=C_SENSOR, ec="black", lw=0.8))  # GMR chip
ax.add_patch(Rectangle((-0.12, -0.12), 0.24, 0.24, fc="#8a8a8a", ec="none"))     # sensing area

slice_pts = [(-1.6, -0.7), (-1.1, 0.95), (0.2, 1.25), (1.5, 0.8), (1.7, -0.5), (0.6, -1.15), (-0.9, -1.1)]
ax.add_patch(Polygon(slice_pts, closed=True, fc=C_SLICE, ec=C_SLICE_EDGE, lw=1, alpha=0.75))
t = [(-0.9 + 1.5 * k / 20, 0.15 + 0.55 * (1 - ((k - 10) / 10) ** 2)) for k in range(21)]   # faint CA arc
wire(ax, t, color=C_SLICE_EDGE)

pip = [(-4.4, 0.42), (-4.4, 0.62), (-0.55, 0.30), (-0.50, 0.27)]                   # recording pipette (left)
ax.add_patch(Polygon(pip, closed=True, fc=C_GLASS, ec=C_GLASS_EDGE, lw=1))
wire(ax, [(-4.4, 0.52), (-2.2, 0.40)], color="#888888")                           # Ag/AgCl wire inside pipette

ax.add_patch(Circle((-2.35, -1.85), 0.22, fc=C_AGCL, ec="black", lw=0.8))         # pellet (left)
wire(ax, [(-2.35, -1.85), (-3.6, -2.6), (-4.6, -2.6)])

ax.plot([4.5, 0.55], [-0.33, -0.18], color=C_TUNGSTEN, lw=2.2, solid_capstyle="butt")   # tungsten (right)
ax.plot([0.55, 0.42], [-0.18, -0.175], color=C_TUNGSTEN, lw=1.0)

wire(ax, [(2.35, 1.85), (3.5, 2.6), (4.6, 2.6)], color=C_AGCL)                    # Ag/AgCl wire (right)
ax.plot([2.2, 2.5], [1.75, 1.95], color=C_AGCL, lw=2.5)

box(ax, -6.15, 0.05, 1.35, 0.95, "Amplifier\n(FP channel)")
wire(ax, [(-4.4, 0.52), (-4.8, 0.52)])
wire(ax, [(-4.6, -2.6), (-5.55, -2.6), (-5.55, 0.05)])
box(ax, 4.9, -0.85, 1.25, 0.95, "Stimulus\nisolator")
wire(ax, [(4.5, -0.33), (4.9, -0.33)])
wire(ax, [(4.6, 2.6), (5.6, 2.6), (5.6, 0.1)])
box(ax, -0.9, -4.85, 1.8, 0.9, "GMR readout\n(bias + amplifier)")
wire(ax, [(0, -3.0), (0, -3.95)], color=C_GOLD)

label(ax, "Recording pipette (glass)", (-3.2, 0.5), (-3.3, 1.7))
label(ax, "Ag/AgCl pellet\n(reference, recording)", (-2.35, -1.85), (-4.3, -1.55))
label(ax, "Monopolar tungsten\nstimulation electrode", (3.2, -0.29), (3.6, -1.6))
label(ax, "Ag/AgCl wire\n(return, stimulation)", (2.45, 1.92), (4.4, 3.5))
label(ax, "GMR sensor chip\n(fixed to chamber floor)", (0.3, -0.3), (2.3, -3.6))
label(ax, "Hippocampal slice", (0.9, 1.0), (1.2, 3.6))

if SHOW_CURRENT_PATH:
    kw = dict(arrowstyle="-|>", color=C_RED, lw=1.4, mutation_scale=12)
    ax.annotate("", xy=(2.3, 1.75), xytext=(0.6, -0.1), arrowprops=dict(**kw, connectionstyle="arc3,rad=-0.3"))
    ax.annotate("", xy=(-2.15, -1.75), xytext=(0.3, -0.3), arrowprops=dict(**kw, ls="--", connectionstyle="arc3,rad=-0.2"))
    ax.annotate("", xy=(0, -0.5), xytext=(0.45, -0.2), arrowprops=dict(**kw, ls="--"))
    ax.text(-1.0, -2.45, "suspected leak path\n(stimulus current → recording ground / sensor)", color=C_RED,
            fontsize=8, ha="center")

# ---------------- B: side view ----------------
ax = axB
ax.set_xlim(-5.6, 5.6)
ax.set_ylim(-3.6, 4.6)
ax.set_title("B   Side view (cross-section, not to scale)", loc="left", fontsize=11, fontweight="bold")

ax.add_patch(Rectangle((-4.0, -1.2), 8.0, 0.5, fc="#d9d9d9", ec="black", lw=1))     # chamber floor
ax.add_patch(Rectangle((-4.0, -0.7), 0.25, 2.4, fc="#d9d9d9", ec="black", lw=1))    # walls
ax.add_patch(Rectangle((3.75, -0.7), 0.25, 2.4, fc="#d9d9d9", ec="black", lw=1))
ax.add_patch(Rectangle((-3.75, -0.7), 7.5, 1.9, fc=C_SALINE, ec="none"))            # ACSF
ax.plot([-3.75, 3.75], [1.2, 1.2], color=C_SALINE_EDGE, lw=1.5)
ax.text(-3.4, 0.95, "ACSF", fontsize=8.5, color="#3b6f94")

ax.add_patch(Rectangle((-0.9, -0.95), 1.8, 0.35, fc=C_SENSOR, ec="black", lw=0.8))   # GMR chip in floor
wire(ax, [(0, -0.95), (0, -2.65)], color=C_GOLD)
ax.add_patch(Ellipse((0, -0.47), 3.4, 0.26, fc=C_SLICE, ec=C_SLICE_EDGE, lw=1))      # slice on chip

ax.add_patch(Polygon([(-3.75, 4.2), (-3.45, 4.2), (-0.55, -0.32), (-0.62, -0.34)], closed=True,
                     fc=C_GLASS, ec=C_GLASS_EDGE, lw=1))                               # pipette from left
ax.plot([3.6, 0.6], [4.2, -0.33], color=C_TUNGSTEN, lw=2.2)                           # tungsten from right

ax.add_patch(Circle((-3.1, -0.35), 0.2, fc=C_AGCL, ec="black", lw=0.8))              # pellet (left)
wire(ax, [(-3.1, -0.15), (-3.1, 2.3), (-4.9, 2.3)])
wire(ax, [(3.05, -0.3), (3.05, 2.3), (4.9, 2.3)], color=C_AGCL)                       # AgCl wire (right)
ax.plot([3.05, 3.05], [-0.3, 0.6], color=C_AGCL, lw=2.5)

box(ax, -1.1, -3.5, 2.2, 0.85, "GMR readout\n(bias + amplifier)")

label(ax, "Recording pipette", (-1.8, 1.5), (-1.4, 3.6))
label(ax, "Tungsten stim.\nelectrode", (1.8, 1.5), (1.4, 3.6))
label(ax, "Slice lies directly\non the sensor", (0.8, -0.45), (2.0, -1.9))
label(ax, "GMR chip", (-0.6, -0.8), (-2.2, -1.9))
label(ax, "Ag/AgCl pellet", (-3.1, -0.35), (-4.6, 0.3))
label(ax, "Ag/AgCl wire", (3.05, 0.2), (4.7, 0.9))
ax.text(-4.9, 2.45, "→ FP amplifier", fontsize=8, va="bottom")
ax.text(4.9, 2.45, "→ stimulus isolator", fontsize=8, va="bottom", ha="right")

fig.tight_layout()
out = Path("plots")
out.mkdir(exist_ok=True)
suffix = "_current_path" if SHOW_CURRENT_PATH else ""
for ext in ("svg", "png"):
    fig.savefig(out / f"setup_29102026{suffix}.{ext}", dpi=200, bbox_inches="tight")
print(f"Saved plots/setup_29102026{suffix}.svg / .png")
