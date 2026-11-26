"""Figure 1: Delta ASR-BLEU vs Delta speech LAAL relative to B, per direction (data/results.json)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = json.load(open("data/results.json"))
BLUE, GRAY, INK, MUTED = "#2a78d6", "#8a8a85", "#1f1f1e", "#6b6b66"
SYS = [("LA-2", "p3x", None, "o"), ("LA-2+CV", "p3hk", "N_vs_B_b", "s"), ("verifier as\ntranslator", "p3t", None, "v"),
       ("hard gate\n(support)", "p3v_b", None, "D"), ("hard gate\n(joint)", "p3w", None, "^"), ("HSMM,\nunanchored", "p3n1", None, "P")]
OURS = [("p3hk", "ASC (quality)"), ("p3hk2", "ASC (latency)")]
OFF = {"en2zh": {"LA-2": (6, -9), "LA-2+CV": (6, 2), "verifier as\ntranslator": (-8, -20), "hard gate\n(support)": (-44, -10),
                 "hard gate\n(joint)": (5, 4), "HSMM,\nunanchored": (6, -17), "ASC (quality)": (6, 4), "ASC (latency)": (6, -12)},
       "zh2en": {"LA-2": (6, -9), "LA-2+CV": (6, 2), "verifier as\ntranslator": (-8, -20), "hard gate\n(support)": (-44, -10),
                 "hard gate\n(joint)": (6, 3), "HSMM,\nunanchored": (6, -19), "ASC (quality)": (-30, 8), "ASC (latency)": (-74, -2)}}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.45))
for ax, d, title in zip(axes, ("en2zh", "zh2en"), ("English$\\rightarrow$Chinese", "Chinese$\\rightarrow$English")):
    ax.axhline(0, color="#d9d9d6", lw=0.8, zorder=0); ax.axvline(0, color="#d9d9d6", lw=0.8, zorder=0)
    ax.plot([0], [0], marker="x", color=INK, ms=7, mew=1.6, ls="none", zorder=3)
    ax.annotate("B", (0, 0), textcoords="offset points", xytext=(5, -10), color=INK)
    for name, root, key, mk in SYS:
        x = R[root][key or "NEW_vs_B_b"][d]["d2"]["diff"]; y = R[root][key or "NEW_vs_B_b"][d]["asr"]["diff"]
        ax.plot([x], [y], marker=mk, color=GRAY, ms=5.5, ls="none", mec="white", mew=0.8, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=OFF[d][name], color=MUTED, fontsize=6.5, linespacing=0.9)
    xs = [R[r]["NEW_vs_B_b"][d]["d2"]["diff"] for r, _ in OURS]; ys = [R[r]["NEW_vs_B_b"][d]["asr"]["diff"] for r, _ in OURS]
    ax.plot(xs, ys, color=BLUE, lw=2, zorder=2)
    ax.plot(xs, ys, marker="*", color=BLUE, ms=10, ls="none", mec="white", mew=0.8, zorder=4)
    for (r, lab), x, y in zip(OURS, xs, ys):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=OFF[d][lab], color=INK, fontsize=7, fontweight="bold")
    ax.set_title(title, fontsize=8.5, color=INK)
    ax.set_xlabel("$\\Delta$ speech LAAL vs. B (s)  $\\leftarrow$ earlier")
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("$\\Delta$ ASR-BLEU vs. B")
axes[0].set_xlim(-0.45, 0.9); axes[1].set_xlim(-0.3, 0.75)
axes[0].set_ylim(-0.3, 1.8); axes[1].set_ylim(-0.3, 3.5)
fig.tight_layout(w_pad=1.5)
fig.savefig("figures/frontier.pdf", bbox_inches="tight"); fig.savefig("figures/frontier.png", dpi=200, bbox_inches="tight")
print("ok")
