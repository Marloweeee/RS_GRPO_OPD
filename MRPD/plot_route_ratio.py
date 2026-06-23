#!/usr/bin/env python
"""Plot EMA-smoothed good/failed/ambiguous route ratios over MRPD training.

Source: on-policy per-step route log of the main MRPD run
(gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_metric_gaussian_e1-20260526-225857).
Caveat: ratios are on-policy (8 candidates/step, quantized to 1/8), so per-step
is noisy and confounded by minibatch difficulty; we report the EMA trend.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = "/data/codes/gui_grounding/data/logs/0528_mainline/route_mrpd_metric_reference/route_stats_per_step.csv"
OUT_PNG = os.path.join(HERE, "route_ratio_curve.png")
OUT_CSV = os.path.join(HERE, "route_ratio_ema.csv")
SPAN = 100  # EMA span in steps

df = pd.read_csv(CSV).sort_values("step").reset_index(drop=True)
cols = {
    "good": "sdpo/route_good",
    "failed": "sdpo/route_failed",
    "ambiguous": "sdpo/route_ambiguous",
}
iou_col = "sdpo/iou_mean"

# segment boundaries (early 0-20% / middle 20-80% / late 80-100%)
N = len(df)
b1, b2 = int(0.2 * N), int(0.8 * N)


def seeded_ema(series, span, seed_window):
    """EMA seeded with the mean of the first `seed_window` points, so the
    curve starts at the early-segment distribution instead of the noisy
    first raw value (e.g. ambiguous=0 at step 1)."""
    alpha = 2.0 / (span + 1)
    prev = float(series.iloc[:seed_window].mean())
    out = []
    for x in series:
        prev = alpha * float(x) + (1 - alpha) * prev
        out.append(prev)
    return out


ema = pd.DataFrame({"step": df["step"]})
for name, c in cols.items():
    ema[name] = seeded_ema(df[c], SPAN, b1)
ema.to_csv(OUT_CSV, index=False)

segs = {"early(1-20%)": df.iloc[:b1], "middle(20-80%)": df.iloc[b1:b2], "late(80-100%)": df.iloc[b2:]}
seg_txt = []
for sname, chunk in segs.items():
    seg_txt.append(
        f"{sname}: good={chunk[cols['good']].mean():.3f} "
        f"failed={chunk[cols['failed']].mean():.3f} "
        f"amb={chunk[cols['ambiguous']].mean():.3f} "
        f"iou={chunk[iou_col].mean():.3f}"
    )

colors = {"good": "#167a3c", "failed": "#c0392b", "ambiguous": "#9a5b00"}
labels = {"good": "good", "failed": "failed", "ambiguous": "ambiguous"}
fig, ax = plt.subplots(figsize=(10, 5.4))

for name in cols:
    ax.plot(ema["step"], ema[name], color=colors[name], lw=2.6, label=labels[name])
ax.axvline(df["step"].iloc[b1], color="#94a3b8", ls="--", lw=1, alpha=0.5)
ax.axvline(df["step"].iloc[b2], color="#94a3b8", ls="--", lw=1, alpha=0.5)
ax.set_xlabel("training step")
ax.set_ylabel("route ratio")
ax.set_ylim(0, 0.60)
ax.set_xlim(df["step"].min(), df["step"].max())
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right", framealpha=0.92, fontsize=10)
ax.set_title("MRPD route composition over training (on-policy, EMA span=%d)" % SPAN)

fig.text(0.012, -0.02, "  |  ".join(seg_txt)
         + "\nNote: on-policy ratios (8 candidates/step, quantized to 1/8); "
           "EMA-smoothed, seeded at early-segment mean. Not a fixed-set re-eval.",
         fontsize=7.5, color="#475569")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print("wrote", OUT_PNG)
print("wrote", OUT_CSV)
print("\nsegment means:")
for t in seg_txt:
    print("  " + t)
# early->late deltas on EMA
e_lo = ema.iloc[:b1][["good", "failed", "ambiguous"]].mean()
e_hi = ema.iloc[b2:][["good", "failed", "ambiguous"]].mean()
print("\nearly->late EMA delta:")
for k in ["good", "failed", "ambiguous"]:
    print(f"  {k}: {e_lo[k]:.4f} -> {e_hi[k]:.4f}  (delta {e_hi[k]-e_lo[k]:+.4f})")
