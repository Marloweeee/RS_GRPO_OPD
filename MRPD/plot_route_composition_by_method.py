#!/usr/bin/env python
"""Visualize fixed-set route composition across methods.

The route buckets are derived from each method's per-sample student IoU:
    good      : IoU >= 0.5
    ambiguous : 0.3 < IoU < 0.5
    failed    : IoU <= 0.3

This is a fixed-set student-quality view, not a training-time route log. The
ambiguous bucket is therefore a pure IoU band, not the teacher-delta route.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("GUI_GROUNDING_ROOT", "/data/codes/gui_grounding"))
LOCAL_JSONL_DIR = HERE / "route_composition_jsonl"
OUT_PNG = HERE / "route_composition_by_method.png"

TAU_GOOD = 0.5
TAU_FAIL = 0.3

METHODS = [
    (
        "SFT-only",
        "data/logs/eval/qwen3-4b-rrsisd-sft-20260526-220915-checkpoint-96-test-greedy/per_sample.jsonl",
    ),
    (
        "OPSD-only",
        "data/logs/eval/q4b-sft-opsd-only-gaussian-20260527-085701-test/per_sample.jsonl",
    ),
    (
        "GRPO-only",
        "data/logs/eval/q4b-sft-grpo-only-20260527-082554-test/per_sample.jsonl",
    ),
    (
        "MRPD",
        "data/logs/eval/gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_metric_gaussian_e1-20260526-225857_test/per_sample.jsonl",
    ),
]

ROUTES = [
    ("good", "good", "#197B45", "white"),
    ("ambiguous", "ambiguous", "#C8D1DC", "#253044"),
    ("failed", "failed", "#C9473D", "white"),
]


@dataclass(frozen=True)
class MethodStats:
    name: str
    n: int
    good: float
    ambiguous: float
    failed: float
    miou: float
    source: Path


def resolve_source(method: str, legacy_rel: str) -> Path:
    """Prefer the local snapshot, but keep the original eval-log path usable."""
    local = LOCAL_JSONL_DIR / f"{method}_per_sample.jsonl"
    if local.exists():
        return local
    return ROOT / legacy_rel


def read_ious(path: Path) -> list[float]:
    ious: list[float] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                ious.append(float(json.loads(line)["iou"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid IoU record in {path}:{line_no}") from exc
    if not ious:
        raise ValueError(f"No IoU records found in {path}")
    return ious


def compute_stats() -> list[MethodStats]:
    rows: list[MethodStats] = []
    for name, legacy_rel in METHODS:
        source = resolve_source(name, legacy_rel)
        ious = read_ious(source)
        n = len(ious)
        good = sum(i >= TAU_GOOD for i in ious) / n * 100.0
        failed = sum(i <= TAU_FAIL for i in ious) / n * 100.0
        ambiguous = 100.0 - good - failed
        rows.append(
            MethodStats(
                name=name,
                n=n,
                good=good,
                ambiguous=ambiguous,
                failed=failed,
                miou=sum(ious) / n,
                source=source,
            )
        )
    return rows


def signed_pp(value: float) -> str:
    return f"{value:+.1f}"


def delta_color(value: float, higher_is_better: bool) -> str:
    if abs(value) < 0.05:
        return "#64748B"
    improved = value > 0 if higher_is_better else value < 0
    return "#197B45" if improved else "#B42318"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D7DE",
            "axes.labelcolor": "#111827",
            "axes.titlecolor": "#111827",
            "xtick.color": "#475569",
            "ytick.color": "#111827",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "savefig.facecolor": "white",
        }
    )


def label_segment(ax, left: float, width: float, y: int, text_color: str) -> None:
    label = f"{width:.1f}%"
    if width >= 4.6:
        ax.text(
            left + width / 2,
            y,
            label,
            ha="center",
            va="center",
            color=text_color,
            fontsize=10,
            fontweight="bold" if text_color == "white" else "normal",
        )
    else:
        ax.text(
            left + width + 0.8,
            y,
            label,
            ha="left",
            va="center",
            color="#334155",
            fontsize=9,
        )


def draw_figure(rows: list[MethodStats]) -> None:
    setup_style()

    fig = plt.figure(figsize=(10.8, 5.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.9, 1.8], wspace=0.03)
    ax = fig.add_subplot(gs[0, 0])
    meta_ax = fig.add_subplot(gs[0, 1], sharey=ax)

    y_positions = list(range(len(rows)))
    for y, row in zip(y_positions, rows):
        left = 0.0
        for key, _, color, text_color in ROUTES:
            width = getattr(row, key)
            ax.barh(
                y,
                width,
                left=left,
                height=0.58,
                color=color,
                edgecolor="white",
                linewidth=1.2,
            )
            label_segment(ax, left, width, y, text_color)
            left += width

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r.name for r in rows], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of samples (%)", labelpad=8)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.grid(True, color="#E7EBF0", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    legend_items = [
        Patch(facecolor=color, edgecolor="none", label=f"{label} ({threshold_text(key)})")
        for key, label, color, _ in ROUTES
    ]
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=3,
        frameon=False,
        columnspacing=1.8,
        handlelength=1.3,
    )

    baseline = rows[0]
    meta_ax.set_xlim(0, 1)
    meta_ax.set_ylim(ax.get_ylim())
    meta_ax.axis("off")
    meta_ax.text(0.00, -0.72, "mIoU", fontsize=9, color="#64748B", fontweight="bold")
    meta_ax.text(0.34, -0.72, "Δ good", fontsize=9, color="#64748B", fontweight="bold")
    meta_ax.text(0.67, -0.72, "Δ fail", fontsize=9, color="#64748B", fontweight="bold")

    for y, row in zip(y_positions, rows):
        dg = row.good - baseline.good
        df = row.failed - baseline.failed
        fw = "bold" if row.name == "MRPD" else "normal"
        meta_ax.text(0.00, y, f"{row.miou:.4f}", ha="left", va="center", fontsize=10.5, fontweight=fw)
        meta_ax.text(
            0.35,
            y,
            signed_pp(dg),
            ha="left",
            va="center",
            fontsize=10.5,
            color=delta_color(dg, higher_is_better=True),
            fontweight=fw,
        )
        meta_ax.text(
            0.68,
            y,
            signed_pp(df),
            ha="left",
            va="center",
            fontsize=10.5,
            color=delta_color(df, higher_is_better=False),
            fontweight=fw,
        )

    n_values = {row.n for row in rows}
    n_text = f"n={rows[0].n:,}" if len(n_values) == 1 else "mixed n"
    fig.suptitle("Route composition by method", x=0.08, y=0.985, ha="left", fontsize=16, fontweight="bold")
    fig.text(
        0.08,
        0.925,
        f"Fixed RRSIS-D test, greedy decoding, {n_text}. Deltas are percentage points vs SFT-only.",
        ha="left",
        fontsize=10,
        color="#475569",
    )
    fig.text(
        0.08,
        0.035,
        "Buckets use student IoU thresholds: good >= 0.5, ambiguous between 0.3 and 0.5, failed <= 0.3.",
        ha="left",
        fontsize=8.7,
        color="#64748B",
    )

    fig.subplots_adjust(left=0.13, right=0.965, top=0.83, bottom=0.16)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")


def threshold_text(key: str) -> str:
    if key == "good":
        return "IoU >= 0.5"
    if key == "ambiguous":
        return "0.3 < IoU < 0.5"
    if key == "failed":
        return "IoU <= 0.3"
    raise KeyError(key)


def main() -> None:
    rows = compute_stats()
    draw_figure(rows)

    print("wrote", OUT_PNG)
    print(f"\n{'method':12}{'n':>7}{'good%':>8}{'amb%':>8}{'failed%':>9}{'mIoU':>9}{'Δgood':>9}{'Δfail':>9}")
    base = rows[0]
    for r in rows:
        print(
            f"{r.name:12}{r.n:7d}{r.good:8.2f}{r.ambiguous:8.2f}{r.failed:9.2f}"
            f"{r.miou:9.4f}{r.good - base.good:9.2f}{r.failed - base.failed:9.2f}"
        )


if __name__ == "__main__":
    main()
