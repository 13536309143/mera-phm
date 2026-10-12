"""Render published aggregate validation; no dataset, model training or API needed."""

import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent.parent


def main():
    result = json.loads(
        (ROOT / "experiments/numerical/tree-groupcv/summary.json").read_text(encoding="utf-8")
    )
    folds = [
        json.loads(
            (ROOT / f"experiments/numerical/tree-groupcv/fold-{i}.json").read_text(encoding="utf-8")
        )
        for i in range(1, 6)
    ]
    fig = Figure(figsize=(13, 5.3), dpi=160)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.05]})
    methods = [
        "static_fusion",
        "extra_trees_fusion_30",
        "random_forest_fusion_30",
        "history_fusion",
        "router_without_freshness",
        "router_without_reset",
        "learned_router",
    ]
    labels = [
        "Static fusion (25 devices)",
        "Static fusion (30 devices)",
        "Random forest (30 devices)",
        "History fusion",
        "Router without freshness",
        "Router without reset",
        "Constrained router",
    ]
    values = [result["metrics"][p]["mae"] for p in methods]
    axes[0].barh(labels, values, color=["#a8b4c2"] * 6 + ["#265f93"])
    for i, method in enumerate(methods):
        axes[0].scatter(
            [f["metrics"][method]["mae"] for f in folds],
            [i] * 5,
            s=14,
            color="#243648",
            alpha=0.65,
            zorder=3,
        )
        axes[0].text(
            max(f["metrics"][method]["mae"] for f in folds) + 0.06,
            i,
            f"{values[i]:.2f}",
            va="center",
            fontsize=8,
        )
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(f["metrics"][m]["mae"] for f in folds for m in methods) + 0.4)
    axes[0].set_xlabel("OOF MAE (adu); dots show individual fold MAE")
    axes[0].set_title("A  Every device tested once", loc="left", fontsize=12)
    kinds = list(result["stress"])
    stress_labels = [
        "Image outage",
        "Text outage",
        "Image + text outage",
        "Stale history",
        "Extra random image masking",
        "Contiguous image outage",
        "Curve feature perturbation",
    ]
    y = np.arange(len(kinds))
    height = 0.23
    for j, (name, label, color) in enumerate(
        [
            ("static_fusion", "Static, 25", "#a8b4c2"),
            ("extra_trees_fusion_30", "Static, 30", "#cf9b50"),
            ("learned_router", "Constrained", "#265f93"),
        ]
    ):
        axes[1].barh(
            y + (j - 1) * height,
            [result["stress"][k][name]["mae"] for k in kinds],
            height,
            label=label,
            color=color,
        )
    axes[1].set_yticks(y, stress_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 28)
    axes[1].set_xlabel("MAE (adu)")
    axes[1].set_title("B  Causal modality stress tests", loc="left", fontsize=12)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    fig.text(
        0.02,
        0.02,
        "Previously explored synthetic dataset: internal validation, not external confirmation. Dot dispersion is not an inferential confidence interval.",
        fontsize=8,
        color="#586a7a",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = ROOT / "docs/assets"
    out.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        fig.savefig(out / f"validation.{extension}", bbox_inches="tight")


if __name__ == "__main__":
    main()
