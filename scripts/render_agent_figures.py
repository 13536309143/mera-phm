"""Render measured process completion and API cost; never interpret these as MAE gains."""

import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent.parent


def main():
    result = json.loads(
        (ROOT / "experiments/agents/paired-api/summary.json").read_text(encoding="utf-8")
    )
    modes = ["deterministic_control", "single_agent", "multi_agent"]
    labels = ["Local control", "Single Agent", "Multi-Agent"]
    aggregate = result["aggregate"]
    fig = Figure(figsize=(10, 4.6), constrained_layout=True)
    FigureCanvasAgg(fig)
    left, right = fig.subplots(1, 2)
    completed = [aggregate[m]["completed"] for m in modes]
    total = [aggregate[m]["cases"] for m in modes]
    bars = left.bar(labels, completed, color=["#929fa6", "#2d728f", "#c27b35"])
    left.set_ylim(0, max(total) + 3)
    left.set_ylabel("Completed default-budget sessions")
    left.set_title("24 paired cases; fixed numerical model")
    for bar, n, denominator in zip(bars, completed, total):
        left.text(bar.get_x() + bar.get_width() / 2, n + 0.4, f"{n}/{denominator}", ha="center")
    costs = [aggregate[m]["tokens"] / aggregate[m]["cases"] for m in modes]
    bars = right.bar(labels, costs, color=["#929fa6", "#2d728f", "#c27b35"])
    right.set_ylabel("Mean measured API tokens per case")
    right.set_title("Includes incomplete runs; excludes critic/rescue")
    right.set_ylim(0, max(costs) * 1.2)
    for bar, value in zip(bars, costs):
        right.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(costs) * 0.025,
            f"{value:,.0f}",
            ha="center",
        )
    for ax in [left, right]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Agent process completion and cost do not establish language correctness", fontsize=12
    )
    output = ROOT / "docs/assets"
    fig.savefig(output / "agent-validation.png", dpi=180)
    fig.savefig(output / "agent-validation.pdf")


if __name__ == "__main__":
    main()
