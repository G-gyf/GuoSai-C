"""Hourly mechanism figures for the forecast-update value interpretation.

Figure 1: per-hour-of-day cost deltas V6 (M6 - M0) and V12 (M612 - M6),
split into retained / down / up / emergency components (yuan).
Figure 2: M612 adjustment direction and emergency volume by hour (kWh).
Run: python -m src.reporting.question3_mechanism_figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/question3/current/paper/figures"


def load(path):
    return pd.read_csv(path, index_col=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d6 = load(ROOT / "outputs/question3/analysis/mechanism/M6_minus_M0.csv")
    d12 = load(ROOT / "outputs/question3/analysis/mechanism/M612_minus_M6.csv")
    raw = pd.read_csv(
        ROOT / "outputs/question3/analysis/mechanism/hourly_cost_breakdown.csv",
        index_col=[0, 1],
    )
    m612 = raw.loc["M612"]
    hours = np.arange(24)
    labels = [f"{h}:00" for h in hours]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), dpi=300)
    width = 0.38
    for ax, d, title in [
        (axes[0], d6, "6:00 更新的逐小时费用变化 (M6 − M0，元)"),
        (axes[1], d12, "12:00 更新的逐小时费用变化 (M612 − M6，元)"),
    ]:
        x = np.arange(24)
        ax.bar(x - 1.5 * width / 2, d.retained, width, label="常规结算费变化", color="#4C72B0")
        ax.bar(x - 0.5 * width / 2, d.down, width, label="下调违约费", color="#DD8452")
        ax.bar(x + 0.5 * width / 2, d.up, width, label="上调新增费", color="#55A868")
        ax.bar(x + 1.5 * width / 2, d.emergency, width, label="紧急购电费", color="#C44E52")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, ncol=4, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "图_预报更新价值的逐小时分解.png")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(9, 4.0), dpi=300)
    x = np.arange(24)
    ax2.bar(x - 0.2, m612.down_kwh, 0.4, label="下调量", color="#DD8452")
    ax2.bar(x + 0.2, m612.up_kwh, 0.4, label="上调量", color="#55A868")
    ax2.plot(x, m612.em_kwh, "o-", color="#C44E52", label="紧急购电量", ms=4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("kWh")
    ax2.set_title("主方案 M612 的逐小时调整方向与紧急购电量", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT / "图_调整方向与紧急购电逐小时分布.png")
    plt.close(fig2)
    print("figures written:", list(OUT.glob("*.png")))


if __name__ == "__main__":
    main()
