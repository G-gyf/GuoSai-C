"""Generate Chinese figures for the load forecast comparison report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "question2" / "analysis" / "load_forecast"
OUT.mkdir(parents=True, exist_ok=True)

SCHEMES = {
    "b0": "昨日持久化",
    "b1": "周持久化",
    "b2": "七天均值",
    "gk": "相似日高斯核",
}
COLORS = {"b0": "#9e9e9e", "b1": "#1f77b4", "b2": "#d62728", "gk": "#2ca02c"}


def main() -> None:
    ten = pd.read_parquet(ROOT / "data" / "processed" / "load_day_ahead_10min.parquet")
    metrics = pd.read_csv(OUT / "model_metrics.csv")
    key_dates = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=300)
    for ax, date_text in zip(axes.ravel(), key_dates):
        day = ten[ten["plan_date"].eq(pd.Timestamp(date_text))].sort_values("slot_index")
        x = np.arange(144)
        ax.plot(x, day["load_actual_kw"], color="black", lw=2.2, label="实际")
        for name, label in SCHEMES.items():
            ax.plot(
                x,
                day[f"load_{name}_kw"],
                color=COLORS[name],
                lw=1.1,
                ls="-" if name != "gk" else "--",
                alpha=0.9,
                label=label,
            )
        ax.set_title(f"{date_text}  负荷曲线对比")
        ax.set_xlabel("时段（10 分钟）")
        ax.set_ylabel("负荷 (kW)")
        ax.grid(alpha=0.3)
        if date_text == key_dates[0]:
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("四个重点日期：各方案负荷预测 vs 实际", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "figure1_key_dates_forecasts.png", dpi=300)
    plt.close(fig)

    formal = metrics[metrics["scope"].eq("formal")].set_index("scheme")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300)
    order = ["b0", "b1", "b2", "gk"]
    labels = [SCHEMES[name] for name in order]
    mape = formal.loc[order, "mape_pct"]
    mae = formal.loc[order, "mae_kw"]
    colors = [COLORS[name] for name in order]
    axes[0].bar(labels, mape, color=colors)
    axes[0].set_title("正式期 MAPE（%）")
    axes[0].set_ylabel("MAPE (%)")
    for i, value in enumerate(mape):
        axes[0].text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(labels, mae, color=colors)
    axes[1].set_title("正式期 MAE（kW）")
    axes[1].set_ylabel("MAE (kW)")
    for i, value in enumerate(mae):
        axes[1].text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("2025-02-01—12-31 正式期各方案预测精度（334 天）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "figure2_formal_metrics.png", dpi=300)
    plt.close(fig)

    by_type = metrics[
        metrics["scope"].isin(["workday", "sat_class", "sunday", "holiday"])
    ].pivot(index="scope", columns="scheme", values="mape_pct")
    type_names = {"workday": "工作日", "sat_class": "周六类", "sunday": "周日", "holiday": "节假日"}
    x = np.arange(len(type_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    for j, name in enumerate(order):
        values = [by_type.loc[scope, name] for scope in type_names]
        ax.bar(x + (j - 1.5) * width, values, width, label=SCHEMES[name], color=COLORS[name])
    ax.set_xticks(x)
    ax.set_xticklabels([type_names[scope] for scope in type_names])
    ax.set_ylabel("MAPE (%)")
    ax.set_title("正式期分日类型 MAPE（%）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figure3_mape_by_day_type.png", dpi=300)
    plt.close(fig)

    print("figures written to", OUT)


if __name__ == "__main__":
    main()
