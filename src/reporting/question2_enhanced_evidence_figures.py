"""Build explanatory evidence tables and publication-style figures for Q2."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "docs" / "问题二" / "figures_增强论证"
DATA_DIR = ROOT / "docs" / "问题二" / "增强论证数据"
SENS_DIR = DATA_DIR / "sensitivity"

INK = "#34424F"
MUTED = "#687681"
GRID = "#E5EAF0"
PAPER = "#FCFDFE"
BLUE = "#7FA9C4"
BLUE_DARK = "#527C99"
SKY = "#B8D3E2"
PEACH = "#E7B49E"
GOLD = "#D8C58E"
SAGE = "#9DB8A5"
LAVENDER = "#B9B1CF"
ROSE = "#D8A9B2"
NEUTRAL = "#C8D0D7"


def setup_style() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    font = next((x for x in candidates if x in installed), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font],
            "axes.unicode_minus": False,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "axes.edgecolor": "#AEB8C2",
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": 600,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.85,
            "savefig.facecolor": PAPER,
        }
    )


def finish(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / name, dpi=320, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def add_subtitle(fig: plt.Figure, text: str, y: float = 0.955) -> None:
    fig.text(0.06, y, text, fontsize=9.2, color=MUTED, va="top")


def workflow_figure() -> None:
    fig, ax = plt.subplots(figsize=(14.2, 4.8))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.suptitle("问题二的因果决策链与证据闭环", x=0.06, y=0.98, ha="left", fontsize=17, fontweight=650)
    add_subtitle(fig, "所有日前量在 0:00 锁定；日内规则只读取已经完成区间的观测，实际期末 SOC 传递到下一天。", 0.92)

    boxes = [
        (0.3, 2.75, 1.55, 0.92, "历史信息", "截至 d−1"),
        (2.25, 2.75, 1.65, 0.92, "点预测", "负载＋光伏"),
        (4.3, 2.75, 1.75, 0.92, "残差情景", "最近 M 日路径"),
        (6.45, 2.75, 1.75, 0.92, "风险曲线", "max{点预测,Q0.8}"),
        (8.6, 2.75, 1.75, 0.92, "日前 LP", "计划 g 与参考 SOC"),
        (10.75, 2.75, 1.75, 0.92, "规则校准", "δ、λ 的经验目标"),
        (12.65, 1.25, 1.25, 0.92, "日内执行", "储能＋补购"),
    ]
    colors = [SKY, SKY, LAVENDER, GOLD, BLUE, SAGE, PEACH]
    for (x, y, w, h, title, sub), color in zip(boxes, colors):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.09",
            linewidth=1.15, edgecolor=INK, facecolor=color, alpha=0.82
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + 0.58, title, ha="center", va="center", fontsize=11, fontweight=650)
        ax.text(x + w / 2, y + 0.28, sub, ha="center", va="center", fontsize=8.5, color=MUTED)
    for i in range(6):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i + 1][0]
        y = boxes[i][1] + boxes[i][3] / 2
        if i == 5:
            arrow = FancyArrowPatch((x1, y), (13.25, 2.18), arrowstyle="-|>", mutation_scale=12, lw=1.3, color=BLUE_DARK)
        else:
            arrow = FancyArrowPatch((x1 + 0.05, y), (x2 - 0.05, y), arrowstyle="-|>", mutation_scale=12, lw=1.3, color=BLUE_DARK)
        ax.add_patch(arrow)

    lock = FancyBboxPatch((8.78, 1.1), 3.25, 0.78, boxstyle="round,pad=0.03,rounding_size=0.08", linewidth=1.05, edgecolor=BLUE_DARK, facecolor="#EEF5F8")
    ax.add_patch(lock)
    ax.text(10.405, 1.55, "0:00 锁定：全天计划＋规则参数", ha="center", fontsize=10.2, fontweight=650)
    ax.text(10.405, 1.27, "日内不得回看未来实际负载或光伏", ha="center", fontsize=8.5, color=MUTED)
    ax.add_patch(FancyArrowPatch((11.6, 2.72), (11.4, 1.92), arrowstyle="-|>", mutation_scale=11, lw=1.1, color=BLUE_DARK))
    ax.add_patch(FancyArrowPatch((12.05, 1.48), (12.62, 1.65), arrowstyle="-|>", mutation_scale=11, lw=1.1, color=BLUE_DARK))

    soc = FancyBboxPatch((4.95, 0.25), 3.25, 0.78, boxstyle="round,pad=0.03,rounding_size=0.08", linewidth=1.05, edgecolor="#6F8C76", facecolor="#F0F5F1")
    ax.add_patch(soc)
    ax.text(6.575, 0.70, "跨日状态：实际期末 SOC", ha="center", fontsize=10.2, fontweight=650)
    ax.text(6.575, 0.42, "作为下一日 LP 与规则评价的初始库存", ha="center", fontsize=8.5, color=MUTED)
    ax.add_patch(FancyArrowPatch((12.95, 1.22), (8.25, 0.70), connectionstyle="arc3,rad=-0.11", arrowstyle="-|>", mutation_scale=11, lw=1.2, color="#6F8C76"))
    ax.add_patch(FancyArrowPatch((4.95, 0.63), (1.10, 2.70), connectionstyle="arc3,rad=-0.18", arrowstyle="-|>", mutation_scale=11, lw=1.2, color="#6F8C76"))
    ax.text(0.35, 4.25, "信息边界", fontsize=9, color=BLUE_DARK, fontweight=650)
    ax.plot([0.35, 12.4], [4.08, 4.08], color=BLUE_DARK, lw=1.0, ls="--", alpha=0.65)
    ax.text(12.45, 4.08, "↓ 真实路径随后揭示", fontsize=8.5, color=MUTED, va="center")
    finish(fig, "fig01_workflow.png")


def strategy_cost_figure(strategy: pd.DataFrame) -> None:
    labels = ["β=1 固定保留", "β=0 即时补缺", "情景价值控制", "LDR 主方案"]
    strategy = strategy.copy()
    strategy["label"] = labels
    order = [3, 2, 1, 0]
    d = strategy.iloc[order].reset_index(drop=True)
    x = np.arange(len(d))
    planned = d["planned_cost"].to_numpy() / 1e4
    emerg = d["emergency_cost"].to_numpy() / 1e4
    total = d["total_cost"].to_numpy() / 1e4
    ldr_total = float(strategy.iloc[3]["total_cost"])
    delta = (strategy["total_cost"].to_numpy() - ldr_total) / 1e4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle("四种策略的正式期费用与相对差额", x=0.06, y=0.99, ha="left", fontsize=16, fontweight=650)
    add_subtitle(fig, "2025-02-01—12-31；左图从零起轴展示真实费用构成，右图聚焦相对 LDR 的差额。", 0.935)
    ax1.bar(x, planned, width=0.62, color=BLUE, edgecolor="#FFFFFF", lw=0.8, label="计划购电费")
    ax1.bar(x, emerg, width=0.62, bottom=planned, color=PEACH, edgecolor="#FFFFFF", lw=0.8, label="紧急购电费")
    for xi, value in zip(x, total):
        ax1.text(xi, value + 20, f"{value:,.1f}", ha="center", fontsize=9, color=INK)
    ax1.set_xticks(x, d["label"], rotation=8)
    ax1.set_ylabel("费用 / 万元")
    ax1.set_ylim(0, total.max() * 1.13)
    ax1.grid(axis="y")
    ax1.legend(loc="upper left", ncol=2)
    ax1.set_title("费用构成（绝对尺度）", loc="left")

    dd = pd.DataFrame({"label": labels, "delta": delta}).iloc[[0, 1, 2, 3]]
    y = np.arange(len(dd))
    colors = [SKY, GOLD, LAVENDER, BLUE_DARK]
    ax2.barh(y, dd["delta"], color=colors, height=0.56, edgecolor="#FFFFFF")
    ax2.axvline(0, color=INK, lw=1)
    for yi, v in zip(y, dd["delta"]):
        ax2.text(v + (0.22 if v >= 0 else -0.22), yi, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=9.3)
    ax2.set_yticks(y, dd["label"])
    ax2.set_xlabel("相对 LDR 多支出 / 万元")
    ax2.grid(axis="x")
    ax2.set_title("聚焦尺度：LDR 与对照的差额", loc="left")
    ax2.text(0.02, -0.22, "情景价值控制仅高 0.514 万元（0.0368%），应表述为数值近似持平。", transform=ax2.transAxes, fontsize=8.7, color=MUTED)
    fig.tight_layout(rect=(0.03, 0.03, 0.99, 0.90))
    finish(fig, "fig02_strategy_costs.png")


def forecast_tradeoff_figure(metrics: pd.DataFrame) -> None:
    formal = metrics[metrics["scope"] == "formal"].copy().set_index("scheme").loc[["b0", "b1", "b2", "gk"]]
    names = ["昨日持久化", "周持久化", "七天均值", "相似日高斯核"]
    colors = [NEUTRAL, BLUE_DARK, GOLD, LAVENDER]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7))
    fig.suptitle("负载预测方案的多指标权衡", x=0.06, y=0.99, ha="left", fontsize=16, fontweight=650)
    add_subtitle(fig, "正式期 334 天；高斯核的 MAPE 更低，周持久化的 RMSE 与日电量误差更稳，不能只按单一指标选型。", 0.93)
    specs = [
        ("mape_pct", "MAPE / %", "越低越好"),
        ("rmse_kw", "RMSE / kW", "越低越好"),
        ("daily_energy_mae_kwh", "日电量 MAE / kWh", "越低越好"),
    ]
    y = np.arange(4)
    for ax, (field, title, note) in zip(axes, specs):
        vals = formal[field].to_numpy()
        ax.hlines(y, 0, vals, color=GRID, lw=2)
        ax.scatter(vals, y, s=72, c=colors, edgecolors="#FFFFFF", linewidths=1.0, zorder=3)
        for yi, v in zip(y, vals):
            label = f"{v:,.2f}" if field == "mape_pct" else f"{v:,.0f}"
            ax.text(v, yi - 0.23, label, ha="center", va="top", fontsize=8.5)
        ax.set_yticks(y, names if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.grid(axis="x")
        ax.set_title(title, loc="left")
        ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=8.4, color=MUTED)
    fig.tight_layout(rect=(0.03, 0.03, 0.99, 0.89))
    finish(fig, "fig03_forecast_tradeoffs.png")


def risk_calibration_figure(schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = schedule.copy()
    s["date"] = pd.to_datetime(s["date"])
    s["month"] = s["date"].dt.month
    s["actual_net_kwh"] = s["load_kwh"] - s["pv_kwh"]
    s["point_net_kwh"] = s["forecast_load_kwh"] - s["forecast_pv_kwh"]
    s["risk_covered"] = s["actual_net_kwh"] <= s["planning_net_kwh"] + 1e-9
    s["point_covered"] = s["actual_net_kwh"] <= s["point_net_kwh"] + 1e-9
    s["risk_buffer_kwh"] = s["planning_net_kwh"] - s["point_net_kwh"]
    monthly = s.groupby("month", as_index=False).agg(
        intervals=("slot", "size"),
        risk_coverage=("risk_covered", "mean"),
        point_coverage=("point_covered", "mean"),
        mean_risk_buffer_kwh=("risk_buffer_kwh", "mean"),
    )
    stage = s.groupby("stage", as_index=False).agg(
        intervals=("slot", "size"),
        risk_coverage=("risk_covered", "mean"),
        point_coverage=("point_covered", "mean"),
        mean_risk_buffer_kwh=("risk_buffer_kwh", "mean"),
    )
    monthly.to_csv(DATA_DIR / "risk_coverage_by_month.csv", index=False, encoding="utf-8-sig")
    stage.to_csv(DATA_DIR / "risk_coverage_by_stage.csv", index=False, encoding="utf-8-sig")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.9), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.suptitle("80%风险曲线的样本外区间覆盖表现", x=0.06, y=0.99, ha="left", fontsize=16, fontweight=650)
    add_subtitle(fig, "覆盖率按 48,096 个正式期10分钟区间计算；它是逐区间校准指标，不是全天可靠率。", 0.93)
    ax1.plot(monthly["month"], monthly["risk_coverage"] * 100, marker="o", color=BLUE_DARK, lw=2, label="风险曲线")
    ax1.plot(monthly["month"], monthly["point_coverage"] * 100, marker="s", mfc=PAPER, color=PEACH, lw=1.6, ls="--", label="点预测")
    ax1.axhline(80, color=INK, lw=1.1, ls=":", label="名义 80% 参考线")
    ax1.set_xticks(range(2, 13))
    ax1.set_xlabel("月份")
    ax1.set_ylabel("实际净负荷不超过曲线的区间占比 / %")
    ax1.grid(axis="y")
    ax1.legend(loc="lower left")
    ax1.set_title("分月覆盖率", loc="left")

    x = np.arange(4)
    ax2.bar(x - 0.18, stage["point_coverage"] * 100, width=0.36, color=PEACH, label="点预测")
    ax2.bar(x + 0.18, stage["risk_coverage"] * 100, width=0.36, color=BLUE, label="风险曲线")
    ax2.axhline(80, color=INK, lw=1.1, ls=":")
    ax2.set_xticks(x, ["0—6时", "6—12时", "12—18时", "18—24时"])
    ax2.set_ylabel("区间覆盖率 / %")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y")
    ax2.set_title("分阶段覆盖率", loc="left")
    for xi, v in zip(x, stage["risk_coverage"] * 100):
        ax2.text(xi + 0.18, v + 1.6, f"{v:.1f}%", ha="center", fontsize=8.2)
    fig.tight_layout(rect=(0.03, 0.03, 0.99, 0.89))
    finish(fig, "fig04_risk_calibration.png")
    return monthly, stage


def controller_figure(schedule: pd.DataFrame) -> pd.DataFrame:
    s = schedule.copy()
    tol = 1e-6
    s["clip_state"] = np.where(
        np.abs(s["reserve_kwh"] - 1200) <= tol,
        "下限截断",
        np.where(np.abs(s["reserve_kwh"] - 10800) <= tol, "上限截断", "区间内部"),
    )
    s["feedback_shift_kwh"] = s["ldr_lambda"] * s["stage_error_mean_kwh"]
    s["total_threshold_shift_kwh"] = s["reserve_kwh"] - s["plan_soc_kwh"]
    comp = s.groupby(["stage", "clip_state"]).size().unstack(fill_value=0)
    comp = comp.reindex(columns=["下限截断", "区间内部", "上限截断"], fill_value=0)
    comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100
    daily_stage = s.drop_duplicates(["date", "stage"])
    summary = daily_stage.groupby("stage", as_index=False).agg(
        day_stage_observations=("date", "size"),
        feedback_q05=("feedback_shift_kwh", lambda x: x.quantile(0.05)),
        feedback_median=("feedback_shift_kwh", "median"),
        feedback_q95=("feedback_shift_kwh", lambda x: x.quantile(0.95)),
        threshold_shift_median=("total_threshold_shift_kwh", "median"),
    )
    for state in comp_pct.columns:
        summary[state + "_pct"] = summary["stage"].map(comp_pct[state])
    summary.to_csv(DATA_DIR / "controller_stage_summary.csv", index=False, encoding="utf-8-sig")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.9), gridspec_kw={"width_ratios": [1.05, 1.25]})
    fig.suptitle("截断仿射保留规则的实际工作状态", x=0.06, y=0.99, ha="left", fontsize=16, fontweight=650)
    add_subtitle(fig, "左图按正式期区间统计；右图每个日期×阶段保留一个反馈值，避免144点重复计权。", 0.93)
    x = np.arange(4)
    bottom = np.zeros(4)
    for state, color in zip(["下限截断", "区间内部", "上限截断"], [PEACH, BLUE, LAVENDER]):
        vals = comp_pct[state].to_numpy()
        ax1.bar(x, vals, bottom=bottom, width=0.62, color=color, edgecolor="#FFFFFF", label=state)
        for xi, b, v in zip(x, bottom, vals):
            if v >= 6:
                ax1.text(xi, b + v / 2, f"{v:.1f}%", ha="center", va="center", fontsize=8.2, color=INK)
        bottom += vals
    ax1.set_xticks(x, ["0—6时", "6—12时", "12—18时", "18—24时"])
    ax1.set_ylabel("区间占比 / %")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper center", ncol=3)
    ax1.set_title("阈值截断状态构成", loc="left")

    sub = daily_stage[daily_stage["stage"] > 1]
    data = [sub.loc[sub["stage"] == k, "feedback_shift_kwh"].to_numpy() for k in [2, 3, 4]]
    bp = ax2.boxplot(data, patch_artist=True, widths=0.5, showfliers=False, whis=(5, 95), medianprops={"color": INK, "lw": 1.4})
    for patch, color in zip(bp["boxes"], [SKY, SAGE, GOLD]):
        patch.set_facecolor(color)
        patch.set_edgecolor(BLUE_DARK)
    for item in bp["whiskers"] + bp["caps"]:
        item.set_color(BLUE_DARK)
    ax2.axhline(0, color=INK, lw=1, ls=":")
    ax2.set_xticks([1, 2, 3], ["6—12时", "12—18时", "18—24时"])
    ax2.set_ylabel("反馈项 λ·a / kWh")
    ax2.grid(axis="y")
    ax2.set_title("反馈项的日—阶段分布（5%—95%）", loc="left")
    fig.tight_layout(rect=(0.03, 0.03, 0.99, 0.89))
    finish(fig, "fig05_controller_interpretability.png")
    return summary


def paired_savings_figure(paired: pd.DataFrame) -> pd.DataFrame:
    p = paired.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p[p["date"] >= "2025-02-01"].copy()
    p["saving_vs_beta1"] = p["beta1_total_cost"] - p["ldr_total_cost"]
    p["saving_vs_beta0"] = p["beta0_total_cost"] - p["ldr_total_cost"]
    p["rolling_beta1"] = p["saving_vs_beta1"].rolling(14, min_periods=1).mean()
    p["rolling_beta0"] = p["saving_vs_beta0"].rolling(14, min_periods=1).mean()
    p["cum_beta1"] = p["saving_vs_beta1"].cumsum()
    p["cum_beta0"] = p["saving_vs_beta0"].cumsum()
    p["month"] = p["date"].dt.month
    monthly = p.groupby("month", as_index=False).agg(
        days=("date", "size"),
        saving_vs_beta1=("saving_vs_beta1", "sum"),
        saving_vs_beta0=("saving_vs_beta0", "sum"),
        ldr_better_beta1_days=("saving_vs_beta1", lambda x: int((x > 1e-8).sum())),
        ldr_better_beta0_days=("saving_vs_beta0", lambda x: int((x > 1e-8).sum())),
    )
    monthly.to_csv(DATA_DIR / "paired_monthly_summary.csv", index=False, encoding="utf-8-sig")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.8, 7.2), sharex=True)
    fig.suptitle("同计划、同日初库存配对试验的节费路径", x=0.06, y=0.99, ha="left", fontsize=16, fontweight=650)
    add_subtitle(fig, "正值表示 LDR 当日费用更低；配对按日重置，不能替代相同初始状态的连续策略回测。", 0.947)
    ax1.plot(p["date"], p["rolling_beta1"] / 1e3, color=BLUE_DARK, lw=1.8, label="相对 β=1（14日均值）")
    ax1.plot(p["date"], p["rolling_beta0"] / 1e3, color=PEACH, lw=1.8, ls="--", label="相对 β=0（14日均值）")
    ax1.axhline(0, color=INK, lw=1, ls=":")
    ax1.set_ylabel("日均节费 / 千元")
    ax1.grid(axis="y")
    ax1.legend(loc="upper left", ncol=2)
    ax1.set_title("14日滚动平均", loc="left")
    ax2.plot(p["date"], p["cum_beta1"] / 1e4, color=BLUE_DARK, lw=2, label="相对 β=1")
    ax2.plot(p["date"], p["cum_beta0"] / 1e4, color=PEACH, lw=2, ls="--", label="相对 β=0")
    ax2.axhline(0, color=INK, lw=1, ls=":")
    ax2.set_ylabel("累计节费 / 万元")
    ax2.grid(axis="y")
    ax2.set_title("累计节费", loc="left")
    ax2.text(p["date"].iloc[-1], p["cum_beta1"].iloc[-1] / 1e4, f"  {p['cum_beta1'].iloc[-1]/1e4:.2f}", va="center", fontsize=9)
    ax2.text(p["date"].iloc[-1], p["cum_beta0"].iloc[-1] / 1e4, f"  {p['cum_beta0'].iloc[-1]/1e4:.2f}", va="center", fontsize=9)
    fig.tight_layout(rect=(0.03, 0.03, 0.99, 0.91))
    finish(fig, "fig06_paired_savings.png")
    return monthly


def sensitivity_figure(main_summary: dict) -> pd.DataFrame:
    rows = []
    main = main_summary[0]
    rows.append(
        {
            "case": "main_21d_seed12_iter8",
            "label": "主方案：21日 / seed12 / 8代",
            "group": "主方案",
            "total_cost": main["formal_period"]["total_cost"],
            "initial_soc_kwh": main["formal_period"]["initial_soc_kwh"],
            "final_soc_kwh": main["formal_period"]["final_soc_kwh"],
        }
    )
    label_map = {
        "seed_20250911": ("随机种子 20250911", "随机种子"),
        "seed_20250913": ("随机种子 20250913", "随机种子"),
        "residual_14d": ("残差窗口 14 日", "残差窗口"),
        "residual_28d": ("残差窗口 28 日", "残差窗口"),
        "maxiter_4": ("搜索预算 4 代", "搜索预算"),
        "maxiter_12": ("搜索预算 12 代", "搜索预算"),
        "delta_4800": ("δ 边界 ±4800 kWh", "参数边界"),
        "lambda_1": ("λ 边界 ±1", "参数边界"),
        "delta_only_lambda0": ("仅截距：λ=0", "结构消融"),
    }
    for path in sorted(SENS_DIR.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj["case"] not in label_map:
            continue
        label, group = label_map[obj["case"]]
        fp = obj["formal_period"]
        rows.append(
            {
                "case": obj["case"],
                "label": label,
                "group": group,
                "total_cost": fp["total_cost"],
                "initial_soc_kwh": fp["initial_soc_kwh"],
                "final_soc_kwh": fp["final_soc_kwh"],
            }
        )
    df = pd.DataFrame(rows)
    base = float(df.loc[df["case"] == "main_21d_seed12_iter8", "total_cost"].iloc[0])
    df["delta_vs_main_yuan"] = df["total_cost"] - base
    order = [
        "residual_14d", "maxiter_12", "maxiter_4", "delta_only_lambda0", "main_21d_seed12_iter8",
        "lambda_1", "seed_20250913", "delta_4800", "residual_28d", "seed_20250911",
    ]
    df["sort"] = df["case"].map({k: i for i, k in enumerate(order)})
    df = df.sort_values("sort").drop(columns="sort")
    df.to_csv(DATA_DIR / "sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    group_colors = {
        "主方案": BLUE_DARK,
        "随机种子": SKY,
        "残差窗口": GOLD,
        "搜索预算": SAGE,
        "参数边界": LAVENDER,
        "结构消融": PEACH,
    }
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    fig.suptitle("LDR主方案的有限敏感性与结构消融", x=0.08, y=0.99, ha="left", fontsize=16, fontweight=650)
    add_subtitle(fig, "横轴为相对主方案的正式期费用变化；负值表示该替代配置在本年度回测费用更低。", 0.945)
    y = np.arange(len(df))
    vals = df["delta_vs_main_yuan"].to_numpy() / 1e4
    colors = [group_colors[g] for g in df["group"]]
    ax.barh(y, vals, color=colors, height=0.57, edgecolor="#FFFFFF")
    ax.axvline(0, color=INK, lw=1.1)
    for yi, v in zip(y, vals):
        ax.text(v + (0.12 if v >= 0 else -0.12), yi, f"{v:+.2f}", ha="left" if v >= 0 else "right", va="center", fontsize=8.8)
    ax.set_yticks(y, df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("相对主方案费用变化 / 万元")
    ax.grid(axis="x")
    handles = [plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=c, markeredgecolor="none", markersize=8, label=g) for g, c in group_colors.items()]
    ax.legend(handles=handles, ncol=3, loc="lower right")
    ax.text(0.01, -0.13, "这些是同一年度的事后敏感性结果，不构成独立外部验证；14日窗口更低，说明21日并非经充分证明的最优窗口。", transform=ax.transAxes, fontsize=8.7, color=MUTED)
    fig.tight_layout(rect=(0.05, 0.06, 0.99, 0.91))
    finish(fig, "fig07_sensitivity.png")
    return df


def key_dates_figure(schedule: pd.DataFrame) -> None:
    dates = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
    fig, axes = plt.subplots(4, 2, figsize=(13, 12.5), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1]})
    fig.suptitle("四个指定日期的供需响应与库存保留", x=0.06, y=0.995, ha="left", fontsize=16, fontweight=650)
    add_subtitle(fig, "左列比较实际净负荷、计划购电与紧急购电；右列单独展示 SOC 与保留阈值，避免双轴混读。", 0.967)
    for row, date in enumerate(dates):
        g = schedule[schedule["date"] == date].copy()
        t = np.arange(len(g))
        net = g["load_kwh"] - g["pv_kwh"]
        ax = axes[row, 0]
        ax.plot(t, net, color=INK, lw=1.25, label="实际净负荷")
        ax.plot(t, g["grid_kwh"], color=BLUE_DARK, lw=1.35, label="计划购电")
        ax.fill_between(t, 0, g["emergency_kwh"], step="mid", color=PEACH, alpha=0.9, label="紧急购电")
        ax.axhline(0, color="#AEB8C2", lw=0.8)
        ax.set_ylabel("区间电量 / kWh")
        ax.grid(axis="y")
        ax.set_title(f"{date}  供需与补购", loc="left", fontsize=11.5)
        axr = axes[row, 1]
        axr.plot(t, g["soc_end_kwh"], color=SAGE, lw=1.55, label="实际 SOC")
        axr.plot(t, g["reserve_kwh"], color=GOLD, lw=1.25, ls="--", label="保留阈值")
        axr.fill_between(t, g["reserve_kwh"], g["soc_end_kwh"], where=g["soc_end_kwh"] >= g["reserve_kwh"], color=SKY, alpha=0.24)
        axr.set_ylim(1000, 11000)
        axr.set_ylabel("库存 / kWh")
        axr.grid(axis="y")
        axr.set_title(f"{date}  SOC 与阈值", loc="left", fontsize=11.5)
    ticks = np.arange(0, 145, 24)
    ticklabels = [f"{h:02d}:00" for h in range(0, 25, 4)]
    for ax in axes[-1, :]:
        ax.set_xticks(ticks, ticklabels)
        ax.set_xlabel("时间")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[0, 1].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, ncol=5, loc="upper center", bbox_to_anchor=(0.57, 0.943))
    fig.tight_layout(rect=(0.03, 0.025, 0.99, 0.93))
    finish(fig, "fig08_key_dates.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    strategy = pd.read_csv(ROOT / "outputs" / "question2" / "current" / "paper" / "tables" / "strategy_comparison.csv")
    schedule = pd.read_csv(ROOT / "outputs" / "question2" / "current" / "ldr" / "question2_schedule.csv")
    paired = pd.read_csv(ROOT / "outputs" / "question2" / "current" / "ldr" / "same_plan_controller_comparison.csv")
    metrics = pd.read_csv(ROOT / "outputs" / "question2" / "analysis" / "load_forecast" / "model_metrics.csv")
    main_summary = json.loads((ROOT / "outputs" / "question2" / "current" / "ldr" / "question2_summary.json").read_text(encoding="utf-8"))

    workflow_figure()
    strategy_cost_figure(strategy)
    forecast_tradeoff_figure(metrics)
    risk_month, risk_stage = risk_calibration_figure(schedule)
    controller = controller_figure(schedule)
    paired_month = paired_savings_figure(paired)
    sensitivity = sensitivity_figure(main_summary)
    key_dates_figure(schedule)

    summary = {
        "risk_coverage_overall": float(((schedule["load_kwh"] - schedule["pv_kwh"]) <= schedule["planning_net_kwh"] + 1e-9).mean()),
        "point_coverage_overall": float(((schedule["load_kwh"] - schedule["pv_kwh"]) <= (schedule["forecast_load_kwh"] - schedule["forecast_pv_kwh"]) + 1e-9).mean()),
        "reserve_at_min_pct": float((np.abs(schedule["reserve_kwh"] - 1200) <= 1e-6).mean() * 100),
        "reserve_at_max_pct": float((np.abs(schedule["reserve_kwh"] - 10800) <= 1e-6).mean() * 100),
        "feedback_shift_over_1kwh_pct": float((np.abs(schedule["ldr_lambda"] * schedule["stage_error_mean_kwh"]) > 1).mean() * 100),
        "paired_saving_beta1_yuan": float((paired.loc[pd.to_datetime(paired["date"]) >= "2025-02-01", "beta1_total_cost"] - paired.loc[pd.to_datetime(paired["date"]) >= "2025-02-01", "ldr_total_cost"]).sum()),
        "paired_saving_beta0_yuan": float((paired.loc[pd.to_datetime(paired["date"]) >= "2025-02-01", "beta0_total_cost"] - paired.loc[pd.to_datetime(paired["date"]) >= "2025-02-01", "ldr_total_cost"]).sum()),
        "generated_figures": sorted(p.name for p in FIG_DIR.glob("*.png")),
    }
    (DATA_DIR / "enhanced_evidence_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
