"""Generate the six Question 2 academic figures requested by the writing guide.

The visual language deliberately follows the Question 1 paper figures:
Okabe-Ito blue/sky/orange/green/vermillion/purple on a white background,
quiet grey grids, direct labels, and 400 dpi PNG plus print-ready PDF exports.

All quantitative panels are rebuilt from the saved Question 2 tables or from
the same causal scenario constructor used by the optimization model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import fitz
import numpy as np
import pandas as pd

from src.optimization.question2 import forecasts, load_forecast_weekly_persist, load_inputs
from src.optimization.question2_g_search import scenario_net_matrix


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "question2" / "academic_figures"
TABLES = ROOT / "outputs" / "question2" / "current" / "paper" / "tables"
SCHEDULE_PATH = ROOT / "outputs" / "question2" / "current" / "ldr" / "question2_schedule.csv"

# Palette copied from the shipped Question 1 figures.
COLORS = {
    "ink": "#222222",
    "muted": "#6B7280",
    "grid": "#D9DEE7",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#7A5195",
    "yellow": "#F0E442",
    "blue_light": "#DDEEF7",
    "orange_light": "#F8E8C6",
    "grey_light": "#EEF1F4",
    "white": "#FFFFFF",
}

KEY_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans SC",
                "Source Han Sans CN",
                "Microsoft YaHei",
                "SimHei",
                "DejaVu Sans",
            ],
            "font.size": 8.5,
            "axes.titlesize": 10.2,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "axes.unicode_minus": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "legend.frameon": False,
            "lines.linewidth": 1.45,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
        }
    )


def clean_axes(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3.0, width=0.7)


def panel_title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontweight="bold", pad=7)


def figure_title(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.suptitle(title, x=0.5, y=0.988, fontsize=12.5, fontweight="bold")
    if subtitle:
        fig.text(
            0.5,
            0.951,
            subtitle,
            ha="center",
            va="top",
            fontsize=7.7,
            color=COLORS["muted"],
        )


def add_note(fig: plt.Figure, text: str, y: float = 0.012) -> None:
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=6.8, color=COLORS["muted"])


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    # Matplotlib's Windows CJK font subsetting is not portable across all PDF
    # viewers. Embed the already-QA'd 400 dpi image on a one-page PDF instead;
    # this keeps every Chinese glyph identical to the PNG readers will see.
    pix = fitz.Pixmap(str(png))
    page_width = pix.width * 72.0 / 400.0
    page_height = pix.height * 72.0 / 400.0
    pix = None
    document = fitz.open()
    page = document.new_page(width=page_width, height=page_height)
    page.insert_image(page.rect, filename=str(png), keep_proportion=True)
    document.save(str(pdf), deflate=True, garbage=4)
    document.close()
    return {"png": str(png), "pdf": str(pdf)}


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strategy = pd.read_csv(TABLES / "strategy_comparison.csv")
    monthly = pd.read_csv(TABLES / "monthly_summary.csv")
    paired = pd.read_csv(TABLES / "monthly_paired_savings.csv")
    schedule = pd.read_csv(SCHEDULE_PATH, parse_dates=["date", "interval_start"])
    paired = paired.rename(columns={"date": "month"})
    for frame, columns in [
        (strategy, ["planned_cost", "emergency_cost", "total_cost", "emergency_kwh"]),
        (monthly, ["planned_cost", "emergency_cost", "total_cost"]),
        (paired, ["saving_vs_beta1", "saving_vs_beta0"]),
        (
            schedule,
            [
                "forecast_load_kwh",
                "forecast_pv_kwh",
                "planning_net_kwh",
                "load_kwh",
                "pv_kwh",
                "grid_kwh",
                "emergency_kwh",
                "soc_start_kwh",
                "soc_end_kwh",
                "reserve_kwh",
                "total_cost",
            ],
        ),
    ]:
        if frame[columns].isna().any().any():
            raise ValueError(f"Missing values detected in required fields: {columns}")
    if len(strategy) != 4:
        raise ValueError(f"Expected four strategy rows, found {len(strategy)}")
    return strategy, monthly, paired, schedule


def _workflow_node(
    ax: plt.Axes,
    x: float,
    title: str,
    detail: str,
    *,
    edge: str,
    fill: str,
    number: int,
) -> None:
    w, h, y = 0.105, 0.34, 0.37
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.012,
        y + h - 0.045,
        f"{number:02d}",
        color=edge,
        fontsize=7.0,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        x + w / 2,
        y + 0.225,
        title,
        ha="center",
        va="center",
        fontsize=8.1,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        x + w / 2,
        y + 0.095,
        detail,
        ha="center",
        va="center",
        fontsize=6.3,
        color=COLORS["muted"],
        linespacing=1.25,
    )


def _workflow_arrow(ax: plt.Axes, x0: float, x1: float, color: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, 0.54),
            (x1, 0.54),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color=color,
        )
    )


def figure_1_workflow() -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(11.8, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    figure_title(
        fig,
        "图1  问题二：从历史信息到日内因果执行的闭环",
        "全部计划与参数在每日00:00锁定；日内只使用当时已经揭示的信息",
    )

    xs = np.linspace(0.035, 0.86, 8)
    nodes = [
        ("历史信息", "负载与光伏\n仅截至 $d-1$", COLORS["blue"], COLORS["blue_light"]),
        ("点预测", "周持久化负载\n＋光伏7日均值", COLORS["blue"], COLORS["white"]),
        ("残差情景", "最近21日成对残差\n构造净负荷情景", COLORS["purple"], "#EEE8F2"),
        ("风险曲线", "经验80%分位数\n对应五倍补购惩罚", COLORS["vermillion"], "#F7E8E2"),
        ("日前两阶段LP", "锁定计划购电 $g^Q$\n与参考库存 $E^p$", COLORS["blue"], COLORS["blue_light"]),
        ("规则校准", "差分进化搜索\n7个LDR阈值参数", COLORS["purple"], "#EEE8F2"),
        ("日内执行", "0/6/12/18时更新 $a_k$\n逐10分钟因果响应", COLORS["orange"], COLORS["orange_light"]),
        ("库存传递", "SOC跨日连续\n进入下一日状态", COLORS["green"], "#E2F2EC"),
    ]
    for i, (x, (title, detail, edge, fill)) in enumerate(zip(xs, nodes), start=1):
        _workflow_node(ax, x, title, detail, edge=edge, fill=fill, number=i)
    for i in range(7):
        _workflow_arrow(ax, xs[i] + 0.107, xs[i + 1] - 0.004, COLORS["muted"])

    # Closed-loop state carryover, drawn below the main information flow.
    loop = FancyArrowPatch(
        (xs[-1] + 0.052, 0.355),
        (xs[0] + 0.052, 0.355),
        connectionstyle="arc3,rad=-0.18",
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=1.0,
        linestyle=(0, (4, 2)),
        color=COLORS["green"],
    )
    ax.add_patch(loop)
    ax.text(
        0.50,
        0.105,
        "滚动闭环：跨日库存与新增历史观测共同进入下一日决策",
        ha="center",
        va="center",
        fontsize=7.8,
        color=COLORS["green"],
        fontweight="bold",
    )
    ax.text(
        0.035,
        0.80,
        "日前层  ·  预测 -> 风险 -> 计划",
        fontsize=7.6,
        color=COLORS["blue"],
        fontweight="bold",
    )
    ax.text(
        0.624,
        0.80,
        "执行层  ·  校准 -> 控制 -> 状态",
        fontsize=7.6,
        color=COLORS["orange"],
        fontweight="bold",
    )
    return save_figure(fig, "图1_问题二求解闭环流程")


def _strategy_labels(names: Iterable[str]) -> list[str]:
    mapping = {
        "分位数＋β=1": "分位数\nβ=1",
        "分位数＋β=0": "分位数\nβ=0",
        "情景价值控制": "情景价值\n控制",
        "分位数＋LDR": "分位数\nLDR",
    }
    return [mapping.get(str(name), str(name)) for name in names]


def figure_2_strategy_costs(strategy: pd.DataFrame) -> dict[str, str]:
    labels = _strategy_labels(strategy["方案"])
    planned = strategy["planned_cost"].to_numpy(float) / 1e4
    emergency = strategy["emergency_cost"].to_numpy(float) / 1e4
    total = strategy["total_cost"].to_numpy(float) / 1e4
    x = np.arange(len(strategy))
    ldr = strategy["方案"].astype(str).str.contains("LDR").to_numpy()

    fig = plt.figure(figsize=(8.2, 4.8))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.58, 1.0],
        left=0.085,
        right=0.98,
        bottom=0.19,
        top=0.80,
        wspace=0.30,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    figure_title(
        fig,
        "图2  四种策略正式期费用对比",
        "2025-02-01 至 12-31；各策略独立连续运行，柱形均从零开始",
    )

    plan_colors = np.where(ldr, COLORS["blue"], COLORS["blue_light"])
    plan_edges = np.where(ldr, COLORS["blue"], COLORS["sky"])
    emergency_colors = np.where(ldr, COLORS["vermillion"], COLORS["orange_light"])
    emergency_edges = np.where(ldr, COLORS["vermillion"], COLORS["orange"])
    ax1.bar(x, planned, width=0.62, color=plan_colors, edgecolor=plan_edges, linewidth=0.8)
    ax1.bar(
        x,
        emergency,
        width=0.62,
        bottom=planned,
        color=emergency_colors,
        edgecolor=emergency_edges,
        linewidth=0.8,
        hatch=["//" if not flag else "///" for flag in ldr],
    )
    for xi, value in zip(x, total):
        ax1.text(xi, value + 17, f"{value:,.2f}", ha="center", va="bottom", fontsize=7.7)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("正式期费用 / 万元")
    ax1.set_ylim(0, max(total) * 1.13)
    panel_title(ax1, "(a) 实际账单构成")
    clean_axes(ax1, "y")
    ax1.legend(
        handles=[
            Patch(facecolor=COLORS["blue"], edgecolor=COLORS["blue"], label="计划购电费"),
            Patch(
                facecolor=COLORS["orange_light"],
                edgecolor=COLORS["orange"],
                hatch="//",
                label="紧急购电费",
            ),
        ],
        loc="upper left",
        ncol=2,
    )

    e_colors = np.where(ldr, COLORS["vermillion"], COLORS["orange_light"])
    e_edges = np.where(ldr, COLORS["vermillion"], COLORS["orange"])
    ax2.bar(x, emergency, width=0.62, color=e_colors, edgecolor=e_edges, linewidth=0.9)
    for xi, value in zip(x, emergency):
        ax2.text(xi, value + 2.0, f"{value:,.2f}", ha="center", va="bottom", fontsize=7.6)
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("紧急购电费 / 万元")
    ax2.set_ylim(0, max(emergency) * 1.23)
    panel_title(ax2, "(b) 紧急购电费用放大")
    clean_axes(ax2, "y")
    ldr_total = float(total[ldr][0])
    beta1_total = float(total[strategy["方案"].eq("分位数＋β=1")][0])
    saving = beta1_total - ldr_total
    ax2.text(
        0.04,
        0.96,
        f"LDR相对β=1\n节费 {saving:.2f} 万元（{saving / beta1_total:.2%}）",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        color=COLORS["blue"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": COLORS["grey_light"], "edgecolor": "none"},
    )
    add_note(
        fig,
        "注：实际账单仅计计划购电费与五倍紧急购电费；深色柱为LDR。总费用差异较小，故右图单独放大紧急购电费。",
    )
    return save_figure(fig, "图2_四策略正式期费用对比")


def figure_3_monthly_and_paired(monthly: pd.DataFrame, paired: pd.DataFrame) -> dict[str, str]:
    monthly = monthly.sort_values("month")
    paired = paired.sort_values("month")
    months = monthly["month"].to_numpy(int)
    x = np.arange(len(months))
    planned = monthly["planned_cost"].to_numpy(float) / 1e4
    emergency = monthly["emergency_cost"].to_numpy(float) / 1e4
    s1 = paired["saving_vs_beta1"].to_numpy(float) / 1e4
    s0 = paired["saving_vs_beta0"].to_numpy(float) / 1e4

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8.2, 6.4),
        gridspec_kw={"height_ratios": [1.18, 1.0], "hspace": 0.34},
    )
    figure_title(
        fig,
        "图3  LDR月度账单与同计划配对节费",
        "上图为独立连续运行账单；下图固定同一计划与同日初库存，仅比较控制规则",
    )
    ax1.bar(x, planned, width=0.62, color=COLORS["blue"], label="计划购电费")
    ax1.bar(
        x,
        emergency,
        width=0.62,
        bottom=planned,
        color=COLORS["orange"],
        edgecolor=COLORS["vermillion"],
        linewidth=0.45,
        hatch="//",
        label="紧急购电费",
    )
    ax1.set_xticks(x, [f"{m}月" for m in months])
    ax1.set_ylabel("LDR月度费用 / 万元")
    panel_title(ax1, "(a) LDR月度实际账单构成")
    ax1.legend(loc="upper left", ncol=2)
    clean_axes(ax1, "y")
    totals = planned + emergency
    for xi in np.argsort(totals)[-3:]:
        ax1.text(xi, totals[xi] + 2.1, f"{totals[xi]:.1f}", ha="center", fontsize=7.1)

    ax2.axhline(0, color=COLORS["ink"], linewidth=0.75)
    ax2.plot(
        x,
        s1,
        color=COLORS["blue"],
        marker="o",
        markersize=4.3,
        markerfacecolor=COLORS["white"],
        markeredgewidth=1.0,
        label="相对β=1",
    )
    ax2.plot(
        x,
        s0,
        color=COLORS["orange"],
        linestyle=(0, (4, 2)),
        marker="s",
        markersize=4.1,
        markerfacecolor=COLORS["white"],
        markeredgewidth=1.0,
        label="相对β=0",
    )
    ax2.fill_between(x, 0, s1, color=COLORS["sky"], alpha=0.10)
    ax2.set_xticks(x, [f"{m}月" for m in months])
    ax2.set_xlabel("月份")
    ax2.set_ylabel("月度配对节费 / 万元")
    panel_title(ax2, "(b) 同计划、同日初库存的月度配对节费（正值为LDR更低）")
    ax2.legend(loc="upper right", ncol=2)
    clean_axes(ax2, "y")
    ax2.text(
        0.02,
        0.94,
        f"正式期合计：相对β=1节费 {s1.sum():.2f} 万元；相对β=0节费 {s0.sum():.2f} 万元",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=COLORS["muted"],
    )
    add_note(fig, "注：两面板统计口径不同，不应将配对节费逐月相加后替代独立连续运行的年度账单差额。")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.88)
    return save_figure(fig, "图3_月度账单与配对节费")


def _format_day_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 6), [f"{h:02d}:00" for h in range(0, 25, 6)])
    ax.axhline(0, color=COLORS["muted"], linewidth=0.65)
    clean_axes(ax, "y")


def figure_4_key_dates(schedule: pd.DataFrame) -> dict[str, str]:
    fig, axes = plt.subplots(2, 2, figsize=(11.1, 7.2), sharex=True)
    figure_title(
        fig,
        "图4  四个指定日期的计划购电、储能状态与紧急响应",
        "左轴为每10分钟电量，右轴为SOC与保留阈值；红色窄柱标出紧急购电",
    )
    legend_handles: list[Line2D | Patch] | None = None
    for ax, date in zip(axes.ravel(), KEY_DATES):
        day = schedule.loc[schedule["date"].eq(pd.Timestamp(date))].sort_values("slot").copy()
        if len(day) != 144:
            raise ValueError(f"Expected 144 intervals for {date}, found {len(day)}")
        t = day["slot"].to_numpy(float) / 6.0
        net = day["load_kwh"].to_numpy(float) - day["pv_kwh"].to_numpy(float)
        plan = day["grid_kwh"].to_numpy(float)
        emergency = day["emergency_kwh"].to_numpy(float)
        soc = day["soc_end_kwh"].to_numpy(float)
        reserve = day["reserve_kwh"].to_numpy(float)
        total_cost = day["total_cost"].sum()

        line_net = ax.plot(t, net, color=COLORS["ink"], linewidth=1.20, label="实际净负荷")[0]
        line_plan = ax.step(
            t,
            plan,
            where="post",
            color=COLORS["blue"],
            linewidth=1.05,
            alpha=0.92,
            label="计划购电",
        )[0]
        ax.bar(
            t,
            emergency,
            width=1 / 6 * 0.90,
            color=COLORS["vermillion"],
            linewidth=0,
            zorder=4,
            label="紧急购电",
        )
        active = emergency > 1e-9
        if active.any():
            ax.scatter(
                t[active],
                net[active],
                s=28,
                marker="v",
                color=COLORS["vermillion"],
                edgecolor=COLORS["white"],
                linewidth=0.55,
                zorder=6,
            )
            for slot in t[active]:
                ax.axvspan(slot, slot + 1 / 6, color=COLORS["vermillion"], alpha=0.08, linewidth=0)

        ax2 = ax.twinx()
        line_soc = ax2.plot(t, soc, color=COLORS["green"], linewidth=1.25, label="SOC")[0]
        line_reserve = ax2.step(
            t,
            reserve,
            where="post",
            color=COLORS["orange"],
            linestyle=(0, (4, 2)),
            linewidth=1.05,
            label="保留阈值",
        )[0]
        ax2.set_ylim(0, 11200)
        ax2.set_yticks([1200, 6000, 10800])
        ax2.tick_params(direction="out", length=2.5, width=0.6, colors=COLORS["muted"], labelsize=7.0)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_color(COLORS["grid"])
        ax2.spines["left"].set_visible(False)
        ax2.set_ylabel("SOC / kWh", color=COLORS["muted"], fontsize=7.5)
        ax.set_title(f"{date}  ·  当日总费 {total_cost / 1e4:.2f} 万元", loc="left", fontweight="bold")
        ax.set_ylabel("每10分钟电量 / kWh")
        _format_day_axis(ax)
        if legend_handles is None:
            legend_handles = [
                line_net,
                line_plan,
                Patch(facecolor=COLORS["vermillion"], label="紧急购电"),
                line_soc,
                line_reserve,
            ]

    for ax in axes[-1, :]:
        ax.set_xlabel("时刻")
    if legend_handles:
        fig.legend(
            legend_handles,
            [item.get_label() for item in legend_handles],
            ncol=5,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.915),
            columnspacing=1.5,
            handlelength=2.4,
        )
    add_note(
        fig,
        "注：计划购电量在00:00锁定；保留阈值是停止继续放电的参考而非强制补电目标。2025-09-23的红色标记对应两次短时紧急购电。",
    )
    fig.subplots_adjust(left=0.075, right=0.94, bottom=0.10, top=0.84, hspace=0.29, wspace=0.28)
    return save_figure(fig, "图4_四个指定日期日内运行")


def figure_5_risk_curve(schedule: pd.DataFrame) -> dict[str, str]:
    date = "2025-09-23"
    day = schedule.loc[schedule["date"].eq(pd.Timestamp(date))].sort_values("slot").copy()
    t = day["slot"].to_numpy(float) / 6.0
    point = day["forecast_load_kwh"].to_numpy(float) - day["forecast_pv_kwh"].to_numpy(float)
    risk = day["planning_net_kwh"].to_numpy(float)
    actual = day["load_kwh"].to_numpy(float) - day["pv_kwh"].to_numpy(float)
    emergency = day["emergency_kwh"].to_numpy(float) > 1e-9

    fig, ax = plt.subplots(figsize=(8.2, 4.35))
    figure_title(
        fig,
        "图5  关键日点预测、80%分位风险曲线与实际净负荷",
        "示例日2025-09-23；风险修正仅使用此前21个完整历史日的成对残差",
    )
    ax.fill_between(
        t,
        point,
        risk,
        where=risk >= point,
        color=COLORS["sky"],
        alpha=0.18,
        interpolate=True,
        label="风险抬升区间",
    )
    ax.fill_between(
        t,
        point,
        risk,
        where=risk < point,
        color=COLORS["orange"],
        alpha=0.12,
        interpolate=True,
        label="风险下调区间",
    )
    ax.plot(t, point, color=COLORS["muted"], linestyle=(0, (4, 2)), linewidth=1.05, label="点预测净负荷")
    ax.plot(t, risk, color=COLORS["blue"], linewidth=1.45, label="80%分位风险曲线")
    ax.plot(t, actual, color=COLORS["ink"], linewidth=1.25, label="实际净负荷")
    ax.scatter(
        t[emergency],
        actual[emergency],
        s=40,
        marker="v",
        color=COLORS["vermillion"],
        edgecolor=COLORS["white"],
        linewidth=0.7,
        zorder=6,
        label="紧急购电时段",
    )
    for slot, value in zip(t[emergency], actual[emergency]):
        hh = int(slot)
        mm = int(round((slot - hh) * 60))
        ax.annotate(
            f"{hh:02d}:{mm:02d}",
            xy=(slot, value),
            xytext=(0, 16),
            textcoords="offset points",
            ha="center",
            fontsize=7.0,
            color=COLORS["vermillion"],
            arrowprops={"arrowstyle": "-", "color": COLORS["vermillion"], "lw": 0.65},
        )
    ax.axhline(0, color=COLORS["muted"], linewidth=0.7)
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4), [f"{h:02d}:00" for h in range(0, 25, 4)])
    ax.set_xlabel("时刻")
    ax.set_ylabel("每10分钟净负荷 / kWh")
    ax.legend(loc="upper left", ncol=3, handlelength=2.3, columnspacing=1.2)
    clean_axes(ax, "y")
    add_note(
        fig,
        "注：80%分位数由普通购电价与五倍紧急购电价的单时段报童权衡导出；在储能耦合下它是风险设计，不等同于全天80%可靠性保证。",
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.84)
    return save_figure(fig, "图5_关键日风险修正与实际净负荷")


def figure_6_scenario_distribution(schedule: pd.DataFrame) -> dict[str, str]:
    dates, load, pv, _ = load_inputs()
    fl = load_forecast_weekly_persist(load)
    fv = forecasts(load, pv)[1]
    target_date = pd.Timestamp("2025-09-23")
    date_index = int(np.flatnonzero(dates == target_date)[0])
    scenarios, count = scenario_net_matrix(date_index, load, pv, fl, fv, residual_days=21)
    if scenarios is None or count != 21:
        raise ValueError(f"Expected 21 scenarios, found {count}")
    slot = 90
    samples = np.asarray(scenarios[:, slot], dtype=float)
    point = float(fl[date_index, slot] - fv[date_index, slot])
    q80 = float(np.quantile(samples, 0.8, method="linear"))
    actual_row = schedule.loc[
        schedule["date"].eq(target_date) & schedule["slot"].eq(slot)
    ].iloc[0]
    actual = float(actual_row["load_kwh"] - actual_row["pv_kwh"])

    fig = plt.figure(figsize=(8.2, 4.35))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.42, 1.0],
        left=0.09,
        right=0.98,
        bottom=0.18,
        top=0.80,
        wspace=0.30,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    figure_title(
        fig,
        "图6  单时段净负荷情景的经验分布与80%分位定位",
        "2025-09-23 15:00 至 15:10；21条情景来自此前21个完整历史日的成对残差",
    )

    bins = np.histogram_bin_edges(samples, bins=8)
    ax1.hist(
        samples,
        bins=bins,
        color=COLORS["blue_light"],
        edgecolor=COLORS["blue"],
        linewidth=0.8,
        label="历史残差情景",
    )
    rug_y = np.full_like(samples, -0.16, dtype=float)
    ax1.scatter(samples, rug_y, marker="|", s=45, color=COLORS["blue"], alpha=0.75, clip_on=False)
    ax1.axvline(point, color=COLORS["muted"], linestyle=(0, (4, 2)), linewidth=1.2, label=f"点预测 {point:.1f}")
    ax1.axvline(q80, color=COLORS["orange"], linewidth=1.55, label=f"80%分位 {q80:.1f}")
    ax1.axvline(actual, color=COLORS["vermillion"], linewidth=1.15, linestyle=(0, (2, 2)), label=f"实际 {actual:.1f}")
    ax1.set_xlabel("净负荷情景值 / kWh")
    ax1.set_ylabel("情景数")
    panel_title(ax1, "(a) 情景直方图与样本刻度")
    ax1.legend(loc="upper left")
    clean_axes(ax1, "y")

    ordered = np.sort(samples)
    ecdf = np.arange(1, len(ordered) + 1) / len(ordered)
    ax2.step(ordered, ecdf, where="post", color=COLORS["blue"], linewidth=1.5)
    ax2.scatter(
        ordered,
        ecdf,
        s=20,
        facecolor=COLORS["white"],
        edgecolor=COLORS["blue"],
        linewidth=0.75,
        zorder=3,
    )
    ax2.axhline(0.8, color=COLORS["muted"], linestyle=(0, (3, 2)), linewidth=0.9)
    ax2.axvline(q80, color=COLORS["orange"], linewidth=1.2)
    ax2.scatter([q80], [0.8], s=42, color=COLORS["orange"], edgecolor=COLORS["white"], linewidth=0.6, zorder=4)
    ax2.annotate(
        rf"$q_{{0.8}}$ = {q80:.1f} kWh",
        xy=(q80, 0.8),
        xytext=(10, -28),
        textcoords="offset points",
        fontsize=7.5,
        color=COLORS["orange"],
        arrowprops={"arrowstyle": "-", "color": COLORS["orange"], "lw": 0.7},
    )
    ax2.set_ylim(0, 1.03)
    ax2.set_xlabel("净负荷情景值 / kWh")
    ax2.set_ylabel("经验累积概率")
    panel_title(ax2, "(b) 经验分布函数")
    clean_axes(ax2, "both")
    ax2.text(
        0.04,
        0.97,
        "临界分位：\n1 - c/(5c) = 0.80",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        color=COLORS["purple"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#EEE8F2", "edgecolor": "none"},
    )
    add_note(
        fig,
        "注：经验分位使用线性插值；红色实际值仅作事后说明，不参与当日计划生成。该时段实际净负荷高于风险分位并发生紧急购电。",
    )
    return save_figure(fig, "图6_净负荷情景经验分布")


def contact_sheet(png_paths: list[Path]) -> str:
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 13.8), facecolor="#F4F6F8")
    for ax, path in zip(axes.ravel(), png_paths):
        ax.imshow(plt.imread(path))
        ax.set_title(path.stem, loc="left", fontsize=9.2, fontweight="bold", pad=7)
        ax.axis("off")
    fig.suptitle("问题二学术图表总览", x=0.055, y=0.994, ha="left", fontsize=14.0, fontweight="bold")
    fig.text(
        0.055,
        0.976,
        "问题一同源配色；最终排版请使用独立400 dpi PNG或矢量PDF，不使用本缩略图。",
        fontsize=8.0,
        color=COLORS["muted"],
    )
    fig.tight_layout(rect=(0.03, 0.02, 0.98, 0.965), h_pad=2.0, w_pad=1.2)
    path = OUT / "问题二_学术图表总览.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return str(path)


def write_guide(outputs: list[dict[str, str]], preview: str) -> str:
    captions = [
        ("F1 / 图1", "问题二求解流程：预测、情景、日前计划、规则校准与日内执行", "整体求解流程如图1所示，全流程只使用决策时刻之前的信息。"),
        ("F2 / 图2", "四种策略正式期费用对比（堆叠柱为计划费与紧急费，柱顶为总费）", "图2显示LDR的总费与紧急购电费均低于三种对照，其中紧急费用下降是主要节费来源。"),
        ("F3 / 图3", "(a) LDR分月实际账单；(b) 同计划配对试验的月度节费", "图3同时区分独立年度账单与配对控制器归因，避免混用两种口径。"),
        ("F4 / 图4", "四个指定日期的净负荷、计划购电、紧急购电、SOC与保留阈值", "图4显示9月23日午间光伏盈余推动储能充电，15:00与20:10的短时缺口触发紧急购电。"),
        ("F5 / 图5", "2025-09-23点预测、80%分位风险曲线与实际净负荷", "图5中风险曲线相对点预测形成分时修正，实际净负荷在两个时段触发紧急补购。"),
        ("F6 / 图6", "2025-09-23 15:00 至 15:10净负荷情景的经验分布", "图6从经验分布角度展示80%分位数在计划与补购成本权衡中的位置。"),
    ]
    lines = [
        "# 问题二学术图表使用说明",
        "",
        "- 配色：沿用问题一的蓝、天蓝、橙、绿、朱红、紫色盲友好色板。",
        "- 输出：每图同时提供 400 dpi PNG 与中文字体稳定的印刷版 PDF。",
        "- 数据：F2-F5读取当前结果 CSV；F6调用与模型一致的21日因果情景构造。",
        "- 总览：仅用于快速审图，论文排版请用独立图件。",
        "",
        "| 图号 | 建议图注 | 正文引图句 |",
        "|---|---|---|",
    ]
    for number, caption, sentence in captions:
        lines.append(f"| {number} | {caption} | {sentence} |")
    lines += ["", "## 文件", ""]
    for item in outputs:
        lines.append(f"- `{Path(item['png']).name}` / `{Path(item['pdf']).name}`")
    lines.append(f"- `{Path(preview).name}`")
    path = OUT / "图表使用说明.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def main() -> None:
    configure_style()
    strategy, monthly, paired, schedule = read_inputs()
    outputs = [
        figure_1_workflow(),
        figure_2_strategy_costs(strategy),
        figure_3_monthly_and_paired(monthly, paired),
        figure_4_key_dates(schedule),
        figure_5_risk_curve(schedule),
        figure_6_scenario_distribution(schedule),
    ]
    preview = contact_sheet([Path(item["png"]) for item in outputs])
    guide = write_guide(outputs, preview)
    payload = {"figures": outputs, "preview": preview, "guide": guide}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
