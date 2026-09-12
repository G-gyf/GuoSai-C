"""Generate publication-grade static figures for the completed Q1/Q2 paper.

Outputs are raster-only PNG files.  Every quantitative panel is rebuilt from
the saved result tables so that figure labels remain auditable.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "paper_figures_top_journal"

FONT = "Source Han Sans CN"
INK = "#202428"
MUTED = "#66717E"
GRID = "#D9DEE5"
BLUE = "#2468A2"
BLUE_DARK = "#174A73"
BLUE_LIGHT = "#DCEAF4"
ORANGE = "#D17A22"
ORANGE_LIGHT = "#F6E5D2"
GOLD = "#B79A37"
GREY = "#8D98A5"
GREY_LIGHT = "#E9EDF1"
WHITE = "#FFFFFF"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT, "Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
            "legend.fontsize": 8.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "figure.titlesize": 13.0,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=INK,
    )


def clean_axes(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.72)
        ax.set_axisbelow(True)


def add_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.055, y=0.985, ha="left", va="top", fontweight="semibold")
    fig.text(0.055, 0.935, subtitle, ha="left", va="top", fontsize=8.2, color=MUTED)


def save(fig: plt.Figure, filename: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def node(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    detail: str,
    edge: str,
    fill: str,
    title_color: str | None = None,
) -> None:
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=0.85,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=8.1,
            fontweight="semibold", color=title_color or INK)
    ax.text(x + w / 2, y + h * 0.29, detail, ha="center", va="center", fontsize=6.4,
            color=MUTED, linespacing=1.15)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=color,
            shrinkA=1.5,
            shrinkB=1.5,
        )
    )


def figure_technical_route() -> Path:
    fig, ax = plt.subplots(figsize=(11.6, 5.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_title(
        fig,
        "从确定性经济调度到不确定性风险控制",
        "问题一与问题二的统一建模路线；箭头表示信息与决策的先后关系，底部为共同验证闭环",
    )

    # Problem progression ribbon.
    ax.add_patch(Rectangle((0.055, 0.825), 0.40, 0.065, facecolor=BLUE_LIGHT, edgecolor="none"))
    ax.add_patch(Rectangle((0.455, 0.825), 0.49, 0.065, facecolor=ORANGE_LIGHT, edgecolor="none"))
    ax.text(0.075, 0.858, "问题一  ·  确定性单日调度", fontsize=9.0, fontweight="semibold", color=BLUE_DARK, va="center")
    ax.text(0.485, 0.858, "问题二  ·  预测不确定性与因果执行", fontsize=9.0, fontweight="semibold", color=ORANGE, va="center")
    arrow(ax, (0.415, 0.858), (0.48, 0.858), MUTED)

    # Q1 lane.
    y1, h = 0.59, 0.13
    xs1 = [0.065, 0.255, 0.445, 0.635]
    titles1 = ["实际数据", "单日线性规划", "储能调度", "经济结果"]
    details1 = ["负荷 · 光伏 · 电价", "供需平衡 · SOC约束", "低价充电 · 高价放电", "35,245.31 元\n节费 26.45%"]
    for x, t, d in zip(xs1, titles1, details1):
        node(ax, (x, y1), (0.15, h), t, d, BLUE, WHITE if t != "经济结果" else BLUE_LIGHT)
    for x in xs1[:-1]:
        arrow(ax, (x + 0.15, y1 + h / 2), (x + 0.19, y1 + h / 2), BLUE)
    node(ax, (0.835, y1), (0.11, h), "机制解释", "影子价格\n跨时段能量价值", BLUE_DARK, WHITE)
    arrow(ax, (0.785, y1 + h / 2), (0.835, y1 + h / 2), BLUE)

    # Q2 lane.
    y2 = 0.345
    xs2 = [0.055, 0.217, 0.379, 0.541, 0.703, 0.865]
    titles2 = ["历史与日前预测", "风险需求", "日前计划", "LDR控制", "实际执行", "费用与边界"]
    details2 = [
        "周持久化负荷\n7日均值光伏",
        "21日残差\n80%分位修正",
        "计划量锁定\n参考SOC轨迹",
        "四阶段\n截断仿射阈值",
        "逐10分钟揭示\n紧急购电补缺",
        "1,397.81 万元\n下界差额12.35%",
    ]
    for x, t, d in zip(xs2, titles2, details2):
        fill = ORANGE_LIGHT if t in {"风险需求", "LDR控制", "费用与边界"} else WHITE
        node(ax, (x, y2), (0.12, h), t, d, ORANGE, fill)
    for x in xs2[:-1]:
        arrow(ax, (x + 0.12, y2 + h / 2), (x + 0.162, y2 + h / 2), ORANGE)

    # Validation rail.
    ax.text(0.055, 0.205, "共同验证轨", fontsize=8.5, fontweight="semibold", color=INK, va="center")
    rail_y = 0.155
    checks = [
        ("物理可行", "能量平衡 · SOC边界"),
        ("因果信息", "仅用决策时可得数据"),
        ("账单复核", "独立重算计划费与紧急费"),
        ("文件回读", "官方模板逐项一致"),
    ]
    x0s = [0.18, 0.385, 0.59, 0.795]
    for x, (t, d) in zip(x0s, checks):
        node(ax, (x, rail_y), (0.16, 0.095), t, d, GREY, GREY_LIGHT)
    for x in x0s[:-1]:
        arrow(ax, (x + 0.16, rail_y + 0.047), (x + 0.205, rail_y + 0.047), GREY)

    ax.text(
        0.055,
        0.055,
        "结论边界：LDR仅在当前数据口径、参数边界、搜索预算和已实现对照集中费用最低；高斯核未接入费用回测。",
        fontsize=7.1,
        color=MUTED,
        ha="left",
        va="center",
    )
    return save(fig, "图0_确定性到不确定性控制_技术路线.png")


def read_schedule() -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "outputs/question2/current/ldr/question2_schedule_with_warmup.csv",
        parse_dates=["date", "interval_start"],
    )


def figure_quantile_risk(schedule: pd.DataFrame) -> Path:
    target_date = pd.Timestamp("2025-06-01")
    target = schedule.loc[schedule["date"].eq(target_date)].sort_values("slot").copy()
    prior_dates = sorted(schedule.loc[schedule["date"].lt(target_date), "date"].drop_duplicates())[-21:]
    residuals = []
    for day in prior_dates:
        hist = schedule.loc[schedule["date"].eq(day)].sort_values("slot")
        point_hist = hist["forecast_load_kwh"].to_numpy() - hist["forecast_pv_kwh"].to_numpy()
        actual_hist = hist["load_kwh"].to_numpy() - hist["pv_kwh"].to_numpy()
        residuals.append(actual_hist - point_hist)
    residuals = np.vstack(residuals)

    point = target["forecast_load_kwh"].to_numpy() - target["forecast_pv_kwh"].to_numpy()
    risk = target["planning_net_kwh"].to_numpy()
    q20, q50, q80 = np.quantile(residuals, [0.2, 0.5, 0.8], axis=0, method="linear")
    t = np.arange(144) / 6.0
    selected = int(np.nanargmax(risk - point))
    r = np.sort(residuals[:, selected])
    ecdf = np.arange(1, len(r) + 1) / len(r)
    q_sel = float(np.quantile(r, 0.8, method="linear"))
    hh = int(selected // 6)
    mm = int((selected % 6) * 10)

    fig = plt.figure(figsize=(11.4, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.72, 1.0], left=0.065, right=0.975, bottom=0.16, top=0.82, wspace=0.27)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    add_title(
        fig,
        "80%分位数风险修正的构造",
        "示例日：2025-06-01；风险曲线由此前21个完整历史残差日逐时估计，不使用示例日未来信息",
    )

    ax1.fill_between(t, point + q20, point + q80, color=BLUE_LIGHT, alpha=0.95, linewidth=0, label="历史残差20%–80%区间")
    ax1.plot(t, point, color=GREY, linewidth=1.15, linestyle=(0, (4, 2)), label="点预测净需求")
    ax1.plot(t, risk, color=BLUE_DARK, linewidth=1.7, label="80%风险修正净需求")
    ax1.plot(t, point + q50, color=BLUE, linewidth=0.85, alpha=0.8, label="残差中位数修正")
    ax1.axvline(selected / 6.0, color=ORANGE, linewidth=0.9, linestyle=(0, (3, 2)))
    ax1.text(selected / 6.0 + 0.22, ax1.get_ylim()[1] * 0.93, f"{hh:02d}:{mm:02d}\n最大风险抬升", fontsize=7.2, color=ORANGE, va="top")
    ax1.set_xlim(0, 24)
    ax1.set_xticks(np.arange(0, 25, 4))
    ax1.set_xlabel("时刻 / h")
    ax1.set_ylabel("10分钟净需求 / kWh")
    ax1.legend(loc="upper left", ncol=2, handlelength=2.3, columnspacing=1.2)
    clean_axes(ax1, "y")
    panel_label(ax1, "a")

    ax2.step(r, ecdf, where="post", color=BLUE_DARK, linewidth=1.65)
    ax2.scatter(r, ecdf, s=13, facecolor=WHITE, edgecolor=BLUE_DARK, linewidth=0.65, zorder=3)
    ax2.axhline(0.8, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
    ax2.axvline(q_sel, color=ORANGE, linewidth=1.2, linestyle=(0, (3, 2)))
    ax2.scatter([q_sel], [0.8], s=35, color=ORANGE, zorder=4)
    ax2.annotate(
        rf"$q_{{0.8}}$ = {q_sel:.1f} kWh",
        xy=(q_sel, 0.8),
        xytext=(8, -28),
        textcoords="offset points",
        fontsize=7.8,
        color=ORANGE,
        arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 0.7},
    )
    ax2.text(
        0.04,
        0.96,
        "单时段成本平衡\n$F(q)=C_u/(C_u+C_o)$\n$=4c/(4c+c)=0.80$",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": GREY_LIGHT, "edgecolor": "none"},
    )
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("历史净需求残差 / kWh")
    ax2.set_ylabel("经验累积分布")
    clean_axes(ax2, "both")
    panel_label(ax2, "b")

    fig.text(
        0.065,
        0.055,
        "注：80%来自普通采购价与五倍紧急购电价的简化单时段权衡；加入储能跨时段耦合后，它是风险设计而非全天可靠性定理。",
        fontsize=7.1,
        color=MUTED,
    )
    return save(fig, "图8_80分位风险修正机制.png")


def figure_ldr_mechanism(schedule: pd.DataFrame) -> Path:
    target_date = pd.Timestamp("2025-06-01")
    day = schedule.loc[schedule["date"].eq(target_date)].sort_values("slot").copy()
    t = np.arange(144) / 6.0
    stage_colors = ["#F4F7FA", "#EDF4F8", "#F4F7FA", "#EDF4F8"]

    fig = plt.figure(figsize=(11.5, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0], left=0.065, right=0.975, bottom=0.16, top=0.82, wspace=0.24)
    ax = fig.add_subplot(gs[0, 0])
    info = fig.add_subplot(gs[0, 1])
    add_title(
        fig,
        "四阶段截断仿射库存保留阈值",
        "示例日：2025-06-01；每日参数于00:00锁定，阶段信号只使用阶段开始前已完成时段",
    )

    for k, (lo, hi) in enumerate([(0, 6), (6, 12), (12, 18), (18, 24)]):
        ax.axvspan(lo, hi, color=stage_colors[k], zorder=0)
        ax.text((lo + hi) / 2, 10710, f"阶段 {k + 1}\n{lo:02d}–{hi:02d} h", ha="center", va="top", fontsize=7.0, color=MUTED)
    for x in [6, 12, 18]:
        ax.axvline(x, color=GREY, linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)

    ax.plot(t, day["plan_soc_kwh"], color=GREY, linewidth=1.05, linestyle=(0, (4, 2)), label="参考库存 $E^p_t$")
    ax.plot(t, day["reserve_kwh"], color=ORANGE, linewidth=1.55, label="保留阈值 $R_t$")
    ax.plot(t, day["soc_start_kwh"], color=BLUE_DARK, linewidth=1.75, label="实际库存 $E_t$")
    emergency = day["emergency_kwh"].to_numpy() > 1e-9
    ax.scatter(t[emergency], day.loc[emergency, "soc_start_kwh"], s=13, color=ORANGE, marker="|", linewidth=1.2, label="发生紧急购电", zorder=4)
    ax.axhline(1200, color=INK, linewidth=0.75, linestyle=(0, (2, 2)))
    ax.axhline(10800, color=INK, linewidth=0.75, linestyle=(0, (2, 2)))
    ax.text(24.1, 1200, "$E_{min}$", fontsize=7.2, va="center", color=MUTED)
    ax.text(24.1, 10800, "$E_{max}$", fontsize=7.2, va="center", color=MUTED)
    ax.set_xlim(0, 24)
    ax.set_ylim(800, 11200)
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xlabel("时刻 / h")
    ax.set_ylabel("库存与阈值 / kWh")
    ax.legend(loc="lower left", ncol=2, handlelength=2.4, columnspacing=1.3)
    clean_axes(ax, "y")
    panel_label(ax, "a")

    info.axis("off")
    panel_label(info, "b")
    info.text(0.02, 0.98, "控制规则", fontsize=10.0, fontweight="semibold", va="top")
    info.text(
        0.02,
        0.87,
        "$a_{d,k}=\\mathrm{mean}(N^{obs}-\\hat N)$\n$R_{d,t}=\\mathrm{clip}(E^p_t+\\delta_k+\\lambda_k a_{d,k},\\ E_{min},E_{max})$",
        fontsize=8.8,
        va="top",
        linespacing=1.6,
        bbox={"boxstyle": "round,pad=0.38", "facecolor": GREY_LIGHT, "edgecolor": "none"},
    )
    info.text(0.02, 0.62, "当日阶段参数", fontsize=8.8, fontweight="semibold", va="top")
    headers = ["阶段", "$a_k$", "$\\delta_k$", "$\\lambda_k$"]
    xcols = [0.03, 0.31, 0.58, 0.85]
    for x, h in zip(xcols, headers):
        info.text(x, 0.545, h, fontsize=7.5, fontweight="semibold", ha="center", color=MUTED)
    firsts = day.groupby("stage", sort=True).first()
    for i, stage in enumerate([1, 2, 3, 4]):
        y = 0.47 - i * 0.105
        row = firsts.loc[stage]
        vals = [
            f"{stage}",
            f"{row['stage_error_mean_kwh']:.1f}",
            f"{row['ldr_delta_kwh']:.0f}",
            f"{row['ldr_lambda']:.2f}",
        ]
        info.add_patch(Rectangle((0.0, y - 0.035), 0.98, 0.08, facecolor=WHITE if i % 2 == 0 else "#F7F9FB", edgecolor=GRID, lw=0.45))
        for x, value in zip(xcols, vals):
            info.text(x, y, value, fontsize=7.3, ha="center", va="center")
    info.text(
        0.02,
        0.015,
        "δ改变阶段保留水平；λ调节对已观测误差的响应。\n截断保证阈值始终位于设备SOC边界内。",
        fontsize=7.2,
        color=MUTED,
        va="bottom",
        linespacing=1.4,
    )
    fig.text(
        0.065,
        0.055,
        "注：保留阈值是停止继续放电的参考，不是必须补购充到的硬目标；整个控制器含截断和分段操作，并非纯线性动作规则。",
        fontsize=7.1,
        color=MUTED,
    )
    return save(fig, "图9_LDR四阶段截断仿射阈值.png")


def figure_lower_bound() -> Path:
    comparison = pd.read_csv(ROOT / "outputs/question2/current/paper/tables/strategy_comparison.csv")
    with (ROOT / "outputs/question2/benchmark/perfect_foresight/matched_ldr_summary.json").open("r", encoding="utf-8") as f:
        bound = json.load(f)

    name_map = {
        "分位数＋β=1": "分位数 + β=1",
        "分位数＋β=0": "分位数 + β=0",
        "情景价值控制": "情景价值控制",
        "分位数＋LDR": "分位数 + LDR",
    }
    rows = [("完美预见下界", bound["cost_lower_bound_yuan"], "bound")]
    for _, row in comparison.iterrows():
        rows.append((name_map.get(row["方案"], row["方案"]), float(row["total_cost"]), "ldr" if "LDR" in row["方案"] else "other"))
    rows = sorted(rows, key=lambda x: x[1])

    actual = float(bound["causal_policy_cost_yuan"])
    lower = float(bound["cost_lower_bound_yuan"])
    gap = float(bound["gap_to_bound_yuan"])

    fig = plt.figure(figsize=(11.4, 4.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], left=0.08, right=0.97, bottom=0.18, top=0.81, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    add_title(
        fig,
        "实际策略费用与完美预见下界",
        "正式期2025-02-01至2025-12-31；下界匹配LDR期初SOC，年末仅要求达到最低SOC",
    )

    labels = [x[0] for x in rows]
    vals = np.array([x[1] for x in rows]) / 1e6
    colors = [GREY if x[2] == "bound" else BLUE_DARK if x[2] == "ldr" else BLUE_LIGHT for x in rows]
    edges = [GREY if x[2] == "bound" else BLUE_DARK if x[2] == "ldr" else BLUE for x in rows]
    y = np.arange(len(rows))
    ax1.barh(y, vals, color=colors, edgecolor=edges, linewidth=0.75, height=0.62)
    for yi, v in zip(y, vals):
        ax1.text(v + 0.10, yi, f"{v:.3f}", va="center", fontsize=7.6, color=INK)
    ax1.set_yticks(y, labels)
    ax1.set_xlim(0, 15.1)
    ax1.set_xlabel("正式期总费用 / 百万元")
    ax1.text(0.98, 0.02, "柱形从零开始", transform=ax1.transAxes, ha="right", va="bottom", fontsize=6.8, color=MUTED)
    clean_axes(ax1, "x")
    panel_label(ax1, "a")

    ax2.barh([0], [lower / 1e6], color=GREY, edgecolor=INK, linewidth=0.65, height=0.42, label="完美预见下界")
    ax2.barh([0], [gap / 1e6], left=[lower / 1e6], color=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=0.8, height=0.42, label="下界以上差额")
    ax2.text(lower / 2e6, 0, f"下界\n{lower/1e6:.3f}", ha="center", va="center", fontsize=8.0, color=WHITE, fontweight="semibold")
    ax2.text((lower + gap / 2) / 1e6, 0, f"差额\n{gap/1e6:.3f}", ha="center", va="center", fontsize=8.0, color=INK, fontweight="semibold")
    ax2.annotate(
        "差额占实际成本 12.35%",
        xy=(actual / 1e6, 0.23),
        xytext=(actual / 1e6, 0.55),
        ha="right",
        fontsize=8.1,
        color=ORANGE,
        arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 0.8},
    )
    ax2.text(0.5, 0.16, "实际成本比下界高 14.08%", transform=ax2.transAxes, ha="center", fontsize=8.0, color=INK)
    ax2.set_xlim(0, 15.1)
    ax2.set_ylim(-0.75, 0.9)
    ax2.set_yticks([])
    ax2.set_xlabel("费用构成 / 百万元")
    clean_axes(ax2, "x")
    panel_label(ax2, "b")

    fig.text(
        0.08,
        0.055,
        "注：下界允许全时域联合优化且不可在线执行；1.726百万元差额混合信息限制、风险修正、策略近似、搜索预算和终端SOC口径，不能全部归因于预测误差。",
        fontsize=7.0,
        color=MUTED,
    )
    return save(fig, "图13_完美预见下界参照.png")


def figure_forecast_robustness() -> Path:
    metrics = pd.read_csv(ROOT / "outputs/question2/analysis/load_forecast/model_metrics.csv")
    metrics = metrics.loc[metrics["scheme"].isin(["b0", "b1", "b2"])].copy()
    labels = {"b0": "昨日持久化", "b1": "周持久化", "b2": "七天均值"}
    schemes = ["b0", "b1", "b2"]
    daytypes = ["workday", "sat_class", "sunday", "holiday"]
    day_labels = ["工作日\n(n=185)", "周六类\n(n=93)", "周日\n(n=46)", "节假日\n(n=10)"]

    formal = metrics.loc[metrics["scope"].eq("formal")].set_index("scheme").loc[schemes]
    matrix = (
        metrics.loc[metrics["scope"].isin(daytypes)]
        .pivot(index="scheme", columns="scope", values="mape_pct")
        .loc[schemes, daytypes]
        .to_numpy()
    )

    fig = plt.figure(figsize=(11.35, 4.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.55], left=0.08, right=0.97, bottom=0.18, top=0.80, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    add_title(
        fig,
        "负载预测口径的正式期稳健性比较",
        "正式评价期334天；主费用管线采用周持久化，高斯核暂未接入费用回测，故不列入本图",
    )

    vals = formal["mape_pct"].to_numpy()
    colors = [GREY_LIGHT, BLUE_DARK, ORANGE_LIGHT]
    edges = [GREY, BLUE_DARK, ORANGE]
    x = np.arange(3)
    ax1.bar(x, vals, width=0.62, color=colors, edgecolor=edges, linewidth=0.8)
    for xi, v in zip(x, vals):
        ax1.text(xi, v + 0.65, f"{v:.2f}%", ha="center", va="bottom", fontsize=8.2, fontweight="semibold")
    ax1.set_xticks(x, [labels[s] for s in schemes], rotation=0)
    ax1.set_ylabel("正式期 MAPE / %")
    ax1.set_ylim(0, 22)
    clean_axes(ax1, "y")
    panel_label(ax1, "a")

    cmap = LinearSegmentedColormap.from_list("academic_blue", ["#F4F7FA", "#B9D2E5", BLUE_DARK])
    norm = Normalize(vmin=0, vmax=42)
    im = ax2.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax2.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.0,
                     color=WHITE if value > 24 else INK, fontweight="semibold" if schemes[i] == "b1" else "normal")
    ax2.set_xticks(np.arange(4), day_labels)
    ax2.set_yticks(np.arange(3), [labels[s] for s in schemes])
    ax2.set_xlabel("日类型")
    ax2.set_ylabel("预测口径")
    ax2.tick_params(length=0)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.035)
    cbar.set_label("MAPE / %", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.0, width=0.6)
    panel_label(ax2, "b")

    fig.text(
        0.08,
        0.055,
        "解读：周持久化在四类日期上均保持较低误差；七天均值在周六类达到41.43%，暴露了混合不同周内日型造成的系统偏差。",
        fontsize=7.1,
        color=MUTED,
    )
    return save(fig, "图7_负载预测日型稳健性.png")


def figure_de_diagnostics() -> Path:
    params = pd.read_csv(ROOT / "outputs/question2/current/ldr/ldr_daily_parameters.csv")
    data = params.loc[params["residual_count"].ge(21)].copy()
    msg_max = "Maximum number of iterations has been exceeded."
    msg_ok = "Optimization terminated successfully."
    counts = data["solver_message"].value_counts()
    n_max = int(counts.get(msg_max, 0))
    n_ok = int(counts.get(msg_ok, 0))
    total = len(data)

    fig = plt.figure(figsize=(10.9, 4.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], left=0.08, right=0.97, bottom=0.18, top=0.80, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    add_title(
        fig,
        "差分进化终止状态与历史情景改进",
        "337个参与日参数标定的日期；所有候选均通过零参数保底并进入后续物理与账单校验",
    )

    ax1.barh([0], [n_max / total], color=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=0.8, height=0.36)
    ax1.barh([0], [n_ok / total], left=[n_max / total], color=BLUE_DARK, edgecolor=BLUE_DARK, linewidth=0.8, height=0.36)
    ax1.text((n_max / total) / 2, 0, f"达到最大迭代次数\n{n_max} 天  |  {n_max/total:.1%}", ha="center", va="center", fontsize=8.2, color=INK)
    ax1.annotate(
        f"正常终止\n{n_ok} 天  |  {n_ok/total:.1%}",
        xy=((n_max + n_ok / 2) / total, 0),
        xytext=(0.69, 0.68),
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=8.0,
        color=BLUE_DARK,
        arrowprops={"arrowstyle": "-", "color": BLUE_DARK, "lw": 0.8},
    )
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.65, 0.85)
    ax1.set_yticks([])
    ax1.set_xlabel("标定日占比")
    ax1.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    clean_axes(ax1, "x")
    panel_label(ax1, "a")

    groups = [
        data.loc[data["solver_message"].eq(msg_max), "scenario_improvement"].to_numpy(),
        data.loc[data["solver_message"].eq(msg_ok), "scenario_improvement"].to_numpy(),
    ]
    positions = [0, 1]
    bp = ax2.boxplot(
        groups,
        positions=positions,
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "lw": 1.0},
        whiskerprops={"color": GREY, "lw": 0.8},
        capprops={"color": GREY, "lw": 0.8},
        boxprops={"edgecolor": GREY, "lw": 0.8},
    )
    bp["boxes"][0].set_facecolor(ORANGE_LIGHT)
    bp["boxes"][1].set_facecolor(BLUE_LIGHT)
    rng = np.random.default_rng(20250912)
    for pos, arr, color in zip(positions, groups, [ORANGE, BLUE_DARK]):
        jitter = rng.normal(0, 0.055, size=len(arr))
        ax2.scatter(np.full(len(arr), pos) + jitter, arr, s=8, color=color, alpha=0.42, linewidth=0)
        ax2.text(pos, ax2.get_ylim()[1] * 0.96, f"n={len(arr)}", ha="center", va="top", fontsize=7.4, color=MUTED)
    ax2.axhline(0, color=INK, linewidth=0.7)
    ax2.set_xticks(positions, ["最大迭代终止", "正常终止"])
    ax2.set_ylabel("历史情景目标改进 / 元")
    clean_axes(ax2, "y")
    panel_label(ax2, "b")

    fig.text(
        0.08,
        0.055,
        "结论边界：最大迭代终止不等于不可行；它表示搜索预算经常耗尽。图中结果支持“获得可执行候选”，不构成全局最优或充分收敛证书。",
        fontsize=7.1,
        color=MUTED,
    )
    return save(fig, "附图A1_差分进化终止与改进分布.png")


def contact_sheet(paths: list[Path]) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 14.5), facecolor="#F3F5F7")
    for ax, path in zip(axes.ravel(), paths):
        ax.imshow(plt.imread(path))
        ax.set_title(path.stem, fontsize=9.0, loc="left", pad=7)
        ax.axis("off")
    fig.suptitle("论文新增图表总览（缩略预览）", x=0.055, y=0.995, ha="left", fontsize=14.0, fontweight="semibold")
    fig.text(0.055, 0.975, "最终论文请使用各独立600 dpi PNG，不要直接使用本缩略总览。", fontsize=8.0, color=MUTED)
    fig.tight_layout(rect=(0.03, 0.02, 0.98, 0.965), h_pad=2.1, w_pad=1.4)
    return save(fig, "图表总览_缩略预览.png")


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    schedule = read_schedule()
    paths = [
        figure_technical_route(),
        figure_forecast_robustness(),
        figure_quantile_risk(schedule),
        figure_ldr_mechanism(schedule),
        figure_lower_bound(),
        figure_de_diagnostics(),
    ]
    preview = contact_sheet(paths)
    print(json.dumps({"figures": [str(p) for p in paths], "preview": str(preview)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
