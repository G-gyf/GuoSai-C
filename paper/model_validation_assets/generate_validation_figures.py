from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 140,
    }
)

BLUE = "#356A9A"
ORANGE = "#C77C2B"
DARK = "#25313C"
GRID = "#D9DEE5"
PALE = "#E9EDF2"


# Figure 1: numerical residuals, read directly from the selected formal outputs.
q1 = load_json("outputs/question1/question1_summary.json")["validation"]
q2_all = load_json("outputs/question2/current/ldr/question2_summary.json")
q2 = q2_all[0]["validation"]
q3_all = load_json("outputs/question3/current/question3_summary.json")
q3 = next(row for row in q3_all if row["settings"]["strategy"] == "M612")["validation"]
q4 = load_json("outputs/question4/v2/result4-2/run_summary.json")["main"]["validation"]

labels = ["问题一", "问题二", "问题三 M612", "问题四主方案"]
balance = np.array(
    [
        q1["max_balance_residual_kwh"],
        q2["max_balance_error_kwh"],
        q3["max_balance_error_kwh"],
        q4["max_balance_error_kwh"],
    ]
)
state_raw = np.array(
    [
        q1["max_soc_transition_residual_kwh"],
        q2["max_state_error_kwh"],
        q3["max_state_error_kwh"],
        q4["max_state_error_kwh"],
    ]
)
floor = 1e-15
state_plot = np.where(state_raw == 0, floor, state_raw)

fig, ax = plt.subplots(figsize=(10.2, 5.2))
x = np.arange(len(labels))
ax.scatter(x - 0.10, balance, s=75, color=BLUE, marker="o", label="供需平衡最大残差", zorder=3)
ax.scatter(
    x + 0.10,
    state_plot,
    s=75,
    facecolors="white",
    edgecolors=ORANGE,
    linewidths=1.8,
    marker="s",
    label="SOC 递推最大残差",
    zorder=3,
)
for i, value in enumerate(balance):

    ax.annotate(f"{value:.2e}", (x[i] - 0.10, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=DARK)
for i, value in enumerate(state_raw):
    label = "0" if value == 0 else f"{value:.2e}"
    ax.annotate(label, (x[i] + 0.10, state_plot[i]), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8, color=DARK)
ax.set_yscale("log")
ax.set_ylim(6e-16, 2e-12)
ax.set_xticks(x, labels)
ax.set_ylabel("最大绝对残差 kWh 对数尺度 越低越好")
fig.suptitle("四问主方案的数值约束残差", y=0.97, fontsize=15, color=DARK)
fig.text(0.5, 0.91, "零残差按 1e-15 显示；全部结果远低于 1e-6 的可行性容差", ha="center", color="#5B6570")
ax.grid(axis="y", which="both", color=GRID, linewidth=0.7)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(bottom=0.22, top=0.84)
save(fig, "fig1_constraint_residuals")


# Figure 2: coverage map of checks already implemented in the current workflow.
checks = [
    "数据契约与时间轴",
    "逐区间能量平衡",
    "SOC 递推与边界",
    "无同时充放电",
    "跨日状态连续",
    "费用独立复算",
    "未来信息篡改",
    "共同起点或分支回放",
    "发布执行账本桥接",
    "完美信息下界",
]
matrix = np.array(
    [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [np.nan, 1, 1, 1],
        [1, 1, 1, 1],
        [np.nan, 1, 1, 1],
        [np.nan, 1, 1, 1],
        [np.nan, np.nan, np.nan, 1],
        [np.nan, np.nan, 1, 1],
    ],
    dtype=float,
)
display = np.nan_to_num(matrix, nan=0.0)
fig, ax = plt.subplots(figsize=(9.7, 6.7))
ax.imshow(display, cmap=ListedColormap([PALE, BLUE]), vmin=0, vmax=1, aspect="auto")
ax.set_xticks(np.arange(4), ["问题一", "问题二", "问题三", "问题四"])
ax.set_yticks(np.arange(len(checks)), checks)
fig.suptitle("当前模型检验的证据覆盖矩阵", y=0.97, fontsize=15, color=DARK)
fig.text(0.5, 0.92, "蓝色表示已有结果通过；灰色表示该问题口径下不适用，并非检验失败", ha="center", color="#5B6570")
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        passed = not np.isnan(matrix[i, j])
        ax.text(j, i, "通过" if passed else "不适用", ha="center", va="center", color="white" if passed else "#69737D", fontsize=9)
ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(checks), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.6)
ax.tick_params(which="minor", bottom=False, left=False)
ax.spines[:].set_visible(False)
fig.subplots_adjust(left=0.28, right=0.97, top=0.86, bottom=0.08)
save(fig, "fig2_validation_evidence_matrix")


# Figure 3: economic lower-bound checks under matched initial states and scopes.
q3_bound = load_json("outputs/question3/benchmark/matched_m612_summary.json")
q4_run = load_json("outputs/question4/v2/result4-2/run_summary.json")
lower = np.array([q3_bound["cost_lower_bound_yuan"], q4_run["bound"]["total_cost"]]) / 1e6
actual = np.array([q3_bound["causal_policy_cost_yuan"], q4_run["main"]["formal_period"]["total_cost"]]) / 1e6
gap_pct_lower = (actual - lower) / lower * 100

fig, ax = plt.subplots(figsize=(9.4, 5.5))
x = np.arange(2)
width = 0.32
bars1 = ax.bar(x - width / 2, lower, width, color=PALE, edgecolor=DARK, linewidth=1.0, label="完美信息下界")
bars2 = ax.bar(x + width / 2, actual, width, color=BLUE, edgecolor=BLUE, linewidth=1.0, label="当前因果主策略")
for bars in (bars1, bars2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.10, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for i, pct in enumerate(gap_pct_lower):
    ax.annotate(f"相对下界高 {pct:.2f}%", xy=(x[i] + width / 2, actual[i]), xytext=(0, 28), textcoords="offset points", ha="center", color=ORANGE, fontsize=10, fontweight="bold")
ax.set_xticks(x, ["问题三 M612", "问题四第二问主方案"])
ax.set_ylabel("正式期总费用 百万元")
ax.set_ylim(0, max(actual) * 1.18)
fig.suptitle("共同口径下当前策略与完美信息下界", y=0.97, fontsize=15, color=DARK)
fig.text(0.5, 0.91, "各组在同一期初 SOC 和同一评价区间内比较；两组之间不作绝对费用横向排名", ha="center", color="#5B6570")
ax.grid(axis="y", color=GRID, linewidth=0.7)
ax.legend(loc="upper left", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(top=0.82, bottom=0.14)
save(fig, "fig3_perfect_information_bounds")


print("generated:")
for path in sorted(OUT.glob("fig*.*")):
    print(path)
