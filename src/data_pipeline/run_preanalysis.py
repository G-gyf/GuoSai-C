"""End-to-end preprocessing and pre-analysis report generation."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .build_actuals import build_dispatch_10min
from .build_forecasts import (
    build_day_ahead_baseline,
    build_pv_forecast_10min,
    build_pv_forecast_hourly,
)
from .ingest import PROJECT_ROOT, build_inventory, load_contract, read_attachments
from .validate_data import quality_report, validate_datasets


def _configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _endpoint_label(slot_index: int) -> str:
    minute = int(slot_index) * 10
    if minute == 24 * 60:
        return "0:00+1"
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _profile_summary(dispatch: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "load_actual_kw", "pv_actual_kw", "net_load_actual_kw",
        "price_fixed_yuan_per_kwh", "price_variable_yuan_per_kwh",
    ]
    grouped = dispatch.groupby("slot_index")[variables].agg(["mean", "std", "min", "max"])
    grouped.columns = [f"{variable}_{stat}" for variable, stat in grouped.columns]
    grouped = grouped.reset_index()
    grouped.insert(1, "observation_time_label", grouped["slot_index"].map(_endpoint_label))
    return grouped


def _forecast_evaluation(hourly: pd.DataFrame, dispatch: pd.DataFrame) -> pd.DataFrame:
    actual = dispatch[["observation_ts", "pv_actual_kw"]].rename(
        columns={"observation_ts": "target_ts"}
    )
    evaluated = hourly.merge(actual, on="target_ts", how="left", validate="many_to_one")
    evaluated = evaluated[evaluated["pv_actual_kw"].notna()].copy()
    evaluated["error_kw"] = evaluated["pv_forecast_kw"] - evaluated["pv_actual_kw"]
    evaluated["abs_error_kw"] = evaluated["error_kw"].abs()
    evaluated["squared_error_kw2"] = evaluated["error_kw"].pow(2)
    evaluated["pv_nonzero"] = evaluated["pv_actual_kw"].gt(0)
    return evaluated


def _error_by_horizon(evaluated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, group in evaluated.groupby("horizon_hour", sort=True):
        rows.append(
            {
                "horizon_hour": int(horizon),
                "count": int(len(group)),
                "mae_kw": group["abs_error_kw"].mean(),
                "rmse_kw": math.sqrt(group["squared_error_kw2"].mean()),
                "bias_kw": group["error_kw"].mean(),
                "absolute_error_q50_kw": group["abs_error_kw"].quantile(0.50),
                "absolute_error_q90_kw": group["abs_error_kw"].quantile(0.90),
                "absolute_error_q95_kw": group["abs_error_kw"].quantile(0.95),
                "overestimate_ratio": group["error_kw"].gt(0).mean(),
                "underestimate_ratio": group["error_kw"].lt(0).mean(),
                "nonzero_pv_mae_kw": group.loc[group["pv_nonzero"], "abs_error_kw"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _update_gain(evaluated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = evaluated.sort_values(["target_ts", "issue_ts"]).copy()
    comparisons["prior_issue_ts"] = comparisons.groupby("target_ts")["issue_ts"].shift(1)
    comparisons["prior_abs_error_kw"] = comparisons.groupby("target_ts")["abs_error_kw"].shift(1)
    comparisons = comparisons[comparisons["prior_issue_ts"].notna()].copy()
    comparisons["new_issue_hour"] = comparisons["issue_ts"].dt.hour
    comparisons["lead_reduction_hours"] = (
        comparisons["issue_ts"] - comparisons["prior_issue_ts"]
    ).dt.total_seconds().div(3600)
    comparisons["abs_error_gain_kw"] = comparisons["prior_abs_error_kw"] - comparisons["abs_error_kw"]
    rows: list[dict[str, Any]] = []
    for issue_hour, group in comparisons.groupby("new_issue_hour", sort=True):
        rows.append(
            {
                "new_issue_time": f"{int(issue_hour):02d}:00",
                "comparison_count": int(len(group)),
                "prior_mae_kw": group["prior_abs_error_kw"].mean(),
                "new_mae_kw": group["abs_error_kw"].mean(),
                "mean_abs_error_gain_kw": group["abs_error_gain_kw"].mean(),
                "improvement_ratio": group["abs_error_gain_kw"].gt(0).mean(),
                "mean_lead_reduction_hours": group["lead_reduction_hours"].mean(),
            }
        )
    return pd.DataFrame(rows), comparisons


def _key_dates_summary(
    dispatch: pd.DataFrame, ten_minute: pd.DataFrame, key_dates: list[str]
) -> pd.DataFrame:
    selected = dispatch[dispatch["plan_date"].isin(pd.to_datetime(key_dates))]
    rows: list[dict[str, Any]] = []
    for date, group in selected.groupby("plan_date", sort=True):
        surplus = (-group["net_load_actual_kwh"].clip(upper=0)).sum()
        issues = ten_minute[
            ten_minute["issue_ts"].isin(
                [date + pd.Timedelta(hours=hour) for hour in (0, 6, 12, 18)]
            )
        ]
        day_forecasts = issues.merge(
            group[["observation_ts", "pv_actual_kw"]],
            left_on="target_ts", right_on="observation_ts", how="inner"
        )
        day_forecasts["abs_error_kw"] = (
            day_forecasts["pv_forecast_kw"] - day_forecasts["pv_actual_kw"]
        ).abs()
        midnight = day_forecasts[day_forecasts["issue_ts"].dt.hour.eq(0)]
        rows.append(
            {
                "plan_date": date.date().isoformat(),
                "interval_count": int(len(group)),
                "load_energy_kwh": group["load_actual_kwh"].sum(),
                "pv_energy_kwh": group["pv_actual_kwh"].sum(),
                "net_load_energy_kwh": group["net_load_actual_kwh"].sum(),
                "pv_surplus_energy_kwh": surplus,
                "load_peak_kw": group["load_actual_kw"].max(),
                "pv_peak_kw": group["pv_actual_kw"].max(),
                "net_load_peak_kw": group["net_load_actual_kw"].max(),
                "fixed_price_min": group["price_fixed_yuan_per_kwh"].min(),
                "fixed_price_max": group["price_fixed_yuan_per_kwh"].max(),
                "variable_price_min": group["price_variable_yuan_per_kwh"].min(),
                "variable_price_max": group["price_variable_yuan_per_kwh"].max(),
                "midnight_pv_forecast_mae_kw": midnight["abs_error_kw"].mean(),
                "forecast_issue_count": int(issues["issue_ts"].nunique()),
                "evaluated_forecast_points": int(len(day_forecasts)),
                "first_interval_start": group["interval_start"].min(),
                "last_interval_start": group["interval_start"].max(),
            }
        )
    return pd.DataFrame(rows)


def _anomaly_flags(dispatch: pd.DataFrame) -> pd.DataFrame:
    """Flag robust outer-fence observations without changing source values."""
    variables = [
        "load_actual_kw", "pv_actual_kw", "net_load_actual_kw",
        "price_variable_yuan_per_kwh",
    ]
    blocks: list[pd.DataFrame] = []
    for variable in variables:
        series = dispatch[variable]
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        mask = series.lt(lower) | series.gt(upper)
        flagged = dispatch.loc[mask, ["plan_date", "slot_index", "interval_start", variable]].copy()
        flagged = flagged.rename(columns={variable: "value"})
        flagged.insert(0, "variable", variable)
        flagged["lower_outer_fence"] = lower
        flagged["upper_outer_fence"] = upper
        flagged["rule"] = "Tukey outer fence: Q1-3IQR or Q3+3IQR"
        blocks.append(flagged)
    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True).sort_values(["variable", "interval_start"])


def _daily_energy(dispatch: pd.DataFrame) -> pd.DataFrame:
    daily = dispatch.groupby("plan_date", as_index=False).agg(
        load_energy_kwh=("load_actual_kwh", "sum"),
        pv_energy_kwh=("pv_actual_kwh", "sum"),
        net_load_energy_kwh=("net_load_actual_kwh", "sum"),
    )
    return daily


def _make_figures(
    dispatch: pd.DataFrame,
    ten_minute: pd.DataFrame,
    profile: pd.DataFrame,
    error_horizon: pd.DataFrame,
    update_gain: pd.DataFrame,
    key_dates: list[str],
    figures_dir: Path,
) -> None:
    _configure_plots()
    endpoint_hours = profile["slot_index"] * 10 / 60
    start_hours = (profile["slot_index"] - 1) * 10 / 60

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.plot(endpoint_hours, profile["load_actual_kw_mean"], label="平均负载", linewidth=2)
    ax.plot(endpoint_hours, profile["pv_actual_kw_mean"], label="平均光伏", linewidth=2)
    ax.plot(endpoint_hours, profile["net_load_actual_kw_mean"], label="平均净负荷", linewidth=2)
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set(xlabel="区间终点时刻 / h", ylabel="功率 / kW", title="全年同一终点时刻平均功率曲线")
    ax.set_xlim(0, 24)
    ax.legend(ncol=3)
    _save_figure(fig, figures_dir / "average_dispatch_profile.png")

    net = np.sort(dispatch["net_load_actual_kw"].to_numpy())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    axes[0].plot(np.linspace(0, 100, len(net)), net, color="#315a91")
    axes[0].axhline(0, color="#555", linewidth=0.8)
    axes[0].set(xlabel="时段累计比例 / %", ylabel="净负荷 / kW", title="净负荷持续曲线")
    surplus = dispatch.loc[dispatch["net_load_actual_kw"].lt(0), "net_load_actual_kw"]
    axes[1].hist(-surplus, bins=45, color="#58a55c", alpha=0.85)
    axes[1].set(xlabel="光伏剩余功率 / kW", ylabel="时段数", title="负净荷时段的光伏剩余分布")
    _save_figure(fig, figures_dir / "net_load_surplus.png")

    fig, ax = plt.subplots(figsize=(9, 3.9))
    ax.step(start_hours, profile["price_fixed_yuan_per_kwh_mean"], where="post", color="#c44e52", linewidth=1.8)
    ax.set(xlabel="交易区间起点 / h", ylabel="电价 / 元/kWh", title="附件1固定电价曲线")
    ax.set_xlim(0, 24)
    _save_figure(fig, figures_dir / "fixed_price_profile.png")

    variable_group = dispatch.groupby("slot_index")["price_variable_yuan_per_kwh"]
    q10, q50, q90 = (variable_group.quantile(q) for q in (0.1, 0.5, 0.9))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    axes[0].hist(
        dispatch.loc[dispatch["price_variable_available"], "price_variable_yuan_per_kwh"],
        bins=55,
        color="#8172b3",
        alpha=0.85,
    )
    axes[0].set(xlabel="波动电价 / 元/kWh", ylabel="时段数", title="全年波动电价分布")
    axes[1].fill_between(start_hours, q10.to_numpy(), q90.to_numpy(), alpha=0.25, label="10%—90%分位带")
    axes[1].plot(start_hours, q50.to_numpy(), linewidth=1.8, label="中位数")
    axes[1].set(xlabel="交易区间起点 / h", ylabel="电价 / 元/kWh", title="同一起点时段波动价格范围")
    axes[1].legend()
    _save_figure(fig, figures_dir / "variable_price_distribution.png")

    daily = _daily_energy(dispatch)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.boxplot(
        [daily["load_energy_kwh"], daily["pv_energy_kwh"], daily["net_load_energy_kwh"]],
        labels=["负载电量", "光伏电量", "净负荷电量"], showfliers=False,
    )
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set(ylabel="每日总电量 / kWh", title="每日能量总体分布（不按日历类别分组）")
    _save_figure(fig, figures_dir / "daily_energy_distribution.png")

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(error_horizon["horizon_hour"], error_horizon["mae_kw"], marker="o", label="MAE")
    ax.plot(error_horizon["horizon_hour"], error_horizon["rmse_kw"], marker="s", label="RMSE")
    ax.set(xlabel="预测步长 / h", ylabel="误差 / kW", title="光伏预报误差随预测步长的变化")
    ax.set_xticks(range(1, 25))
    ax.legend()
    _save_figure(fig, figures_dir / "forecast_error_by_horizon.png")

    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    colors = ["#55a868" if value >= 0 else "#c44e52" for value in update_gain["mean_abs_error_gain_kw"]]
    ax.bar(update_gain["new_issue_time"], update_gain["mean_abs_error_gain_kw"], color=colors)
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set(xlabel="新预报发布时间", ylabel="绝对误差平均改善 / kW", title="同一交付时刻新旧预报比较")
    _save_figure(fig, figures_dir / "forecast_update_gain.png")

    fig, axes = plt.subplots(4, 3, figsize=(13, 13), sharex="col")
    forecast_colors = {0: "#4c72b0", 6: "#dd8452", 12: "#55a868", 18: "#c44e52"}
    for row, date_text in enumerate(key_dates):
        group = dispatch[dispatch["plan_date"].eq(pd.Timestamp(date_text))]
        endpoint_hours = group["slot_index"] * 10 / 60
        start_hours = (group["slot_index"] - 1) * 10 / 60
        axes[row, 0].plot(endpoint_hours, group["load_actual_kw"], label="负载", linewidth=1.3)
        axes[row, 0].plot(endpoint_hours, group["pv_actual_kw"], label="光伏", linewidth=1.3)
        axes[row, 0].plot(endpoint_hours, group["net_load_actual_kw"], label="净负荷", linewidth=1.3)
        axes[row, 0].axhline(0, color="#555", linewidth=0.7)
        axes[row, 0].set_ylabel(f"{date_text}\n功率 / kW")

        axes[row, 1].plot(start_hours, group["price_fixed_yuan_per_kwh"], label="固定电价", linewidth=1.2)
        axes[row, 1].plot(start_hours, group["price_variable_yuan_per_kwh"], label="波动电价", linewidth=1.2)
        axes[row, 1].set_ylabel("元/kWh")

        axes[row, 2].plot(endpoint_hours, group["pv_actual_kw"], color="#222", linewidth=1.8, label="实际光伏")
        start = pd.Timestamp(date_text)
        end = start + pd.Timedelta(days=1)
        issues = ten_minute[
            ten_minute["issue_ts"].isin([start + pd.Timedelta(hours=h) for h in (0, 6, 12, 18)])
            & ten_minute["target_ts"].between(start + pd.Timedelta(minutes=10), end)
        ]
        for issue_hour, forecast in issues.groupby(issues["issue_ts"].dt.hour):
            forecast_x = (forecast["target_ts"] - start).dt.total_seconds().div(3600)
            axes[row, 2].plot(
                forecast_x, forecast["pv_forecast_kw"], linestyle="--", linewidth=1.1,
                color=forecast_colors[int(issue_hour)], label=f"{int(issue_hour):02d}:00预报"
            )
        for col in range(3):
            axes[row, col].set_xlim(0, 24)
    axes[0, 0].set_title("实际负载、光伏与净负荷")
    axes[0, 1].set_title("固定与波动电价")
    axes[0, 2].set_title("实际光伏与日内预报更新")
    axes[0, 0].legend(ncol=3, fontsize=7)
    axes[0, 1].legend(ncol=2, fontsize=7)
    axes[0, 2].legend(ncol=3, fontsize=7)
    axes[-1, 0].set_xlabel("区间终点时刻 / h")
    axes[-1, 1].set_xlabel("交易区间起点 / h")
    axes[-1, 2].set_xlabel("预测目标时刻 / h")
    fig.suptitle("四个必交日期的数据与预报映射核查", y=1.005)
    _save_figure(fig, figures_dir / "key_dates_profiles.png")


def _summary_metrics(dispatch: pd.DataFrame, evaluated: pd.DataFrame) -> dict[str, float]:
    surplus_mask = dispatch["net_load_actual_kw"].lt(0)
    return {
        "load_min": dispatch["load_actual_kw"].min(),
        "load_max": dispatch["load_actual_kw"].max(),
        "load_mean": dispatch["load_actual_kw"].mean(),
        "pv_min": dispatch["pv_actual_kw"].min(),
        "pv_max": dispatch["pv_actual_kw"].max(),
        "pv_mean": dispatch["pv_actual_kw"].mean(),
        "net_min": dispatch["net_load_actual_kw"].min(),
        "net_max": dispatch["net_load_actual_kw"].max(),
        "net_mean": dispatch["net_load_actual_kw"].mean(),
        "surplus_ratio": surplus_mask.mean(),
        "surplus_energy": (-dispatch.loc[surplus_mask, "net_load_actual_kwh"]).sum(),
        "forecast_mae": evaluated["abs_error_kw"].mean(),
        "forecast_rmse": math.sqrt(evaluated["squared_error_kw2"].mean()),
        "forecast_bias": evaluated["error_kw"].mean(),
    }


def _table_html(frame: pd.DataFrame, columns: list[str] | None = None, rows: int = 12) -> str:
    shown = frame if columns is None else frame.loc[:, columns]
    return shown.head(rows).to_html(index=False, border=0, classes="data-table", float_format=lambda x: f"{x:,.4f}")


def _write_html_report(
    output_path: Path,
    inventory: pd.DataFrame,
    checks: pd.DataFrame,
    profile: pd.DataFrame,
    error_horizon: pd.DataFrame,
    update_gain: pd.DataFrame,
    key_summary: pd.DataFrame,
    dispatch: pd.DataFrame,
    evaluated: pd.DataFrame,
    representative: pd.DataFrame,
    anomaly_flags: pd.DataFrame,
) -> None:
    metrics = _summary_metrics(dispatch, evaluated)
    actual_profile = dispatch.groupby("slot_index").agg(
        load_mean=("load_actual_kw", "mean"), pv_mean=("pv_actual_kw", "mean"), price_mean=("price_variable_yuan_per_kwh", "mean")
    )
    rep_load = pd.to_numeric(representative.iloc[:, 2]).to_numpy()
    rep_pv = pd.to_numeric(representative.iloc[:, 3]).to_numpy()
    rep_price_source = pd.to_numeric(representative.iloc[:, 1]).to_numpy()
    rep_price = np.r_[rep_price_source[-1], rep_price_source[:-1]]
    relation = pd.DataFrame(
        {
            "指标": ["负载", "光伏", "电价"],
            "平均绝对差": [
                np.mean(np.abs(rep_load - actual_profile["load_mean"])),
                np.mean(np.abs(rep_pv - actual_profile["pv_mean"])),
                np.mean(np.abs(rep_price - actual_profile["price_mean"])),
            ],
            "最大绝对差": [
                np.max(np.abs(rep_load - actual_profile["load_mean"])),
                np.max(np.abs(rep_pv - actual_profile["pv_mean"])),
                np.max(np.abs(rep_price - actual_profile["price_mean"])),
            ],
        }
    )
    issue_stats = evaluated.groupby(evaluated["issue_ts"].dt.strftime("%H:%M")).agg(
        样本数=("error_kw", "size"),
        MAE_kW=("abs_error_kw", "mean"),
        偏差_kW=("error_kw", "mean"),
        非零光伏MAE_kW=("abs_error_kw", lambda x: x[evaluated.loc[x.index, "pv_nonzero"]].mean()),
    ).reset_index(names="发布时间")
    correlations = dispatch[[
        "price_variable_yuan_per_kwh", "load_actual_kw", "pv_actual_kw", "net_load_actual_kw"
    ]].corr()["price_variable_yuan_per_kwh"].drop("price_variable_yuan_per_kwh")
    high_price = dispatch["price_variable_yuan_per_kwh"].ge(dispatch["price_variable_yuan_per_kwh"].quantile(0.9))
    high_net = dispatch["net_load_actual_kw"].ge(dispatch["net_load_actual_kw"].quantile(0.9))
    price_stats = pd.DataFrame(
        {
            "指标": ["波动电价—负载相关系数", "波动电价—光伏相关系数", "波动电价—净负荷相关系数", "高价与高净负荷同时出现比例", "固定电价变异系数", "波动电价变异系数"],
            "数值": [
                correlations["load_actual_kw"], correlations["pv_actual_kw"], correlations["net_load_actual_kw"],
                (high_price & high_net).mean(),
                dispatch["price_fixed_yuan_per_kwh"].std() / dispatch["price_fixed_yuan_per_kwh"].mean(),
                dispatch["price_variable_yuan_per_kwh"].std() / dispatch["price_variable_yuan_per_kwh"].mean(),
            ],
        }
    )
    mean_profile = dispatch.groupby("slot_index")[["load_actual_kw", "pv_actual_kw", "net_load_actual_kw"]].mean()
    peak_table = pd.DataFrame(
        {
            "序列": ["平均负载", "平均光伏", "平均净负荷"],
            "峰值时段": [
                f"{int(mean_profile['load_actual_kw'].idxmax() * 10 // 60) % 24:02d}:{int(mean_profile['load_actual_kw'].idxmax() * 10 % 60):02d}",
                f"{int(mean_profile['pv_actual_kw'].idxmax() * 10 // 60) % 24:02d}:{int(mean_profile['pv_actual_kw'].idxmax() * 10 % 60):02d}",
                f"{int(mean_profile['net_load_actual_kw'].idxmax() * 10 // 60) % 24:02d}:{int(mean_profile['net_load_actual_kw'].idxmax() * 10 % 60):02d}",
            ],
            "峰值_kW": [mean_profile["load_actual_kw"].max(), mean_profile["pv_actual_kw"].max(), mean_profile["net_load_actual_kw"].max()],
        }
    )

    fail_count = int(checks["status"].eq("FAIL").sum())
    escaped_generation = html.escape(pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M %Z"))
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>数据预处理与前置分析报告</title>
<style>
body{{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;line-height:1.7;color:#253047;max-width:1120px;margin:0 auto;padding:32px;background:#f4f6f9}}
main{{background:#fff;padding:34px 44px;box-shadow:0 2px 16px #dfe4ec}}h1,h2,h3{{color:#16375b}}h1{{border-bottom:3px solid #2c6eaa;padding-bottom:12px}}
.meta,.note{{color:#5c6678}}.pass{{color:#1d7a46;font-weight:700}}.fail{{color:#b52b2b;font-weight:700}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.card{{background:#eef4fa;padding:12px 16px;border-left:4px solid #2c6eaa}}img{{display:block;width:100%;height:auto;margin:16px 0 30px}}
.data-table{{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}}th,td{{border:1px solid #d8dee8;padding:7px 9px;white-space:nowrap}}th{{background:#edf3f8}}
code{{background:#eef1f5;padding:2px 5px}}@media(max-width:760px){{main{{padding:20px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>数据预处理与前置分析报告</h1>
<p class="meta">生成时间：{escaped_generation}。本报告仅使用题目附件，不引入日历、节假日、天气或其他外部特征；不包含优化策略与求解结果。</p>
<h2>1. 数据契约与质量结论</h2>
<p>日历物理区间主表共 <strong>{len(dispatch):,}</strong> 条记录，正式结果期共 <strong>{int(dispatch['is_result_period'].sum()):,}</strong> 个10分钟物理区间。质量检查共 {len(checks)} 项，失败项为 <span class="{'pass' if fail_count == 0 else 'fail'}">{fail_count}</span>。</p>
{_table_html(checks, rows=len(checks))}
<h3>原始数据登记</h3>{_table_html(inventory, rows=len(inventory))}
<h2>2. 代表性日与全年数据关系</h2>
<p>附件1与全年数据按统一物理时刻比较：负荷和光伏记录对应10分钟区间终点，购电价格对应区间起点。负载及电价差异仅体现源表精度，光伏差异集中于黎明、傍晚的小功率区间，说明附件1适合作为问题1的确定性代表日，并为问题1—3提供固定价格；后续实际运行与预测评价仍以附件2—4为准。</p>
{_table_html(relation, rows=3)}
<h2>3. 负载、光伏与净负荷</h2>
<div class="grid"><div class="card">负载范围<br><strong>{metrics['load_min']:,.2f}—{metrics['load_max']:,.2f} kW</strong></div><div class="card">光伏范围<br><strong>{metrics['pv_min']:,.2f}—{metrics['pv_max']:,.2f} kW</strong></div><div class="card">净负荷范围<br><strong>{metrics['net_min']:,.2f}—{metrics['net_max']:,.2f} kW</strong></div></div>
<p>光伏高于负载的时段占比为 <strong>{metrics['surplus_ratio']:.2%}</strong>，对应全年光伏剩余电量为 <strong>{metrics['surplus_energy']:,.2f} kWh</strong>。负净荷作为有效运行状态保留，未进行截断。</p>
{_table_html(peak_table, rows=3)}
<img src="figures/average_dispatch_profile.png" alt="平均功率曲线"><img src="figures/net_load_surplus.png" alt="净负荷与光伏剩余">
<img src="figures/daily_energy_distribution.png" alt="每日能量总体分布">
<h2>4. 电价特征</h2>
<p>固定电价与波动电价均按交易区间起点对齐，波动电价保留逐日逐时段差异。前置分析仅讨论其分布、时段范围及与供需变量的统计关系，不据此预判优化策略。</p>
{_table_html(price_stats, rows=6)}
<img src="figures/fixed_price_profile.png" alt="固定电价"><img src="figures/variable_price_distribution.png" alt="波动电价分布">
<h2>5. 光伏预报质量</h2>
<p>误差定义为预测值减实际值，并仅在预测目标时刻存在对应实际观测时计算。可评价样本的总体 MAE 为 <strong>{metrics['forecast_mae']:,.2f} kW</strong>，RMSE 为 <strong>{metrics['forecast_rmse']:,.2f} kW</strong>，偏差为 <strong>{metrics['forecast_bias']:,.2f} kW</strong>。</p>
{_table_html(issue_stats, rows=4)}
<img src="figures/forecast_error_by_horizon.png" alt="分步长预报误差">
<p>新旧预报价值采用同一交付时刻配对比较，避免不同发布时间覆盖的昼夜结构差异造成混淆。绝对误差改善定义为旧预报绝对误差减去新预报绝对误差。</p>
{_table_html(update_gain, rows=len(update_gain))}<img src="figures/forecast_update_gain.png" alt="预报更新改善">
<h2>6. 四个必交日期核查</h2>
{_table_html(key_summary, rows=4)}<img src="figures/key_dates_profiles.png" alt="关键日期曲线">
<h2>7. 数据使用边界</h2>
<p>日初基线在 2025-01-01 使用附件1，随后仅使用决策时刻之前已经完成的同终点时刻观测；第8日起固定使用最近7个完整日。问题3与问题4-3中，负载预测固定为0:00版本，光伏可在0:00、6:00、12:00、18:00按附件3更新。实际同日负载与光伏仅供执行仿真、结算和回测使用。</p>
<h2>8. 异常标记</h2>
<p>采用 Tukey 外围栏（低于 Q1−3IQR 或高于 Q3+3IQR）形成统计标记，共识别 <strong>{len(anomaly_flags)}</strong> 条记录。该标记仅用于提示复核，不等同于数据错误；所有原值均保留，未删除、平滑或缩尾。</p>
{_table_html(anomaly_flags, rows=20) if len(anomaly_flags) else '<p>未发现外围栏标记。</p>'}
<p class="note">完整标记见 <code>tables/anomaly_flags.csv</code>；其他机器可读结果见 <code>quality_report.json</code>、<code>quality_checks.csv</code> 与 <code>tables/</code>。</p>
</main></body></html>"""
    output_path.write_text(document, encoding="utf-8")


def run_pipeline(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    project_root = Path(project_root).resolve()
    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "outputs" / "preanalysis"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    for directory in (processed_dir, output_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    contract = load_contract(project_root)
    raw = read_attachments(project_root)
    inventory = build_inventory(project_root)
    dispatch = build_dispatch_10min(raw)
    hourly = build_pv_forecast_hourly(raw["attachment_3"])
    ten_minute = build_pv_forecast_10min(hourly, dispatch)
    baseline = build_day_ahead_baseline(dispatch, raw["attachment_1"])
    checks = validate_datasets(dispatch, hourly, ten_minute, baseline, raw, contract)

    paths = {
        "dispatch": processed_dir / "dispatch_10min.parquet",
        "hourly": processed_dir / "pv_forecast_hourly.parquet",
        "ten_minute": processed_dir / "pv_forecast_10min.parquet",
        "baseline": processed_dir / "day_ahead_baseline_10min.parquet",
        "report": output_dir / "preanalysis.html",
        "quality_json": output_dir / "quality_report.json",
        "quality_csv": output_dir / "quality_checks.csv",
    }
    dispatch.to_parquet(paths["dispatch"], index=False)
    hourly.to_parquet(paths["hourly"], index=False)
    ten_minute.to_parquet(paths["ten_minute"], index=False)
    baseline.to_parquet(paths["baseline"], index=False)
    checks.to_csv(paths["quality_csv"], index=False, encoding="utf-8-sig")
    paths["quality_json"].write_text(json.dumps(quality_report(checks), ensure_ascii=False, indent=2), encoding="utf-8")

    profile = _profile_summary(dispatch)
    evaluated = _forecast_evaluation(hourly, dispatch)
    error_horizon = _error_by_horizon(evaluated)
    update_gain, _ = _update_gain(evaluated)
    key_summary = _key_dates_summary(dispatch, ten_minute, contract["key_dates"])
    anomaly_flags = _anomaly_flags(dispatch)
    inventory.to_csv(tables_dir / "data_inventory.csv", index=False, encoding="utf-8-sig")
    profile.to_csv(tables_dir / "profile_summary.csv", index=False, encoding="utf-8-sig")
    error_horizon.to_csv(tables_dir / "forecast_error_by_horizon.csv", index=False, encoding="utf-8-sig")
    update_gain.to_csv(tables_dir / "forecast_update_gain.csv", index=False, encoding="utf-8-sig")
    key_summary.to_csv(tables_dir / "key_dates_summary.csv", index=False, encoding="utf-8-sig")
    anomaly_flags.to_csv(tables_dir / "anomaly_flags.csv", index=False, encoding="utf-8-sig")
    _make_figures(dispatch, ten_minute, profile, error_horizon, update_gain, contract["key_dates"], figures_dir)
    _write_html_report(paths["report"], inventory, checks, profile, error_horizon, update_gain, key_summary, dispatch, evaluated, raw["attachment_1"], anomaly_flags)

    if checks["status"].eq("FAIL").any():
        failed = checks.loc[checks["status"].eq("FAIL"), "check_id"].tolist()
        raise RuntimeError(f"Data quality checks failed: {failed}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed data and the pre-analysis package")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    paths = run_pipeline(args.project_root)
    print(json.dumps({key: str(path) for key, path in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
