"""附件4 电价规律分析与"问题二式"日前预测回测.

分析目标（对应两个提问）:
1. 附件4 的 2025 年逐日 10 分钟电价是否呈某种规律:
   - 日内双峰形态、工作日/周末差异、月度/季节变化;
   - 同日同时段的前后日/前 7 日相关性与自相关结构;
   - 分布形态、近零与极端高价时点;
   - 电价与负载的同期耦合。
2. 若把问题二的负载/光伏预测方法搬到电价上, 准确性如何:
   - 与问题二完全同构的四套日前方案: 昨日持久化 b0、周持久化 b1、
     七天均值 b2、相似日高斯核 gk(四类日型, k/h/tau 同负载口径);
   - 补充: 全历史均值剖面 c0、二类日型(工作日/周末)核 gk_wd;
   - 在 1/20-1/27 调参窗上按负载口径 72 组合网格复验价格专用核参数;
   - 正式期 2025-02-01..2025-12-31 评估 MAPE/sMAPE/MAE/RMSE/偏差/相关,
     并给出按负载加权的"日前电价误差对购电费的影响"金额口径。

口径说明:
- 价格取 dispatch_10min.parquet 的 price_variable_yuan_per_kwh,
  即附件4 按"电价=区间起点"对齐后的 365x144 矩阵(与项目统一口径一致);
  2025-01-01 第 1 个时段无前一日 0:00+1 值, 用附件1 代表日 0:00 电价填充
  (该日不在正式评估期内).
- 所有方案严格因果: 第 d 日 0:00 的预测只使用 d 日之前的信息;
  冷启动只用附件1 代表日 0:00 电价(仅价格; 负载/光伏冷启动不使用附件1,
  1/1 为无预测/零计划).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.data_pipeline.ingest import PROJECT_ROOT, load_contract, read_attachments
from src.forecasting.load_day_ahead import (
    DAY_TYPES,
    GRID_H,
    GRID_K,
    GRID_TAU,
    KERNEL_H,
    KERNEL_K,
    KERNEL_TAU,
    build_day_type_frame,
    gaussian_kernel_forecasts,
)

PRICE_SIGMA_FLOOR = 1e-4  # 元/kWh, 约为 0.03%*0.77 元/kWh(负载 1.0 kW 相对均值的同比例)
FORMAL_START = "2025-02-01"
FORMAL_END = "2025-12-31"
TUNE_START = "2025-01-20"
TUNE_END = "2025-01-27"


def _use_chinese_labels() -> bool:
    from matplotlib import font_manager

    names = {Path(f).name.lower() for f in font_manager.findSystemFonts()}
    return any("msyh" in n or "simhei" in n or "simsun" in n for n in names)


def _setup_fonts() -> None:
    if _use_chinese_labels():
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False


def load_price_matrix(dispatch: pd.DataFrame, representative: np.ndarray) -> tuple[np.ndarray, pd.DatetimeIndex]:
    ordered = dispatch.sort_values(["plan_date", "slot_index"])
    dates = pd.DatetimeIndex(ordered["plan_date"].drop_duplicates()).sort_values()
    matrix = (
        ordered.pivot(index="plan_date", columns="slot_index", values="price_variable_yuan_per_kwh")
        .reindex(index=dates, columns=np.arange(1, 145))
        .to_numpy(float)
    )
    # 2025-01-01 第一个时段 (00:00-00:10) 无前一日 0:00+1 价格: 用附件1 代表日
    # 0:00 电价填充(representative[0] = 附件1 末行 0:00+1 值 = 区间 00:00-00:10
    # 的区间起点价)。注意:该回填仅用于本分析矩阵;结果管线 R0 对该格采用
    # "1/1 当日 00:10 标签价" 回填,两处均只影响预热期单格,不进正式期评估。
    nan_mask = ~np.isfinite(matrix)
    if nan_mask.sum() == 1 and np.isnan(matrix[0, 0]):
        matrix[0, 0] = representative[0]
    elif nan_mask.any():
        raise ValueError(f"Unexpected NaN pattern in price matrix: {int(nan_mask.sum())}")
    return matrix, dates


def load_representative_price(raw: dict[str, Any]) -> np.ndarray:
    source = pd.to_numeric(raw["attachment_1"].iloc[:, 1], errors="raise").to_numpy(float)
    if len(source) != 144:
        raise ValueError("Attachment 1 must contain 144 representative price slots")
    # 附件1 行序 00:10..0:00+1; 区间起点口径: 第1段取最后一行的 0:00+1 值
    return np.r_[source[-1], source[:-1]]


def _daily_summary_eda(matrix: np.ndarray, dates: pd.DatetimeIndex, out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for month in range(1, 13):
        mask = dates.month == month
        block = matrix[mask]
        rows.append(
            {
                "month": month,
                "days": int(mask.sum()),
                "mean": float(block.mean()),
                "std": float(block.std()),
                "min": float(block.min()),
                "max": float(block.max()),
                "p05": float(np.quantile(block, 0.05)),
                "p50": float(np.quantile(block, 0.50)),
                "p95": float(np.quantile(block, 0.95)),
                "share_below_0p1": float((block < 0.1).mean()),
                "share_above_1p5": float((block > 1.5).mean()),
            }
        )
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out_dir / "eda_monthly_stats.csv", index=False, encoding="utf-8-sig")

    daily_mean = matrix.mean(axis=1)
    daily_std = matrix.std(axis=1)
    daily_max = matrix.max(axis=1)
    daily_min = matrix.min(axis=1)
    daily = pd.DataFrame(
        {
            "plan_date": dates,
            "daily_mean": daily_mean,
            "daily_std": daily_std,
            "daily_min": daily_min,
            "daily_max": daily_max,
            "daily_range": daily_max - daily_min,
        }
    )
    daily.to_csv(out_dir / "eda_daily_stats.csv", index=False, encoding="utf-8-sig")

    intraday = pd.DataFrame(
        {
            "slot_index": np.arange(1, 145),
            "mean": matrix.mean(axis=0),
            "std": matrix.std(axis=0),
            "p05": np.quantile(matrix, 0.05, axis=0),
            "p50": np.quantile(matrix, 0.50, axis=0),
            "p95": np.quantile(matrix, 0.95, axis=0),
        }
    )
    intraday.to_csv(out_dir / "eda_intraday_profile.csv", index=False, encoding="utf-8-sig")

    # 周内形态: 周一..周日 平均曲线
    weekday_profiles = {}
    for wd in range(7):
        weekday_profiles[wd] = matrix[dates.dayofweek == wd].mean(axis=0)
    weekday_frame = pd.DataFrame(weekday_profiles, index=np.arange(1, 145))
    weekday_frame.columns = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday_frame.to_csv(out_dir / "eda_weekday_profiles.csv", encoding="utf-8-sig")

    # 同日同时段前后相关性 (全样本)
    lag1_corrs, lag7_corrs = [], []
    for s in range(matrix.shape[1]):
        series = matrix[:, s]
        lag1_corrs.append(np.corrcoef(series[1:], series[:-1])[0, 1])
        lag7_corrs.append(np.corrcoef(series[7:], series[:-7])[0, 1])
    slot_lag = pd.DataFrame(
        {
            "slot_index": np.arange(1, 145),
            "corr_lag1d": lag1_corrs,
            "corr_lag7d": lag7_corrs,
        }
    )
    slot_lag.to_csv(out_dir / "eda_slot_lag_correlations.csv", index=False, encoding="utf-8-sig")

    # 日均价自相关 ACF (lag 1..21)
    acf = {
        lag: float(np.corrcoef(daily_mean[lag:], daily_mean[:-lag])[0, 1])
        for lag in range(1, 22)
    }
    acf_frame = pd.DataFrame({"lag_days": list(acf), "acf": list(acf.values())})
    acf_frame.to_csv(out_dir / "eda_daily_mean_acf.csv", index=False, encoding="utf-8-sig")

    # 近零价与极端高价时点
    near_zero = (matrix < 0.05).sum(axis=1)
    high = (matrix > 1.5).sum(axis=1)
    extremes = pd.DataFrame(
        {
            "plan_date": dates,
            "slots_below_0p05": near_zero,
            "slots_above_1p5": high,
        }
    )
    extremes = extremes[(extremes["slots_below_0p05"] > 0) | (extremes["slots_above_1p5"] > 0)]
    extremes.to_csv(out_dir / "eda_extreme_slots.csv", index=False, encoding="utf-8-sig")

    summary = {
        "year": {"mean": float(matrix.mean()), "std": float(matrix.std()),
                 "min": float(matrix.min()), "max": float(matrix.max()),
                 "p05": float(np.quantile(matrix, 0.05)),
                 "p50": float(np.quantile(matrix, 0.50)),
                 "p95": float(np.quantile(matrix, 0.95))},
        "share_below_0p1": float((matrix < 0.1).mean()),
        "share_above_1p5": float((matrix > 1.5).mean()),
        "mean_slot_lag1d_corr": float(np.mean(lag1_corrs)),
        "mean_slot_lag7d_corr": float(np.mean(lag7_corrs)),
        "daily_mean_acf_lag1": acf[1],
        "daily_mean_acf_lag7": acf[7],
        "daily_mean_acf_lag14": acf[14],
        "weekday_weekend_mean_diff": float(
            matrix[dates.dayofweek < 5].mean() - matrix[dates.dayofweek >= 5].mean()
        ),
    }
    return summary


def _eda_figures(
    matrix: np.ndarray,
    dates: pd.DatetimeIndex,
    load_matrix: np.ndarray,
    out_dir: Path,
) -> None:
    cn = _use_chinese_labels()
    hours = np.arange(1, 145) / 6.0

    # 图1: 日内剖面均值±1σ 与 5/95 分位
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    ax.fill_between(hours, np.quantile(matrix, 0.05, axis=0), np.quantile(matrix, 0.95, axis=0),
                    color="tab:blue", alpha=0.18, label="5%-95%")
    ax.fill_between(hours, matrix.mean(axis=0) - matrix.std(axis=0), matrix.mean(axis=0) + matrix.std(axis=0),
                    color="tab:blue", alpha=0.35, label="均值±1σ")
    ax.plot(hours, matrix.mean(axis=0), color="tab:blue", lw=1.8, label="日内均值")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel("时刻 (h)" if not cn else "时刻 (小时)")
    ax.set_ylabel("电价 (元/kWh)" if not cn else "电价（元/kWh）")
    ax.set_title("2025年电价日内剖面" if cn else "2025 price intraday profile")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_intraday_profile.png")
    plt.close(fig)

    # 图2: 工作日/周末 + 季节
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=200)
    wd = dates.dayofweek < 5
    axes[0].plot(hours, matrix[wd].mean(axis=0), label="工作日" if cn else "Weekday", lw=1.8)
    axes[0].plot(hours, matrix[~wd].mean(axis=0), label="周末" if cn else "Weekend", lw=1.8)
    axes[0].set_xlim(0, 24)
    axes[0].set_xlabel("时刻 (h)" if not cn else "时刻（小时）")
    axes[0].set_ylabel("电价 (元/kWh)" if not cn else "电价（元/kWh）")
    axes[0].set_title("工作日 vs 周末" if cn else "Weekday vs weekend")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    season = {"冬(1-2月)": (1, 2), "春(3-5月)": (3, 5), "夏(6-8月)": (6, 8), "秋(9-11月)": (9, 11)}
    colors = ["tab:blue", "tab:green", "tab:red", "tab:orange"]
    for (label, (m0, m1)), color in zip(season.items(), colors):
        mask = (dates.month >= m0) & (dates.month <= m1)
        axes[1].plot(hours, matrix[mask].mean(axis=0), label=label if cn else f"{m0}-{m1}", color=color, lw=1.6)
    axes[1].set_xlim(0, 24)
    axes[1].set_xlabel("时刻 (h)" if not cn else "时刻（小时）")
    axes[1].set_title("季节剖面" if cn else "Seasonal profiles")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_weekday_season.png")
    plt.close(fig)

    # 图3: 日均价走势 + 波动
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=200)
    daily_mean = matrix.mean(axis=1)
    ax.plot(dates, daily_mean, lw=0.9, color="tab:blue", label="日均价" if cn else "Daily mean")
    ax.plot(dates, pd.Series(daily_mean).rolling(7, center=True).mean(), lw=1.6, color="tab:red",
            label="7日滑动均值" if cn else "7-day MA")
    ax.set_ylabel("电价 (元/kWh)" if not cn else "电价（元/kWh）")
    ax.set_title("2025年日均电价走势" if cn else "2025 daily mean price")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_daily_mean.png")
    plt.close(fig)

    # 图4: ACF 与同槽滞后相关
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=200)
    lags = np.arange(1, 22)
    acf = [np.corrcoef(daily_mean[l:], daily_mean[:-l])[0, 1] for l in lags]
    axes[0].stem(lags, acf)
    axes[0].set_xlabel("滞后 (天)" if cn else "Lag (days)")
    axes[0].set_ylabel("日均价自相关" if cn else "Daily-mean ACF")
    axes[0].grid(alpha=0.3)
    lag1, lag7 = [], []
    for s in range(matrix.shape[1]):
        series = matrix[:, s]
        lag1.append(np.corrcoef(series[1:], series[:-1])[0, 1])
        lag7.append(np.corrcoef(series[7:], series[:-7])[0, 1])
    axes[1].plot(hours, lag1, label="滞后1天" if cn else "lag 1 day")
    axes[1].plot(hours, lag7, label="滞后7天" if cn else "lag 7 days")
    axes[1].set_xlim(0, 24)
    axes[1].set_xlabel("时刻 (h)" if not cn else "时刻（小时）")
    axes[1].set_ylabel("同槽相关系数" if cn else "Same-slot correlation")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_autocorrelation.png")
    plt.close(fig)

    # 图5: 分布直方图
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=200)
    ax.hist(matrix.ravel(), bins=80, color="tab:blue", alpha=0.8, edgecolor="white")
    ax.set_xlabel("电价 (元/kWh)" if not cn else "电价（元/kWh）")
    ax.set_ylabel("时段数" if cn else "Slots")
    ax.set_title("2025年电价分布" if cn else "2025 price histogram")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_histogram.png")
    plt.close(fig)

    # 图6: 电价-负载同期耦合 (标准化后叠加)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=200)
    p_norm = (matrix - matrix.mean()) / matrix.std()
    l_norm = (load_matrix - load_matrix.mean()) / load_matrix.std()
    ax.plot(hours, p_norm.mean(axis=0), label="电价(标准化)" if cn else "Price (z)", lw=1.8, color="tab:red")
    ax.plot(hours, l_norm.mean(axis=0), label="负载(标准化)" if cn else "Load (z)", lw=1.8, color="tab:blue")
    ax.set_xlim(0, 24)
    ax.set_xlabel("时刻 (h)" if not cn else "时刻（小时）")
    ax.set_ylabel("标准化均值" if cn else "Standardized mean")
    ax.set_title("电价与负载的日内耦合" if cn else "Intraday price-load coupling")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_price_load_coupling.png")
    plt.close(fig)


def build_schemes(
    price_matrix: np.ndarray,
    day_types: np.ndarray,
    representative: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """与问题二负载四方案同构的日前电价预测矩阵."""
    days, slots = price_matrix.shape
    b0 = np.empty_like(price_matrix)
    b1 = np.empty_like(price_matrix)
    b2 = np.empty_like(price_matrix)
    c0 = np.empty_like(price_matrix)
    for d in range(days):
        b0[d] = price_matrix[d - 1] if d > 0 else representative
        b1[d] = price_matrix[d - 7] if d >= 7 else representative
        b2[d] = price_matrix[max(0, d - 7) : d].mean(axis=0) if d > 0 else representative
        c0[d] = price_matrix[:d].mean(axis=0) if d > 0 else representative
    gk, pool_sizes = gaussian_kernel_forecasts(
        price_matrix, day_types, k=KERNEL_K, h=KERNEL_H, tau=KERNEL_TAU,
        sigma_floor=PRICE_SIGMA_FLOOR,
    )
    for d in range(days):
        if not np.isfinite(gk[d]).all():
            gk[d] = price_matrix[d - 1] if d > 0 else representative
    # 二类日型(工作日/周末)核
    wd_types = np.where(np.isin(day_types, ("workday",)), "workday", "weekend")
    gk_wd, _ = gaussian_kernel_forecasts(
        price_matrix, wd_types, k=KERNEL_K, h=KERNEL_H, tau=KERNEL_TAU,
        sigma_floor=PRICE_SIGMA_FLOOR,
    )
    for d in range(days):
        if not np.isfinite(gk_wd[d]).all():
            gk_wd[d] = price_matrix[d - 1] if d > 0 else representative
    return {
        "c0": c0,
        "b0": b0,
        "b1": b1,
        "b2": b2,
        "gk": gk,
        "gk_wd": gk_wd,
    }, pool_sizes


def _scheme_metrics(
    dates: pd.DatetimeIndex,
    day_types: np.ndarray,
    price_matrix: np.ndarray,
    schemes: dict[str, np.ndarray],
    result_days: np.ndarray,
    load_kwh_matrix: np.ndarray | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [("formal", result_days)] + [
        (day_type, (result_days & (day_types == day_type))) for day_type in DAY_TYPES
    ]
    for name, matrix in schemes.items():
        for scope, mask in scopes:
            error = matrix[mask] - price_matrix[mask]
            actual = price_matrix[mask]
            mape_full = np.mean(np.abs(error) / actual) * 100.0
            keep = actual > 0.02
            mape_trim = np.mean(np.abs(error[keep]) / actual[keep]) * 100.0 if keep.any() else np.nan
            smape = np.mean(2 * np.abs(error) / (np.abs(matrix[mask]) + np.abs(actual))) * 100.0
            corr = np.corrcoef(matrix[mask].ravel(), actual.ravel())[0, 1]
            rows.append(
                {
                    "scheme": name,
                    "scope": scope,
                    "days": int(mask.sum()),
                    "mape_pct": float(mape_full),
                    "mape_trim_pct": float(mape_trim),
                    "smape_pct": float(smape),
                    "mae_yuan_per_kwh": float(np.mean(np.abs(error))),
                    "rmse_yuan_per_kwh": float(np.sqrt(np.mean(error**2))),
                    "bias_yuan_per_kwh": float(np.mean(error)),
                    "corr": float(corr),
                }
            )
    metrics = pd.DataFrame(rows)

    if load_kwh_matrix is not None:
        cost_rows: list[dict[str, Any]] = []
        daily_actual_cost = (price_matrix * load_kwh_matrix).sum(axis=1)
        for name, matrix in schemes.items():
            daily_cost_error = ((matrix - price_matrix) * load_kwh_matrix).sum(axis=1)
            for scope, mask in [("formal", result_days)]:
                c = daily_cost_error[mask]
                a = daily_actual_cost[mask]
                cost_rows.append(
                    {
                        "scheme": name,
                        "scope": scope,
                        "days": int(mask.sum()),
                        "mean_daily_cost_error_yuan": float(np.mean(c)),
                        "mean_abs_daily_cost_error_yuan": float(np.mean(np.abs(c))),
                        "rmse_daily_cost_error_yuan": float(np.sqrt(np.mean(c**2))),
                        "mean_daily_actual_cost_yuan": float(np.mean(a)),
                        "abs_error_share_of_cost_pct": float(np.mean(np.abs(c)) / np.mean(a) * 100.0),
                    }
                )
        cost_metrics = pd.DataFrame(cost_rows)
    else:
        cost_metrics = pd.DataFrame()
    return metrics, cost_metrics


def _monthly_metrics(
    dates: pd.DatetimeIndex,
    price_matrix: np.ndarray,
    schemes: dict[str, np.ndarray],
    result_days: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month in range(2, 13):
        mask = result_days & (dates.month == month)
        for name, matrix in schemes.items():
            error = matrix[mask] - price_matrix[mask]
            actual = price_matrix[mask]
            rows.append(
                {
                    "month": month,
                    "scheme": name,
                    "days": int(mask.sum()),
                    "mape_pct": float(np.mean(np.abs(error) / actual) * 100.0),
                    "mae_yuan_per_kwh": float(np.mean(np.abs(error))),
                    "rmse_yuan_per_kwh": float(np.sqrt(np.mean(error**2))),
                    "bias_yuan_per_kwh": float(np.mean(error)),
                }
            )
    return pd.DataFrame(rows)


def _key_date_metrics(
    dates: pd.DatetimeIndex,
    price_matrix: np.ndarray,
    schemes: dict[str, np.ndarray],
    key_dates: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date_text in key_dates:
        date = pd.Timestamp(date_text)
        day = int(dates.get_loc(date))
        actual = price_matrix[day]
        for name, matrix in schemes.items():
            error = matrix[day] - actual
            rows.append(
                {
                    "plan_date": date,
                    "scheme": name,
                    "mape_pct": float(np.mean(np.abs(error) / actual) * 100.0),
                    "smape_pct": float(np.mean(2 * np.abs(error) / (np.abs(matrix[day]) + np.abs(actual))) * 100.0),
                    "mae_yuan_per_kwh": float(np.mean(np.abs(error))),
                    "rmse_yuan_per_kwh": float(np.sqrt(np.mean(error**2))),
                    "corr": float(np.corrcoef(matrix[day], actual)[0, 1]),
                    "spearman": float(stats.spearmanr(matrix[day], actual).statistic),
                }
            )
    return pd.DataFrame(rows)


def _daily_shape_metrics(
    dates: pd.DatetimeIndex,
    price_matrix: np.ndarray,
    schemes: dict[str, np.ndarray],
    result_days: np.ndarray,
) -> pd.DataFrame:
    """逐日剖面形态匹配: 平均逐日 Pearson/Spearman 相关(评估'峰谷位置'把握)."""
    rows: list[dict[str, Any]] = []
    for name, matrix in schemes.items():
        pears, spears = [], []
        for d in np.flatnonzero(result_days):
            pears.append(np.corrcoef(matrix[d], price_matrix[d])[0, 1])
            spears.append(stats.spearmanr(matrix[d], price_matrix[d]).statistic)
        rows.append(
            {
                "scheme": name,
                "mean_daily_pearson": float(np.mean(pears)),
                "mean_daily_spearman": float(np.mean(spears)),
            }
        )
    return pd.DataFrame(rows)


def verify_price_kernel_grid(
    price_matrix: np.ndarray,
    day_types: np.ndarray,
    dates: pd.DatetimeIndex,
    tune_start: str = TUNE_START,
    tune_end: str = TUNE_END,
) -> pd.DataFrame:
    start, end = pd.Timestamp(tune_start), pd.Timestamp(tune_end)
    window_mask = (dates >= start) & (dates <= end)
    truncate = int(np.flatnonzero(window_mask)[-1]) + 1
    truncated = price_matrix[:truncate]
    truncated_types = day_types[:truncate]
    mask = window_mask[:truncate]
    actual = truncated[mask]
    rows: list[dict[str, Any]] = []
    for k in GRID_K:
        for h in GRID_H:
            for tau in GRID_TAU:
                forecasts, _ = gaussian_kernel_forecasts(
                    truncated, truncated_types, k=k, h=h, tau=tau,
                    sigma_floor=PRICE_SIGMA_FLOOR,
                )
                error = forecasts[mask] - actual
                keep = actual > 0.02
                mape = float(np.mean(np.abs(error[keep]) / actual[keep]) * 100.0)
                rows.append({"k": k, "h": h, "tau": tau, "mape_pct": mape})
    return pd.DataFrame(rows).sort_values("mape_pct").reset_index(drop=True)


def run_price_analysis(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    project_root = Path(project_root).resolve()
    out_dir = project_root / "outputs" / "question4" / "analysis" / "price_forecast"
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = project_root / "data" / "processed"
    contract = load_contract(project_root)
    raw = read_attachments(project_root)
    dispatch = pd.read_parquet(processed_dir / "dispatch_10min.parquet")
    representative = load_representative_price(raw)

    price_matrix, dates = load_price_matrix(dispatch, representative)
    load_kwh_matrix = (
        dispatch.sort_values(["plan_date", "slot_index"])
        .pivot(index="plan_date", columns="slot_index", values="load_actual_kwh")
        .reindex(index=dates, columns=np.arange(1, 145))
        .to_numpy(float)
    )
    load_kw_matrix = load_kwh_matrix * 6.0
    day_types = build_day_type_frame(dates)["day_type"].to_numpy()
    result_days = (dates >= pd.Timestamp(FORMAL_START)) & (dates <= pd.Timestamp(FORMAL_END))

    # 保存 365x144 对齐矩阵 (计划日 x 时段)
    long = pd.DataFrame(
        {
            "plan_date": np.repeat(dates.to_numpy(), 144),
            "slot_index": np.tile(np.arange(1, 145), len(dates)),
            "price_yuan_per_kwh": price_matrix.reshape(-1),
        }
    )
    long.to_parquet(processed_dir / "price_10min.parquet", index=False)

    _setup_fonts()
    eda_summary = _daily_summary_eda(price_matrix, dates, out_dir)
    _eda_figures(price_matrix, dates, load_kw_matrix, out_dir)

    schemes, pool_sizes = build_schemes(price_matrix, day_types, representative)
    metrics, cost_metrics = _scheme_metrics(
        dates, day_types, price_matrix, schemes, result_days, load_kwh_matrix
    )
    monthly = _monthly_metrics(dates, price_matrix, schemes, result_days)
    key_dates = _key_date_metrics(dates, price_matrix, schemes, contract["key_dates"])
    shape_metrics = _daily_shape_metrics(dates, price_matrix, schemes, result_days)
    grid = verify_price_kernel_grid(price_matrix, day_types, dates)

    metrics.to_csv(out_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    cost_metrics.to_csv(out_dir / "cost_impact_metrics.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(out_dir / "monthly_metrics.csv", index=False, encoding="utf-8-sig")
    key_dates.to_csv(out_dir / "key_dates_metrics.csv", index=False, encoding="utf-8-sig")
    shape_metrics.to_csv(out_dir / "daily_shape_metrics.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(out_dir / "kernel_grid_verification.csv", index=False, encoding="utf-8-sig")

    # 重点日期预测-实际对比图
    _key_date_figures(dates, price_matrix, schemes, contract["key_dates"], out_dir)

    best = grid.iloc[0]
    summary = {
        "eda": eda_summary,
        "kernel_grid": {
            "tuning_window": [TUNE_START, TUNE_END],
            "combinations": int(len(grid)),
            "best": {"k": int(best["k"]), "h": float(best["h"]), "tau": float(best["tau"]),
                     "mape_pct": float(best["mape_pct"])},
            "load_params_k10_h2_tau14_mape_pct": float(
                grid[(grid["k"] == 10) & (grid["h"] == 2.0) & (grid["tau"] == 14)]["mape_pct"].iloc[0]
            ),
        },
        "formal_metrics": metrics.to_dict(orient="records"),
        "cost_impact": cost_metrics.to_dict(orient="records"),
        "daily_shape": shape_metrics.to_dict(orient="records"),
    }
    paths = {
        "summary": out_dir / "price_analysis_summary.json",
        "long_price": processed_dir / "price_10min.parquet",
        "metrics": out_dir / "model_metrics.csv",
        "cost_metrics": out_dir / "cost_impact_metrics.csv",
        "monthly": out_dir / "monthly_metrics.csv",
        "key_dates": out_dir / "key_dates_metrics.csv",
        "shape": out_dir / "daily_shape_metrics.csv",
        "grid": out_dir / "kernel_grid_verification.csv",
        "eda_monthly": out_dir / "eda_monthly_stats.csv",
        "eda_daily": out_dir / "eda_daily_stats.csv",
        "eda_intraday": out_dir / "eda_intraday_profile.csv",
        "eda_weekday": out_dir / "eda_weekday_profiles.csv",
        "eda_slot_lag": out_dir / "eda_slot_lag_correlations.csv",
        "eda_acf": out_dir / "eda_daily_mean_acf.csv",
        "eda_extremes": out_dir / "eda_extreme_slots.csv",
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths


def _key_date_figures(
    dates: pd.DatetimeIndex,
    price_matrix: np.ndarray,
    schemes: dict[str, np.ndarray],
    key_dates: list[str],
    out_dir: Path,
) -> None:
    cn = _use_chinese_labels()
    hours = np.arange(1, 145) / 6.0
    names = {
        "b0": "昨日持久化" if cn else "yesterday",
        "b1": "周持久化" if cn else "weekly",
        "b2": "七天均值" if cn else "7d mean",
        "gk": "相似日核" if cn else "kernel",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=200)
    for ax, date_text in zip(axes.ravel(), key_dates):
        date = pd.Timestamp(date_text)
        day = int(dates.get_loc(date))
        ax.plot(hours, price_matrix[day], color="black", lw=2.0, label="实际" if cn else "actual")
        for name, color in zip(("b0", "b1", "b2", "gk"), ("tab:gray", "tab:blue", "tab:green", "tab:red")):
            ax.plot(hours, schemes[name][day], color=color, lw=1.2, alpha=0.85, label=names[name])
        ax.set_xlim(0, 24)
        ax.set_title(f"{date_text}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("重点日期日前电价预测 vs 实际" if cn else "Key dates: day-ahead price forecasts")
    fig.tight_layout()
    fig.savefig(out_dir / "fig7_key_dates_forecasts.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="附件4 电价规律分析与问题二式日前预测回测")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    paths = run_price_analysis(args.project_root)
    print(json.dumps({key: str(path) for key, path in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
