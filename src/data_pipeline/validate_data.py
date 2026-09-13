"""Contract and business-rule validation for processed datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def validate_datasets(
    dispatch: pd.DataFrame,
    hourly: pd.DataFrame,
    ten_minute: pd.DataFrame,
    baseline: pd.DataFrame,
    raw: dict[str, Any],
    contract: dict[str, Any],
) -> pd.DataFrame:
    checks: list[dict[str, str]] = []

    def add(check_id: str, condition: bool, observed: Any, expected: Any, details: str) -> None:
        checks.append(
            {
                "check_id": check_id,
                "status": "PASS" if bool(condition) else "FAIL",
                "observed": str(observed),
                "expected": str(expected),
                "details": details,
            }
        )

    expected = contract["expected_rows"]
    source_shapes = {
        "attachment_1": tuple(raw["attachment_1"].shape),
        "attachment_2_load": tuple(raw["attachment_2_load"].shape),
        "attachment_2_pv": tuple(raw["attachment_2_pv"].shape),
        "attachment_3": tuple(raw["attachment_3"].shape),
        "attachment_4": tuple(raw["attachment_4"].shape),
    }
    expected_shapes = {
        "attachment_1": (144, 4),
        "attachment_2_load": (365, 145),
        "attachment_2_pv": (365, 145),
        "attachment_3": (1460, 26),
        "attachment_4": (365, 145),
    }
    add("source_table_dimensions", source_shapes == expected_shapes, source_shapes, expected_shapes, "五张源数据表的行列数符合数据契约")
    add("dispatch_row_count", len(dispatch) == expected["dispatch_10min"], len(dispatch), expected["dispatch_10min"], "全年日历物理区间主表行数")
    add("hourly_forecast_row_count", len(hourly) == expected["pv_forecast_hourly"], len(hourly), expected["pv_forecast_hourly"], "附件3小时长表行数")
    add("ten_min_forecast_row_count", len(ten_minute) == expected["pv_forecast_10min"], len(ten_minute), expected["pv_forecast_10min"], "线性插值后的10分钟预报行数")
    add("baseline_row_count", len(baseline) == expected["day_ahead_baseline_10min"], len(baseline), expected["day_ahead_baseline_10min"], "日初基线预报行数")

    actual_key_unique = not dispatch.duplicated(["plan_date", "slot_index"]).any()
    forecast_key_unique = not ten_minute.duplicated(["issue_ts", "horizon_10min"]).any()
    add("dispatch_primary_key", actual_key_unique, actual_key_unique, True, "plan_date与slot_index组合键唯一")
    add("forecast_primary_key", forecast_key_unique, forecast_key_unique, True, "issue_ts与horizon_10min组合键唯一")

    counts = dispatch.groupby("plan_date")["slot_index"].count()
    add("daily_slot_completeness", counts.eq(144).all(), f"min={counts.min()}, max={counts.max()}", "all=144", "每个源日期均含144个日历物理区间")
    diffs = dispatch.groupby("plan_date")["interval_start"].diff().dropna().dt.total_seconds().div(60)
    add("ten_minute_continuity", diffs.eq(10).all(), sorted(diffs.unique().tolist()), [10.0], "同一源日期相邻区间起点间隔恒为10分钟")
    boundary = dispatch.groupby("plan_date").agg(
        first_start=("interval_start", "first"),
        first_end=("interval_end", "first"),
        last_start=("interval_start", "last"),
        last_end=("interval_end", "last"),
    )
    boundary_ok = (
        (boundary["first_start"] - boundary.index).eq(pd.Timedelta(0))
        & (boundary["first_end"] - boundary.index).eq(pd.Timedelta(minutes=10))
        & (boundary["last_start"] - boundary.index).eq(pd.Timedelta(hours=23, minutes=50))
        & (boundary["last_end"] - boundary.index).eq(pd.Timedelta(days=1))
    ).all()
    add("calendar_day_boundaries", boundary_ok, boundary_ok, True, "源日期覆盖00:00至次日00:00的144个物理区间")

    endpoint_ok = (
        dispatch["observation_ts"].eq(dispatch["interval_end"]).all()
        and dispatch["load_pv_source_ts"].eq(dispatch["interval_end"]).all()
    )
    price_start_ok = dispatch["price_source_ts"].eq(dispatch["interval_start"]).all()
    add("load_pv_endpoint_semantics", endpoint_ok, endpoint_ok, True, "负荷与光伏观测时刻严格等于对应物理区间终点")
    add("price_start_semantics", price_start_ok, price_start_ok, True, "购电价格时刻严格等于对应物理区间起点")

    slot1 = dispatch["slot_index"].eq(1)
    template_mapping_ok = (
        dispatch.loc[slot1, "template_slot_index"].eq(144).all()
        and dispatch.loc[slot1, "template_plan_date"].eq(
            dispatch.loc[slot1, "plan_date"] - pd.Timedelta(days=1)
        ).all()
        and dispatch.loc[~slot1, "template_slot_index"].eq(
            dispatch.loc[~slot1, "slot_index"] - 1
        ).all()
        and dispatch.loc[~slot1, "template_plan_date"].eq(
            dispatch.loc[~slot1, "plan_date"]
        ).all()
    )
    add("template_mapping", template_mapping_ok, template_mapping_ok, True, "日历物理区间到正式结果模板采用明确的日期与行号映射")

    numeric_columns = [
        "load_actual_kw", "pv_actual_kw", "net_load_actual_kw", "load_actual_kwh",
        "pv_actual_kwh", "net_load_actual_kwh", "price_fixed_yuan_per_kwh",
    ]
    finite = np.isfinite(dispatch[numeric_columns].to_numpy(float)).all()
    add("actual_numeric_finite", finite, finite, True, "负荷、光伏、净负荷与固定电价字段无空值或无穷值")
    variable_available = dispatch["price_variable_available"].to_numpy(bool)
    variable_values = dispatch["price_variable_yuan_per_kwh"].to_numpy(float)
    variable_availability_ok = (
        np.array_equal(variable_available, np.isfinite(variable_values))
        and int((~variable_available).sum()) == 1
        and not variable_available[0]
        and pd.Timestamp(dispatch["interval_start"].iloc[0]) == pd.Timestamp("2025-01-01 00:00")
    )
    add("variable_price_availability", variable_availability_ok, int(variable_available.sum()), len(dispatch) - 1, "波动电价可用性标志与源数据覆盖范围一致")
    forecast_finite = np.isfinite(hourly["pv_forecast_kw"].to_numpy(float)).all() and np.isfinite(ten_minute["pv_forecast_kw"].to_numpy(float)).all()
    baseline_forecast = baseline[["load_forecast_kw", "pv_forecast_kw"]].to_numpy(float)
    baseline_day0 = baseline["plan_date"].eq(pd.Timestamp("2025-01-01")).to_numpy()
    baseline_finite = np.isfinite(baseline_forecast[~baseline_day0]).all()
    baseline_day0_nan = (~np.isfinite(baseline_forecast[baseline_day0])).all()
    add("forecast_numeric_finite", forecast_finite, forecast_finite, True, "小时及10分钟光伏预报无空值或无穷值")
    add("baseline_numeric_finite", baseline_finite, baseline_finite, True, "日初负载与光伏基线除2025-01-01外无空值或无穷值")
    add("baseline_day0_no_forecast", baseline_day0_nan, baseline_day0_nan, True, "2025-01-01日初基线为无预测（NaN），与1/1冻结（无计划、电池不动作、不入统计）口径一致")
    nonnegative = (
        dispatch[["load_actual_kw", "pv_actual_kw", "price_fixed_yuan_per_kwh"]].ge(0).all().all()
        and dispatch.loc[dispatch["price_variable_available"], "price_variable_yuan_per_kwh"].ge(0).all()
    )
    add("source_values_nonnegative", nonnegative, nonnegative, True, "负载、光伏和两类电价非负；净负荷允许为负")

    energy_ok = all(
        np.allclose(dispatch[p], dispatch[e] * 6.0, atol=1e-10, rtol=1e-10)
        for p, e in [
            ("load_actual_kw", "load_actual_kwh"),
            ("pv_actual_kw", "pv_actual_kwh"),
            ("net_load_actual_kw", "net_load_actual_kwh"),
        ]
    )
    add("power_energy_conversion", energy_ok, energy_ok, True, "10分钟电量严格等于功率乘以1/6")

    fixed_source = pd.to_numeric(raw["attachment_1"].iloc[:, 1], errors="raise").to_numpy(float)
    fixed_expected = np.r_[fixed_source[-1], fixed_source[:-1]]
    fixed_ok = np.allclose(
        dispatch["price_fixed_yuan_per_kwh"].to_numpy().reshape(-1, 144),
        fixed_expected[None, :],
    )
    variable_source = raw["attachment_4"].iloc[:, 1:145].apply(pd.to_numeric, errors="raise").to_numpy(float)
    variable_expected = np.full_like(variable_source, np.nan, dtype=float)
    variable_expected[:, 1:] = variable_source[:, :-1]
    variable_expected[1:, 0] = variable_source[:-1, -1]
    variable_ok = np.allclose(
        dispatch["price_variable_yuan_per_kwh"].to_numpy().reshape(-1, 144),
        variable_expected,
        equal_nan=True,
    )
    add("fixed_price_alignment", fixed_ok, fixed_ok, True, "固定价格按物理区间起点与附件1时刻标签对齐")
    add("variable_price_alignment", variable_ok, variable_ok, True, "波动价格按物理区间起点与附件4时刻标签对齐")

    issue_counts = hourly.groupby("issue_ts")["horizon_hour"].count()
    issue_hours = hourly[["issue_ts"]].drop_duplicates()["issue_ts"].dt.hour.value_counts().sort_index().to_dict()
    issue_ok = issue_counts.eq(24).all() and issue_hours == {0: 365, 6: 365, 12: 365, 18: 365}
    add("hourly_issue_structure", issue_ok, issue_hours, "{0,6,12,18}:365 each; 24 horizons", "每个发布时间均含24个步长")
    target_ok = (hourly["target_ts"] == hourly["issue_ts"] + pd.to_timedelta(hourly["horizon_hour"], unit="h")).all()
    add("hourly_target_mapping", target_ok, target_ok, True, "预测目标时刻等于发布时间加预测步长")

    anchor_rows = ten_minute[ten_minute["lead_minutes"].mod(60).eq(0)]
    anchor_join = anchor_rows.merge(
        hourly[["issue_ts", "target_ts", "pv_forecast_kw"]],
        on=["issue_ts", "target_ts"], how="left", suffixes=("_10min", "_hourly"), validate="one_to_one"
    )
    anchor_ok = anchor_join["pv_forecast_kw_hourly"].notna().all() and np.allclose(
        anchor_join["pv_forecast_kw_10min"], anchor_join["pv_forecast_kw_hourly"], atol=1e-10
    )
    add("interpolation_anchor_fidelity", anchor_ok, anchor_ok, True, "每个整点插值严格还原附件3小时预报")
    add("interpolation_nonnegative", ten_minute["pv_forecast_kw"].ge(0).all(), float(ten_minute["pv_forecast_kw"].min()), ">=0", "插值结果非负")

    result_count = int(dispatch["is_result_period"].sum())
    add("result_period_count", result_count == expected["result_period_intervals"], result_count, expected["result_period_intervals"], "正式结果期日历物理区间数")
    baseline_counts = baseline.groupby("plan_date")["slot_index"].count()
    baseline_causal = (baseline.loc[baseline["training_end"].notna(), "training_end"] < baseline.loc[baseline["training_end"].notna(), "issue_ts"]).all()
    add("baseline_daily_completeness", baseline_counts.eq(144).all(), f"min={baseline_counts.min()}, max={baseline_counts.max()}", "all=144", "每日基线含144个时段")
    add("baseline_causality", baseline_causal, baseline_causal, True, "训练来源计划日期严格早于发布时间")

    key_dates = pd.to_datetime(contract["key_dates"])
    key_ok = dispatch[dispatch["plan_date"].isin(key_dates)].groupby("plan_date").size().eq(144).all()
    add("key_dates_extractable", key_ok, int(dispatch[dispatch["plan_date"].isin(key_dates)].shape[0]), 576, "四个必交日期均可无歧义提取")
    return pd.DataFrame(checks)


def quality_report(checks: pd.DataFrame) -> dict[str, Any]:
    counts = checks["status"].value_counts().to_dict()
    return {
        "overall_status": "PASS" if not checks["status"].eq("FAIL").any() else "FAIL",
        "check_count": int(len(checks)),
        "status_counts": {key: int(value) for key, value in counts.items()},
        "failed_checks": checks.loc[checks["status"].eq("FAIL"), "check_id"].tolist(),
        "note": "异常值仅标记，不删除、不平滑、不缩尾。",
    }
