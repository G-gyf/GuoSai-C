"""Causal day-ahead PV forecasts for question 2.

The implementation separates the deterministic solar envelope from a KPV
weather index.  Every forecast origin uses only completed plan dates strictly
before its 00:00 issue timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


CANDIDATE_NAMES = (
    "a1_7",
    "a1_30",
    "a2",
    "a3",
    "a4_k5",
    "a4_k10",
    "a4_k20",
)
COMPLEX_CANDIDATES = CANDIDATE_NAMES[2:]


@dataclass(frozen=True)
class PVDayAheadConfig:
    """Fixed, auditable controls for the PV forecasting pipeline."""

    hard_daylight_start_minute: int = 4 * 60 + 10
    hard_daylight_end_minute: int = 19 * 60 + 30
    positive_threshold_kw: float = 1.0
    boundary_window_days: int = 30
    boundary_buffer_minutes: int = 60
    solar_grid_points: int = 121
    shape_quantile: float = 0.90
    shape_savgol_window: int = 11
    shape_savgol_order: int = 3
    amplitude_window_days: int = 30
    amplitude_quantile: float = 0.90
    normalization_floor_ratio: float = 0.02
    ar_window_days: int = 90
    ar_min_samples: int = 14
    ridge_alpha: float = 1.0
    knn_window_days: int = 120
    knn_lag_days: int = 7
    knn_neighbors: tuple[int, ...] = (5, 10, 20)
    boundary_thresholds: tuple[float, ...] = (0.10, 0.15, 0.20)
    selection_window_days: int = 30
    selection_min_days: int = 14
    selection_improvement_ratio: float = 0.01
    scenario_window_days: int = 60
    scenario_count: int = 50
    scenario_seed: int = 20_250_911

    @classmethod
    def from_contract(cls, contract: dict[str, Any]) -> "PVDayAheadConfig":
        raw = contract.get("forecast", {}).get("pv_day_ahead", {})

        def minute(text: str) -> int:
            hour, value = text.split(":")
            return int(hour) * 60 + int(value)

        return cls(
            hard_daylight_start_minute=minute(raw.get("hard_daylight_start", "04:10")),
            hard_daylight_end_minute=minute(raw.get("hard_daylight_end", "19:30")),
            positive_threshold_kw=float(raw.get("positive_threshold_kw", 1.0)),
            boundary_window_days=int(raw.get("boundary_window_days", 30)),
            boundary_buffer_minutes=int(raw.get("boundary_buffer_minutes", 60)),
            solar_grid_points=int(raw.get("solar_grid_points", 121)),
            shape_quantile=float(raw.get("shape_quantile", 0.90)),
            shape_savgol_window=int(raw.get("shape_savgol_window", 11)),
            shape_savgol_order=int(raw.get("shape_savgol_order", 3)),
            amplitude_window_days=int(raw.get("amplitude_window_days", 30)),
            amplitude_quantile=float(raw.get("amplitude_quantile", 0.90)),
            normalization_floor_ratio=float(raw.get("normalization_floor_ratio", 0.02)),
            ar_window_days=int(raw.get("ar_window_days", 90)),
            ar_min_samples=int(raw.get("ar_min_samples", 14)),
            ridge_alpha=float(raw.get("ridge_alpha", 1.0)),
            knn_window_days=int(raw.get("knn_window_days", 120)),
            knn_lag_days=int(raw.get("knn_lag_days", 7)),
            knn_neighbors=tuple(int(value) for value in raw.get("knn_neighbors", (5, 10, 20))),
            boundary_thresholds=tuple(float(value) for value in raw.get("boundary_thresholds", (0.10, 0.15, 0.20))),
            selection_window_days=int(raw.get("selection_window_days", 30)),
            selection_min_days=int(raw.get("selection_min_days", 14)),
            selection_improvement_ratio=float(raw.get("selection_improvement_ratio", 0.01)),
            scenario_window_days=int(raw.get("scenario_window_days", 60)),
            scenario_count=int(raw.get("scenario_count", 50)),
            scenario_seed=int(raw.get("scenario_seed", 20_250_911)),
        )


@dataclass
class PVForecastResult:
    hourly: pd.DataFrame
    ten_minute: pd.DataFrame
    residuals: pd.DataFrame
    metrics: pd.DataFrame
    key_dates: pd.DataFrame
    summary: dict[str, Any]


def _daily_geometry(
    matrix: np.ndarray, threshold_kw: float, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    days = matrix.shape[0]
    first = np.full(days, np.nan)
    last = np.full(days, np.nan)
    peaks = matrix.max(axis=1)
    valid = np.zeros(days, dtype=bool)
    shapes = np.full((days, len(grid)), np.nan)
    slot_minutes = np.arange(1, 145, dtype=float) * 10.0
    for day in range(days):
        positive = np.flatnonzero(matrix[day] > threshold_kw)
        if len(positive) == 0 or peaks[day] <= 0:
            continue
        start, end = int(positive[0]), int(positive[-1])
        first[day] = slot_minutes[start]
        last[day] = slot_minutes[end]
        segment = matrix[day, start : end + 1]
        shapes[day] = np.interp(grid, np.linspace(0.0, 1.0, len(segment)), segment) / peaks[day]
        valid[day] = True
    return first, last, peaks, valid, shapes


def _smooth_shape(values: np.ndarray, config: PVDayAheadConfig) -> np.ndarray:
    smoothed = savgol_filter(
        values,
        window_length=config.shape_savgol_window,
        polyorder=config.shape_savgol_order,
        mode="interp",
    )
    smoothed = np.maximum(smoothed, 0.0)
    smoothed[[0, -1]] = 0.0
    peak = float(smoothed.max())
    if peak <= 0:
        raise ValueError("The empirical solar shape has no positive values")
    return smoothed / peak


def _ridge_predict(x: np.ndarray, y: np.ndarray, target: np.ndarray, alpha: float) -> float:
    center = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - center) / scale
    z_target = (target - center) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return float(np.r_[1.0, z_target] @ coefficients)


def _ar1_predict(previous_x: np.ndarray, y: np.ndarray, latest: float) -> float:
    design = np.column_stack([np.ones(len(previous_x)), previous_x])
    intercept, phi = np.linalg.lstsq(design, y, rcond=None)[0]
    phi = float(np.clip(phi, -0.95, 0.95))
    return float(intercept + phi * latest)


def _knn_predict(
    x: np.ndarray, y: np.ndarray, target: np.ndarray, neighbors: int
) -> float:
    center = np.median(x, axis=0)
    q25, q75 = np.quantile(x, [0.25, 0.75], axis=0)
    scale = q75 - q25
    scale[scale < 1e-8] = 1.0
    distances = np.sqrt(np.sum(((x - target) / scale) ** 2, axis=1))
    chosen = np.argsort(distances)[: min(neighbors, len(distances))]
    weights = 1.0 / (distances[chosen] + 1e-6)
    return float(np.average(y[chosen], weights=weights))


def restore_hourly_to_10min(hourly_kw: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Restore 24 hourly block means while conserving every block mean."""
    hourly = np.asarray(hourly_kw, dtype=float)
    weights = np.asarray(template, dtype=float)
    if hourly.shape != (24,) or weights.shape != (144,):
        raise ValueError("Expected a 24-value hourly curve and a 144-value template")
    restored = np.maximum(np.repeat(hourly, 6) * weights, 0.0)
    for hour in range(24):
        block = slice(hour * 6, (hour + 1) * 6)
        target = max(float(hourly[hour]), 0.0)
        current = float(restored[block].mean())
        if target == 0.0:
            restored[block] = 0.0
        elif current > 0.0:
            restored[block] *= target / current
        else:
            raise ValueError(f"Positive hourly forecast has no active template slots in block {hour + 1}")
    return restored


def _normalized_day_loss(
    forecast: np.ndarray,
    actual: np.ndarray,
    hard_mask: np.ndarray,
    amplitude_kw: float,
) -> float:
    return float(np.mean(np.abs(forecast[hard_mask] - actual[hard_mask])) / max(amplitude_kw, 1.0))


def _choose_threshold(
    losses: np.ndarray,
    day: int,
    config: PVDayAheadConfig,
) -> int:
    start = max(0, day - config.selection_window_days)
    history = losses[start:day]
    valid_days = np.isfinite(history).all(axis=1)
    default = int(np.argmin(np.abs(np.asarray(config.boundary_thresholds) - 0.15)))
    if int(valid_days.sum()) < config.selection_min_days:
        return default
    return int(np.argmin(np.mean(history[valid_days], axis=0)))


def _select_policy(
    day: int,
    candidate_10min: dict[str, np.ndarray],
    daily_losses: np.ndarray,
    actual_10min: np.ndarray,
    hard_mask: np.ndarray,
    amplitude: np.ndarray,
    config: PVDayAheadConfig,
) -> tuple[str, dict[str, float], dict[str, float], float, float]:
    """Strict winner-take-all selection on the rolling loss history.

    Default policy is ``a1_7``.  A complex candidate (a2/a3/a4_k*) replaces it
    only when its rolling mean loss is strictly lower than that of ``a1_7``
    (numerical tolerance 1e-12); the winner is then the complex candidate with
    the lowest rolling loss.  ``a1_30`` is reported for comparison only and
    never participates in switching.  No ensemble, no improvement gate and no
    hysteresis.
    """
    candidate_index = {name: idx for idx, name in enumerate(CANDIDATE_NAMES)}
    start = max(0, day - config.selection_window_days)
    history = daily_losses[start:day]
    valid_rows = np.isfinite(history).all(axis=1)
    losses = {name: np.nan for name in CANDIDATE_NAMES}
    if int(valid_rows.sum()) < config.selection_min_days:
        weights = {name: float(name == "a1_7") for name in CANDIDATE_NAMES}
        return "a1_7", losses, weights, np.nan, np.nan

    averages = np.mean(history[valid_rows], axis=0)
    losses = {name: float(averages[idx]) for name, idx in candidate_index.items()}
    default_loss = losses["a1_7"]
    tolerance = 1e-12
    challengers = [
        name
        for name in COMPLEX_CANDIDATES
        if losses[name] < default_loss - tolerance
    ]
    if not challengers:
        winner = "a1_7"
    else:
        winner = min(challengers, key=lambda name: losses[name])
        if losses[winner] >= default_loss - tolerance:
            winner = "a1_7"
    weights = {name: float(name == winner) for name in CANDIDATE_NAMES}
    selected_loss = losses[winner] if np.isfinite(losses[winner]) else np.nan
    return winner, losses, weights, np.nan, selected_loss


def _scenario_sample(
    plan_date: pd.Timestamp,
    point_kw: np.ndarray,
    pcs_norm_kw: np.ndarray,
    effective_mask: np.ndarray,
    residual_history: np.ndarray,
    available_days: np.ndarray,
    scenario_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(available_days) == 0:
        return np.repeat(point_kw[None, :], scenario_count, axis=0), np.full(scenario_count, -1)
    rng = np.random.default_rng(seed + int(pd.Timestamp(plan_date).strftime("%Y%m%d")))
    source_days = rng.choice(available_days, size=scenario_count, replace=True)
    scenarios = point_kw[None, :] + pcs_norm_kw[None, :] * residual_history[source_days]
    scenarios = np.maximum(scenarios, 0.0)
    scenarios[:, ~effective_mask] = 0.0
    return scenarios, source_days


def _forecast_metrics(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    forecasts: dict[str, np.ndarray],
    result_mask: np.ndarray,
    hard_mask: np.ndarray,
    mixed_slots: np.ndarray,
    threshold_kw: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_actual = actual[result_mask]
    selected_hard = np.broadcast_to(hard_mask, selected_actual.shape)
    for name, values in forecasts.items():
        prediction = values[result_mask]
        error = prediction - selected_actual
        hard_error = error[selected_hard]
        daily_actual_energy = selected_actual.sum(axis=1) / 6.0
        daily_forecast_energy = prediction.sum(axis=1) / 6.0
        boundary = mixed_slots[result_mask] & (selected_actual <= threshold_kw)
        false_positive = (prediction > threshold_kw) & boundary
        rows.append(
            {
                "model": name,
                "evaluation_start": dates[result_mask].min(),
                "evaluation_end": dates[result_mask].max(),
                "mae_hard_daylight_kw": float(np.mean(np.abs(hard_error))),
                "rmse_hard_daylight_kw": float(np.sqrt(np.mean(hard_error**2))),
                "mae_all_day_kw": float(np.mean(np.abs(error))),
                "daily_energy_mae_kwh": float(np.mean(np.abs(daily_forecast_energy - daily_actual_energy))),
                "bias_hard_daylight_kw": float(np.mean(hard_error)),
                "boundary_false_positive_ratio": float(false_positive.sum() / max(boundary.sum(), 1)),
            }
        )
    metrics = pd.DataFrame(rows)
    baseline_rows = metrics[metrics["model"].isin(["a1_7", "a1_30"])]
    baseline_mae = float(baseline_rows["mae_hard_daylight_kw"].min())
    metrics["skill_vs_best_a1"] = 1.0 - metrics["mae_hard_daylight_kw"] / baseline_mae
    return metrics


def build_pv_day_ahead_forecasts(
    dispatch: pd.DataFrame,
    config: PVDayAheadConfig | None = None,
    key_dates: list[str] | None = None,
) -> PVForecastResult:
    """Build causal hourly, 10-minute, residual, and probabilistic forecasts.

    The envelope uses only completed plan dates strictly before each 00:00
    issue timestamp.  Attachment 1 is not used anywhere in this pipeline; on
    days without any completed history (only the very first day) a zero shadow
    forecast is emitted and flagged via ``cold_start_shadow``.
    """
    config = config or PVDayAheadConfig()
    ordered = dispatch.sort_values(["plan_date", "slot_index"]).copy()
    dates = pd.DatetimeIndex(ordered["plan_date"].drop_duplicates()).sort_values()
    counts = ordered.groupby("plan_date").size().reindex(dates)
    if not counts.eq(144).all():
        raise ValueError("Every plan date must contain exactly 144 slots")
    actual_10min = (
        ordered.pivot(index="plan_date", columns="slot_index", values="pv_actual_kw")
        .reindex(index=dates, columns=np.arange(1, 145))
        .to_numpy(float)
    )
    actual_hourly = actual_10min.reshape(len(dates), 24, 6).mean(axis=2)

    grid = np.linspace(0.0, 1.0, config.solar_grid_points)
    first, last, peaks, valid_geometry, daily_shapes = _daily_geometry(
        actual_10min, config.positive_threshold_kw, grid
    )

    days = len(dates)
    slot_minutes = np.arange(1, 145, dtype=float) * 10.0
    clock_minutes = np.where(slot_minutes == 1440.0, 0.0, slot_minutes)
    hard_mask = (
        (clock_minutes >= config.hard_daylight_start_minute)
        & (clock_minutes <= config.hard_daylight_end_minute)
    )
    annual_angle = 2.0 * np.pi * dates.dayofyear.to_numpy(float) / 365.25
    annual_sin, annual_cos = np.sin(annual_angle), np.cos(annual_angle)

    amplitude = np.zeros(days)
    sunrise_hat = np.zeros(days)
    sunset_hat = np.zeros(days)
    pcs_raw_10 = np.zeros((days, 144))
    pcs_norm_10 = np.zeros((days, 144))
    pcs_raw_hour = np.zeros((days, 24))
    pcs_norm_hour = np.zeros((days, 24))
    templates = np.zeros((days, 144))
    effective_masks = np.zeros((days, 144), dtype=bool)
    solar_hour = np.zeros((days, 24))
    shape_values = np.zeros((days, config.solar_grid_points))
    kpv_actual = np.full((days, 24), np.nan)
    a2_base = np.zeros((days, 24))
    a2_residual = np.full((days, 24), np.nan)
    boundary_mixed = np.zeros((days, 24), dtype=bool)
    cold_start_shadow = np.zeros(days, dtype=bool)
    a3_fallback_hours = np.zeros(days, dtype=int)
    a4_fallback_hours = {neighbors: np.zeros(days, dtype=int) for neighbors in config.knn_neighbors}

    candidate_hour = {name: np.zeros((days, 24)) for name in CANDIDATE_NAMES}
    candidate_10 = {name: np.zeros((days, 144)) for name in CANDIDATE_NAMES}
    daily_losses = np.full((days, len(CANDIDATE_NAMES)), np.nan)
    tau_a2_losses = np.full((days, len(config.boundary_thresholds)), np.nan)
    tau_a3_losses = np.full((days, len(config.boundary_thresholds)), np.nan)
    selected_hour = np.zeros((days, 24))
    selected_10 = np.zeros((days, 144))
    selected_policy = np.full(days, "a1_7", dtype=object)
    selected_kpv = np.zeros((days, 24))
    model_losses = {name: np.full(days, np.nan) for name in CANDIDATE_NAMES}
    model_skills = {name: np.full(days, np.nan) for name in CANDIDATE_NAMES}
    model_weights = {name: np.zeros(days) for name in CANDIDATE_NAMES}
    ensemble_loss = np.full(days, np.nan)
    selected_loss = np.full(days, np.nan)
    tau_a2 = np.full(days, 0.15)
    tau_a3 = np.full(days, 0.15)
    p10 = np.zeros((days, 144))
    p50 = np.zeros((days, 144))
    p90 = np.zeros((days, 144))
    scenario_history_days = np.zeros(days, dtype=int)
    residual_kpv = np.zeros((days, 144))
    residual_available = np.zeros((days, 144), dtype=bool)

    for day, plan_date in enumerate(dates):
        boundary_valid = valid_geometry[max(0, day - config.boundary_window_days) : day]
        historical_first = first[max(0, day - config.boundary_window_days) : day][boundary_valid]
        historical_last = last[max(0, day - config.boundary_window_days) : day][boundary_valid]
        historical_peaks = peaks[max(0, day - config.amplitude_window_days) : day]
        prior_shapes = daily_shapes[:day][valid_geometry[:day]]
        if len(historical_first) == 0:
            # No completed history at all: zero shadow forecast, flagged.
            cold_start_shadow[day] = True
            sunrise_hat[day] = 0.0
            sunset_hat[day] = 0.0
            amplitude[day] = 0.0
            shape_values[day] = 0.0
        else:
            sunrise_hat[day] = float(np.median(historical_first))
            sunset_hat[day] = float(np.median(historical_last))
            amplitude[day] = float(np.quantile(historical_peaks, config.amplitude_quantile))
            empirical = np.quantile(prior_shapes, config.shape_quantile, axis=0)
            shape_values[day] = _smooth_shape(empirical, config)

        duration = max(sunset_hat[day] - sunrise_hat[day], 1.0)
        solar_position = (slot_minutes - sunrise_hat[day]) / duration
        inside_solar_day = (solar_position >= 0.0) & (solar_position <= 1.0)
        raw_shape = np.zeros(144)
        raw_shape[inside_solar_day] = np.interp(solar_position[inside_solar_day], grid, shape_values[day])
        soft_mask = (
            (clock_minutes >= sunrise_hat[day] - config.boundary_buffer_minutes)
            & (clock_minutes <= sunset_hat[day] + config.boundary_buffer_minutes)
        )
        effective = hard_mask & soft_mask
        norm_shape = np.where(effective, np.maximum(raw_shape, config.normalization_floor_ratio), 0.0)
        pcs_raw_10[day] = amplitude[day] * raw_shape
        pcs_norm_10[day] = amplitude[day] * norm_shape
        effective_masks[day] = effective
        pcs_raw_hour[day] = pcs_raw_10[day].reshape(24, 6).mean(axis=1)
        pcs_norm_hour[day] = pcs_norm_10[day].reshape(24, 6).mean(axis=1)
        solar_hour[day] = solar_position.reshape(24, 6).mean(axis=1)

        for hour in range(24):
            block = slice(hour * 6, (hour + 1) * 6)
            basis = raw_shape[block].copy()
            basis[~effective[block]] = 0.0
            if basis.sum() > 0:
                templates[day, block] = 6.0 * basis / basis.sum()
            elif effective[block].any():
                templates[day, block] = 6.0 * effective[block].astype(float) / effective[block].sum()

        history_7 = actual_hourly[max(0, day - 7) : day]
        history_30 = actual_hourly[max(0, day - 30) : day]
        a1_7 = np.zeros(24) if day == 0 else history_7.mean(axis=0)
        a1_30 = np.zeros(24) if day == 0 else history_30.mean(axis=0)

        k_start = max(0, day - 30)
        k_history = kpv_actual[k_start:day]
        k_mean = np.zeros(24)
        for hour in range(24):
            values = k_history[:, hour]
            values = values[np.isfinite(values)]
            k_mean[hour] = float(values.mean()) if len(values) else 0.0
        a2_base[day] = pcs_norm_hour[day] * k_mean
        raw_a2 = a2_base[day].copy()
        raw_a3 = a2_base[day].copy()
        a4_predictions = {neighbors: a2_base[day].copy() for neighbors in config.knn_neighbors}

        for hour in range(24):
            # A2: AR(1) correction on the rolling-origin power residuals.
            transition_start = max(1, day - config.ar_window_days)
            transitions = np.arange(transition_start, day)
            ar_valid = np.isfinite(a2_residual[transitions, hour]) & np.isfinite(
                a2_residual[transitions - 1, hour]
            )
            ar_days = transitions[ar_valid]
            if len(ar_days) >= config.ar_min_samples:
                raw_a2[hour] += _ar1_predict(
                    a2_residual[ar_days - 1, hour],
                    a2_residual[ar_days, hour],
                    a2_residual[day - 1, hour],
                )

            # A3: ridge KPV regression; on insufficient samples fall back to
            # the FULL A2 (including its AR(1) correction).
            k_valid = np.isfinite(kpv_actual[transitions, hour]) & np.isfinite(
                kpv_actual[transitions - 1, hour]
            )
            k_days = transitions[k_valid]
            if len(k_days) >= config.ar_min_samples and np.isfinite(kpv_actual[day - 1, hour]):
                x = np.column_stack(
                    [
                        kpv_actual[k_days - 1, hour],
                        annual_sin[k_days],
                        annual_cos[k_days],
                    ]
                )
                target = np.array([kpv_actual[day - 1, hour], annual_sin[day], annual_cos[day]])
                k_prediction = _ridge_predict(
                    x, kpv_actual[k_days, hour], target, config.ridge_alpha
                )
                raw_a3[hour] = pcs_norm_hour[day, hour] * k_prediction
            else:
                raw_a3[hour] = raw_a2[hour]
                a3_fallback_hours[day] += 1

            # A4: distance-weighted kNN on KPV features; on insufficient
            # samples fall back to the FULL A2 (which itself degrades to
            # a2_base when the AR(1) term lacks samples, and to zero outside
            # active hours).
            training_start = max(config.knn_lag_days, day - config.knn_window_days)
            features: list[np.ndarray] = []
            targets: list[float] = []
            for sample_day in range(training_start, day):
                lag_values = kpv_actual[
                    sample_day - config.knn_lag_days : sample_day, hour
                ][::-1]
                feature = np.r_[
                    lag_values,
                    solar_hour[sample_day, hour],
                    annual_sin[sample_day],
                    annual_cos[sample_day],
                ]
                if np.isfinite(feature).all() and np.isfinite(kpv_actual[sample_day, hour]):
                    features.append(feature)
                    targets.append(float(kpv_actual[sample_day, hour]))
            target_feature = np.r_[
                kpv_actual[max(0, day - config.knn_lag_days) : day, hour][::-1],
                solar_hour[day, hour],
                annual_sin[day],
                annual_cos[day],
            ]
            if (
                day >= config.knn_lag_days
                and len(features) >= max(config.knn_neighbors)
                and len(target_feature) == config.knn_lag_days + 3
                and np.isfinite(target_feature).all()
            ):
                feature_matrix = np.asarray(features)
                target_vector = np.asarray(targets)
                for neighbors in config.knn_neighbors:
                    k_prediction = _knn_predict(
                        feature_matrix, target_vector, target_feature, neighbors
                    )
                    a4_predictions[neighbors][hour] = pcs_norm_hour[day, hour] * k_prediction
            else:
                for neighbors in config.knn_neighbors:
                    a4_predictions[neighbors][hour] = raw_a2[hour]
                    a4_fallback_hours[neighbors][day] += 1

        if day:
            recent = actual_hourly[max(0, day - config.boundary_window_days) : day]
            zero_share = np.mean(recent <= config.positive_threshold_kw, axis=0)
            boundary_mixed[day] = (zero_share > 0.0) & (zero_share < 1.0)
        active_hours = templates[day].reshape(24, 6).sum(axis=1) > 0.0
        a1_7[~active_hours] = 0.0
        a1_30[~active_hours] = 0.0
        raw_a2[~active_hours] = 0.0
        raw_a3[~active_hours] = 0.0
        for values in a4_predictions.values():
            values[~active_hours] = 0.0

        a2_tau_hours: list[np.ndarray] = []
        a3_tau_hours: list[np.ndarray] = []
        a2_tau_ten: list[np.ndarray] = []
        a3_tau_ten: list[np.ndarray] = []
        for threshold in config.boundary_thresholds:
            gated_a2 = np.maximum(raw_a2.copy(), 0.0)
            gated_a3 = np.maximum(raw_a3.copy(), 0.0)
            valid_pcs = pcs_norm_hour[day] > 0.0
            a2_kpv = np.divide(gated_a2, pcs_norm_hour[day], out=np.zeros(24), where=valid_pcs)
            a3_kpv = np.divide(gated_a3, pcs_norm_hour[day], out=np.zeros(24), where=valid_pcs)
            gated_a2[boundary_mixed[day] & (a2_kpv < threshold)] = 0.0
            gated_a3[boundary_mixed[day] & (a3_kpv < threshold)] = 0.0
            a2_tau_hours.append(gated_a2)
            a3_tau_hours.append(gated_a3)
            a2_tau_ten.append(restore_hourly_to_10min(gated_a2, templates[day]))
            a3_tau_ten.append(restore_hourly_to_10min(gated_a3, templates[day]))

        selected_a2_tau = _choose_threshold(tau_a2_losses, day, config)
        selected_a3_tau = _choose_threshold(tau_a3_losses, day, config)
        tau_a2[day] = config.boundary_thresholds[selected_a2_tau]
        tau_a3[day] = config.boundary_thresholds[selected_a3_tau]

        current_hour = {
            "a1_7": np.maximum(a1_7, 0.0),
            "a1_30": np.maximum(a1_30, 0.0),
            "a2": a2_tau_hours[selected_a2_tau],
            "a3": a3_tau_hours[selected_a3_tau],
            **{
                f"a4_k{neighbors}": np.maximum(a4_predictions[neighbors], 0.0)
                for neighbors in config.knn_neighbors
            },
        }
        for name in CANDIDATE_NAMES:
            candidate_hour[name][day] = current_hour[name]
            if name == "a2":
                candidate_10[name][day] = a2_tau_ten[selected_a2_tau]
            elif name == "a3":
                candidate_10[name][day] = a3_tau_ten[selected_a3_tau]
            else:
                candidate_10[name][day] = restore_hourly_to_10min(
                    current_hour[name], templates[day]
                )

        policy, losses, weights, current_ensemble_loss, current_selected_loss = _select_policy(
            day,
            candidate_10,
            daily_losses,
            actual_10min,
            hard_mask,
            amplitude,
            config,
        )
        selected_policy[day] = policy
        ensemble_loss[day] = current_ensemble_loss
        selected_loss[day] = current_selected_loss
        baseline_values = [losses["a1_7"], losses["a1_30"]]
        finite_baselines = [value for value in baseline_values if np.isfinite(value)]
        baseline_loss = min(finite_baselines) if finite_baselines else np.nan
        for name in CANDIDATE_NAMES:
            model_losses[name][day] = losses[name]
            model_skills[name][day] = (
                1.0 - losses[name] / baseline_loss
                if np.isfinite(losses[name]) and np.isfinite(baseline_loss) and baseline_loss > 0
                else np.nan
            )
            model_weights[name][day] = weights[name]
        selected_hour[day] = sum(weights[name] * candidate_hour[name][day] for name in CANDIDATE_NAMES)
        selected_10[day] = sum(weights[name] * candidate_10[name][day] for name in CANDIDATE_NAMES)
        selected_kpv[day] = np.divide(
            selected_hour[day],
            pcs_norm_hour[day],
            out=np.zeros(24),
            where=pcs_norm_hour[day] > 0,
        )

        scenario_start = max(0, day - config.scenario_window_days)
        available_days = np.arange(scenario_start, day, dtype=int)
        scenario_history_days[day] = len(available_days)
        scenarios, _ = _scenario_sample(
            plan_date,
            selected_10[day],
            pcs_norm_10[day],
            effective_masks[day],
            residual_kpv,
            available_days,
            config.scenario_count,
            config.scenario_seed,
        )
        p10[day], p50[day], p90[day] = np.quantile(scenarios, [0.10, 0.50, 0.90], axis=0)

        kpv_actual[day] = np.divide(
            actual_hourly[day],
            pcs_norm_hour[day],
            out=np.full(24, np.nan),
            where=pcs_norm_hour[day] > 0,
        )
        if day == 0:
            # Cold-start shadow day: no usable history, so leave the A2 AR(1)
            # residual series and daily losses undefined instead of feeding
            # a degenerate zero-envelope residual into later training windows.
            a2_residual[day] = np.nan
            daily_losses[day, :] = np.nan
            tau_a2_losses[day, :] = np.nan
            tau_a3_losses[day, :] = np.nan
        else:
            a2_residual[day] = actual_hourly[day] - a2_base[day]
        residual_available[day] = pcs_norm_10[day] > 0
        residual_kpv[day] = np.divide(
            actual_10min[day] - selected_10[day],
            pcs_norm_10[day],
            out=np.zeros(144),
            where=residual_available[day],
        )
        if day == 0:
            continue
        for model_index, name in enumerate(CANDIDATE_NAMES):
            daily_losses[day, model_index] = _normalized_day_loss(
                candidate_10[name][day], actual_10min[day], hard_mask, amplitude[day]
            )
        for threshold_index in range(len(config.boundary_thresholds)):
            tau_a2_losses[day, threshold_index] = _normalized_day_loss(
                a2_tau_ten[threshold_index], actual_10min[day], hard_mask, amplitude[day]
            )
            tau_a3_losses[day, threshold_index] = _normalized_day_loss(
                a3_tau_ten[threshold_index], actual_10min[day], hard_mask, amplitude[day]
            )

    plan_values = np.repeat(dates.to_numpy(), 144)
    slot_index = np.tile(np.arange(1, 145), days)
    issue_values = plan_values.copy()
    target_values = plan_values + pd.to_timedelta(slot_index * 10, unit="m").to_numpy()
    hour_block = np.tile(np.repeat(np.arange(1, 25), 6), days)
    result_start = pd.Timestamp("2025-02-01")
    result_end = pd.Timestamp("2025-12-31")
    result_days = (dates >= result_start) & (dates <= result_end)
    repeated_policy = np.repeat(selected_policy, 144)
    mixed_slots = np.repeat(boundary_mixed, 6, axis=1)
    residual_frame = pd.DataFrame(
        {
            "plan_date": plan_values,
            "slot_index": slot_index,
            "issue_ts": issue_values,
            "target_ts": target_values,
            "pv_actual_kw": actual_10min.reshape(-1),
            "pv_point_kw": selected_10.reshape(-1),
            "residual_kw": (actual_10min - selected_10).reshape(-1),
            "residual_kpv": residual_kpv.reshape(-1),
            "residual_available": residual_available.reshape(-1),
            "boundary_class": np.where(
                mixed_slots.reshape(-1),
                "mixed",
                np.where(effective_masks.reshape(-1), "core", "night"),
            ),
            "annual_sin": np.repeat(annual_sin, 144),
            "annual_cos": np.repeat(annual_cos, 144),
            "selected_model": repeated_policy,
            "is_result_period": np.repeat(result_days, 144),
        }
    )
    ten_minute = pd.DataFrame(
        {
            "plan_date": plan_values,
            "slot_index": slot_index,
            "hour_block": hour_block,
            "issue_ts": issue_values,
            "target_ts": target_values,
            "pv_point_kw": selected_10.reshape(-1),
            "pv_p10_kw": p10.reshape(-1),
            "pv_p50_kw": p50.reshape(-1),
            "pv_p90_kw": p90.reshape(-1),
            "pcs_raw_kw": pcs_raw_10.reshape(-1),
            "pcs_norm_kw": pcs_norm_10.reshape(-1),
            "hard_daylight_mask": np.tile(hard_mask, days),
            "effective_daylight_mask": effective_masks.reshape(-1),
            "boundary_mixed": mixed_slots.reshape(-1),
            "selected_model": repeated_policy,
            "scenario_history_days": np.repeat(scenario_history_days, 144),
            "scenario_seed": np.repeat(
                config.scenario_seed + dates.strftime("%Y%m%d").astype(int).to_numpy(), 144
            ),
            "training_end": np.repeat(
                np.r_[np.datetime64("NaT"), dates[:-1].to_numpy()], 144
            ),
            "cold_start_shadow": np.repeat(cold_start_shadow, 144),
            "is_result_period": np.repeat(result_days, 144),
        }
    )

    hourly_plan = np.repeat(dates.to_numpy(), 24)
    hourly_block = np.tile(np.arange(1, 25), days)
    block_start = hourly_plan + pd.to_timedelta((hourly_block - 1) * 60, unit="m").to_numpy()
    training_starts = np.array(
        [np.datetime64("NaT") if day == 0 else dates[max(0, day - config.knn_window_days)].to_datetime64() for day in range(days)]
    )
    training_ends = np.r_[np.datetime64("NaT"), dates[:-1].to_numpy()]
    hourly_data: dict[str, Any] = {
        "plan_date": hourly_plan,
        "hour_block": hourly_block,
        "issue_ts": hourly_plan,
        "block_start_ts": block_start,
        "block_end_ts": block_start + pd.to_timedelta(60, unit="m"),
        "training_start": np.repeat(training_starts, 24),
        "training_end": np.repeat(training_ends, 24),
        "history_days": np.repeat(np.arange(days), 24),
        "sunrise_hat_minute": np.repeat(sunrise_hat, 24),
        "sunset_hat_minute": np.repeat(sunset_hat, 24),
        "amplitude_kw": np.repeat(amplitude, 24),
        "annual_sin": np.repeat(annual_sin, 24),
        "annual_cos": np.repeat(annual_cos, 24),
        "solar_time": solar_hour.reshape(-1),
        "pcs_raw_hour_kw": pcs_raw_hour.reshape(-1),
        "pcs_norm_hour_kw": pcs_norm_hour.reshape(-1),
        "boundary_mixed": boundary_mixed.reshape(-1),
        "tau_a2": np.repeat(tau_a2, 24),
        "tau_a3": np.repeat(tau_a3, 24),
        "cold_start_shadow": np.repeat(cold_start_shadow, 24),
        "a3_a2_fallback_hours": np.repeat(a3_fallback_hours, 24),
    }
    for neighbors in config.knn_neighbors:
        hourly_data[f"a4_a2_fallback_hours_k{neighbors}"] = np.repeat(
            a4_fallback_hours[neighbors], 24
        )
    for name in CANDIDATE_NAMES:
        hourly_data[f"pv_{name}_kw"] = candidate_hour[name].reshape(-1)
        hourly_data[f"loss_{name}"] = np.repeat(model_losses[name], 24)
        hourly_data[f"skill_{name}"] = np.repeat(model_skills[name], 24)
        hourly_data[f"weight_{name}"] = np.repeat(model_weights[name], 24)
    hourly_data.update(
        {
            "ensemble_loss": np.repeat(ensemble_loss, 24),
            "selected_loss": np.repeat(selected_loss, 24),
            "selected_model": np.repeat(selected_policy, 24),
            "selected_kpv": selected_kpv.reshape(-1),
            "pv_selected_hour_kw": selected_hour.reshape(-1),
            "is_result_period": np.repeat(result_days, 24),
        }
    )
    hourly = pd.DataFrame(hourly_data)

    metric_forecasts = {name: candidate_10[name] for name in CANDIDATE_NAMES}
    metric_forecasts["selected"] = selected_10
    metrics = _forecast_metrics(
        dates,
        actual_10min,
        metric_forecasts,
        result_days,
        hard_mask,
        mixed_slots,
        config.positive_threshold_kw,
    )

    key_dates = key_dates or ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
    key_rows: list[dict[str, Any]] = []
    for date_text in key_dates:
        date = pd.Timestamp(date_text)
        if date not in dates:
            continue
        day = int(dates.get_loc(date))
        error = selected_10[day] - actual_10min[day]
        key_rows.append(
            {
                "plan_date": date,
                "selected_model": selected_policy[day],
                "mae_hard_daylight_kw": float(np.mean(np.abs(error[hard_mask]))),
                "rmse_hard_daylight_kw": float(np.sqrt(np.mean(error[hard_mask] ** 2))),
                "mae_all_day_kw": float(np.mean(np.abs(error))),
                "actual_energy_kwh": float(actual_10min[day].sum() / 6.0),
                "forecast_energy_kwh": float(selected_10[day].sum() / 6.0),
                "p10_energy_kwh": float(p10[day].sum() / 6.0),
                "p90_energy_kwh": float(p90[day].sum() / 6.0),
            }
        )
    key_frame = pd.DataFrame(key_rows)
    formal_actual = actual_10min[result_days]
    formal_p10 = p10[result_days]
    formal_p90 = p90[result_days]
    formal_hard = np.broadcast_to(hard_mask, formal_actual.shape)
    coverage = (formal_actual >= formal_p10) & (formal_actual <= formal_p90)
    selection_counts = pd.Series(selected_policy[result_days]).value_counts().sort_index()
    summary = {
        "calendar_start": dates.min().date().isoformat(),
        "calendar_end": dates.max().date().isoformat(),
        "result_start": result_start.date().isoformat(),
        "result_end": result_end.date().isoformat(),
        "hourly_rows": int(len(hourly)),
        "ten_minute_rows": int(len(ten_minute)),
        "residual_rows": int(len(residual_frame)),
        "hard_daylight_p10_p90_coverage": float(coverage[formal_hard].mean()),
        "selection_policy": "strict_winner_take_all_default_a1_7",
        "selection_counts": {str(key): int(value) for key, value in selection_counts.items()},
        "cold_start_shadow_days": int(cold_start_shadow.sum()),
        "a3_a2_fallback_hours": int(a3_fallback_hours.sum()),
        "a4_a2_fallback_hours": {
            f"k{neighbors}": int(a4_fallback_hours[neighbors].sum())
            for neighbors in config.knn_neighbors
        },
        "no_upper_clipping": True,
        "uses_attachment_1": False,
        "uses_attachment_3": False,
        "uses_external_weather": False,
    }
    return PVForecastResult(hourly, ten_minute, residual_frame, metrics, key_frame, summary)


def build_pv_scenarios(
    plan_date: str | pd.Timestamp,
    ten_minute: pd.DataFrame,
    residuals: pd.DataFrame,
    scenario_count: int = 50,
    seed: int = 20_250_911,
    window_days: int = 60,
) -> pd.DataFrame:
    """Generate reproducible whole-day residual-bootstrap PV scenarios."""
    date = pd.Timestamp(plan_date).normalize()
    target = ten_minute[ten_minute["plan_date"].eq(date)].sort_values("slot_index")
    if len(target) != 144:
        raise ValueError(f"No complete 144-slot forecast found for {date.date()}")
    prior = residuals[residuals["plan_date"].lt(date)].copy()
    prior_dates = pd.DatetimeIndex(prior["plan_date"].drop_duplicates()).sort_values()
    prior_dates = prior_dates[-window_days:]
    if len(prior_dates):
        matrix = (
            prior[prior["plan_date"].isin(prior_dates)]
            .pivot(index="plan_date", columns="slot_index", values="residual_kpv")
            .reindex(index=prior_dates, columns=np.arange(1, 145))
            .to_numpy(float)
        )
        available = np.arange(len(prior_dates), dtype=int)
    else:
        matrix = np.zeros((0, 144))
        available = np.array([], dtype=int)
    scenarios, source_indices = _scenario_sample(
        date,
        target["pv_point_kw"].to_numpy(float),
        target["pcs_norm_kw"].to_numpy(float),
        target["effective_daylight_mask"].to_numpy(bool),
        matrix,
        available,
        scenario_count,
        seed,
    )
    source_dates = [pd.NaT if index < 0 else prior_dates[index] for index in source_indices]
    return pd.DataFrame(
        {
            "plan_date": np.repeat(date, scenario_count * 144),
            "scenario_id": np.repeat(np.arange(1, scenario_count + 1), 144),
            "probability": np.repeat(1.0 / scenario_count, scenario_count * 144),
            "slot_index": np.tile(np.arange(1, 145), scenario_count),
            "target_ts": np.tile(target["target_ts"].to_numpy(), scenario_count),
            "pv_scenario_kw": scenarios.reshape(-1),
            "source_residual_date": np.repeat(source_dates, 144),
            "seed": np.repeat(seed + int(date.strftime("%Y%m%d")), scenario_count * 144),
        }
    )


def validate_pv_day_ahead(
    result: PVForecastResult,
    expected_hourly_rows: int = 8_760,
    expected_ten_minute_rows: int = 52_560,
    expected_result_rows: int = 48_096,
) -> pd.DataFrame:
    """Return machine-readable acceptance checks for the forecast artifacts."""
    hourly = result.hourly
    ten = result.ten_minute
    residuals = result.residuals
    checks: list[dict[str, Any]] = []

    def add(check_id: str, condition: bool, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "check_id": check_id,
                "status": "PASS" if bool(condition) else "FAIL",
                "observed": str(observed),
                "expected": str(expected),
            }
        )

    add("hourly_row_count", len(hourly) == expected_hourly_rows, len(hourly), expected_hourly_rows)
    add("ten_minute_row_count", len(ten) == expected_ten_minute_rows, len(ten), expected_ten_minute_rows)
    add("residual_row_count", len(residuals) == expected_ten_minute_rows, len(residuals), expected_ten_minute_rows)
    add("result_period_row_count", int(ten["is_result_period"].sum()) == expected_result_rows, int(ten["is_result_period"].sum()), expected_result_rows)
    hourly_counts = hourly.groupby("plan_date").size()
    ten_counts = ten.groupby("plan_date").size()
    add("daily_structure", hourly_counts.eq(24).all() and ten_counts.eq(144).all(), f"hour={hourly_counts.min()}-{hourly_counts.max()}, ten={ten_counts.min()}-{ten_counts.max()}", "24 and 144")
    trained = ten[ten["training_end"].notna()]
    causal = (trained["training_end"] < trained["issue_ts"]).all()
    add("causality", causal, causal, True)
    nonnegative = ten[["pv_point_kw", "pv_p10_kw", "pv_p50_kw", "pv_p90_kw"]].ge(0).all().all()
    add("nonnegative", nonnegative, nonnegative, True)
    numeric_columns = [
        "pv_point_kw",
        "pv_p10_kw",
        "pv_p50_kw",
        "pv_p90_kw",
        "pcs_raw_kw",
        "pcs_norm_kw",
    ]
    finite = np.isfinite(ten[numeric_columns].to_numpy(dtype=float)).all()
    add("forecast_numeric_finite", finite, finite, True)
    quantiles = (ten["pv_p10_kw"] <= ten["pv_p50_kw"]).all() and (ten["pv_p50_kw"] <= ten["pv_p90_kw"]).all()
    add("quantile_order", quantiles, quantiles, True)
    zero_mask = ~ten["effective_daylight_mask"]
    masked_zero = ten.loc[zero_mask, ["pv_point_kw", "pv_p10_kw", "pv_p50_kw", "pv_p90_kw"]].eq(0).all().all()
    add("daylight_mask_zero", masked_zero, masked_zero, True)
    restored = ten.groupby(["plan_date", "hour_block"], sort=True)["pv_point_kw"].mean().to_numpy()
    hourly_values = hourly.sort_values(["plan_date", "hour_block"])["pv_selected_hour_kw"].to_numpy()
    max_difference = float(np.max(np.abs(restored - hourly_values)))
    add("hourly_restoration_conservation", max_difference <= 1e-8, max_difference, "<=1e-8 kW")
    above_raw_count = int((ten["pv_point_kw"] > ten["pcs_raw_kw"] + 1e-9).sum())
    add("no_upper_envelope_clipping_observed", above_raw_count > 0, above_raw_count, ">0 rows")
    add("hourly_primary_key", not hourly.duplicated(["plan_date", "hour_block"]).any(), not hourly.duplicated(["plan_date", "hour_block"]).any(), True)
    add("ten_minute_primary_key", not ten.duplicated(["plan_date", "slot_index"]).any(), not ten.duplicated(["plan_date", "slot_index"]).any(), True)
    add("strict_selection_no_ensemble", not hourly["selected_model"].eq("ensemble").any(), int(hourly["selected_model"].eq("ensemble").sum()), 0)
    add("cold_start_shadow_present", "cold_start_shadow" in ten.columns, "cold_start_shadow" in ten.columns, True)
    add("cold_start_shadow_only_first_day", int(ten["cold_start_shadow"].sum()) == 144, int(ten["cold_start_shadow"].sum()), 144)
    fallback_columns = [column for column in hourly.columns if "fallback_hours" in column]
    fallback_finite = bool(fallback_columns) and all(
        np.isfinite(hourly[column].to_numpy(dtype=float)).all() for column in fallback_columns
    )
    add("fallback_counters_finite", fallback_finite, fallback_finite, True)
    return pd.DataFrame(checks)
