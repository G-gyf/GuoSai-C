"""Acceptance tests for the causal question 2 load forecast comparison."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.load_day_ahead import (
    DAY_TYPES,
    GRID_H,
    GRID_K,
    GRID_TAU,
    KERNEL_H,
    KERNEL_K,
    KERNEL_TAU,
    build_day_type_frame,
    build_load_day_ahead,
    gaussian_kernel_forecasts,
    rolling_type_quantile_residuals,
    verify_kernel_grid,
)
from src.forecasting.question2_load import run_load_pipeline

ROOT = Path(__file__).resolve().parents[1]


class LoadDayAheadAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        run_load_pipeline(ROOT)
        cls.ten = pd.read_parquet(ROOT / "data/processed/load_day_ahead_10min.parquet")
        cls.metrics = pd.read_csv(
            ROOT / "outputs/question2/analysis/load_forecast/model_metrics.csv"
        )
        cls.grid = pd.read_csv(
            ROOT / "outputs/question2/analysis/load_forecast/kernel_grid_verification.csv"
        )

    def test_day_type_mapping_rules(self) -> None:
        dates = pd.DatetimeIndex(
            pd.read_parquet(ROOT / "data/processed/dispatch_10min.parquet")[
                "plan_date"
            ].drop_duplicates()
        ).sort_values()
        frame = build_day_type_frame(dates)
        self.assertEqual(len(frame), 365)
        counts = frame["day_type"].value_counts().to_dict()
        self.assertEqual(
            counts,
            {"workday": 200, "sat_class": 102, "sunday": 49, "holiday": 14},
        )
        expected = {
            "2025-01-01": "holiday",
            "2025-01-26": "workday",  # makeup Sunday
            "2025-01-28": "holiday",
            "2025-02-01": "sat_class",  # 初四 Saturday
            "2025-02-03": "holiday",
            "2025-02-08": "workday",  # makeup Saturday
            "2025-04-04": "sat_class",  # 清明 Friday -> rule ②
            "2025-05-01": "holiday",
            "2025-05-05": "holiday",
            "2025-06-02": "holiday",
            "2025-10-01": "holiday",
            "2025-10-11": "workday",  # makeup Saturday
            "2025-12-31": "workday",
        }
        lookup = dict(zip(frame["plan_date"].dt.strftime("%Y-%m-%d"), frame["day_type"]))
        for date_text, day_type in expected.items():
            self.assertEqual(lookup[date_text], day_type, date_text)

    def test_structure_and_causality(self) -> None:
        self.assertEqual(len(self.ten), 365 * 144)
        self.assertEqual(int(self.ten["is_result_period"].sum()), 334 * 144)
        self.assertTrue(self.ten.groupby("plan_date").size().eq(144).all())
        trained = self.ten[self.ten["training_end"].notna()]
        self.assertTrue((trained["training_end"] < trained["issue_ts"]).all())
        columns = ["load_b0_kw", "load_b1_kw", "load_b2_kw", "load_gk_kw"]
        self.assertTrue(self.ten[columns].ge(0).all().all())
        self.assertTrue(np.isfinite(self.ten[columns].to_numpy(dtype=float)).all())

    def test_kernel_pool_is_same_type_and_prior_only(self) -> None:
        matrix = np.abs(np.random.default_rng(7).normal(5000, 500, (60, 144)))
        day_types = np.array(["a"] * 20 + ["b"] * 40)
        forecasts, pool_sizes = gaussian_kernel_forecasts(matrix, day_types)
        # Day 0 has no computable r7 and is excluded from every pool: pool
        # sizes grow as 0,0,1,2,...; days 21..59 (type b) grow to 39.
        self.assertTrue(np.array_equal(pool_sizes[:20], np.r_[0, np.arange(19)]))
        self.assertTrue((pool_sizes[21:] == np.arange(1, 40)).all())
        self.assertEqual(pool_sizes[20], 0)
        self.assertTrue(np.isnan(forecasts[20]).all())
        # Forecast of day 40 (type b) must stay within the range of type-b
        # rows (a convex combination cannot leave the per-slot range).
        lo = matrix[20:40].min(axis=0)
        hi = matrix[20:40].max(axis=0)
        self.assertTrue((forecasts[40] >= lo - 1e-9).all())
        self.assertTrue((forecasts[40] <= hi + 1e-9).all())
        # Never uses type-a rows: day 25's forecast is a combination of rows
        # 20..24 (all type b).
        lo = matrix[20:25].min(axis=0)
        hi = matrix[20:25].max(axis=0)
        self.assertTrue((forecasts[25] >= lo - 1e-9).all())
        self.assertTrue((forecasts[25] <= hi + 1e-9).all())

    def test_bounds_need_three_same_type_days(self) -> None:
        residuals = np.full((10, 6), 100.0)
        day_types = np.array(["a", "a", "b", "a", "b", "b", "b", "a", "a", "a"])
        q90 = rolling_type_quantile_residuals(residuals, day_types, 0.90, min_days=3)
        for d in range(6):
            self.assertTrue(np.isnan(q90[d]).all(), f"day {d}")
        self.assertTrue(np.isfinite(q90[6]).all())
        self.assertTrue((q90[6] == 100.0).all())
        for d in (7, 8, 9):
            self.assertTrue(np.isfinite(q90[d]).all(), f"day {d}")

    def test_grid_verification_contains_72_combos_and_docx_optimum(self) -> None:
        self.assertEqual(len(self.grid), 72)
        self.assertEqual(len(GRID_K) * len(GRID_H) * len(GRID_TAU), 72)
        optimum = self.grid[
            (self.grid["k"] == KERNEL_K)
            & (self.grid["h"] == KERNEL_H)
            & (self.grid["tau"] == KERNEL_TAU)
        ]
        self.assertEqual(len(optimum), 1)
        self.assertTrue(np.isfinite(optimum["mape_pct"].iloc[0]))
        self.assertTrue(self.grid["tuning_days"].eq(8).all())

    def test_future_actuals_do_not_change_earlier_forecasts(self) -> None:
        dispatch = pd.read_parquet(ROOT / "data/processed/dispatch_10min.parquet")
        subset = dispatch[dispatch["plan_date"].lt(pd.Timestamp("2025-03-02"))].copy()
        changed = subset.copy()
        cutoff = pd.Timestamp("2025-02-15")
        changed.loc[changed["plan_date"].ge(cutoff), "load_actual_kw"] *= 1.25
        representative = pd.to_numeric(
            subset.iloc[0:144]["load_actual_kw"], errors="raise"
        ).to_numpy(float)
        original = build_load_day_ahead(subset, representative)
        perturbed = build_load_day_ahead(changed, representative)
        before = original.ten_minute[original.ten_minute["plan_date"].lt(cutoff)]
        before_changed = perturbed.ten_minute[perturbed.ten_minute["plan_date"].lt(cutoff)]
        columns = ["load_b0_kw", "load_b1_kw", "load_b2_kw", "load_gk_kw"]
        self.assertTrue(
            np.allclose(
                before[columns].to_numpy(float),
                before_changed[columns].to_numpy(float),
                equal_nan=True,
            )
        )

    def test_saved_artifacts_pass_quality_checks(self) -> None:
        checks = pd.read_csv(
            ROOT / "outputs/question2/analysis/load_forecast/quality_checks.csv"
        )
        self.assertTrue(checks["status"].eq("PASS").all())
        formal = self.metrics[self.metrics["scope"].eq("formal")]
        self.assertEqual(set(formal["scheme"]), {"b0", "b1", "b2", "gk"})
        self.assertTrue(formal["mape_pct"].between(0, 100).all())


if __name__ == "__main__":
    unittest.main()
