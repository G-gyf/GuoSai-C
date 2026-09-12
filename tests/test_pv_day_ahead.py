"""Acceptance tests for the causal question 2 PV forecast pipeline."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.ingest import read_attachments
from src.forecasting.pv_day_ahead import (
    CANDIDATE_NAMES,
    PVDayAheadConfig,
    PVForecastResult,
    build_pv_day_ahead_forecasts,
    build_pv_scenarios,
    restore_hourly_to_10min,
    validate_pv_day_ahead,
)
from src.forecasting.question2_pv import run_pv_forecast_pipeline


ROOT = Path(__file__).resolve().parents[1]


class PVDayAheadAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        run_pv_forecast_pipeline(ROOT)
        cls.hourly = pd.read_parquet(ROOT / "data/processed/pv_day_ahead_hourly.parquet")
        cls.ten = pd.read_parquet(ROOT / "data/processed/pv_day_ahead_10min.parquet")
        cls.residuals = pd.read_parquet(ROOT / "data/processed/pv_day_ahead_residuals.parquet")

    def test_structure_causality_and_nonnegative_values(self) -> None:
        self.assertEqual(len(self.hourly), 8_760)
        self.assertEqual(len(self.ten), 52_560)
        self.assertEqual(len(self.residuals), 52_560)
        self.assertEqual(int(self.ten["is_result_period"].sum()), 48_096)
        self.assertTrue(self.hourly.groupby("plan_date").size().eq(24).all())
        self.assertTrue(self.ten.groupby("plan_date").size().eq(144).all())
        self.assertTrue(
            (
                self.hourly["block_start_ts"]
                == self.hourly["plan_date"]
                + pd.to_timedelta((self.hourly["hour_block"] - 1) * 60, unit="m")
            ).all()
        )
        self.assertTrue(
            (self.hourly["block_end_ts"] == self.hourly["block_start_ts"] + pd.Timedelta(hours=1)).all()
        )
        trained = self.ten[self.ten["training_end"].notna()]
        self.assertTrue((trained["training_end"] < trained["issue_ts"]).all())
        self.assertTrue(
            self.ten[["pv_point_kw", "pv_p10_kw", "pv_p50_kw", "pv_p90_kw"]]
            .ge(0)
            .all()
            .all()
        )

    def test_hourly_restoration_is_exact_and_masks_are_zero(self) -> None:
        restored = self.ten.groupby(["plan_date", "hour_block"])["pv_point_kw"].mean()
        expected = self.hourly.set_index(["plan_date", "hour_block"])["pv_selected_hour_kw"]
        self.assertLessEqual(float((restored - expected).abs().max()), 1e-8)
        masked = self.ten[~self.ten["effective_daylight_mask"]]
        self.assertTrue(masked[["pv_point_kw", "pv_p10_kw", "pv_p50_kw", "pv_p90_kw"]].eq(0).all().all())

    def test_no_upper_envelope_clipping(self) -> None:
        self.assertTrue((self.ten["pv_point_kw"] > self.ten["pcs_raw_kw"] + 1e-9).any())
        hourly = np.r_[200.0, np.zeros(23)]
        template = np.r_[np.ones(6), np.zeros(138)]
        restored = restore_hourly_to_10min(hourly, template)
        self.assertTrue(np.allclose(restored[:6], 200.0))

    def test_dynamic_selection_only_uses_positive_skill_challengers(self) -> None:
        daily = self.hourly.drop_duplicates("plan_date")
        for name in CANDIDATE_NAMES[2:]:
            selected = daily[daily["selected_model"].eq(name)]
            self.assertTrue((selected[f"skill_{name}"] > 0).all())
        ensemble = daily[daily["selected_model"].eq("ensemble")]
        baseline = ensemble[["loss_a1_7", "loss_a1_30"]].min(axis=1)
        self.assertTrue((ensemble["ensemble_loss"] < baseline).all())

    def test_scenarios_are_reproducible_and_causal(self) -> None:
        first = build_pv_scenarios("2025-06-21", self.ten, self.residuals, scenario_count=5, seed=71)
        second = build_pv_scenarios("2025-06-21", self.ten, self.residuals, scenario_count=5, seed=71)
        different = build_pv_scenarios("2025-06-21", self.ten, self.residuals, scenario_count=5, seed=72)
        self.assertTrue(first.equals(second))
        self.assertFalse(np.array_equal(first["pv_scenario_kw"], different["pv_scenario_kw"]))
        self.assertEqual(len(first), 5 * 144)
        self.assertTrue((first["source_residual_date"] < pd.Timestamp("2025-06-21")).all())
        self.assertTrue(first["pv_scenario_kw"].ge(0).all())

    def test_future_actuals_do_not_change_earlier_forecasts(self) -> None:
        dispatch = pd.read_parquet(ROOT / "data/processed/dispatch_10min.parquet")
        subset = dispatch[dispatch["plan_date"].lt(pd.Timestamp("2025-03-02"))].copy()
        changed = subset.copy()
        cutoff = pd.Timestamp("2025-02-15")
        changed.loc[changed["plan_date"].ge(cutoff), "pv_actual_kw"] *= 0.05
        raw = read_attachments(ROOT)
        config = PVDayAheadConfig(scenario_count=5)
        original_result = build_pv_day_ahead_forecasts(subset, raw["attachment_1"], config=config)
        changed_result = build_pv_day_ahead_forecasts(changed, raw["attachment_1"], config=config)
        original_before = original_result.ten_minute[
            original_result.ten_minute["plan_date"].lt(cutoff)
        ][["pv_point_kw", "pv_p10_kw", "pv_p50_kw", "pv_p90_kw", "pcs_raw_kw", "pcs_norm_kw"]]
        changed_before = changed_result.ten_minute[
            changed_result.ten_minute["plan_date"].lt(cutoff)
        ][original_before.columns]
        self.assertTrue(np.allclose(original_before, changed_before, equal_nan=True))

    def test_saved_artifacts_pass_quality_checks(self) -> None:
        metrics = pd.read_csv(ROOT / "outputs/question2/analysis/forecast/model_metrics.csv")
        key_dates = pd.read_csv(ROOT / "outputs/question2/analysis/forecast/key_dates_metrics.csv")
        result = PVForecastResult(
            hourly=self.hourly,
            ten_minute=self.ten,
            residuals=self.residuals,
            metrics=metrics,
            key_dates=key_dates,
            summary={},
        )
        checks = validate_pv_day_ahead(result)
        self.assertTrue(checks["status"].eq("PASS").all())


if __name__ == "__main__":
    unittest.main()
