"""Acceptance tests for the preprocessing outputs.

Run with: ``python -m unittest tests.test_data_pipeline -v``
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.run_preanalysis import run_pipeline


ROOT = Path(__file__).resolve().parents[1]


class DataPipelineAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        run_pipeline(ROOT)
        cls.dispatch = pd.read_parquet(ROOT / "data/processed/dispatch_10min.parquet")
        cls.hourly = pd.read_parquet(ROOT / "data/processed/pv_forecast_hourly.parquet")
        cls.ten_minute = pd.read_parquet(ROOT / "data/processed/pv_forecast_10min.parquet")
        cls.baseline = pd.read_parquet(ROOT / "data/processed/day_ahead_baseline_10min.parquet")

    def test_expected_row_counts(self) -> None:
        self.assertEqual(len(self.dispatch), 52_560)
        self.assertEqual(len(self.hourly), 35_040)
        self.assertEqual(len(self.ten_minute), 210_240)
        self.assertEqual(len(self.baseline), 52_560)
        self.assertEqual(int(self.dispatch["is_result_period"].sum()), 48_096)

    def test_plan_time_axis(self) -> None:
        counts = self.dispatch.groupby("plan_date").size()
        self.assertTrue(counts.eq(144).all())
        first = self.dispatch.groupby("plan_date").first()
        last = self.dispatch.groupby("plan_date").last()
        self.assertTrue((first["interval_start"] - first.index == pd.Timedelta(minutes=10)).all())
        self.assertTrue((last["interval_start"] - last.index == pd.Timedelta(days=1)).all())
        self.assertFalse(self.dispatch.duplicated(["plan_date", "slot_index"]).any())

    def test_units_and_net_load(self) -> None:
        self.assertTrue(np.allclose(self.dispatch["load_actual_kw"] / 6, self.dispatch["load_actual_kwh"]))
        self.assertTrue(np.allclose(self.dispatch["pv_actual_kw"] / 6, self.dispatch["pv_actual_kwh"]))
        self.assertTrue(np.allclose(self.dispatch["load_actual_kw"] - self.dispatch["pv_actual_kw"], self.dispatch["net_load_actual_kw"]))

    def test_hourly_targets_and_interpolation_anchors(self) -> None:
        expected_targets = self.hourly["issue_ts"] + pd.to_timedelta(self.hourly["horizon_hour"], unit="h")
        self.assertTrue((self.hourly["target_ts"] == expected_targets).all())
        anchors = self.ten_minute[self.ten_minute["lead_minutes"].mod(60).eq(0)]
        joined = anchors.merge(
            self.hourly[["issue_ts", "target_ts", "pv_forecast_kw"]],
            on=["issue_ts", "target_ts"], suffixes=("_10min", "_hourly"), validate="one_to_one"
        )
        self.assertTrue(np.allclose(joined["pv_forecast_kw_10min"], joined["pv_forecast_kw_hourly"]))
        self.assertTrue(self.ten_minute["pv_forecast_kw"].ge(0).all())

    def test_baseline_is_causal_and_has_documented_cold_start(self) -> None:
        trained = self.baseline[self.baseline["training_end"].notna()]
        self.assertTrue((trained["training_end"] < trained["issue_ts"]).all())
        first_day = self.baseline[self.baseline["plan_date"].eq(pd.Timestamp("2025-01-01"))]
        self.assertTrue(first_day["forecast_source"].eq("attachment_1_cold_start").all())
        self.assertTrue(self.baseline[self.baseline["plan_date"].ge(pd.Timestamp("2025-01-08"))]["history_days"].eq(7).all())

    def test_quality_report_and_key_dates(self) -> None:
        report = json.loads((ROOT / "outputs/preanalysis/quality_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["overall_status"], "PASS")
        for date_text in ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]:
            self.assertEqual(len(self.dispatch[self.dispatch["plan_date"].eq(pd.Timestamp(date_text))]), 144)


if __name__ == "__main__":
    unittest.main()
