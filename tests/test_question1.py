"""Tests for the continuous LP implementation of Question 1."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimization.question1 import (
    Q1Parameters,
    load_question1_inputs,
    solve_question1,
)


ROOT = Path(__file__).resolve().parents[1]


class Question1LPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parameters = Q1Parameters()
        cls.inputs = load_question1_inputs(ROOT / "附件" / "附件1.xlsx")
        cls.solution = solve_question1(cls.inputs, cls.parameters)

    def test_input_contract_and_units(self) -> None:
        self.assertEqual(len(self.inputs), 144)
        self.assertEqual(self.inputs.iloc[0]["time_label"], "00:10")
        self.assertEqual(self.inputs.iloc[-1]["time_label"], "0:00+1")
        self.assertTrue((self.inputs[["price_yuan_per_kwh", "load_kw", "pv_kw"]] >= 0).all().all())
        self.assertTrue(
            np.allclose(
                self.solution.schedule["load_kwh"],
                self.inputs["load_kw"] / 6.0,
            )
        )

    def test_full_solution_is_feasible(self) -> None:
        validation = self.solution.validation
        self.assertEqual(validation["status"], "PASS")
        self.assertLessEqual(validation["max_balance_residual_kwh"], 1e-6)
        self.assertLessEqual(validation["max_soc_transition_residual_kwh"], 1e-6)
        self.assertEqual(validation["simultaneous_charge_discharge_slots"], 0)
        self.assertAlmostEqual(validation["terminal_soc_kwh"], 6000.0, places=6)
        self.assertGreaterEqual(validation["min_soc_kwh"], 1200.0 - 1e-6)
        self.assertLessEqual(validation["max_soc_kwh"], 10800.0 + 1e-6)
        self.assertLessEqual(validation["max_charge_kwh"], 5000.0 / 6.0 + 1e-6)
        self.assertLessEqual(validation["max_discharge_kwh"], 5000.0 / 6.0 + 1e-6)

    def test_regression_values(self) -> None:
        summary = self.solution.summary
        self.assertAlmostEqual(summary["primary_optimal_cost_yuan"], 35126.9485892896, places=5)
        self.assertAlmostEqual(summary["total_purchase_kwh"], 59482.6989900444, places=4)
        self.assertAlmostEqual(summary["total_charge_kwh"], 20740.6660879530, places=4)
        self.assertAlmostEqual(summary["total_discharge_kwh"], 16799.9395312419, places=4)
        self.assertLessEqual(
            summary["secondary_cost_yuan"] - summary["primary_optimal_cost_yuan"],
            summary["cost_tolerance_yuan"] + 1e-6,
        )

    def test_four_hour_blocks_and_selected_intervals(self) -> None:
        blocks = self.solution.block_summary
        self.assertEqual(len(blocks), 6)
        self.assertAlmostEqual(blocks["charge_kwh"].sum(), self.solution.summary["total_charge_kwh"], places=6)
        self.assertAlmostEqual(blocks["discharge_kwh"].sum(), self.solution.summary["total_discharge_kwh"], places=6)
        first_block = self.solution.schedule[self.solution.schedule["block"].eq("0:00-4:00")]
        self.assertEqual(len(first_block), 24)
        self.assertTrue(first_block["is_next_day"].any())
        self.assertEqual(
            self.solution.selected_intervals["time_label"].tolist(),
            ["10:00", "12:00", "14:00", "16:00", "18:00", "20:00"],
        )

    def test_toy_low_price_charge_high_price_discharge(self) -> None:
        n = 144
        toy = pd.DataFrame(
            {
                "slot_index": np.arange(1, n + 1),
                "time_label": [f"toy-{i}" for i in range(n)],
                "clock_minute": np.arange(n) * 10 % 1440,
                "is_next_day": [False] * n,
                "price_yuan_per_kwh": np.r_[0.2, np.full(n - 2, 0.8), 2.0],
                "load_kw": np.full(n, 1200.0),
                "pv_kw": np.zeros(n),
            }
        )
        vector, metadata = __import__(
            "src.optimization.question1", fromlist=["_solve_two_stage_lp"]
        )._solve_two_stage_lp(toy, self.parameters)
        self.assertEqual(len(vector), 5 * n)
        self.assertLessEqual(metadata["secondary_cost_yuan"], metadata["primary_optimal_cost_yuan"] + metadata["cost_tolerance_yuan"] + 1e-6)


if __name__ == "__main__":
    unittest.main()
