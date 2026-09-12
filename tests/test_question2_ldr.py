"""Regression tests for the revised four-stage clipped-affine LDR."""
import unittest

import numpy as np

from src.optimization.question2 import EMIN, T
from src.optimization.question2_ldr import (
    ScenarioObjective,
    execute_actual_day,
    rule_transition,
    stage_error_means,
)


class Question2LDRTests(unittest.TestCase):
    def test_stage_signals_use_only_completed_intervals(self):
        forecast = np.zeros(T)
        path = np.arange(T, dtype=float)
        signals = stage_error_means(path, forecast)[0]
        changed = path.copy()
        changed[36:] += 1e6
        changed_signals = stage_error_means(changed, forecast)[0]
        self.assertEqual(signals[0], 0.0)
        self.assertEqual(signals[1], changed_signals[1])
        self.assertNotEqual(signals[2], changed_signals[2])
        self.assertAlmostEqual(signals[1], path[:36].mean())

    def test_zero_parameters_equal_beta_one_threshold(self):
        load = np.linspace(300, 900, T)
        pv = np.r_[np.zeros(36), np.full(72, 350.0), np.zeros(36)]
        grid = np.full(T, 400.0)
        plan_soc = np.linspace(EMIN, 9000.0, T)
        prices = np.linspace(0.3, 1.0, T)
        forecast = load - pv - 20.0
        result = execute_actual_day(
            load, pv, grid, plan_soc, prices, 6000.0, forecast, np.zeros(7)
        )
        np.testing.assert_allclose(result["reserve_kwh"], plan_soc)
        for t in [0, 35, 36, 71, 72, 107, 108, 143]:
            expected = rule_transition(
                load[t] - pv[t], grid[t], result["soc_start_kwh"][t], plan_soc[t]
            )
            self.assertAlmostEqual(result["charge_kwh"][t], float(expected[0]))
            self.assertAlmostEqual(result["discharge_kwh"][t], float(expected[1]))
            self.assertAlmostEqual(result["emergency_kwh"][t], float(expected[2]))

    def test_future_actual_perturbation_leaves_prior_actions_unchanged(self):
        rng = np.random.default_rng(17)
        load = rng.uniform(300, 1000, T)
        pv = rng.uniform(0, 500, T)
        grid = rng.uniform(200, 700, T)
        plan_soc = rng.uniform(EMIN, 10000, T)
        forecast = rng.uniform(100, 500, T)
        prices = rng.uniform(0.2, 1.1, T)
        theta = np.array([100, -200, 300, -400, 0.4, -0.2, 0.7])
        base = execute_actual_day(load, pv, grid, plan_soc, prices, 6000, forecast, theta)
        altered = load.copy()
        altered[90:] += 1e6
        changed = execute_actual_day(altered, pv, grid, plan_soc, prices, 6000, forecast, theta)
        for key in ["reserve_kwh", "charge_kwh", "discharge_kwh", "emergency_kwh", "soc_end_kwh"]:
            np.testing.assert_allclose(base[key][:90], changed[key][:90])

    def test_scenario_score_matches_same_path_actual_execution(self):
        rng = np.random.default_rng(23)
        net = rng.uniform(-300, 900, T)
        forecast = rng.uniform(0, 500, T)
        grid = rng.uniform(100, 700, T)
        plan_soc = rng.uniform(EMIN, 9500, T)
        prices = rng.uniform(0.2, 1.1, T)
        theta = np.array([-500, 250, 700, -300, 0.2, -0.4, 0.8])
        nu = 0.4
        score = ScenarioObjective(
            net[None, :], forecast, grid, plan_soc, prices, 6000, nu
        )(theta)
        actual = execute_actual_day(
            net, np.zeros(T), grid, plan_soc, prices, 6000, forecast, theta
        )
        expected = actual["emergency_cost"].sum() - nu * actual["soc_end_kwh"][-1]
        self.assertAlmostEqual(score, expected, places=8)


if __name__ == "__main__":
    unittest.main()
