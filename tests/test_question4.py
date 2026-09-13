# -*- coding: utf-8 -*-
"""Question 4 fluctuating-price rule tests (data layer, scenario pairing and
backward-compatible pipeline extensions)."""
import unittest

import numpy as np
import pandas as pd

from src.data_pipeline.question4_prices import (
    build_price_forecast,
    load_price_actual,
    load_fixed_price,
    price_residual,
    price_scenario_paths,
)
from src.optimization.question2 import load_inputs, run_case, Settings
from src.optimization.question2_g_search import scenario_net_matrix
from src.optimization.question2_ldr import (
    LDRSettings,
    ScenarioObjective,
    run_ldr,
)
from src.optimization.question3 import run_rule_horizon, rule_transition  # noqa: F401
from src.optimization.question3 import Q3Settings  # noqa: F401

ROOT = None


class TestPriceData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dates, cls.actual = load_price_actual()
        cls.forecast = build_price_forecast(cls.actual, load_fixed_price())
        cls.residual = price_residual(cls.actual, cls.forecast)

    def test_alignment_and_fill(self):
        import pandas as pd
        raw = pd.read_excel(r"附件/附件4.xlsx", sheet_name=0).iloc[:, 1:145].to_numpy(float)
        # day d slots 2..144 <- row d columns 0..142
        np.testing.assert_allclose(self.actual[5, 1:], raw[5, :143])
        # day d slot 1 <- row d-1 last column
        np.testing.assert_allclose(self.actual[5, 0], raw[4, 143])
        # 1/1 slot 1 fill = its own 00:10 label price (confirmed R0)
        self.assertAlmostEqual(self.actual[0, 0], raw[0, 0], places=12)
        self.assertTrue(np.isfinite(self.actual).all())

    def test_forecast_rules(self):
        f = self.forecast
        # weekly persistence from 1/8 (index 7)
        np.testing.assert_allclose(f[7], self.actual[0])
        np.testing.assert_allclose(f[100], self.actual[93])
        # warm-up: d = 1..6 use the available-history mean
        for d in range(1, 7):
            np.testing.assert_allclose(f[d], self.actual[:d].mean(axis=0))
        # 1/1 placeholder equals the attachment-1 fixed curve (inert, zero plan)
        np.testing.assert_allclose(f[0], load_fixed_price())

    def test_residual_out_of_sample(self):
        # residual for day i uses the forecast built from history BEFORE i
        for i in (8, 50, 300):
            np.testing.assert_allclose(self.residual[i], self.actual[i] - self.forecast[i])
        # the forecast of day i never uses day i or later actuals: weekly
        # persistence reads only day i-7, the warm-up only earlier days.
        self.assertTrue(np.allclose(self.forecast[50], self.actual[43]))

    def test_scenario_window_matches_net_load(self):
        dates, load, pv, _ = load_inputs()
        from src.optimization.question2 import load_forecast_weekly_persist
        fl = load_forecast_weekly_persist(load)
        fv = np.full_like(pv, np.nan)
        for d in range(1, len(load)):
            fv[d] = pv[max(0, d - 7):d].mean(axis=0)
        for d in (40, 100, 300):
            scen, count = scenario_net_matrix(d, load, pv, fl, fv, 21)
            paths = price_scenario_paths(d, self.forecast, self.residual, 21)
            self.assertEqual(paths.shape[0], count)
            self.assertEqual(paths.shape, scen.shape)
            self.assertTrue((paths >= 0).all())
            # same-day pairing: row i of price paths uses residual day
            # first+i (the window is [first, d) with first = max(7, d-21))
            first = max(7, d - 21)
            expected = np.maximum(0.0, self.forecast[d] + self.residual[first])
            np.testing.assert_allclose(paths[0], expected)

    def test_scenario_paths_horizon(self):
        paths = price_scenario_paths(100, self.forecast, self.residual, 21, h0=36)
        self.assertEqual(paths.shape[1], 108)
        self.assertIsNone(price_scenario_paths(5, self.forecast, self.residual, 21))


class TestQ2BroadcastEquivalence(unittest.TestCase):
    def test_scenario_objective_broadcast_identical(self):
        rng = np.random.default_rng(7)
        net = rng.normal(50, 20, size=(5, 144))
        forecast = rng.normal(50, 10, size=144)
        grid = np.abs(rng.normal(60, 10, size=144))
        soc = rng.uniform(1500, 9000, size=144)
        prices = np.abs(rng.normal(0.6, 0.15, size=144))
        theta = rng.normal(0, 1000, size=7)
        obj1 = ScenarioObjective(net, forecast, grid, soc, prices, 3000.0, 0.4)
        obj2 = ScenarioObjective(net, forecast, grid, soc,
                                 np.broadcast_to(prices, net.shape), 3000.0, 0.4)
        self.assertEqual(obj1(theta), obj2(theta))
        # scenario-specific prices change the objective
        alt = np.broadcast_to(prices, net.shape).copy()
        alt[0] *= 2.0
        obj3 = ScenarioObjective(net, forecast, grid, soc, alt, 3000.0, 0.4)
        self.assertNotEqual(obj1(theta), obj3(theta))

    def test_run_case_fixed_equivalence(self):
        dates, load, pv, prices = load_inputs()
        n = 40  # must reach the formal period (2025-02-01) for the summary
        dates, load, pv = dates[:n], load[:n], pv[:n]
        f1, d1, s1 = run_case(dates, load, pv, prices, Settings())
        f2, d2, s2 = run_case(
            dates, load, pv, prices, Settings(),
            prices_plan=np.tile(prices, (n, 1)),
            prices_actual=np.tile(prices, (n, 1)),
        )
        pd.testing.assert_frame_equal(f1, f2)
        self.assertEqual(s1["total_cost"], s2["total_cost"])

    def test_run_ldr_short_variable(self):
        dates, load, pv, prices = load_inputs()
        n = 6
        dates, load, pv = dates[:n], load[:n], pv[:n]
        p_act = np.abs(np.random.default_rng(1).normal(0.7, 0.2, size=(n, 144)))
        p_hat = p_act.copy()
        p_hat[3:] = p_act[:3]  # weekly persistence imitation for a tiny run

        def fn(d, m):
            return np.broadcast_to(p_hat[d], (m, 144))

        frame, daily, daily_all, summary, diag, pair = run_ldr(
            dates, load, pv, prices, LDRSettings(), limit=n,
            prices_plan=p_hat, prices_actual=p_act, price_scenarios_fn=fn,
        )
        self.assertTrue(summary["validation"]["passed"])
        # settled bill uses actual prices: planned_cost = sum p_act * g,
        # emergency_cost = 5 * p_act * b (day 0 is frozen and has no rows)
        expected = 0.0
        emergency_expected = 0.0
        for d in range(1, n):
            day = frame[frame.date == dates[d]]
            expected += float(np.dot(p_act[d], day.grid_kwh.to_numpy()))
            emergency_expected += float(5 * np.dot(p_act[d], day.emergency_kwh.to_numpy()))
        self.assertAlmostEqual(frame.planned_cost.sum(), expected, places=6)
        self.assertAlmostEqual(frame.emergency_cost.sum(), emergency_expected, places=6)
        # Correction 2: the 0:00 decision price of the first interval is the
        # realized price, not the forecast; later slots keep the forecast.
        for d in range(1, n):
            day = frame[frame.date == dates[d]]
            np.testing.assert_allclose(
                day.loc[day.slot == 0, "price_forecast"].iloc[0], p_act[d][0], rtol=1e-12
            )
            np.testing.assert_allclose(
                day.loc[day.slot == 1, "price_forecast"].iloc[0], p_hat[d][1], rtol=1e-12
            )
        # 1 January is frozen: no rows, first planning day is 2 January.
        self.assertNotIn(pd.Timestamp("2025-01-01"), set(frame.date))
        self.assertEqual(frame.date.min(), pd.Timestamp("2025-01-02"))

    def test_realized_decision_price_helpers(self):
        from src.optimization.question2 import decision_prices_with_realized as q2_helper
        from src.optimization.question3 import decision_prices_with_realized as q3_helper
        from src.data_pipeline.question4_prices import determinize_scenario_first_column

        p_hat = np.full(144, 0.5)
        p_act = np.linspace(0.6, 0.9, 144)
        row = q2_helper(p_hat, p_act, (0,))
        self.assertAlmostEqual(row[0], p_act[0])
        self.assertTrue((row[1:] == 0.5).all())
        row3 = q3_helper(p_hat, p_act, (0, 36, 72, 108))
        for h0 in (0, 36, 72, 108):
            self.assertAlmostEqual(row3[h0], p_act[h0])
        kept = [t for t in range(144) if t not in (0, 36, 72, 108)]
        self.assertTrue((row3[kept] == 0.5).all())

        paths = np.broadcast_to(np.linspace(0.4, 0.8, 108), (5, 108)).copy()
        original = paths.copy()
        fixed = determinize_scenario_first_column(paths, 1.234)
        self.assertTrue((fixed[:, 0] == 1.234).all())
        self.assertTrue((fixed[:, 1:] == original[:, 1:]).all())
        self.assertTrue((paths == original).all())  # caller's matrix untouched


class TestQ3BroadcastEquivalence(unittest.TestCase):
    def test_run_rule_horizon_broadcast_identical(self):
        rng = np.random.default_rng(11)
        net = rng.normal(40, 18, size=(4, 72))
        q = np.abs(rng.normal(50, 10, size=72))
        ref = rng.uniform(1500, 9000, size=72)
        prices = np.abs(rng.normal(0.6, 0.15, size=72))
        fc = rng.normal(40, 8, size=72)
        theta = rng.normal(0, 800, size=4)
        stages = [(0, 36, True), (36, 72, True)]
        ec1, e1 = run_rule_horizon(net, q, ref, prices, 3000.0, fc, stages,
                                   [np.nan, np.nan], theta)
        ec2, e2 = run_rule_horizon(net, q, ref, np.broadcast_to(prices, net.shape),
                                   3000.0, fc, stages, [np.nan, np.nan], theta)
        np.testing.assert_allclose(ec1, ec2)
        np.testing.assert_allclose(e1, e2)


if __name__ == "__main__":
    unittest.main()
