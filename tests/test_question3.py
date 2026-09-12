"""Acceptance tests for the question-3 pipeline.

Covers: segment net settlement (four canonical cases + LP equivalence),
the adjustment LP economics, PCHIP anchor pass-through, M0 degeneracy into
a no-adjustment strategy, causal invariance under future-data perturbation,
and scenario-evaluation / live-execution consistency.
"""
import unittest

import numpy as np
import pandas as pd

from src.optimization.question2 import T, ETA, EMIN, EMAX, S, load_inputs, load_forecast_weekly_persist
from src.optimization.question3 import (
    Q3Settings,
    adjustment_curve,
    calibrate_staged,
    execute_segment,
    run_rule_horizon,
    run_strategy,
    scenario_matrix,
    settle,
    solve_adjustment,
    validate_question3,
)
from src.data_pipeline.question3_forecasts import (
    build_issuance_curves,
    load_hourly_issuances,
    load_pv_actuals,
    observed_anchor_kw,
)

PRICES = np.linspace(0.3, 0.9, 144)


class TestSettlement(unittest.TestCase):
    def test_no_adjustment(self):
        q0 = np.array([100.0, 50.0])
        qA = q0.copy()
        retained, down, up = settle(q0, qA, np.array([0.5, 0.5]))
        np.testing.assert_allclose(retained, [50.0, 25.0])
        np.testing.assert_allclose(down, 0.0)
        np.testing.assert_allclose(up, 0.0)

    def test_only_down(self):
        q0 = np.array([100.0])
        qA = np.array([60.0])
        p = np.array([0.5])
        retained, down, up = settle(q0, qA, p)
        np.testing.assert_allclose(retained, [30.0])  # c*min
        np.testing.assert_allclose(down, [10.0])      # 0.5*c*(q0-qA)
        np.testing.assert_allclose(up, [0.0])

    def test_only_up(self):
        q0 = np.array([100.0])
        qA = np.array([140.0])
        p = np.array([0.5])
        retained, down, up = settle(q0, qA, p)
        np.testing.assert_allclose(retained, [50.0])
        np.testing.assert_allclose(down, [0.0])
        np.testing.assert_allclose(up, [30.0])        # 1.5*c*(qA-q0)

    def test_mixed_and_identity(self):
        rng = np.random.default_rng(7)
        q0 = rng.uniform(0, 500, size=144)
        qA = rng.uniform(0, 500, size=144)
        retained, down, up = settle(q0, qA, PRICES)
        # identity: c*min + 0.5c*(q0-qA)+ + 1.5c*(qA-q0)+ == c*qA + 0.5c(p+q)
        alt = PRICES * qA + 0.5 * PRICES * (np.maximum(q0 - qA, 0) + np.maximum(qA - q0, 0))
        np.testing.assert_allclose(retained + down + up, alt, rtol=1e-12)
        self.assertTrue((down >= 0).all() and (up >= 0).all())


class TestAdjustmentLP(unittest.TestCase):
    def test_down_adjustment_economics(self):
        # q0 is far above the risk curve and the battery is full, so
        # down-adjusting is unambiguously cheaper than keeping the plan.
        n = 36
        q0 = np.full(n, 800.0)
        risk = np.full(n, 100.0)
        prices = np.full(n, 0.5)
        a, e, obj = solve_adjustment(q0, risk, prices, initial=EMAX, terminal_value=0.3)
        self.assertTrue((a < q0 - 1e-6).all())
        # settlement with the LP solution must be strictly cheaper than c*q0
        retained, down, up = settle(q0, a, prices)
        self.assertLess(float((retained + down + up).sum()), float((prices * q0).sum()) - 1e-6)
        self.assertTrue(np.isfinite(e).all())

    def test_solution_satisfies_physics(self):
        n = 72
        q0 = np.full(n, 600.0)
        risk = np.linspace(300.0, 700.0, n)
        a, e, obj = solve_adjustment(q0, risk, PRICES[:n], initial=3000.0, terminal_value=0.4)
        self.assertEqual(a.shape, (n,))
        self.assertEqual(e.shape, (n,))
        self.assertTrue((a >= -1e-9).all())
        self.assertTrue((e >= EMIN - 1e-6).all() and (e <= EMAX + 1e-6).all())
        # settlement of the LP solution equals the LP's own optimal objective
        # up to the terminal-value term
        retained, down, up = settle(q0, a, PRICES[:n])
        np.testing.assert_allclose((retained + down + up).sum() - 0.4 * e[-1], obj, atol=1e-4)


class TestForecastInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.curves = build_issuance_curves()
        cls.hourly = load_hourly_issuances()
        cls.pv = load_pv_actuals()

    def test_anchor_pass_through(self):
        from scipy.interpolate import PchipInterpolator
        for d in [0, 100, 364]:
            for k, hour in enumerate([0, 6, 12, 18]):
                anchors = np.r_[observed_anchor_kw(self.pv, d, hour),
                                self.hourly[d, k, : 24 - hour]]
                # attachment-3 hourly anchors strictly pass through on the grid
                for j, h in enumerate(range(hour + 1, 25)):
                    curve_kw = self.curves[d, k, 6 * h - 1] * 6.0
                    self.assertLess(abs(curve_kw - anchors[j + 1]), 1e-9)
                # the observed issue-hour anchor is a knot of the interpolant
                interp = PchipInterpolator(60.0 * np.arange(hour, 25), anchors,
                                           extrapolate=False)
                self.assertLess(abs(interp(60.0 * hour) - anchors[0]), 1e-9)

    def test_coverage_and_nonnegativity(self):
        finite = np.isfinite(self.curves)
        for k, hour in enumerate([0, 6, 12, 18]):
            self.assertFalse(finite[:, k, : 6 * hour].any(), f"issuance {hour}:00 leaks before issue")
            self.assertTrue(finite[:, k, 6 * hour:].all())
        self.assertTrue((np.nan_to_num(self.curves, nan=0.0) >= 0).all())


class TestAdjustmentCurve(unittest.TestCase):
    @staticmethod
    def _scen(lo, hi):
        scen = np.empty((10, 36))
        scen[::2] = lo
        scen[1::2] = hi
        return scen

    def setUp(self):
        self.fl = np.zeros(36)
        self.fc = np.zeros(36)
        self.q0 = np.full(36, 100.0)

    def test_down_region_targets_q90(self):
        scen = self._scen(50.0, 80.0)  # Q90=80 < q0
        target = adjustment_curve(self.fl, self.fc, scen, self.q0, 0)
        np.testing.assert_allclose(target, 80.0)

    def test_up_region_targets_q50(self):
        scen = self._scen(150.0, 200.0)  # Q50 (linear interp) = 175 > q0
        target = adjustment_curve(self.fl, self.fc, scen, self.q0, 0)
        np.testing.assert_allclose(target, 175.0)

    def test_kink_region_keeps_plan(self):
        scen = self._scen(50.0, 150.0)  # Q50<q0<Q90
        target = adjustment_curve(self.fl, self.fc, scen, self.q0, 0)
        np.testing.assert_allclose(target, 100.0)

    def test_point_forecast_floor(self):
        scen = self._scen(50.0, 150.0)
        target = adjustment_curve(np.full(36, 120.0), self.fc, scen, self.q0, 0)
        np.testing.assert_allclose(target, 120.0)


class TestRuleConsistency(unittest.TestCase):
    def test_scenario_eval_equals_live_execution(self):
        rng = np.random.default_rng(3)
        n = 36
        net = rng.uniform(-100, 400, size=(1, n))
        q = rng.uniform(0, 400, size=n)
        ref = np.full(n, 2500.0)
        prices = PRICES[:n]
        forecast = rng.uniform(-50, 350, size=n)
        a6 = 12.0
        theta = np.array([-1500.0, 0.4])
        ec, e_end = run_rule_horizon(
            net, q, ref, prices, 2400.0, forecast,
            stages=[(0, n, True)], observed=[a6], theta=theta,
        )
        out = execute_segment(net[0], q, ref, prices, 2400.0, a6, theta)
        np.testing.assert_allclose(ec, np.sum(5.0 * prices * out["emergency_kwh"]), rtol=1e-12)
        np.testing.assert_allclose(e_end, out["soc_end_kwh"][-1], rtol=1e-12)


class TestStrategyProperties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dates, cls.load, cls.pv, cls.prices = load_inputs()
        rep = pd.read_excel("附件/附件1.xlsx", sheet_name=0).iloc[:, 2].to_numpy(float) / 6.0
        cls.fl = load_forecast_weekly_persist(cls.load, rep)
        cls.fc = build_issuance_curves()

    def _run(self, strategy, days):
        settings = Q3Settings(strategy=strategy, search_maxiter=2, search_popsize=3)
        frame, _, _, summary = run_strategy(
            self.dates, self.load, self.pv, self.prices, self.fl, self.fc, settings, limit=days
        )
        return frame, summary

    def test_m0_is_no_adjustment(self):
        frame, summary = self._run("M0", 3)
        self.assertTrue((np.abs(frame.qA_kwh - frame.q0_kwh) < 1e-9).all())
        self.assertAlmostEqual(float(frame.down_cost.sum()), 0.0)
        self.assertAlmostEqual(float(frame.up_cost.sum()), 0.0)
        np.testing.assert_allclose(
            frame.total_cost.to_numpy(), frame.retained_cost.to_numpy() + frame.emergency_cost.to_numpy(),
            rtol=1e-12,
        )
        validate_question3(frame)

    def test_staged_day_is_causal(self):
        frame_a, _ = self._run("M612", 4)
        # perturb a FUTURE day's PV; earlier decisions must not change
        pv2 = self.pv.copy()
        pv2[3] = pv2[3] + 1000.0
        settings = Q3Settings(strategy="M612", search_maxiter=2, search_popsize=3)
        frame_b, _, _, _ = run_strategy(
            self.dates, self.load, pv2, self.prices, self.fl, self.fc, settings, limit=4
        )
        a = frame_a[frame_a.date < self.dates[3]]
        b = frame_b[frame_b.date < self.dates[3]]
        for col in ["q0_kwh", "qA_kwh", "charge_kwh", "discharge_kwh", "emergency_kwh"]:
            np.testing.assert_allclose(a[col].to_numpy(), b[col].to_numpy(), rtol=0, atol=1e-9)

    def test_staged_validation(self):
        for strategy in ["M6", "M612", "M61218-S", "M61218-F"]:
            frame, summary = self._run(strategy, 4)
            checks = validate_question3(frame)
            self.assertTrue(checks["passed"], strategy)
            self.assertEqual(checks["pre_six_executed_adjusted"], 0, strategy)

    def test_warmup_uses_zero_parameters(self):
        settings = Q3Settings(strategy="M612", search_maxiter=2, search_popsize=3)
        frame, _, _, _ = run_strategy(
            self.dates, self.load, self.pv, self.prices, self.fl, self.fc, settings, limit=8
        )
        # before the 21-day residual window every calibration is zero-parameter
        self.assertTrue((np.abs(frame.ldr_delta_kwh) < 1e-9).all())
        self.assertTrue((np.abs(frame.ldr_lambda) < 1e-9).all())


if __name__ == "__main__":
    unittest.main()
