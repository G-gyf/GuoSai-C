# -*- coding: utf-8 -*-
"""Unit tests for the question-4 v2 machinery (docs/问题四/问题四优化实施方案.md).

Fast invariants only; the full-year backtests are exercised through the
driver scripts.  Run:  python -m unittest tests.test_question4_v2 -v
"""
import unittest

import numpy as np

from src.optimization.q4_v2_common import (
    V2Settings,
    calibrate_v2,
    decay_weights,
    plan_horizon,
    price_weighted_quantile,
    residual_columns,
    scenario_mean_price,
    scenario_window,
    unpack_by_stages,
)
from src.optimization.question3 import settle, solve_adjustment
from src.optimization.question2 import ETA, EMIN, EMAX


class TestPriceWeightedQuantile(unittest.TestCase):
    def test_equal_prices_reduces_to_weighted_ordinary(self):
        rng = np.random.default_rng(7)
        net = rng.normal(size=(60, 12))
        price = np.full((60, 12), 0.7)
        w = rng.random(60)
        w /= w.sum()
        got = price_weighted_quantile(net, price, w, 0.8)
        want = np.zeros(12)
        for j in range(12):
            order = np.argsort(net[:, j], kind="stable")
            cum = np.cumsum(w[order])
            k = int(np.searchsorted(cum, 0.8, side="left"))
            want[j] = net[order, j][min(k, 59)]
        self.assertTrue(np.allclose(got, want, atol=1e-12))

    def test_high_price_high_net_raises_quantile(self):
        # deterministic construction: expensive scenarios carry high net load
        rng = np.random.default_rng(11)
        net = rng.normal(size=(200, 1))
        price = 0.2 + 0.8 * (net - net.min()) / (net.max() - net.min())
        w = np.full(200, 1.0 / 200)
        q_p = price_weighted_quantile(net, price, w, 0.8)[0]
        q_o = float(np.quantile(net[:, 0], 0.8))
        self.assertGreaterEqual(q_p, q_o - 1e-12)

    def test_decay_weights_monotone_and_normalized(self):
        w = decay_weights(100, 50, tau=14.0)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
        self.assertTrue(np.all(np.diff(w) > 0))  # more recent -> larger weight


class TestScenarioWindow(unittest.TestCase):
    def test_window(self):
        self.assertEqual(scenario_window(100, 42), (58, 42))
        self.assertEqual(scenario_window(20, 42), (7, 13))
        self.assertEqual(scenario_window(6, 42), (7, 0))


class TestHorizonHelpers(unittest.TestCase):
    def test_plan_horizon(self):
        F = np.arange(24, dtype=float).reshape(2, 12)
        h = plan_horizon(F, 0)
        self.assertTrue(np.allclose(h, np.r_[F[0, 1:12], F[1, 0]]))

    def test_residual_columns_alignment(self):
        rng = np.random.default_rng(3)
        actual = rng.random((10, 144))
        F = rng.random((11, 144))
        e = residual_columns(actual, F, 4)
        self.assertTrue(np.allclose(e[:143], actual[4, 1:144] - F[4, 1:144]))
        self.assertAlmostEqual(e[143], actual[5, 0] - F[5, 0])


class TestAdjustmentSettlement(unittest.TestCase):
    def test_lp_objective_equals_segment_settlement(self):
        rng = np.random.default_rng(5)
        n = 40
        q0 = rng.random(n) * 10
        risk = np.maximum(rng.normal(5, 3, n), 0.0)
        prices = rng.random(n) * 0.3 + 0.1
        a, e, obj = solve_adjustment(q0, risk, prices, 6000.0, 0.0)
        retained, down, up = settle(q0, a, prices)
        self.assertAlmostEqual(float((retained + down + up).sum()), float(obj),
                               delta=1e-5)


class TestCalibrationSampleRules(unittest.TestCase):
    def _scen(self, m):
        rng = np.random.default_rng(9)
        net = rng.normal(0, 4, (m, 145))
        forecast = np.zeros(145)
        q = np.full(145, 3.0)
        ref = np.full(145, 5000.0)
        prices = np.full((m, 145), 0.5)
        return net, forecast, q, ref, prices

    def test_insufficient_history_skips_search(self):
        net, forecast, q, ref, prices = self._scen(10)
        settings = V2Settings(gamma=1e-4)
        stages = [(0, 36, False), (36, 72, True), (72, 108, True), (108, 145, True)]
        cal = calibrate_v2(net, forecast, q, ref, prices, 6000.0, 0.3, stages,
                           [None] * 4, settings, seed=1,
                           signal_mode="cumulative", residual_count=10)
        self.assertEqual(cal["method"], "zero_parameter_warmup")
        self.assertTrue(np.allclose(cal["theta"], 0.0))

    def test_mid_history_deltas_only(self):
        net, forecast, q, ref, prices = self._scen(20)
        settings = V2Settings(gamma=1e-4, search_maxiter=2, search_popsize=3)
        stages = [(0, 36, False), (36, 72, True), (72, 108, True), (108, 145, True)]
        cal = calibrate_v2(net, forecast, q, ref, prices, 6000.0, 0.3, stages,
                           [None] * 4, settings, seed=2,
                           signal_mode="cumulative", residual_count=20)
        # lambdas forced off (pinned to zero) -> full-length theta, lambdas 0
        self.assertEqual(len(cal["theta"]), 7)
        deltas, lambdas = unpack_by_stages(cal["theta"], stages)
        self.assertTrue(np.allclose(lambdas, 0.0))

    def test_full_history_uses_seven_parameters(self):
        net, forecast, q, ref, prices = self._scen(42)
        settings = V2Settings(gamma=1e-4, search_maxiter=2, search_popsize=3)
        stages = [(0, 36, False), (36, 72, True), (72, 108, True), (108, 145, True)]
        cal = calibrate_v2(net, forecast, q, ref, prices, 6000.0, 0.3, stages,
                           [None] * 4, settings, seed=3,
                           signal_mode="cumulative", residual_count=42)
        self.assertEqual(cal["n_active"], 7)

    def test_beta0_floor_encodes_emin_reserve(self):
        from src.optimization.q4_v2_common import floor_theta_min, DELTA_BOUND
        stages = [(0, 36, True), (36, 72, True)]
        theta = floor_theta_min(stages)
        self.assertTrue(np.allclose(theta[::2], -DELTA_BOUND))
        self.assertTrue(np.allclose(theta[1::2], 0.0))


class TestMeanPrice(unittest.TestCase):
    def test_weighted_mean(self):
        price = np.array([[1.0, 2.0], [3.0, 4.0]])
        w = np.array([0.25, 0.75])
        got = scenario_mean_price(price, w)
        self.assertTrue(np.allclose(got, [2.5, 3.5]))

    def test_batched_objective_matches_scalar(self):
        from src.optimization.q4_v2_common import CalibObjective
        rng = np.random.default_rng(13)
        m, h = 42, 145
        net = rng.normal(0, 4, (m, h))
        forecast = np.zeros(h)
        q = np.full(h, 3.0)
        ref = np.full(h, 5000.0)
        prices = np.full((m, h), 0.5)
        stages = [(0, 36, False), (36, 72, True), (72, 108, True), (108, 145, True)]
        obj = CalibObjective(net, forecast, q, ref, prices, 6000.0, 0.3,
                             stages, [None] * 4, 1e-4, "cumulative")
        x = rng.uniform(-100, 100, (5, 7))
        batched = np.atleast_1d(obj(x))
        scalar = np.array([obj(x[i]) for i in range(5)])
        self.assertTrue(np.allclose(batched, scalar, atol=1e-10))


class TestCalibrationBatchWorker(unittest.TestCase):
    def test_batch_map_worker(self):
        from src.optimization.q4_v2_common import CalibObjective
        rng = np.random.default_rng(21)
        m, h = 20, 36
        net = rng.normal(0, 4, (m, h))
        forecast = np.zeros(h)
        q = np.full(h, 3.0)
        ref = np.full(h, 5000.0)
        prices = np.full((m, h), 0.5)
        stages = [(0, 36, True)]
        obj = CalibObjective(net, forecast, q, ref, prices, 6000.0, 0.3,
                             stages, [None], 0.0, "cumulative")
        rows = rng.uniform(-1, 1, (8, 2))
        got = obj(rows)
        self.assertEqual(np.shape(got), (8,))


if __name__ == "__main__":
    unittest.main()
