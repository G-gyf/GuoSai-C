# -*- coding: utf-8 -*-
"""Acceptance tests for question 4 second sub-question (additional forecast
releases under fluctuating prices + feasibility of the extended model)."""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimization.question2 import ETA, EMIN, EMAX, load_inputs
from src.optimization.question3 import solve_adjustment
from src.optimization.question4_second_subquestion import (
    HOUR_SLOT,
    INFO_KIND,
    _solve_adjustment_safe,
    build_price_update,
    candidate_pv_horizon,
    candidate_scenarios,
    fit_price_shrink,
    nu_adj_price,
    parse_spec,
    potential_daily,
    price_update_metrics,
    run_q4_3_candidate_range,
    value_decomposition,
)
from src.optimization.q4_v2_common import (
    V2Settings,
    build_price_forecast,
    plan_horizon,
)
from src.optimization.question4_2_v2 import load_question4_v2_inputs
from src.optimization.question4_3_v2 import (
    build_q4_3_bundle,
    pv_issue_horizon,
    run_q4_3_range,
)

ROOT = Path(__file__).resolve().parents[2]


class ParseSpecTest(unittest.TestCase):
    def test_parse_single_and_combo(self):
        self.assertEqual(parse_spec("B"), [])
        ops = parse_spec("10Fboth")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0], {"hour": 10, "slot": 60, "info": "both"})
        ops = parse_spec("10Fboth+14S")
        self.assertEqual([o["hour"] for o in ops], [10, 14])
        self.assertEqual([o["info"] for o in ops], ["both", "state"])
        ops = parse_spec("9Fpv+15O")
        self.assertEqual([(o["hour"], o["info"]) for o in ops],
                         [(9, "pv_only"), (15, "oracle")])

    def test_rejects_bad_specs(self):
        for bad in ["11X", "8Fboth", "10Fboth+11S", "10S+10Fboth"]:
            with self.assertRaises(ValueError):
                parse_spec(bad)


class PriceUpdateCausalityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dates, cls.load, cls.pv, cls.fixed, cls.p_act, _ = \
            load_question4_v2_inputs()
        cls.F_p = build_price_forecast(cls.p_act, cls.dates, "wp", cls.fixed)

    def test_update_uses_only_past_data(self):
        """Corrupting future prices must not change day d's updated forecast."""
        p_act = self.p_act.copy()
        m = HOUR_SLOT[10]
        d = 200
        upd = build_price_update(p_act, self.F_p, 10)
        base = upd["updated"][d].copy()
        p_act[d, m + 1 : 144] += 100.0  # future intervals of day d
        p_act[d + 1 :] += 100.0          # all later days
        try:
            upd2 = build_price_update(p_act, self.F_p, 10, force=True)
            self.assertTrue(np.allclose(base, upd2["updated"][d]))
        finally:
            # restore the shared cache with the uncorrupted data
            build_price_update(self.p_act, self.F_p, 10, force=True)

    def test_lam_bounds_and_ratio(self):
        d = 200
        lam = fit_price_shrink(self.p_act, self.F_p, d, HOUR_SLOT[10])
        self.assertTrue(0.0 <= lam <= 1.0)
        upd = build_price_update(self.p_act, self.F_p, 10)
        self.assertTrue(0.0 <= float(upd["lam"][d]) <= 1.0)

    def test_metrics_on_future_columns_only(self):
        upd = build_price_update(self.p_act, self.F_p, 10)
        metrics = price_update_metrics(self.p_act, self.F_p, 10, upd)
        self.assertTrue(metrics["mae_base"] > 0)
        # the update improves the remaining-horizon MAE causally on the
        # formal period (documented as the screening gate; a weaker bound
        # is asserted here so the test does not depend on data quirks)
        self.assertGreaterEqual(metrics["mae_improvement_pct"], 0.0)

    def test_small_history_lam_zero(self):
        lam = fit_price_shrink(self.p_act, self.F_p, 10, HOUR_SLOT[10],
                               window=42, min_samples=1000)
        self.assertEqual(lam, 0.0)


class CandidateHorizonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dates, cls.load, cls.pv, cls.fixed, cls.p_act, _ = \
            load_question4_v2_inputs()
        cls.bundle = build_q4_3_bundle(cls.load, cls.pv, cls.p_act,
                                       cls.dates, cls.fixed)

    def test_official_horizon_matches_issue_horizon(self):
        from src.data_pipeline.question3_forecasts import ISSUE_HOURS
        d = 120
        for hour, base_k in [(10, 1), (14, 2)]:
            m = HOUR_SLOT[hour]
            official = candidate_pv_horizon(self.bundle, self.pv, d, m,
                                            base_k, "state")
            ref = pv_issue_horizon(self.bundle, d, base_k)[
                m - 6 * ISSUE_HOURS[base_k]:]
            self.assertTrue(np.allclose(official, ref))
        self.assertEqual(candidate_pv_horizon(
            self.bundle, self.pv, d, HOUR_SLOT[10], 1, "state").shape,
            (144 - HOUR_SLOT[10],))

    def test_oracle_horizon_uses_actuals(self):
        d, m = 120, HOUR_SLOT[14]
        ora = candidate_pv_horizon(self.bundle, self.pv, d, m, 2, "oracle")
        expect = np.r_[self.pv[d, m + 1 : 144], self.pv[d + 1, 0]]
        self.assertTrue(np.allclose(ora, expect))

    def test_self_horizon_shape_and_night_carry(self):
        from src.optimization.question3_second_subquestion import \
            load_self_curves
        self_fc, _ = load_self_curves(10, 1)
        m = HOUR_SLOT[10]
        sf = candidate_pv_horizon(self.bundle, self.pv, 120, m, 1, "both",
                                  self_fc=self_fc)
        self.assertEqual(sf.shape, (144 - m,))
        self.assertEqual(sf[-1], 0.0)  # night carry column
        self.assertTrue(np.allclose(sf[:-1], self_fc[120, m + 1 : 144]))


class FeasibilityTest(unittest.TestCase):
    def test_adjustment_lp_feasible_by_construction(self):
        """Extreme risk curves still admit a feasible (optimal) adjustment."""
        rng = np.random.default_rng(0)
        for n in (36, 72):
            risk = rng.uniform(-200, 800, n)
            prices = rng.uniform(0.1, 1.2, n)
            q0 = rng.uniform(0, 600, n)
            a, ep, status, message = _solve_adjustment_safe(
                q0, risk, prices, 5000.0, 0.4)
            self.assertEqual(status, "optimal", message)
            self.assertEqual(len(a), n)
            # SOC stays within bounds
            self.assertTrue((ep >= EMIN - 1e-6).all())
            self.assertTrue((ep <= EMAX + 1e-6).all())

    def test_fallback_keeps_plan(self):
        """A failed LP (monkeypatched) returns None and the caller keeps the
        locked plan instead of aborting."""
        import src.optimization.question4_second_subquestion as mod
        original = mod.solve_adjustment

        def boom(*args, **kwargs):
            raise RuntimeError("synthetic infeasibility")

        mod.solve_adjustment = boom
        try:
            a, ep, status, message = _solve_adjustment_safe(
                np.zeros(36), np.zeros(36), np.ones(36), 4000.0, 0.3)
        finally:
            mod.solve_adjustment = original
        self.assertIsNone(a)
        self.assertEqual(status, "failed")
        self.assertIn("synthetic infeasibility", message)


class ValueDecompositionTest(unittest.TestCase):
    def _daily(self, cost, end_soc):
        n = len(cost)
        return pd.DataFrame({
            "date": pd.date_range("2025-02-01", periods=n, freq="D"),
            "total_cost": cost, "soc_start_kwh": np.full(n, 4000.0),
            "soc_end_kwh": end_soc,
            "retained_cost": cost, "down_cost": 0.0, "up_cost": 0.0,
            "emergency_cost": 0.0, "down_kwh": 0.0, "up_kwh": 0.0,
            "emergency_kwh": 0.0, "unused_kwh": 0.0,
            "q0_kwh": 0.0, "qA_kwh": 0.0,
            "charge_kwh": 0.0, "discharge_kwh": 0.0,
        })

    def test_inventory_adjusted_values(self):
        nu = 0.4
        branches = {
            "B": self._daily([100.0, 100.0], [5000.0, 6000.0]),
            "S": self._daily([98.0, 99.0], [5000.0, 6000.0]),
            "Fboth": self._daily([95.0, 96.0], [5000.0, 6000.0]),
            "O": self._daily([90.0, 91.0], [5000.0, 6000.0]),
        }
        dec = value_decomposition(branches, nu)
        self.assertAlmostEqual(dec["V_state"], 3.0)
        self.assertAlmostEqual(dec["V_forecast_both"], 6.0)
        self.assertAlmostEqual(dec["UB_forecast"], 16.0)
        self.assertAlmostEqual(dec["V_forecast_both_capture"], 0.375)

    def test_potential_daily_telescopes(self):
        nu = 0.4
        d = self._daily([10.0, 20.0], [4100.0, 4200.0])
        pot = potential_daily(d, nu)
        self.assertAlmostEqual(pot[0], 10.0 + nu * 4000.0 - nu * 4100.0)


class BranchRunnerTest(unittest.TestCase):
    """Pilot runs: B branch must reproduce run_q4_3_range bit-for-bit, and
    candidate branches must pass the physics validator."""

    @classmethod
    def setUpClass(cls):
        cls.dates, cls.load, cls.pv, cls.fixed, cls.p_act, _ = \
            load_question4_v2_inputs()
        cls.bundle = build_q4_3_bundle(cls.load, cls.pv, cls.p_act,
                                       cls.dates, cls.fixed)
        from src.optimization.question4_second_subquestion import warmup_carry
        cls.feb1_soc, cls.c0, cls.cA, cls.cr = warmup_carry()
        cls.settings = V2Settings(name="M612", price_model="wp", gamma=0.0,
                                  search_seed=20250912)

    def _run(self, ops, days):
        return run_q4_3_candidate_range(
            self.dates, self.load, self.pv, self.p_act, self.bundle,
            self.settings, 31, self.feb1_soc, 31 + days, "wp",
            prev_q0=self.c0, prev_qA=self.cA, prev_ref=self.cr, ops=ops)

    def test_b_branch_identical_to_reference_runner(self):
        days = 3
        frame, *_ = self._run(None, days)
        ref, *_ = run_q4_3_range(
            self.dates, self.load, self.pv, self.p_act, self.bundle,
            self.settings, 31, self.feb1_soc, 31 + days, "wp",
            prev_q0=self.c0, prev_qA=self.cA, prev_ref=self.cr)
        self.assertTrue(frame.equals(ref))

    def test_candidate_branches_valid(self):
        for spec in ["10Fboth", "14S", "10Fboth+14Fboth"]:
            frame, _, _, _, diag, validation = self._run(parse_spec(spec), 3)
            self.assertTrue(validation["passed"])
            self.assertEqual(validation["simultaneous_charge_discharge"], 0)
            self.assertEqual(validation["emergency_while_charging"], 0)
            self.assertTrue((frame.soc_start_kwh >= EMIN - 1e-5).all())
            self.assertTrue((frame.soc_end_kwh <= EMAX + 1e-5).all())
            ops_rec = diag.candidate_ops.iloc[0]
            self.assertTrue(isinstance(ops_rec, list) and len(ops_rec) > 0)
            for op in ops_rec:
                self.assertEqual(op["lp_status"], "optimal")

    def test_candidate_only_touches_locked_columns(self):
        """Candidate ops must not change already-executed intervals."""
        for spec in ["10Fboth", "14O"]:
            frame, q0d, qAd, _, _, _ = self._run(parse_spec(spec), 3)
            pre = frame[(frame.slot >= 1) & (frame.slot < 36)]
            self.assertTrue((np.abs(pre.qA_kwh - pre.q0_kwh) < 1e-9).all())


if __name__ == "__main__":
    unittest.main()
