"""Acceptance tests for the question-3 second sub-question.

Covers: spec parsing, self-forecast causality/fallback/range, candidate
curve construction, the integer-hour op inside the M612 day runner
(pre-op rows bit-identical to M612, executed intervals never modified,
physical validation), and the recalibration option.
"""
import unittest

import numpy as np
import pandas as pd

from src.optimization.question2 import (
    load_forecast_weekly_persist,
    load_inputs,
)
from src.optimization.question3 import (
    COLUMNS,
    Q3Settings,
    run_day,
    validate_question3,
)
from src.optimization.question3_second_subquestion import (
    HOUR_SLOT,
    build_ops,
    parse_spec,
    potential_daily,
    spec_branch_name,
)
from src.forecasting.question3_self_forecasts import (
    build_self_forecasts,
    pv_hourly_kw,
)
from src.data_pipeline.question3_forecasts import (
    build_issuance_curves,
    load_hourly_issuances,
    load_pv_actuals,
)

DAY = 40  # 2025-02-10


class TestSpecParsing(unittest.TestCase):
    def test_parse_specs(self):
        self.assertEqual(parse_spec("B"), [])
        self.assertEqual(parse_spec("10F"),
                         [{"slot": 60, "info": "self", "hour": 10}])
        self.assertEqual(parse_spec("14S+10F"),
                         [{"slot": 84, "info": "state", "hour": 14},
                          {"slot": 60, "info": "self", "hour": 10}])
        with self.assertRaises(ValueError):
            parse_spec("12F")

    def test_branch_names(self):
        self.assertEqual(spec_branch_name("10F", 20250912, 1.0, False), "10F")
        self.assertEqual(spec_branch_name("10F", 20250913, 1.0, False),
                         "10F_s20250913")
        self.assertEqual(spec_branch_name("10F", 20250912, 0.5, True),
                         "10F_recal_nu0.5")

    def test_hour_slots(self):
        self.assertEqual(HOUR_SLOT[10], 60)
        self.assertEqual(HOUR_SLOT[14], 84)


class TestSelfForecasts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pv = load_pv_actuals()
        cls.pv_h = pv_hourly_kw(cls.pv)
        cls.hourly = load_hourly_issuances()
        cls.pv_max = float(cls.pv.max() * 6.0)

    def test_shape_and_range(self):
        spec = build_self_forecasts(10, 1, 12, self.pv, self.pv_h, self.hourly)
        curve, hourly_kw = spec["curve_kwh"], spec["hourly_kw"]
        self.assertEqual(curve.shape, (365, 144))
        self.assertTrue(np.isnan(curve[:, :60]).all())
        self.assertTrue(np.isfinite(curve[DAY, 60:]).all())
        self.assertTrue((curve[DAY, 60:] >= 0).all())
        self.assertTrue(np.isnan(hourly_kw[DAY, :10]).all())
        self.assertTrue(np.isfinite(hourly_kw[DAY, 10:]).all())
        self.assertTrue((hourly_kw[DAY, 10:] <= self.pv_max + 1e-9).all())

    def test_fallback_few_samples(self):
        spec = build_self_forecasts(10, 1, 12, self.pv, self.pv_h, self.hourly)
        # day 5 has fewer than 14 training days -> official fallback
        self.assertEqual(spec["fallback"][5], 1)
        # day 40 has a full window: either fitted or validation-gate fallback
        self.assertIn(spec["fallback"][DAY], (0, 3))

    def test_causality_under_future_perturbation(self):
        base = build_self_forecasts(14, 2, 18, self.pv, self.pv_h, self.hourly)
        pv2 = self.pv.copy()
        pv2[DAY + 2:] = pv2[DAY + 2:] * 2.0
        pv_h2 = pv_hourly_kw(pv2)
        pert = build_self_forecasts(14, 2, 18, pv2, pv_h2, self.hourly)
        # forecasts for days <= DAY+1 must not change (training windows only
        # touch unperturbed days and their own observed anchor is intact)
        np.testing.assert_allclose(pert["curve_kwh"][:DAY + 2, 84:],
                                   base["curve_kwh"][:DAY + 2, 84:], rtol=0,
                                   atol=0)
        # day DAY+2 onwards change: own anchor is perturbed for DAY+2 and the
        # rolling training windows of later days include perturbed days
        self.assertTrue(
            np.nanmax(np.abs(pert["curve_kwh"][DAY + 2:, 84:]
                             - base["curve_kwh"][DAY + 2:, 84:])) > 0)


class TestCandidateOpsInDayRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dates, cls.load, cls.pv, cls.prices = load_inputs()
        cls.fl = load_forecast_weekly_persist(cls.load)
        cls.fc = build_issuance_curves()
        cls.settings = Q3Settings(strategy="M612")
        cls.date = cls.dates[DAY]
        cls.energy = 2100.0

    def _frame(self, ops):
        rows, _, diag, _, _, _ = run_day(
            DAY, self.date, self.load[DAY], self.pv[DAY], self.prices,
            self.fl, self.fc, self.energy, self.settings, self.load, self.pv,
            candidate_ops=ops,
        )
        return pd.DataFrame.from_records(rows, columns=COLUMNS), diag

    def test_base_equals_m612_path(self):
        frame, _ = self._frame(None)
        checks = validate_question3(frame)
        self.assertTrue(checks["passed"])

    def test_oracle_op_pre_slot_identical_and_valid(self):
        base, _ = self._frame(None)
        ops = build_ops(parse_spec("10O"))
        frame, diag = self._frame(ops)
        checks = validate_question3(frame)
        self.assertTrue(checks["passed"])
        # rows before the candidate slot are bit-identical to M612
        for col in ["qA_kwh", "soc_start_kwh", "charge_kwh", "discharge_kwh",
                    "emergency_kwh", "total_cost"]:
            np.testing.assert_array_equal(
                frame.loc[frame.slot < 60, col].to_numpy(),
                base.loc[base.slot < 60, col].to_numpy())
        # the locked 10:00-12:00 purchases differ (oracle knows actuals)
        self.assertTrue(
            (frame.loc[(frame.slot >= 60) & (frame.slot < 72), "qA_kwh"]
             .to_numpy() != base.loc[(base.slot >= 60) & (base.slot < 72),
                                     "qA_kwh"].to_numpy()).any())
        # executed intervals never modified: 6:00-10:00 purchases equal q0-adjustment
        # from the 6:00 update in both paths
        np.testing.assert_array_equal(
            frame.loc[(frame.slot >= 36) & (frame.slot < 60), "qA_kwh"]
            .to_numpy(),
            base.loc[(base.slot >= 36) & (base.slot < 60), "qA_kwh"]
            .to_numpy())
        self.assertEqual(diag["candidate_ops"],
                         [{"update_slot": 60, "info": "oracle",
                           "recalibrate": False}])

    def test_self_op_14_and_combo(self):
        ops14 = build_ops(parse_spec("14F"))
        frame, diag = self._frame(ops14)
        self.assertTrue(validate_question3(frame)["passed"])
        combo = build_ops(parse_spec("10F+14S"))
        frame2, diag2 = self._frame(combo)
        self.assertTrue(validate_question3(frame2)["passed"])
        self.assertEqual([o["update_slot"] for o in diag2["candidate_ops"]],
                         [60, 84])

    def test_recalibrate_op_valid(self):
        ops = build_ops(parse_spec("10F"))
        for op in ops:
            op["recalibrate"] = True
        frame, diag = self._frame(ops)
        self.assertTrue(validate_question3(frame)["passed"])
        self.assertTrue(any(c.get("update_slot") == 60
                            for c in diag["calibrations"]))

    def test_potential_daily_telescopes(self):
        daily = pd.DataFrame({
            "total_cost": [100.0, 200.0],
            "soc_start_kwh": [5000.0, 4800.0],
            "soc_end_kwh": [4800.0, 4600.0],
        })
        nu = 0.4
        cbar = potential_daily(daily, nu)
        # sum telescopes to total cost + nu*(E0 - ET)
        self.assertAlmostEqual(cbar.sum(), 300.0 + 0.4 * (5000 - 4600), places=9)


if __name__ == "__main__":
    unittest.main()
