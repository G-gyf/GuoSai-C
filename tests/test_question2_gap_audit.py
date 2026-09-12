import unittest
import pandas as pd
from src.optimization.audit_question2_gap import dispatch


class GapAuditTests(unittest.TestCase):
    def test_emergency_charging_relaxation_changes_counterfactual(self):
        # No planned supply or surplus. The relaxed model buys cheap emergency
        # power to charge for the second slot, while the operational direction
        # convention forbids that action.
        f = pd.DataFrame({'load_kwh': [0., 500.], 'pv_kwh': [0., 0.],
                          'grid_kwh': [0., 0.], 'price': [.1, 1.],
                          'soc_start_kwh': [1200., 1200.]})
        relaxed, strict = dispatch(f, False), dispatch(f, True)
        self.assertAlmostEqual(relaxed['total_cost'], 500/.81*.5, places=5)
        self.assertAlmostEqual(strict['total_cost'], 2500., places=5)
        self.assertEqual(relaxed['emergency_charging_slots'], 1)
        self.assertEqual(strict['emergency_charging_slots'], 0)


if __name__ == '__main__':
    unittest.main()
