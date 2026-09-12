import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.optimization.question2 import EMIN,S
from src.optimization.question2_perfect_foresight import solve_horizon


class PerfectForesightTests(unittest.TestCase):
    def test_known_single_cheap_slot_optimum_and_duality(self):
        load=np.zeros((1,144)); load[0,-1]=800
        price=np.ones(144); price[0]=.2
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            s=solve_horizon(pd.date_range('2025-01-01',periods=1),load,np.zeros_like(load),
                            price,EMIN,'toy',Path(directory))
        expected=.2*S+(800-.81*S)
        self.assertAlmostEqual(s['cost_lower_bound_yuan'],expected,places=5)
        self.assertLess(s['duality_gap_yuan'],1e-6)
        self.assertEqual(s['emergency_kwh'],0)
        self.assertTrue(s['validation']['passed'])

    def test_initial_energy_is_not_reset_at_midnight(self):
        load=np.zeros((2,144)); load[1,0]=90
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            s=solve_horizon(pd.date_range('2025-01-01',periods=2),load,np.zeros_like(load),
                            np.ones(144),EMIN+100,'toy',Path(directory))
        self.assertAlmostEqual(s['cost_lower_bound_yuan'],0,places=6)
        self.assertAlmostEqual(s['final_soc_kwh'],EMIN,places=6)


if __name__=='__main__': unittest.main()
