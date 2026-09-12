"""Behavioral checks for physical feasibility, causal information, and billing."""
import unittest
import numpy as np
import pandas as pd

from src.optimization.question2 import (
    EMAX, EMIN, ETA, S, Settings, execute_slot, forecasts, planning_net,
    solve_plan, summarize_emergency,
)


class Question2Tests(unittest.TestCase):
    def test_forecasts_and_risk_ignore_current_and_future_actuals(self):
        rng = np.random.default_rng(11)
        l, v = rng.uniform(100, 200, (40, 144)), rng.uniform(0, 100, (40, 144))
        fl, fv = forecasts(l, v)
        n, count = planning_net(31, l, v, fl, fv, Settings())
        l[31:], v[31:] = 1e8, 1e7
        fl2, fv2 = forecasts(l, v)
        n2, count2 = planning_net(31, l, v, fl2, fv2, Settings())
        np.testing.assert_allclose(n, n2)
        self.assertEqual(count, 21)
        self.assertEqual(count2, 21)
        np.testing.assert_allclose(fl[31], l[24:31].mean(axis=0))

    def test_residual_warmup_calendar(self):
        l, v = np.ones((35,144)), np.zeros((35,144))
        fl, fv = forecasts(l, v)
        for d, expected in [(7,0),(8,1),(27,20),(28,21),(31,21)]:
            self.assertEqual(planning_net(d,l,v,fl,fv,Settings())[1],expected)

    def test_controller_boundaries_and_emergency(self):
        for load,pv,g,e,r in [(1000,0,0,EMIN,EMIN),(0,1000,0,EMAX,EMIN),
                              (500,0,0,6000,5900),(0,2000,0,6000,1200),
                              (100,0,500,10800,1200)]:
            c,d,b,u,end = execute_slot(load,pv,g,e,r)
            self.assertAlmostEqual(g+pv+d+b,load+c+u)
            self.assertGreaterEqual(end,EMIN)
            self.assertLessEqual(end,EMAX)
            self.assertLessEqual(max(c,d),S)
            self.assertEqual(c*d,0)
            self.assertEqual(c*b,0)
        self.assertEqual(execute_slot(1000,0,0,EMIN,EMIN)[2],1000)
        self.assertAlmostEqual(execute_slot(500,0,0,6000,5900)[1],90)

    def test_lp_arbitrage_and_initial_soc(self):
        net, p = np.array([0.,800.]), np.array([.2,1.])
        x, _ = solve_plan(net,p,EMIN,0)
        self.assertGreater(x[1,0],0)
        self.assertGreater(x[2,1],0)
        self.assertLess(np.dot(p,x[0]),800)
        np.testing.assert_allclose(x[0]-x[1]+x[2]-x[3],net,atol=1e-5)
        self.assertLess(np.minimum(x[1],x[2]).max(),1e-5)

    def test_emergency_runs_preserve_midnight_and_zero_day(self):
        f = pd.DataFrame({'date':np.repeat(pd.date_range('2025-02-01',periods=2),144),
                          'emergency_kwh':np.zeros(288)})
        f.loc[[0,1,143],'emergency_kwh'] = [1,2,4]
        self.assertEqual(summarize_emergency(f),[
            ['2025-02-01','00:00-00:20',3.],
            ['2025-02-01','23:50-24:00',4.],['2025-02-02','无',0.]])


if __name__ == '__main__':
    unittest.main()
