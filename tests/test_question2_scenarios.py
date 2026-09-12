import unittest
import numpy as np
from src.optimization.question2 import forecasts, execute_slot, ETA, EMIN
from src.optimization.question2_scenarios import scenarios, stochastic_plan, value_reserves


class ScenarioTests(unittest.TestCase):
    def test_joint_scenarios_are_causal(self):
        r=np.random.default_rng(8)
        l=r.uniform(100,200,(40,144)); v=r.uniform(0,50,(40,144))
        fl,fv=forecasts(l,v)
        s=scenarios(31,l,v,fl,fv)
        l[31:]=1e9; v[31:]=1e8
        fl2,fv2=forecasts(l,v)
        np.testing.assert_allclose(s,scenarios(31,l,v,fl2,fv2))
        self.assertEqual(s.shape,(21,144))

    def test_newsvendor_eighty_percent_quantile(self):
        # Last-slot inventory at its minimum, no useful terminal credit: no
        # storage recourse. Optimal g is a sample 80th-percentile solution.
        net=np.array([[100.],[200.],[300.],[400.],[500.]])
        g,tr,info=stochastic_plan(net,np.array([1.]),EMIN,0)
        self.assertGreaterEqual(g[0],400-1e-6)
        self.assertLessEqual(g[0],500+1e-6)
        self.assertAlmostEqual(info['direction_feasible_objective'],500,places=4)

    def test_scenario_feasibility_and_direction(self):
        net=np.array([[-500.,600.,900.],[200.,800.,-400.],[150.,300.,1100.]])
        p=np.array([.2,1.,.5])
        g,tr,info=stochastic_plan(net,p,1800,.2/ETA)
        r=net-g
        self.assertTrue((tr[:,1]<=np.maximum(-r,0)+1e-5).all())
        self.assertTrue((tr[:,2]<=np.maximum(r,0)+1e-5).all())
        np.testing.assert_allclose(g+tr[:,0]-tr[:,1]+tr[:,2]-tr[:,3],net,atol=1e-4)
        self.assertGreaterEqual(info['direction_feasible_objective']+1e-5,info['relaxed_objective'])

    def test_value_controller_reserves_for_expensive_future(self):
        net=np.array([[500.,500.]])
        r=value_reserves(net,np.zeros(2),np.array([.1,1.]),0,step=5)
        self.assertGreater(r[0],EMIN+500/ETA-10)
        self.assertEqual(r[1],EMIN)
        energy=EMIN+500/ETA
        first=execute_slot(500,0,0,energy,r[0])
        self.assertLess(first[1],10)
        last=execute_slot(500,0,0,first[-1],r[1])
        self.assertGreater(last[1],490)

    def test_fixed_branch_lp_has_valid_lower_bound_and_physical_actions(self):
        net=np.array([[100.,900.,0.],[300.,200.,-400.],[0.,500.,1200.]])
        p=np.array([.2,1.,.4])
        g,tr,info=stochastic_plan(net,p,1600,.2/ETA,branch_only=True)
        self.assertEqual(info['method'],'fixed_direction_LP_with_global_relaxation_bound')
        self.assertLessEqual(info['relaxed_objective'],info['direction_feasible_objective']+1e-5)
        np.testing.assert_allclose(g+tr[:,0]-tr[:,1]+tr[:,2]-tr[:,3],net,atol=1e-5)
        np.testing.assert_allclose(tr[:,4],1600+np.cumsum(ETA*tr[:,1]-tr[:,2]/ETA,axis=1),atol=1e-5)
        self.assertLess(np.max(np.minimum(tr[:,0],tr[:,1])),1e-5)


if __name__=='__main__': unittest.main()
