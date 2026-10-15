"""Independent kernel identities, covariance counterexample and invalid inputs."""
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research/round49"))
import numpy as np
from fixed_time_features import diffuse,align


class Contracts(unittest.TestCase):
    def test_preserves_raw_kernel_under_alignment(self):
        rng=np.random.default_rng(490012)
        for rank in (2,7,16):
            phi,_=np.linalg.qr(rng.normal(size=(40,rank)));ref=rng.normal(size=phi.shape);lam=np.linspace(0,5,rank)
            z=diffuse(phi,lam,.7);aligned,_=align(ref,z,np.linspace(.5,2,40))
            truth=(phi*np.exp(-.7*lam))@phi.T
            self.assertLess(np.linalg.norm(aligned@aligned.T-truth)/np.linalg.norm(truth),1e-12)

    def test_mixed_basis_with_diagonal_weights_is_different(self):
        angle=.35;q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);d=np.diag(np.exp(-.7*np.array([1.,4.])))
        self.assertGreater(np.linalg.norm(q@d@q.T-d),.1)
        self.assertLess(np.linalg.norm(q@(q.T@d@q)@q.T-d),1e-12)

    def test_rank_deficient_alignment(self):
        z=np.zeros((5,3));result,_=align(z,z,np.ones(5));self.assertTrue(np.array_equal(z,result))

    def test_invalid_contracts(self):
        for lam,t in (([-1,1],.1),([0,1],-1),([0,np.nan],1)):
            with self.assertRaises(ValueError):diffuse(np.eye(2),lam,t)
        with self.assertRaises(ValueError):align(np.eye(2),np.eye(2),np.zeros(2))


if __name__=='__main__':unittest.main()
