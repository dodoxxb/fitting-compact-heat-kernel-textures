"""Construct the manuscript's full-spectrum, lumped-mass FEM counterexample.

A 12-vertex, 10-triangle open strip is assembled from element gradients. The
source-diagonal normalization can exceed one; symmetric PSD normalization
obeys its Cauchy--Schwarz bound. This is a counterexample to a general claim,
not a claim about every closed surface or an argument for replacing a trained
representation without retraining. No GPU, external mesh or stored matrix is
used. Adapted from the study's original Round25 construction.
"""
import json
import unittest
import numpy as np
from scipy.linalg import eigh


def construct():
    xy=np.array([[x,y] for y in (0.,.2) for x in np.linspace(0,1,6)])
    faces=[]
    for i in range(5):
        faces.extend([[i,i+1,i+6],[i+1,i+7,i+6]])
    stiffness=np.zeros((len(xy),len(xy)))
    mass=np.zeros(len(xy))
    for triangle in faces:
        points=xy[triangle]
        area=abs(np.linalg.det(np.column_stack([points[1]-points[0],points[2]-points[0]])))/2
        gradients=np.linalg.inv(np.column_stack([np.ones(3),points]))[1:]
        stiffness[np.ix_(triangle,triangle)]+=area*gradients.T@gradients
        mass[triangle]+=area/3
    eigenvalues,eigenvectors=eigh(stiffness,np.diag(mass))
    heat=(eigenvectors*np.exp(-.1*eigenvalues))@eigenvectors.T
    diagonal=np.diag(heat)
    source_normalized=heat/diagonal[None,:]
    symmetric=heat/np.sqrt(np.outer(diagonal,diagonal))
    location=np.unravel_index(source_normalized.argmax(),source_normalized.shape)
    bound=np.sqrt(diagonal[:,None]/diagonal[None,:])
    return dict(vertices=xy,triangles=np.asarray(faces),stiffness=stiffness,mass=mass,
                eigenvalues=eigenvalues,heat=heat,source_normalized=source_normalized,
                symmetric=symmetric,bound=bound,location=location)


class FEMNormalizationTest(unittest.TestCase):
    def test_full_fem_source_normalization(self):
        data=construct()
        self.assertEqual(data['vertices'].shape,(12,2))
        self.assertEqual(data['triangles'].shape,(10,3))
        self.assertTrue((data['mass']>0).all())
        np.testing.assert_allclose(data['stiffness'].sum(axis=1),0,atol=1e-12)
        self.assertGreaterEqual(data['eigenvalues'].min(),-1e-12)
        self.assertTrue(np.all(abs(data['source_normalized'])<=data['bound']+1e-12))
        self.assertGreater(data['source_normalized'].max(),1.)
        self.assertAlmostEqual(float(data['source_normalized'].max()),1.0977412536731928,places=10)
        self.assertLessEqual(float(abs(data['symmetric']).max()),1+1e-12)


if __name__=='__main__':
    result=construct()
    print(json.dumps(dict(mesh_vertices=len(result['vertices']),mesh_triangles=len(result['triangles']),
                         time=.1,maximum_ratio=float(result['source_normalized'].max()),
                         query_vertex=int(result['location'][0]),source_vertex=int(result['location'][1]),
                         symmetric_max_abs=float(abs(result['symmetric']).max())),indent=2))
    unittest.main()
