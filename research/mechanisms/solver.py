"""Box coordinate descent: cache column norms and share RGB sparse reads.

Specializes the existing independent [-1,1] colours + [0,1] background
problem. No fastmath, reordering of coordinates, or changed stopping rule.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def coordinate_rgb(ptr, idx, data, C, residual, background, max_sweeps):
    inverse = np.zeros(len(C))
    for j in range(len(C)):
        h = 0.
        for k in range(ptr[j], ptr[j+1]):
            h += data[k]*data[k]
        if h >= 1e-20:
            inverse[j] = h  # Keep division, avoiding reciprocal rounding changes.
    history = np.zeros((max_sweeps, 3))
    for sweep in range(max_sweeps):
        largest = 0.
        for j in range(len(C)):
            h = inverse[j]
            if h == 0.:
                continue
            g0 = 0.; g1 = 0.; g2 = 0.
            for k in range(ptr[j], ptr[j+1]):
                i = idx[k]; a = data[k]
                g0 += a*residual[i, 0]
                g1 += a*residual[i, 1]
                g2 += a*residual[i, 2]
            lower = 0. if j == background else -1.
            v0 = min(1., max(lower, C[j, 0]-g0/h))
            v1 = min(1., max(lower, C[j, 1]-g1/h))
            v2 = min(1., max(lower, C[j, 2]-g2/h))
            d0 = v0-C[j, 0]; d1 = v1-C[j, 1]; d2 = v2-C[j, 2]
            C[j, 0] = v0; C[j, 1] = v1; C[j, 2] = v2
            largest = max(largest, abs(d0), abs(d1), abs(d2))
            for k in range(ptr[j], ptr[j+1]):
                i = idx[k]; a = data[k]
                residual[i, 0] += a*d0
                residual[i, 1] += a*d1
                residual[i, 2] += a*d2
        history[sweep, 0] = np.sum(residual*residual)
        history[sweep, 1] = largest
        history[sweep, 2] = sweep+1
        if largest < 1e-6:
            return history[:sweep+1]
    return history


def solve(A, Y, initial, sweeps=100):
    if sweeps < 1:
        raise ValueError('sweeps must be positive')
    A = A.tocsc().astype(np.float64)
    C = np.array(initial, dtype=np.float64, copy=True)
    assert C.shape == (A.shape[1], 3)
    if not (np.isfinite(A.data).all() and np.isfinite(Y).all() and np.isfinite(C).all()):
        raise ValueError('non-finite conditional fitting input')
    residual = A@C-Y
    history = coordinate_rgb(A.indptr, A.indices, A.data, C, residual, len(C)-1, sweeps)
    return C, history
