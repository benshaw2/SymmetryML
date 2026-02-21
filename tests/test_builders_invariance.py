import numpy as np
from symdisc.discovery.builders import getExtendedFeatureMatrix
from symdisc.vector_fields import generate_euclidean_killing_fields

def test_extended_feature_matrix_shapes():
    # Simple circle in R^3: points with z=0, x^2 + y^2 = 1
    theta = np.linspace(0, 2*np.pi, 50, endpoint=False)
    X = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)

    # Suppose constraints g = [x^2 + y^2 - 1] only; Jacobian row = [2x, 2y, 0]
    def J(Xb):
        Xb = np.asarray(Xb)
        N = Xb.shape[0]
        Jv = np.zeros((N, 1, 3))
        Jv[:, 0, 0] = 2*Xb[:, 0]
        Jv[:, 0, 1] = 2*Xb[:, 1]
        return Jv

    kvs = generate_euclidean_killing_fields(d=3)
    A, (N, m, q) = getExtendedFeatureMatrix(X, J, kvs, normalize_rows=True)
    assert A.shape == (N*m, q)
    assert N == X.shape[0] and m == 1 and q == len(kvs)
