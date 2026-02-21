import numpy as np
from symdisc.discovery.lse import LSE

def test_lse_polynomial_fit_and_jacobians():
    # Points on a circle in R^2
    t = np.linspace(0, 2*np.pi, 64, endpoint=False)
    X = np.stack([np.cos(t), np.sin(t)], axis=1)

    lse = LSE(
        mode="polynomial",
        degree=3,
        include_bias=False,
        lowvar_policy="relative",
        rel_tol=1e-8,
        svd_solver="randomized",
        random_state=0,
    ).fit(X)

    # Constraint Jacobian shape
    Jg = lse.get_feature_jacobian(X)
    assert Jg.ndim == 3 and Jg.shape[0] == X.shape[0]
