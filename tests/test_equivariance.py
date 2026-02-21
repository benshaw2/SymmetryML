import numpy as np
from symdisc.discovery.equivariance import EquivariantDiscovery
from symdisc.discovery.builders import getEquivariantResidualMatrix
from symdisc.vector_fields import generate_euclidean_killing_fields

def test_equivariant_residual_and_fit():
    # Identity map F(x)=x, so J_F is I (p=d)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 3))

    F = lambda Xb: Xb
    JF = lambda Xb: np.tile(np.eye(3)[None, :, :], (Xb.shape[0], 1, 1))

    vf_in = generate_euclidean_killing_fields(d=3)
    vf_out = generate_euclidean_killing_fields(d=3)  # aligned for identity

    M, info = getEquivariantResidualMatrix(
        X, F, JF, vf_in, vf_out, coupling="aligned", normalize_rows=True
    )
    assert M.ndim == 2 and M.shape[0] == info["N"] * info["p"]

    # Fit PCA low-variance on M
    ed = EquivariantDiscovery(
        coupling="aligned",
        lowvar_policy="relative",
        rel_tol=1e-8,
        svd_solver="randomized",
        random_state=0,
    ).fit(X, F, JF, vf_in, vf_out, normalize_rows=True)

    assert ed.n_components_found_ >= 1
    assert ed.coefficients_.shape[0] == len(vf_in)
