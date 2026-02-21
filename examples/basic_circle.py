#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from symdisc import (
    LSE,
    getExtendedFeatureMatrix,
    discover_symmetry_coeffs,
    generate_euclidean_killing_fields,
)

def main():
    rng = np.random.default_rng(0)

    # ---- Circle in R^3 (z=0), param t ~ N(0,1)
    t = rng.normal(0.0, 1.0, size=1000)
    X = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])  # (N,3)

    # Visualize (optional)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, c=t, cmap="jet")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    plt.show(block=False)

    # ---- LSE fit (polynomial features)
    lse = LSE(
        mode="polynomial",
        degree=3,
        include_bias=False,
        use_incremental=False,
        lowvar_policy="relative",
        rel_tol=1e-8,
        n_components=None,
        svd_solver="randomized",
        random_state=0,
    ).fit(X)

    # Constraint Jacobians J_g(X): (N, r, d)
    Jg = lse.get_constraint_jacobian(X)

    # Euclidean Killing fields in ambient R^3
    kvs = generate_euclidean_killing_fields(d=X.shape[1])

    # Build extended feature matrix A for invariance discovery
    A, info = getExtendedFeatureMatrix(X, Jg, kvs, normalize_rows=True)
    # A: shape (N*m, q). Here m=r (# constraints), q=#vector fields

    # SVD-based symmetry coefficients (columns)
    C, svals = discover_symmetry_coeffs(A, rtol=1e-8)
    print("Discovered coefficient vectors shape:", C.shape)
    print("Small singular values:", svals)

    plt.show()  # keep plot open when running as script

if __name__ == "__main__":
    main()
