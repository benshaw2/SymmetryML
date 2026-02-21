#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from symdisc import (
    LSE,
    getExtendedFeatureMatrix,
    discover_symmetry_coeffs,
    generate_euclidean_killing_fields,
)

# ----- Sampling utilities -----------------------------------------------------

def sample_uniform_s2_numpy(n=2000, rng=None):
    """
    Uniformly sample points on S^2 using a simple normalized Gaussian method.
    """
    if rng is None:
        rng = np.random.default_rng()
    X = rng.normal(size=(n, 3))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def sample_uniform_s2_geomstats(n=2000, rng=None):
    """
    Try to uniformly sample points on S^2 using geomstats, if available.
    Falls back to the NumPy sampler if geomstats isn't installed/usable.
    """
    if rng is None:
        rng = np.random.default_rng()

    try:
        import geomstats.backend as gs
        from geomstats.geometry.hypersphere import Hypersphere

        sphere = Hypersphere(dim=2)
        # Many geomstats versions expose random_point
        if hasattr(sphere, "random_point"):
            X = sphere.random_point(n_samples=n)
            return np.array(X, dtype=float)

        # Fallback to NumPy sampler if API differs
        raise AttributeError("geomstats Hypersphere.random_point not found")

    except Exception as e:
        print("geomstats sampling unavailable, falling back to NumPy sampler. Reason:", repr(e))
        return sample_uniform_s2_numpy(n=n, rng=rng)

# ----- Main pipeline ----------------------------------------------------------

def main():
    rng = np.random.default_rng(0)

    # 1) Sample the 2-sphere in R^3
    X = sample_uniform_s2_geomstats(n=4000, rng=rng)  # will fallback if needed
    assert X.shape[1] == 3

    # 2) Visualize (optional)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=1, c=X[:, 2], cmap="jet")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Uniform samples on S^2")
    plt.show(block=False)

    # 3) LSE on polynomial features
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

    # 4) Constraint Jacobians J_g(X): (N, r, d)
    Jg = lse.get_constraint_jacobian(X)

    # 5) Ambient Euclidean Killing fields in R^3
    kvs = generate_euclidean_killing_fields(d=X.shape[1])

    # 6) Build extended feature matrix A and discover invariances
    A, info = getExtendedFeatureMatrix(X, Jg, kvs, normalize_rows=True)
    C, svals = discover_symmetry_coeffs(A, rtol=1e-8)

    print("Extended feature matrix A shape:", A.shape)
    print("Discovered coefficient vectors shape:", C.shape)  # (q, r)
    print("Small singular values:", svals)

    plt.show()  # keep plot window open when running directly

if __name__ == "__main__":
    main()
