"""
Level-Set Estimation (LSE).

This module provides the `LSE` class that:
  - Builds/accepts a feature matrix Φ(X) from data X via:
      * precomputed features (mode='precomputed'),
      * PolynomialFeatures (mode='polynomial'),
      * user-supplied callable (mode='callable').
  - Fits PCA (or IncrementalPCA) to Φ(X) and identifies low-variance components,
    which act as approximate constraints g_ell(x) = w_ell^T φ(x).
  - Exposes analytic/numeric feature Jacobians J_φ(X), and constraint Jacobians
    J_g(X) = J_φ(X)^T W for downstream symmetry/equivariance discovery.

Distances along the level set are left as stubs for now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.preprocessing import PolynomialFeatures

# Public types
FeatureMode = Literal["precomputed", "polynomial", "callable"]
LowVarPolicy = Literal["count", "relative", "absolute", "eigengap"]

__all__ = [
    "LSE",
    "FeatureMode",
    "LowVarPolicy",
]


def _iterate_batches(N: int, batch_size: Optional[int]):
    """
    Yield batch slices for range(N) using an optional batch_size.
    """
    if batch_size is None or batch_size >= N:
        yield slice(0, N)
    else:
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            yield slice(start, end)


def _numeric_feature_jacobian(
    X: np.ndarray,
    feature_func: Callable[[np.ndarray], np.ndarray],
    h: Optional[np.ndarray] = None,
    method: Literal["central", "forward"] = "central",
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """
    Numerically estimate J_phi(x) for a general feature map phi using finite differences.

    Args:
        X: (N, d)
        feature_func: callable (N,d)->(N,p) or (1,d)->(1,p)
        h: optional per-entry step sizes, shape (N, d)
        method: 'central' (default) or 'forward'
        batch_size: optional batching for memory

    Returns:
        J_phi: shape (N, p, d)
    """
    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape

    # Probe output dimension p by one call
    F0 = feature_func(X[:1])  # (1, p)
    F0 = np.asarray(F0, dtype=np.float64)
    if F0.ndim != 2:
        raise ValueError("feature_func must return (N, p).")
    p = F0.shape[1]

    if h is None:
        # Per-dimension step sizes: sqrt(machine_eps) * (1 + |x|)
        eps = np.finfo(np.float64).eps
        h = np.sqrt(eps) * (1.0 + np.abs(X))
    else:
        h = np.asarray(h, dtype=np.float64)
        if h.shape != X.shape:
            raise ValueError("h must have the same shape as X (N, d).")

    J = np.zeros((N, p, d), dtype=np.float64)

    for sl in _iterate_batches(N, batch_size):
        Xb = X[sl]
        hb = h[sl]

        # Base evaluation for forward method
        if method == "forward":
            Fb = feature_func(Xb)  # (B, p)

        for k in range(d):
            e_k = np.zeros_like(Xb)
            e_k[:, k] = hb[:, k]

            if method == "central":
                Fp = feature_func(Xb + e_k)  # (B, p)
                Fm = feature_func(Xb - e_k)  # (B, p)
                denom = (2.0 * hb[:, k])[:, None]  # (B, 1)
                J[sl, :, k] = (Fp - Fm) / denom
            elif method == "forward":
                Fp = feature_func(Xb + e_k)  # (B, p)
                denom = hb[:, k][:, None]
                J[sl, :, k] = (Fp - Fb) / denom
            else:
                raise ValueError("Unknown finite-difference method.")
    return J


def _poly_feature_jacobian_batch(
    Xb: np.ndarray, poly: PolynomialFeatures
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytically compute:
      - Fb = poly.transform(Xb)    -> (B, p)
      - J_phi_b = d phi / d x      -> (B, p, d)

    Uses the monomial exponents in poly.powers_.
    """
    Xb = np.asarray(Xb, dtype=np.float64)
    B, d = Xb.shape
    Fb = poly.transform(Xb)  # (B, p)
    powers = poly.powers_  # (p, d), non-negative integers
    p = powers.shape[0]

    # Build Jacobian by derivative of each monomial term.
    # For feature t with exponents a = powers[t], the derivative wrt x_k is:
    #   ∂/∂x_k ∏_j x_j^{a_j} = a_k * ∏_j x_j^{a_j - δ_{jk}}
    # We compute this directly; handles zeros safely (no dividing by x_k).
    Jb = np.ones((B, p, d), dtype=np.float64)

    for k in range(d):
        # For each feature, exponent used for derivative is a_k - 1
        exp_k = powers[:, k] - 1  # (p,)
        # Prepare product over all dims with exponents possibly adjusted at k
        prod = np.ones((B, p), dtype=np.float64)
        for j in range(d):
            exp_j = powers[:, j].copy()
            if j == k:
                exp_j = exp_k
            # Negative exponents mean derivative is zero (because a_k == 0)
            # We'll set x**neg = 0 for those columns.
            mask_neg = exp_j < 0
            exp_j_clipped = np.where(mask_neg, 0, exp_j)
            term = Xb[:, [j]] ** exp_j_clipped[None, :]  # (B, p)
            term[:, mask_neg] = 0.0
            prod *= term
        # Multiply by a_k
        ak = powers[:, k][None, :]  # (1, p)
        Jb[:, :, k] = prod * ak

    return Fb, Jb


@dataclass
class LSE:
    """
    Level-Set Estimation via low-variance directions in a feature matrix.

    - .fit() computes PCA/IncrementalPCA on features φ(X).
    - Low-variance directions (constraints) are selected by a policy.
    - If a feature Jacobian J_φ(x) is available (analytic or numeric),
      we expose constraint Jacobians: J_g(x) = J_φ(x)^T W.

    Distances between *points on the level set* are left as stubs for now.
    """

    # Feature configuration
    mode: FeatureMode = "precomputed"

    # Polynomial features (if mode='polynomial')
    degree: int = 2
    include_bias: bool = False
    interaction_only: bool = False
    order: Literal["C", "F"] = "C"

    # Callable feature map (if mode='callable')
    feature_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
    feature_jacobian: Optional[Callable[[np.ndarray], np.ndarray]] = None
    # Numeric Jacobian fallback for callable features
    numeric_jacobian: bool = False
    fd_method: Literal["central", "forward"] = "central"

    # PCA settings
    use_incremental: bool = False
    batch_size: Optional[int] = None
    n_components: Optional[int] = None
    svd_solver: Literal["auto", "full", "arpack", "randomized"] = "auto"
    random_state: Optional[int] = None

    # Low-variance selection
    lowvar_policy: LowVarPolicy = "relative"
    n_small: Optional[int] = None  # if policy='count'
    rel_tol: float = 1e-6  # if policy='relative' (relative to largest eigenvalue)
    abs_tol: Optional[float] = None  # if policy='absolute'
    eigengap_k: Optional[int] = None  # if policy='eigengap': pick the k smallest by largest gap

    # Internal state
    fitted_: bool = field(default=False, init=False)
    poly_: Optional[PolynomialFeatures] = field(default=None, init=False)
    pca_: Optional[Union[PCA, IncrementalPCA]] = field(default=None, init=False)
    feature_names_: Optional[np.ndarray] = field(default=None, init=False)  # (p,)
    X_: Optional[np.ndarray] = field(default=None, init=False)  # (N, d) if available
    F_: Optional[np.ndarray] = field(default=None, init=False)  # (N, p) if stored (precomputed or small)
    constraint_weights_: Optional[np.ndarray] = field(default=None, init=False)  # (p, r)
    lowvar_indices_: Optional[np.ndarray] = field(default=None, init=False)  # indices into components_
    p_: Optional[int] = field(default=None, init=False)
    d_: Optional[int] = field(default=None, init=False)

    # =========================
    # Core fitting
    # =========================
    def fit(self, X: Optional[np.ndarray] = None, F: Optional[np.ndarray] = None):
        """
        Fit PCA on features φ(X) (or on precomputed features F), then select low-variance directions.

        Args:
            X: (N, d) data matrix (required if mode != 'precomputed')
            F: (N, p) precomputed feature matrix (required if mode == 'precomputed')

        Sets:
            - self.pca_, self.feature_names_, self.constraint_weights_, self.lowvar_indices_
            - self.X_ (if X provided), self.F_ (if small or precomputed)
        """
        if self.mode == "precomputed":
            if F is None:
                raise ValueError("mode='precomputed' requires F.")
            F = np.asarray(F, dtype=np.float64)
            _, p = F.shape
            self.p_ = p
            # We cannot construct J_φ without additional info
            self.feature_names_ = np.array([f"f{j}" for j in range(p)])
            feature_iter = (F,)  # single batch
            self.X_ = X  # may be None

        elif self.mode == "polynomial":
            if X is None:
                raise ValueError("mode='polynomial' requires X.")
            X = np.asarray(X, dtype=np.float64)
            _, d = X.shape
            self.X_ = X
            self.d_ = d
            self.poly_ = PolynomialFeatures(
                degree=self.degree,
                include_bias=self.include_bias,
                interaction_only=self.interaction_only,
                order=self.order,
            )
            # Fit to populate powers_
            self.poly_.fit(np.zeros((1, d)))
            powers = self.poly_.powers_
            p = powers.shape[0]
            self.p_ = p
            # Synthesize feature names (best-effort)
            names = []
            for t in range(p):
                monom = []
                for j in range(d):
                    a = powers[t, j]
                    if a == 0:
                        continue
                    monom.append(f"x{j}^{a}" if a > 1 else f"x{j}")
                if len(monom) == 0:
                    names.append("1" if self.include_bias else "<??>")
                else:
                    names.append("*".join(monom))
            self.feature_names_ = np.array(names)
            feature_iter = self._gen_poly_batches()

        elif self.mode == "callable":
            if X is None:
                raise ValueError("mode='callable' requires X.")
            if self.feature_func is None:
                raise ValueError("mode='callable' requires feature_func.")
            X = np.asarray(X, dtype=np.float64)
            _, d = X.shape
            self.X_ = X
            self.d_ = d
            # Probe dimensionality p
            F0 = self.feature_func(X[:1])
            F0 = np.asarray(F0, dtype=np.float64)
            if F0.ndim != 2:
                raise ValueError("feature_func must return (N, p).")
            p = F0.shape[1]
            self.p_ = p
            self.feature_names_ = np.array([f"φ{j}" for j in range(p)])
            feature_iter = self._gen_callable_batches()

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # PCA / IncrementalPCA
        if self.use_incremental:
            ipca = IncrementalPCA(
                n_components=self.n_components, batch_size=self.batch_size, copy=False
            )
            for Fb in feature_iter:
                ipca.partial_fit(Fb)
            self.pca_ = ipca
        else:
            # Collect features (if not already a single batch)
            if isinstance(feature_iter, tuple):
                F_all = feature_iter[0]
            else:
                F_list = [Fb for Fb in feature_iter]
                F_all = np.vstack(F_list)
            self.F_ = F_all  # store in-memory for small/medium cases
            self.pca_ = PCA(
                n_components=self.n_components, svd_solver=self.svd_solver, random_state=self.random_state
            ).fit(F_all)

        # Determine low-variance components (in PCA ordering: high->low variance)
        low_idx = self._select_low_variance_indices()
        self.lowvar_indices_ = low_idx

        # Extract loadings for low-variance components.
        # sklearn: components_.shape = (n_components_kept, p), ordered high->low variance.
        comps = self.pca_.components_  # (k, p)
        W_small = comps[low_idx, :]  # (r, p)
        # We want columns = constraint vectors in feature space (p, r)
        self.constraint_weights_ = W_small.T

        self.fitted_ = True
        return self

    # =========================
    # Feature generators (batches)
    # =========================
    def _gen_poly_batches(self):
        assert self.poly_ is not None and self.X_ is not None
        X = self.X_
        for sl in _iterate_batches(len(X), self.batch_size):
            Xb = X[sl]
            Fb, _ = _poly_feature_jacobian_batch(Xb, self.poly_)  # only need Fb here
            yield Fb

    def _gen_callable_batches(self):
        assert self.X_ is not None and self.feature_func is not None
        X = self.X_
        for sl in _iterate_batches(len(X), self.batch_size):
            Xb = X[sl]
            Fb = self.feature_func(Xb)
            Fb = np.asarray(Fb, dtype=np.float64)
            yield Fb

    # =========================
    # Low-variance selection
    # =========================
    def _select_low_variance_indices(self) -> np.ndarray:
        """
        Choose the indices (row indices into pca_.components_) of the
        low-variance directions according to policy.
        Note: components_ are ordered high -> low variance.
        """
        ev = np.asarray(self.pca_.explained_variance_, dtype=np.float64)  # length k
        k = ev.shape[0]

        if self.lowvar_policy == "count":
            if not self.n_small:
                raise ValueError("lowvar_policy='count' requires n_small.")
            n = min(self.n_small, k)
            # Low-variance are the *last* n components
            idx = np.arange(k - n, k, dtype=int)
            return idx

        elif self.lowvar_policy == "relative":
            if self.rel_tol is None:
                raise ValueError("lowvar_policy='relative' requires rel_tol.")
            vmax = ev[0] if k > 0 else 1.0
            mask = ev <= (self.rel_tol * vmax)
            idx = np.where(mask)[0]
            if idx.size == 0:
                # fallback: take the smallest one
                idx = np.array([k - 1], dtype=int)
            return idx

        elif self.lowvar_policy == "absolute":
            if self.abs_tol is None:
                raise ValueError("lowvar_policy='absolute' requires abs_tol.")
            mask = ev <= self.abs_tol
            idx = np.where(mask)[0]
            if idx.size == 0:
                idx = np.array([k - 1], dtype=int)
            return idx

        elif self.lowvar_policy == "eigengap":
            # Select the bottom-k by a user-specified integer, or infer by the largest gap near the tail.
            if self.eigengap_k is not None:
                n = min(self.eigengap_k, k)
                idx = np.arange(k - n, k, dtype=int)
                return idx
            # Find largest gap among the smallest half to avoid the head
            diffs = np.diff(ev)  # length k-1, with ev sorted descending
            start = k // 2
            tail_diffs = diffs[start:]
            if tail_diffs.size == 0:
                return np.array([k - 1], dtype=int)
            j = np.argmax(tail_diffs) + start
            # Everything after j is the "low-variance" block
            idx = np.arange(j + 1, k, dtype=int)
            if idx.size == 0:
                idx = np.array([k - 1], dtype=int)
            return idx

        else:
            raise ValueError(f"Unknown lowvar_policy: {self.lowvar_policy}")

    # =========================
    # Jacobians of features and constraints
    # =========================
    def get_feature_jacobian(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return J_phi(X): shape (N, p, d). Requires mode != 'precomputed'.
        - polynomial: analytic Jacobian
        - callable: feature_jacobian if provided; else numeric finite differences if enabled
        """
        if self.mode == "precomputed":
            raise RuntimeError("Feature Jacobian unavailable for precomputed features.")
        if X is None:
            if self.X_ is None:
                raise RuntimeError("No X stored; pass X explicitly.")
            X = self.X_
        X = np.asarray(X, dtype=np.float64)
        N, _ = X.shape

        if self.mode == "polynomial":
            if self.poly_ is None:
                raise RuntimeError("PolynomialFeatures not initialized.")
            # Batch analytic Jacobian
            J_list = []
            for sl in _iterate_batches(N, self.batch_size):
                Xb = X[sl]
                _, Jb = _poly_feature_jacobian_batch(Xb, self.poly_)
                J_list.append(Jb)
            return np.vstack(J_list)  # (N, p, d)

        elif self.mode == "callable":
            if self.feature_jacobian is not None:
                J = self.feature_jacobian(X)
                J = np.asarray(J, dtype=np.float64)
                if J.ndim != 3 or J.shape[0] != X.shape[0]:
                    raise ValueError("feature_jacobian must return (N, p, d).")
                return J
            elif self.numeric_jacobian:
                if self.feature_func is None:
                    raise RuntimeError("feature_func must be provided for numeric_jacobian=True.")
                return _numeric_feature_jacobian(
                    X, self.feature_func, method=self.fd_method, batch_size=self.batch_size
                )
            else:
                raise RuntimeError("No feature_jacobian provided and numeric_jacobian=False.")

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_constraint_jacobian(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Construct Jacobians of learned constraints g_l(x) = w_l^T φ(x).
        Returns:
            J_g(X): shape (N, r, d), where r = number of low-variance components.
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() first.")
        if self.constraint_weights_ is None:
            raise RuntimeError("No constraint weights found.")
        W = self.constraint_weights_  # (p, r)
        J_phi = self.get_feature_jacobian(X)  # (N, p, d)
        # Contract over feature dimension: (N, r, d) with (p,r) via J_phi^T W
        # For each n: (d, p) @ (p, r) -> (d, r) -> transpose to (r, d)
        N = J_phi.shape[0]
        Jg = np.empty((N, W.shape[1], J_phi.shape[2]), dtype=np.float64)
        for n in range(N):
            Jg[n] = (J_phi[n].T @ W).T  # (r, d)
        return Jg

    # =========================
    # Accessors / utilities
    # =========================
    @property
    def n_constraints_(self) -> int:
        if self.constraint_weights_ is None:
            return 0
        return self.constraint_weights_.shape[1]

    def low_variance_loadings(self) -> np.ndarray:
        """
        Return constraint weight matrix W in feature space: shape (p, r).
        Columns are low-variance directions in feature space.
        """
        if self.constraint_weights_ is None:
            raise RuntimeError("Call fit() first.")
        return self.constraint_weights_

    # =========================
    # Distances on the level set (stubs for now)
    # =========================
    def distance(self, P: np.ndarray, Q: np.ndarray, method: str = "geodesic") -> np.ndarray:
        """
        Placeholder: distances *between points on the level set*. To be implemented.
        Options may include:
          - 'graph-geodesic' (kNN graph embedded within near-level points),
          - 'tangent-metric' (induced Riemannian metric via constraints),
          - 'diffusion' (PHATE/DM-like on level-set samples).
        """
        raise NotImplementedError("On-manifold distance is not implemented yet.")

    def project_to_level_set(self, X: np.ndarray) -> np.ndarray:
        """
        Optional future: project off-manifold points onto the level set defined by g_l(x)=const.
        Useful for off-manifold detection but not required now.
        """
        raise NotImplementedError("Projection to level set is not implemented yet.")
