"""
Equivariant discovery.

This module provides the `EquivariantDiscovery` class which discovers
equivariant combinations of input/output generators by running PCA on the
stacked residual/design matrix built from

    r_i(x) = J_F(x) @ X̃_i(x) - X̄_i(F(x))    (aligned coupling)

or using a free coupling design matrix for (c_in, d_out) pairs.

Use together with:
  - builders.getEquivariantResidualMatrix(...)  (residual/design matrix builder)

The workflow mirrors invariance discovery:
  1) Build matrix M with builders.getEquivariantResidualMatrix(...)
  2) Run PCA (columns are centered) on M
  3) Select low-variance components at the tail
  4) Columns of `coefficients_` are discovered coefficient vectors
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA

from .builders import getEquivariantResidualMatrix

Array = np.ndarray
Coupling = Literal["aligned", "free"]
LowVarPolicy = Literal["count", "relative", "absolute", "eigengap"]

__all__ = ["EquivariantDiscovery", "Coupling", "LowVarPolicy"]


@dataclass
class EquivariantDiscovery:
    """
    Discover equivariant combinations of input/output generators via PCA
    on the stacked residual matrix.

    Default behavior matches your preferences:
      - coupling='aligned'
      - PCA (centers columns), select **low-variance** components

    For 'aligned':
        `coefficients_` gives c ∈ R^{q} for each discovered component.
    For 'free':
        `coefficients_` has shape (q_in + q_out, r), and you can use
        `split_coefficients()` to get (c_in_, d_out_) blocks.
    """

    # Discovery configuration
    coupling: Coupling = "aligned"  # shared coefficients by default

    # PCA settings
    use_incremental: bool = False
    batch_size: Optional[int] = None
    n_components: Optional[int] = None
    svd_solver: Literal["auto", "full", "arpack", "randomized"] = "auto"
    random_state: Optional[int] = None

    # Low-variance selection
    lowvar_policy: LowVarPolicy = "relative"
    n_small: Optional[int] = None
    rel_tol: float = 1e-6
    abs_tol: Optional[float] = None
    eigengap_k: Optional[int] = None

    # Internal state
    pca_: Optional[Union[PCA, IncrementalPCA]] = field(default=None, init=False)
    coefficients_: Optional[np.ndarray] = field(default=None, init=False)  # (q_total, r)
    lowvar_indices_: Optional[np.ndarray] = field(default=None, init=False)
    shapes_: Optional[Dict[str, int]] = field(default=None, init=False)
    M_: Optional[np.ndarray] = field(default=None, init=False)  # optional cache (small cases)

    def fit(
        self,
        X: np.ndarray,
        F: Union[Callable[[np.ndarray], np.ndarray], np.ndarray],
        J_F: Union[Callable[[np.ndarray], np.ndarray], np.ndarray],
        vf_in: List[Callable[[np.ndarray], np.ndarray]],
        vf_out: List[Callable[[np.ndarray], np.ndarray]],
        normalize_rows: bool = False,
        row_weights: Optional[np.ndarray] = None,
    ) -> "EquivariantDiscovery":
        """
        Build the residual/design matrix M and run PCA to extract low-variance combinations.

        Parameters
        ----------
        X : (N, d) input samples
        F : callable or array; (N,d)->(N,p) or precomputed (N,p)
        J_F : callable or array; (N,d)->(N,p,d) or precomputed (N,p,d)
        vf_in : list of input-space generator callables on R^d
        vf_out : list of output-space generator callables on R^p
        normalize_rows : bool, optional
            If True, normalize each row of M to unit norm (numerical stabilization).
        row_weights : Optional[(N*p,)]
            Per-row weights applied to M (after optional normalization).

        Returns
        -------
        self
        """
        # Build residual/design matrix (aligned by default)
        M, info = getEquivariantResidualMatrix(
            X=X,
            F=F,
            J_F=J_F,
            vf_in=vf_in,
            vf_out=vf_out,
            coupling=self.coupling,
            normalize_rows=normalize_rows,
            row_weights=row_weights,
        )
        self.shapes_ = info

        # PCA (centers columns automatically)
        if self.use_incremental:
            ipca = IncrementalPCA(
                n_components=self.n_components,
                batch_size=self.batch_size,
                copy=False,
            )
            # For large M you can stream row blocks; here we fit in one go.
            ipca.partial_fit(M)
            self.pca_ = ipca
        else:
            self.M_ = M  # keep for inspection in small/medium cases
            self.pca_ = PCA(
                n_components=self.n_components,
                svd_solver=self.svd_solver,
                random_state=self.random_state,
            ).fit(M)

        # Select low-variance PCs (tail components in PCA ordering)
        idx = self._select_low_variance_indices()
        self.lowvar_indices_ = idx

        # components_.shape = (k, q_total), ordered high->low variance
        comps = self.pca_.components_
        W_small = comps[idx, :]  # (r, q_total)
        self.coefficients_ = W_small.T  # (q_total, r)

        return self

    def _select_low_variance_indices(self) -> np.ndarray:
        """
        Select tail indices of PCA components according to the configured policy.
        """
        ev = np.asarray(self.pca_.explained_variance_, dtype=np.float64)
        k = ev.shape[0]

        if self.lowvar_policy == "count":
            if not self.n_small:
                raise ValueError("policy='count' requires n_small.")
            n = min(self.n_small, k)
            return np.arange(k - n, k, dtype=int)

        if self.lowvar_policy == "relative":
            vmax = ev[0] if k > 0 else 1.0
            idx = np.where(ev <= self.rel_tol * vmax)[0]
            return idx if idx.size > 0 else np.array([k - 1], dtype=int)

        if self.lowvar_policy == "absolute":
            if self.abs_tol is None:
                raise ValueError("policy='absolute' requires abs_tol.")
            idx = np.where(ev <= self.abs_tol)[0]
            return idx if idx.size > 0 else np.array([k - 1], dtype=int)

        if self.lowvar_policy == "eigengap":
            if self.eigengap_k is not None:
                n = min(self.eigengap_k, k)
                return np.arange(k - n, k, dtype=int)
            diffs = np.diff(ev)  # descending ev
            start = k // 2
            tail = diffs[start:]
            if tail.size == 0:
                return np.array([k - 1], dtype=int)
            j = np.argmax(tail) + start
            idx = np.arange(j + 1, k, dtype=int)
            return idx if idx.size > 0 else np.array([k - 1], dtype=int)

        raise ValueError(f"Unknown lowvar_policy: {self.lowvar_policy}")

    # Convenience split for 'free' coupling
    def split_coefficients(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        For coupling='free': returns (c_in_, d_out_).
        Shapes: (q_in, r), (q_out, r).
        """
        if self.coefficients_ is None or self.shapes_ is None:
            raise RuntimeError("Call fit() first.")
        if self.shapes_["coupling"] != "free":
            raise RuntimeError("split_coefficients only valid for coupling='free'.")
        q_in, q_out = self.shapes_["q_in"], self.shapes_["q_out"]
        C = self.coefficients_
        return C[:q_in, :], C[q_in:, :]

    @property
    def n_components_found_(self) -> int:
        return 0 if self.coefficients_ is None else self.coefficients_.shape[1]
