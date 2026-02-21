"""
Invariance discovery utilities.

This module provides:
  - discover_symmetry_coeffs(A, ...): find (approximate) nullspace directions of A
    using SVD, returning the right singular vectors corresponding to the smallest
    singular values. These coefficient vectors combine the provided search-space
    vector fields (e.g., Euclidean Killing vectors) to yield tangent (invariant)
    combinations.

Notes
-----
- Use this together with `getExtendedFeatureMatrix(...)` from `builders.py`,
  which constructs A[(i,k), j] = < J[i,k,:], X_j(x_i) >, flattened across (i,k).
- This function focuses strictly on SVD-based discovery. If you want the
  "low-variance PCA" framing (centered columns), call scikit-learn PCA on A
  directly instead of raw SVD.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["discover_symmetry_coeffs"]


def discover_symmetry_coeffs(
    A: np.ndarray,
    rtol: float = 1e-6,
    max_sym: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find symmetry coefficient vectors via SVD of the extended feature matrix A.

    Given A ∈ R^{M×q}, we compute its SVD A = U diag(S) V^T (NumPy returns S in
    descending order). Right singular vectors associated with the *smallest*
    singular values (near-zero) provide (approximate) nullspace directions. Each
    such vector is a coefficient vector c ∈ R^{q} combining the basis vector
    fields (columns of your design) into an invariant/tangent field.

    Parameters
    ----------
    A : np.ndarray, shape (M, q)
        Extended feature matrix (e.g., output of getExtendedFeatureMatrix).
    rtol : float, default=1e-6
        Relative threshold against the largest singular value S[0].
        We select indices i where S[i] <= rtol * S[0].
    max_sym : Optional[int], default=None
        If provided, keep at most this many smallest singular directions.
        When used together with rtol, we first select by the rtol mask (if any),
        then truncate to at most `max_sym` by the *smallest S*.

    Returns
    -------
    V_small : np.ndarray, shape (q, r)
        Columns are coefficient vectors (each column is one symmetry combination).
    svals : np.ndarray, shape (r,)
        The corresponding singular values (returned in ascending order).

    Notes
    -----
    - Input validation ensures A is a 2-D numeric array.
    - This function uses *raw SVD* (no centering). For a PCA "low-variance"
      approach, center columns first or use sklearn.PCA on A and pick tail
      components.
    """
    # Validate and coerce
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D; got ndim={A.ndim} with shape={getattr(A, 'shape', None)}")
    if A.dtype == object:
        # Attempt coercion to float; raise if not possible.
        try:
            A = A.astype(np.float64)
        except Exception as e:
            raise ValueError("A has dtype=object; ensure numeric arrays throughout.") from e
    else:
        A = A.astype(np.float64, copy=False)

    if A.size == 0:
        raise ValueError("A must be non-empty.")

    # Economy SVD (S sorted descending)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Threshold relative to the largest singular value
    smax = S[0] if S.size > 0 else 1.0
    thresh = rtol * smax
    mask = S <= thresh
    idx = np.where(mask)[0]

    if idx.size == 0:
        # No singular values pass the threshold; keep the single smallest one.
        idx = np.array([S.size - 1], dtype=int)

    # If max_sym is provided, keep the *smallest* max_sym among the selected.
    if max_sym is not None and idx.size > max_sym:
        # Sort the selected indices by their S (ascending) and keep the first max_sym
        sel = idx[np.argsort(S[idx])]
        idx = sel[:max_sym]

    # Build outputs selecting the *actual* indices chosen
    # Right singular vectors are rows of Vt; pick rows in `idx` and transpose.
    V_small = Vt[idx, :].T  # (q, r)
    s_small = S[idx]        # (r,)

    # Return in ascending order of singular value for convenience
    order = np.argsort(s_small)
    V_small = V_small[:, order]
    s_small = s_small[order]

    return V_small, s_small
