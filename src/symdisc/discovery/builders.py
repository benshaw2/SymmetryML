"""
Builders for invariance and equivariance design matrices.

This module contains:
  - getExtendedFeatureMatrix: builds A ∈ R^{(N*m)×q} for invariance/tangency discovery
      A[(i,k), j] = < J[i,k,:], X_j(x_i) >
    where J are Jacobian rows of the constraint(s) and X_j are candidate vector fields.

  - getEquivariantResidualMatrix: builds M ∈ R^{(N*p)×q_total} for equivariance discovery
      aligned: columns are vec(J_F X_i - Xbar_i ∘ F)
      free:    columns are [ vec(J_F X_i) | -vec(Xbar_j ∘ F) ]

All helpers here are internal (prefixed with '_') to keep other modules focused.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union, Literal, Dict

import numpy as np

Array = np.ndarray
Coupling = Literal["aligned", "free"]

__all__ = [
    "getExtendedFeatureMatrix",
    "getEquivariantResidualMatrix",
    "Coupling",
]


# ============================================================================
# Internal helpers
# ============================================================================

def _ensure_vector_fields_values(
    X: Array,
    vector_fields: Union[List[Callable[[Array], Array]], Array],
) -> Array:
    """
    Evaluate vector fields on X (R^d).

    Returns:
        VF_values: (q, N, d) where VF_values[j, i, :] = X_j(x_i).

    Accepts:
      - list of callables: batch (N,d)->(N,d) or pointwise (d,)->(d,)
      - precomputed array of shape (q, N, d)
    """
    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape
    if isinstance(vector_fields, list):
        q = len(vector_fields)
        VF_values = np.zeros((q, N, d), dtype=np.float64)
        for j, f in enumerate(vector_fields):
            # Try batch evaluation
            try:
                val = f(X)  # expected (N, d)
                val = np.asarray(val, dtype=np.float64)
                if val.shape != (N, d):
                    raise ValueError(
                        f"Vector field {j} batch returned shape {val.shape}, expected {(N, d)}"
                    )
                VF_values[j] = val
            except Exception:
                # Fallback: per-point evaluation
                for i in range(N):
                    vi = np.asarray(f(X[i]), dtype=np.float64).reshape(-1)
                    if vi.shape != (d,):
                        raise ValueError(
                            f"Vector field {j} at point {i} returned shape {vi.shape}, "
                            f"expected {(d,)}"
                        )
                    VF_values[j, i, :] = vi
        return VF_values
    else:
        VF_values = np.asarray(vector_fields)
        if VF_values.ndim != 3:
            raise ValueError("Precomputed vector_fields array must have shape (q, N, d).")
        q, N2, d2 = VF_values.shape
        if N2 != X.shape[0] or d2 != X.shape[1]:
            raise ValueError(
                f"vector_fields array has shape {VF_values.shape}, "
                f"but X is {(X.shape[0], X.shape[1])}"
            )
        return VF_values.astype(np.float64, copy=False)


def _ensure_jacobians(
    X: Array,
    J: Union[Callable[[Array], Array], Array],
) -> Array:
    """
    Ensure Jacobians are available with shape (N, m, d), where
    J_values[i, k, :] = ∇g_k(x_i).

    Accepts:
      - callable: batch (N,d)->(N,m,d) or single point (d,)->(m,d)
      - precomputed array of shape (N, m, d)
    """
    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape
    if callable(J):
        # Try batch evaluation first
        try:
            Jv = J(X)
            Jv = np.asarray(Jv, dtype=np.float64)
            if Jv.ndim != 3 or Jv.shape[0] != N or Jv.shape[2] != d:
                raise ValueError
            return Jv
        except Exception:
            # Fallback: per-point evaluation
            rows = []
            m = None
            for i in range(N):
                Ji = np.asarray(J(X[i]), dtype=np.float64)
                if Ji.ndim != 2 or Ji.shape[1] != d:
                    raise ValueError(f"Jacobian at point {i} must have shape (m, d); got {Ji.shape}")
                if m is None:
                    m = Ji.shape[0]
                elif Ji.shape[0] != m:
                    raise ValueError("All Jacobians must have the same number of rows m.")
                rows.append(Ji)
            return np.stack(rows, axis=0)  # (N, m, d)
    else:
        Jv = np.asarray(J, dtype=np.float64)
        if Jv.ndim != 3 or Jv.shape[0] != N or Jv.shape[2] != d:
            raise ValueError(f"Precomputed J must have shape (N, m, d); got {Jv.shape}")
        return Jv


def _eval_vf_domain(X: Array, vf_list: List[Callable[[Array], Array]]) -> Array:
    """
    Evaluate input-space vector fields on X (domain R^d).
    Returns shape (q_in, N, d).
    """
    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape
    q = len(vf_list)
    V = np.zeros((q, N, d), dtype=np.float64)
    for j, f in enumerate(vf_list):
        try:
            val = f(X)  # expect (N, d)
            val = np.asarray(val, dtype=np.float64)
            if val.shape != (N, d):
                raise ValueError
            V[j] = val
        except Exception:
            for i in range(N):
                V[j, i] = np.asarray(f(X[i]), dtype=np.float64).reshape(-1)
    return V


def _eval_vf_codomain(Y: Array, vf_list: List[Callable[[Array], Array]]) -> Array:
    """
    Evaluate output-space vector fields on Y (codomain R^p).
    Returns shape (q_out, N, p).
    """
    Y = np.asarray(Y, dtype=np.float64)
    N, p = Y.shape
    q = len(vf_list)
    W = np.zeros((q, N, p), dtype=np.float64)
    for j, f in enumerate(vf_list):
        try:
            val = f(Y)  # expect (N, p)
            val = np.asarray(val, dtype=np.float64)
            if val.shape != (N, p):
                raise ValueError
            W[j] = val
        except Exception:
            for i in range(N):
                W[j, i] = np.asarray(f(Y[i]), dtype=np.float64).reshape(-1)
    return W


# ============================================================================
# Public builders
# ============================================================================

def getExtendedFeatureMatrix(
    X: Array,  # (N, d)
    J: Union[Callable[[Array], Array], Array],  # callable or (N, m, d)
    vector_fields: Union[List[Callable[[Array], Array]], Array],  # list or (q, N, d)
    normalize_rows: bool = True,  # normalize each Jacobian row
    row_weights: Optional[Array] = None,  # weights of shape (N*m,)
    dtype: Union[np.dtype, type] = np.float64,
) -> Tuple[Array, Tuple[int, int, int]]:
    """
    Build A ∈ R^{(N*m)×q}, with entries:
        A[(i,k), j] = < J[i,k,:], VF[j,i,:] >.

    Parameters
    ----------
    X : (N, d)
    J : callable or array
        Jacobian rows per point; returns/provides (N, m, d).
    vector_fields : list of callables or array
        Input-space vector fields; either a list of batch/pointwise callables
        or a precomputed array (q, N, d).
    normalize_rows : bool, default=True
        If True, normalize each J[i,k,:] by its l2-norm.
    row_weights : Optional[(N*m,)]
        Per-row weights multiplied after building A.
    dtype : numpy dtype, default=float64

    Returns
    -------
    A : (N*m, q)
    shape_info : (N, m, q)
    """
    X = np.asarray(X, dtype=dtype)
    N, d = X.shape

    J_values = _ensure_jacobians(X, J)  # (N, m, d)
    N2, m, d2 = J_values.shape
    if N2 != N or d2 != d:
        raise ValueError(f"Shape mismatch for J: expected (N, m, d)=({N}, ?, {d}); got {J_values.shape}")

    VF_values = _ensure_vector_fields_values(X, vector_fields)  # (q, N, d)
    q, N3, d3 = VF_values.shape
    if N3 != N or d3 != d:
        raise ValueError(f"Shape mismatch for vector_fields vs X: got {(q, N3, d3)} and X {(N, d)}")

    if normalize_rows:
        norms = np.linalg.norm(J_values, axis=2, keepdims=True)  # (N, m, 1)
        eps = 1e-12
        J_values = J_values / np.maximum(norms, eps)

    # Compute A[i,k,j] = dot(J[i,k,:], VF[j,i,:]) using einsum
    # Result shape: (N, m, q)
    A_nmq = np.einsum("imd,qid->imq", J_values, VF_values, optimize=True)

    # Flatten to (N*m, q)
    A = A_nmq.reshape(N * m, q).astype(dtype, copy=False)

    if row_weights is not None:
        row_weights = np.asarray(row_weights, dtype=dtype).reshape(-1)
        if row_weights.shape[0] != N * m:
            raise ValueError(f"row_weights must have length N*m={N*m}, got {row_weights.shape[0]}")
        A = row_weights[:, None] * A

    return A, (N, m, q)


def getEquivariantResidualMatrix(
    X: Array,  # (N, d)
    F: Union[Callable[[Array], Array], Array],  # callable or precomputed Y (N, p)
    J_F: Union[Callable[[Array], Array], Array],  # callable or precomputed (N, p, d)
    vf_in: List[Callable[[Array], Array]],  # input-space basis on R^d
    vf_out: List[Callable[[Array], Array]],  # output-space basis on R^p
    coupling: Coupling = "aligned",
    normalize_rows: bool = False,
    row_weights: Optional[Array] = None,
    dtype: Union[np.dtype, type] = np.float64,
) -> Tuple[Array, Dict[str, int]]:
    """
    Build the stacked residual/design matrix for infinitesimal equivariance.

    aligned (default):
        M = [ vec(J_F X_1 - Xbar_1∘F) | ... | vec(J_F X_q - Xbar_q∘F) ]  ∈ R^{(N*p) × q}

    free:
        M = [ vec(J_F X_1) ... vec(J_F X_q_in) | -vec(Xbar_1∘F) ... -vec(Xbar_q_out∘F) ]
            ∈ R^{(N*p) × (q_in + q_out)}

    Notes
    -----
    - scikit-learn PCA centers columns by default; no manual centering needed here.
    - Use `normalize_rows=True` or `row_weights` if some samples/output coords dominate numerically.

    Returns
    -------
    M : 2-D array with shape (N*p, q_total)
    info : dict with keys {'N','d','p','q_in','q_out','coupling'}
    """
    X = np.asarray(X, dtype=dtype)
    N, d = X.shape

    # Evaluate Y = F(X)
    if callable(F):
        Y = np.asarray(F(X), dtype=dtype)
    else:
        Y = np.asarray(F, dtype=dtype)
    if Y.ndim != 2 or Y.shape[0] != N:
        raise ValueError("F(X) must produce (N, p).")
    _, p = Y.shape

    # Evaluate J_F(X)
    if callable(J_F):
        Jv = np.asarray(J_F(X), dtype=dtype)  # (N, p, d)
    else:
        Jv = np.asarray(J_F, dtype=dtype)
    if Jv.shape != (N, p, d):
        raise ValueError(f"J_F must be (N, p, d); got {Jv.shape}")

    # Input and output generator evaluations
    Xin = _eval_vf_domain(X, vf_in)   # (q_in, N, d)
    Xout = _eval_vf_codomain(Y, vf_out)  # (q_out, N, p)
    q_in = Xin.shape[0]
    q_out = Xout.shape[0]

    # v_in[i, n, :] = Jv[n,:,:] @ Xin[i,n,:]  → (q_in, N, p)
    v_in = np.einsum("npd,qnd->qnp", Jv, Xin, optimize=True)

    if coupling == "aligned":
        if q_in != q_out:
            raise ValueError("aligned coupling requires q_in == q_out.")
        # Residual columns (q, Np) then transpose to (Np, q)
        R_cols = (v_in - Xout).reshape(q_in, N * p)
        M = R_cols.T  # (N*p, q)
    elif coupling == "free":
        Vin_cols = v_in.reshape(q_in, N * p)   # (q_in, Np)
        Vout_cols = Xout.reshape(q_out, N * p) # (q_out, Np)
        M = np.concatenate([Vin_cols, -Vout_cols], axis=0).T  # (Np, q_in+q_out)
    else:
        raise ValueError(f"Unknown coupling: {coupling}")

    if normalize_rows:
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        M = M / np.maximum(norms, 1e-12)

    if row_weights is not None:
        rw = np.asarray(row_weights, dtype=dtype).reshape(-1)
        if rw.shape[0] != M.shape[0]:
            raise ValueError(f"row_weights must have length {M.shape[0]}")
        M = rw[:, None] * M

    info = {"N": N, "d": d, "p": p, "q_in": q_in, "q_out": q_out, "coupling": coupling}
    return M, info
