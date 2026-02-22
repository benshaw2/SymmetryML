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
- This function focuses strictly on SVD-based discovery. For the
  "low-variance PCA" framing (centered columns), call scikit-learn PCA on A
  directly and pick tail components.

PyTorch compatibility
------------------------------
- `backend='auto'` (default) keeps the original NumPy behavior unless
  the input A is a torch.Tensor, in which case we use the torch branch.
- You can force a backend with backend='numpy' or backend='torch'.
"""

from __future__ import annotations

from typing import Optional, Tuple, Literal, Union

import numpy as np

__all__ = ["discover_symmetry_coeffs"]

Backend = Literal["auto", "numpy", "torch"]


# ---------------------------------------------------------------------------
# Backend detection (lazy torch import)
# ---------------------------------------------------------------------------

def _maybe_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


def _is_torch_tensor(x) -> bool:
    # Avoid importing torch unless needed; a light structural check:
    return x.__class__.__module__.startswith("torch") and hasattr(x, "dtype") and hasattr(x, "device")


def _choose_backend(backend: Backend, *, A=None) -> Literal["numpy", "torch"]:
    if backend == "numpy":
        return "numpy"
    if backend == "torch":
        torch = _maybe_import_torch()
        if torch is None:
            raise RuntimeError("backend='torch' requested but PyTorch is not available.")
        return "torch"
    # backend == "auto"
    if A is not None and _is_torch_tensor(A):
        torch = _maybe_import_torch()
        if torch is not None:
            return "torch"
    return "numpy"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_symmetry_coeffs(
    A: Union[np.ndarray, "torch.Tensor"],
    rtol: float = 1e-6,
    max_sym: Optional[int] = None,
    backend: Backend = "auto",
) -> Tuple[Union[np.ndarray, "torch.Tensor"], Union[np.ndarray, "torch.Tensor"]]:
    """
    Find symmetry coefficient vectors via SVD of the extended feature matrix A.

    Given A ∈ R^{M×q}, we compute its SVD A = U diag(S) V^T (descending S).
    Right singular vectors associated with the *smallest* singular values
    (near-zero) provide (approximate) nullspace directions. Each such vector
    is a coefficient vector c ∈ R^{q} combining the basis vector fields into
    an invariant/tangent field.

    Parameters
    ----------
    A : array-like, shape (M, q)
        Extended feature matrix (e.g., output of getExtendedFeatureMatrix).
        Can be a NumPy array or a torch.Tensor.
    rtol : float, default=1e-6
        Relative threshold against the largest singular value S[0].
        We select indices i where S[i] <= rtol * S[0].
    max_sym : Optional[int], default=None
        If provided, keep at most this many smallest singular directions.
        When used together with rtol, we first select by the rtol mask (if any),
        then truncate to at most `max_sym` by the *smallest S*.
    backend : {'auto','numpy','torch'}, default='auto'
        Which backend to use. 'auto' chooses 'torch' if A is a torch.Tensor,
        otherwise 'numpy'.

    Returns
    -------
    V_small : (q, r)
        Columns are coefficient vectors (each column is one symmetry combination).
        Type matches the backend: np.ndarray (numpy) or torch.Tensor (torch).
    svals : (r,)
        Corresponding singular values in ascending order; type matches backend.

    Notes
    -----
    - This function uses *raw SVD* (no centering). For a PCA "low-variance"
      approach, center columns first or use sklearn.PCA on A and pick tail
      components.
    """
    bk = _choose_backend(backend, A=A)

    if bk == "numpy":
        # -------- NumPy path (preserves your original implementation) --------
        A = np.asarray(A)
        if A.ndim != 2:
            raise ValueError(f"A must be 2-D; got ndim={A.ndim} with shape={getattr(A, 'shape', None)}")
        if A.dtype == object:
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
            sel = idx[np.argsort(S[idx])]
            idx = sel[:max_sym]

        # Right singular vectors are rows of Vt; select exactly 'idx'
        V_small = Vt[idx, :].T     # (q, r)
        s_small = S[idx]           # (r,)

        # Return in ascending order of singular value for convenience
        order = np.argsort(s_small)
        V_small = V_small[:, order]
        s_small = s_small[order]
        return V_small, s_small

    # -------- Torch path --------
    torch = _maybe_import_torch()
    if torch is None:
        raise RuntimeError("Torch backend selected but PyTorch is not installed.")

    # Coerce to torch tensor (float)
    A_t = A if isinstance(A, torch.Tensor) else torch.as_tensor(A)
    if A_t.ndim != 2:
        raise ValueError(f"A must be 2-D; got ndim={A_t.ndim} with shape={tuple(A_t.shape)}")
    # Use float dtype for linalg
    if not A_t.is_floating_point():
        A_t = A_t.to(dtype=torch.float32)

    # Economy SVD (S descending)
    U, S, Vh = torch.linalg.svd(A_t, full_matrices=False)

    # Threshold relative to the largest singular value
    smax = S[0] if S.numel() > 0 else torch.tensor(1.0, dtype=S.dtype, device=S.device)
    thresh = rtol * smax
    mask = S <= thresh
    idx = torch.nonzero(mask, as_tuple=False).flatten()

    if idx.numel() == 0:
        idx = torch.tensor([S.numel() - 1], device=S.device, dtype=torch.long)

    if max_sym is not None and idx.numel() > max_sym:
        # Keep the *smallest* max_sym among the selected
        sort_idx = torch.argsort(S.index_select(0, idx))
        idx = idx.index_select(0, sort_idx[:max_sym])

    # Right singular vectors are rows of Vh
    V_small = Vh.index_select(0, idx).T  # (q, r)
    s_small = S.index_select(0, idx)     # (r,)

    # Ascending order of singular value
    order = torch.argsort(s_small)
    V_small = V_small.index_select(1, order)
    s_small = s_small.index_select(0, order)

    return V_small, s_small
