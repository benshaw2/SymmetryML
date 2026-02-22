"""
Euclidean Killing vector fields in R^d.

This module provides utilities to construct the canonical generators of the
Euclidean group in R^d:

- Translations: T_i(x) = e_i  for i=0..d-1
- Rotations:    R_{ij}(x) = x_i ∂/∂x_j - x_j ∂/∂x_i  for 0 <= i < j < d

Each generator is returned as a batch-aware callable that accepts either:
  - a single point x with shape (d,) and returns a vector of shape (d,), or
  - a batch X with shape (N, d) and returns an array of shape (N, d).

Backend behavior
----------------
You can request vector fields that operate in NumPy or PyTorch:

  - backend='auto'  (default): detect from the input (np.ndarray vs torch.Tensor).
  - backend='numpy': always return NumPy arrays (input converted if needed).
  - backend='torch': always return torch tensors (input converted if needed).

By default, `generate_euclidean_killing_fields(d)` returns a list of callables
with ordering:
  [T_0, ..., T_{d-1}, R_{0,1}, R_{0,2}, ..., R_{d-2,d-1}]
"""

from __future__ import annotations

from typing import Callable, List, Tuple, Literal, Union

import numpy as np

__all__ = [
    "generate_euclidean_killing_fields",
    "generate_euclidean_killing_fields_with_names",
]

Backend = Literal["auto", "numpy", "torch"]


# ----------------------------- lazy torch helpers -----------------------------

def _maybe_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


def _is_torch_tensor(x) -> bool:
    return x.__class__.__module__.startswith("torch") and hasattr(x, "dtype") and hasattr(x, "device")


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if _is_torch_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_torch(x, dtype=None, device=None):
    torch = _maybe_import_torch()
    if torch is None:
        raise RuntimeError("backend='torch' requested but PyTorch is not available.")
    if isinstance(x, torch.Tensor):
        if dtype is not None or device is not None:
            return x.to(dtype=dtype or x.dtype, device=device or x.device)
        return x
    # convert from numpy or list
    t = torch.as_tensor(x)
    if dtype is not None or device is not None:
        t = t.to(dtype=dtype or t.dtype, device=device or t.device)
    return t


# ------------------------------- core utilities -------------------------------

def _ensure_batch_numpy(x: np.ndarray) -> Tuple[np.ndarray, bool]:
    x = _to_numpy(x).astype(np.float64, copy=False)
    if x.ndim == 1:
        return x.reshape(1, -1), True
    if x.ndim == 2:
        return x, False
    raise ValueError("Input must be shape (d,) or (N, d).")


def _ensure_batch_torch(x) -> Tuple["object", bool]:
    torch = _maybe_import_torch()
    assert torch is not None
    xt = _to_torch(x)
    if xt.ndim == 1:
        return xt.unsqueeze(0), True
    if xt.ndim == 2:
        return xt, False
    raise ValueError("Input must be shape (d,) or (N, d).")


def _make_translation_numpy(d: int, i: int) -> Callable[[np.ndarray], np.ndarray]:
    e = np.zeros((d,), dtype=np.float64)
    e[i] = 1.0

    def T(x: np.ndarray) -> np.ndarray:
        X, was_single = _ensure_batch_numpy(x)
        out = np.tile(e, (X.shape[0], 1))
        return out[0] if was_single else out

    return T


def _make_translation_torch(d: int, i: int) -> Callable:
    torch = _maybe_import_torch()
    assert torch is not None

    def T(x):
        Xt, was_single = _ensure_batch_torch(x)
        out = torch.zeros_like(Xt)
        out[:, i] = 1.0
        return out[0] if was_single else out

    return T


def _make_rotation_numpy(d: int, i: int, j: int) -> Callable[[np.ndarray], np.ndarray]:
    def R(x: np.ndarray) -> np.ndarray:
        X, was_single = _ensure_batch_numpy(x)
        out = np.zeros_like(X)
        out[:, i] = -X[:, j]
        out[:, j] =  X[:, i]
        return out[0] if was_single else out
    return R


def _make_rotation_torch(d: int, i: int, j: int) -> Callable:
    torch = _maybe_import_torch()
    assert torch is not None

    def R(x):
        Xt, was_single = _ensure_batch_torch(x)
        out = torch.zeros_like(Xt)
        out[:, i] = -Xt[:, j]
        out[:, j] =  Xt[:, i]
        return out[0] if was_single else out
    return R


def _wrap_backend(field_np: Callable, field_torch: Callable, backend: Backend) -> Callable:
    """
    Return a callable that dispatches to numpy or torch implementation.

    - backend='numpy': always convert input to NumPy and return NumPy.
    - backend='torch': always convert input to torch and return torch.
    - backend='auto' : use the input type (torch.Tensor -> torch path; else NumPy).
    """
    if backend == "numpy":
        def f(x):
            return field_np(_to_numpy(x))
        return f

    if backend == "torch":
        torch = _maybe_import_torch()
        if torch is None:
            raise RuntimeError("backend='torch' requested but PyTorch is not available.")
        def f(x):
            x_t = _to_torch(x)
            return field_torch(x_t)
        return f

    # backend == "auto"
    def f(x):
        if _is_torch_tensor(x):
            return field_torch(x)
        return field_np(x)
    return f


# --------------------------------- Public API ---------------------------------

def generate_euclidean_killing_fields(
    d: int,
    include_translations: bool = True,
    include_rotations: bool = True,
    backend: Backend = "auto",
) -> List[Callable]:
    """
    Generate the Euclidean Killing vector fields in R^d.

    Parameters
    ----------
    d : int
        Ambient dimension.
    include_translations : bool, default=True
        Include the d translation generators T_i.
    include_rotations : bool, default=True
        Include the d*(d-1)/2 rotation generators R_{ij}.
    backend : {'auto','numpy','torch'}, default='auto'
        - 'auto'  : dispatch based on input type per call.
        - 'numpy' : always return NumPy arrays.
        - 'torch' : always return torch tensors (requires PyTorch).

    Returns
    -------
    fields : List[Callable]
        A list of batch-aware callables f(X)->(N,d) or f(x)->(d,).
        Ordering (when both included):
            [T_0, ..., T_{d-1}, R_{0,1}, R_{0,2}, ..., R_{d-2,d-1}]
    """
    if d <= 0:
        raise ValueError("Dimension d must be positive.")

    fields: List[Callable] = []

    if include_translations:
        for i in range(d):
            field_np    = _make_translation_numpy(d, i)
            field_torch = _make_translation_torch(d, i)
            fields.append(_wrap_backend(field_np, field_torch, backend))

    if include_rotations:
        for i in range(d):
            for j in range(i + 1, d):
                field_np    = _make_rotation_numpy(d, i, j)
                field_torch = _make_rotation_torch(d, i, j)
                fields.append(_wrap_backend(field_np, field_torch, backend))

    if not fields:
        raise ValueError("At least one of translations or rotations must be included.")

    return fields


def generate_euclidean_killing_fields_with_names(
    d: int,
    include_translations: bool = True,
    include_rotations: bool = True,
    backend: Backend = "auto",
) -> Tuple[List[Callable], List[str]]:
    """
    Same as `generate_euclidean_killing_fields`, but also returns names,
    e.g., "T_0", "R_0_1", etc., parallel to the returned callables.
    """
    fields: List[Callable] = []
    names: List[str] = []

    if include_translations:
        for i in range(d):
            field_np    = _make_translation_numpy(d, i)
            field_torch = _make_translation_torch(d, i)
            fields.append(_wrap_backend(field_np, field_torch, backend))
            names.append(f"T_{i}")

    if include_rotations:
        for i in range(d):
            for j in range(i + 1, d):
                field_np    = _make_rotation_numpy(d, i, j)
                field_torch = _make_rotation_torch(d, i, j)
                fields.append(_wrap_backend(field_np, field_torch, backend))
                names.append(f"R_{i}_{j}")

    if not fields:
        raise ValueError("At least one of translations or rotations must be included.")

    return fields, names
