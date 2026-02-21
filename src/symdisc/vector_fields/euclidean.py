"""
Euclidean Killing vector fields in R^d.

This module provides utilities to construct the canonical generators of the
Euclidean group in R^d:

- Translations: T_i(x) = e_i  for i=0..d-1
- Rotations:    R_{ij}(x) = x_i ∂/∂x_j - x_j ∂/∂x_i  for 0 <= i < j < d

Each generator is returned as a batch-aware callable that accepts either:
  - a single point x with shape (d,) and returns a vector of shape (d,), or
  - a batch X with shape (N, d) and returns an array of shape (N, d).

By default, `generate_euclidean_killing_fields(d)` returns a list of callables
with ordering:
  [T_0, ..., T_{d-1}, R_{0,1}, R_{0,2}, ..., R_{d-2,d-1}]
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

__all__ = [
    "generate_euclidean_killing_fields",
    "generate_euclidean_killing_fields_with_names",
]


def _ensure_batch(x: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Normalize an input vector or batch to (N, d). Returns (X, was_single).

    - If x is (d,), returns (x.reshape(1, -1), True).
    - If x is (N, d), returns (x, False).
    - Else raises ValueError.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x.reshape(1, -1), True
    if x.ndim == 2:
        return x, False
    raise ValueError("Input x must be (d,) or (N, d).")


def generate_euclidean_killing_fields(
    d: int,
    include_translations: bool = True,
    include_rotations: bool = True,
) -> List[Callable[[np.ndarray], np.ndarray]]:
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

    Returns
    -------
    fields : List[Callable]
        A list of batch-aware callables f(X)->(N,d) or f(x)->(d,).
        Ordering (when both included):
            [T_0, ..., T_{d-1}, R_{0,1}, R_{0,2}, ..., R_{d-2,d-1}]
    """
    if d <= 0:
        raise ValueError("Dimension d must be positive.")

    fields: List[Callable[[np.ndarray], np.ndarray]] = []

    # Translations: T_i(x) = e_i
    if include_translations:
        for i in range(d):
            def make_T(i=i):
                e = np.zeros((d,), dtype=np.float64)
                e[i] = 1.0

                def T(x: np.ndarray) -> np.ndarray:
                    X, was_single = _ensure_batch(x)
                    out = np.tile(e, (X.shape[0], 1))
                    return out[0] if was_single else out

                return T

            fields.append(make_T())

    # Rotations: R_{ij}(x) = x_i ∂/∂x_j - x_j ∂/∂x_i
    if include_rotations:
        for i in range(d):
            for j in range(i + 1, d):
                def make_R(i=i, j=j):
                    def R(x: np.ndarray) -> np.ndarray:
                        X, was_single = _ensure_batch(x)
                        out = np.zeros_like(X)
                        out[:, i] = -X[:, j]
                        out[:, j] =  X[:, i]
                        return out[0] if was_single else out

                    return R

                fields.append(make_R())

    if not fields:
        raise ValueError("At least one of translations or rotations must be included.")

    return fields


def generate_euclidean_killing_fields_with_names(
    d: int,
    include_translations: bool = True,
    include_rotations: bool = True,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], List[str]]:
    """
    Same as `generate_euclidean_killing_fields`, but also returns a parallel
    list of human-readable names for each generator, useful for interpreting
    discovered coefficient vectors.

    Returns
    -------
    fields : List[Callable]
    names : List[str]
        Names are like: "T_0", "T_1", ..., "R_0_1", "R_0_2", ...
    """
    fields = []
    names: List[str] = []

    # Translations
    if include_translations:
        for i in range(d):
            def make_T(i=i):
                e = np.zeros((d,), dtype=np.float64)
                e[i] = 1.0

                def T(x: np.ndarray) -> np.ndarray:
                    X, was_single = _ensure_batch(x)
                    out = np.tile(e, (X.shape[0], 1))
                    return out[0] if was_single else out

                return T

            fields.append(make_T())
            names.append(f"T_{i}")

    # Rotations
    if include_rotations:
        for i in range(d):
            for j in range(i + 1, d):
                def make_R(i=i, j=j):
                    def R(x: np.ndarray) -> np.ndarray:
                        X, was_single = _ensure_batch(x)
                        out = np.zeros_like(X)
                        out[:, i] = -X[:, j]
                        out[:, j] =  X[:, i]
                        return out[0] if was_single else out

                    return R

                fields.append(make_R())
                names.append(f"R_{i}_{j}")

    if not fields:
        raise ValueError("At least one of translations or rotations must be included.")

    return fields, names
