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

PyTorch compatibility
------------------------------
- The NumPy path (scikit-learn PCA) is unchanged.
- A torch path is added. If `backend='auto'`, torch is used when M is a
  torch.Tensor (as produced by the builder); otherwise NumPy is used.
- Torch path centers columns explicitly and performs SVD to emulate PCA.
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
Backend = Literal["auto", "numpy", "torch"]

__all__ = ["EquivariantDiscovery", "Coupling", "LowVarPolicy"]


# ------------------------- Backend helpers (lazy torch) -------------------------

def _maybe_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


def _is_torch_tensor(x) -> bool:
    return x.__class__.__module__.startswith("torch") and hasattr(x, "dtype") and hasattr(x, "device")


def _choose_backend(backend: Backend, *, M=None) -> Literal["numpy", "torch"]:
    if backend == "numpy":
        return "numpy"
    if backend == "torch":
        torch = _maybe_import_torch()
        if torch is None:
            raise RuntimeError("backend='torch' requested but PyTorch is not available.")
        return "torch"
    # backend == 'auto'
    if M is not None and _is_torch_tensor(M):
        torch = _maybe_import_torch()
        if torch is not None:
            return "torch"
    return "numpy"


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

    # Backend preference for PCA/SVD ('auto' selects torch if M is a torch.Tensor)
    backend: Backend = "auto"

    # Internal state
    pca_: Optional[Union[PCA, IncrementalPCA]] = field(default=None, init=False)
    coefficients_: Optional[Union[np.ndarray, "torch.Tensor"]] = field(default=None, init=False)  # (q_total, r)
    lowvar_indices_: Optional[Union[np.ndarray, "torch.Tensor"]] = field(default=None, init=False)
    shapes_: Optional[Dict[str, int]] = field(default=None, init=False)
    M_: Optional[Union[np.ndarray, "torch.Tensor"]] = field(default=None, init=False)  # optional cache (small cases)

    def fit(
        self,
        X: Union[np.ndarray, "torch.Tensor"],
        F: Union[Callable[[np.ndarray], np.ndarray], np.ndarray, Callable, "torch.Tensor"],
        J_F: Union[Callable[[np.ndarray], np.ndarray], np.ndarray, Callable, "torch.Tensor"],
        vf_in: List[Callable[[np.ndarray], np.ndarray]],
        vf_out: List[Callable[[np.ndarray], np.ndarray]],
        normalize_rows: bool = False,
        row_weights: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
        backend: Backend = "auto",
    ) -> "EquivariantDiscovery":
        """
        Build the residual/design matrix M and run PCA to extract low-variance combinations.

        Parameters
        ----------
        X : (N, d) input samples (NumPy array or torch.Tensor)
        F : callable or array/tensor; (N,d)->(N,p) or precomputed (N,p)
        J_F : callable or array/tensor; (N,d)->(N,p,d) or precomputed (N,p,d)
        vf_in : list of input-space generator callables on R^d
        vf_out : list of output-space generator callables on R^p
        normalize_rows : bool, optional
            If True, normalize each row of M to unit norm (numerical stabilization).
        row_weights : Optional[(N*p,)]
            Per-row weights applied to M (after optional normalization).
        backend : {'auto','numpy','torch'}
            Backend for the PCA step. 'auto' uses torch if M is torch.Tensor.

        Returns
        -------
        self
        """
        # Build residual/design matrix (aligned by default). It returns NumPy or torch.
        M, info = getEquivariantResidualMatrix(
            X=X,
            F=F,
            J_F=J_F,
            vf_in=vf_in,
            vf_out=vf_out,
            coupling=self.coupling,
            normalize_rows=normalize_rows,
            row_weights=row_weights,
            backend=backend,  # pass through so builders can keep everything in torch if needed
        )
        self.shapes_ = info
        self.M_ = M

        # Decide PCA backend based on M and user preference
        bk = _choose_backend(backend if backend != "auto" else self.backend, M=M)

        if bk == "numpy":
            # ---------------------- NumPy path (unchanged) ----------------------
            if self.use_incremental:
                ipca = IncrementalPCA(
                    n_components=self.n_components,
                    batch_size=self.batch_size,
                    copy=False,
                )
                # For large M you can stream row blocks; here we fit in one go.
                ipca.partial_fit(M)
                self.pca_ = ipca
                comps = ipca.components_
                ev = ipca.explained_variance_
            else:
                pca = PCA(
                    n_components=self.n_components,
                    svd_solver=self.svd_solver,
                    random_state=self.random_state,
                ).fit(M)
                self.pca_ = pca
                comps = pca.components_
                ev = pca.explained_variance_

            idx = self._select_low_variance_indices_numpy(ev)
            self.lowvar_indices_ = idx

            # components_.shape = (k, q_total), ordered high->low variance
            W_small = comps[idx, :]  # (r, q_total)
            self.coefficients_ = W_small.T  # (q_total, r)
            return self

        # ----------------------------- Torch path ------------------------------
        torch = _maybe_import_torch()
        if torch is None:
            raise RuntimeError("Torch backend selected but PyTorch is not installed.")

        if self.use_incremental:
            raise NotImplementedError("use_incremental=True is not supported in the torch backend.")

        M_t = M if isinstance(M, torch.Tensor) else torch.as_tensor(M)
        # Center columns (PCA always centers features)
        M_center = M_t - M_t.mean(dim=0, keepdim=True)

        # SVD: M_center = U S V^T
        U, S, Vh = torch.linalg.svd(M_center, full_matrices=False)

        # Explained variance like sklearn: S^2 / (n_samples - 1)
        n_samples = M_center.shape[0]
        denom = max(1, n_samples - 1)
        ev_t = (S ** 2) / denom  # descending order

        # Select low-variance indices on torch tensor
        idx_t = self._select_low_variance_indices_torch(ev_t)

        # Components are rows of Vh (k, q_total); choose indices, then transpose
        W_small_t = Vh.index_select(0, idx_t)              # (r, q_total)
        self.coefficients_ = W_small_t.T                   # (q_total, r)
        self.lowvar_indices_ = idx_t
        self.pca_ = None  # Not used for torch path
        return self

    # ---------------------- selection helpers (numpy/torch) ----------------------

    def _select_low_variance_indices_numpy(self, ev: np.ndarray) -> np.ndarray:
        """
        Select tail indices of PCA components according to the configured policy.
        ev is descending (as in sklearn PCA).
        """
        ev = np.asarray(ev, dtype=np.float64)
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
            j = int(np.argmax(tail)) + start
            idx = np.arange(j + 1, k, dtype=int)
            return idx if idx.size > 0 else np.array([k - 1], dtype=int)

        raise ValueError(f"Unknown lowvar_policy: {self.lowvar_policy}")

    def _select_low_variance_indices_torch(self, ev_t: "torch.Tensor") -> "torch.Tensor":
        """
        Torch variant of low-variance selection. ev_t is descending.
        Returns a 1-D LongTensor of indices.
        """
        torch = _maybe_import_torch()
        assert torch is not None, "Torch must be available here."
        k = ev_t.shape[0]

        if self.lowvar_policy == "count":
            if not self.n_small:
                raise ValueError("policy='count' requires n_small.")
            n = min(self.n_small, k)
            return torch.arange(k - n, k, dtype=torch.long, device=ev_t.device)

        if self.lowvar_policy == "relative":
            vmax = ev_t[0] if k > 0 else torch.tensor(1.0, device=ev_t.device, dtype=ev_t.dtype)
            mask = ev_t <= (self.rel_tol * vmax)
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            if idx.numel() == 0:
                return torch.tensor([k - 1], dtype=torch.long, device=ev_t.device)
            return idx

        if self.lowvar_policy == "absolute":
            if self.abs_tol is None:
                raise ValueError("policy='absolute' requires abs_tol.")
            mask = ev_t <= self.abs_tol
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            if idx.numel() == 0:
                return torch.tensor([k - 1], dtype=torch.long, device=ev_t.device)
            return idx

        if self.lowvar_policy == "eigengap":
            if self.eigengap_k is not None:
                n = min(self.eigengap_k, k)
                return torch.arange(k - n, k, dtype=torch.long, device=ev_t.device)
            diffs = torch.diff(ev_t)  # descending ev
            start = k // 2
            tail = diffs[start:]
            if tail.numel() == 0:
                return torch.tensor([k - 1], dtype=torch.long, device=ev_t.device)
            j = int(torch.argmax(tail)) + start
            idx = torch.arange(j + 1, k, dtype=torch.long, device=ev_t.device)
            if idx.numel() == 0:
                return torch.tensor([k - 1], dtype=torch.long, device=ev_t.device)
            return idx

        raise ValueError(f"Unknown lowvar_policy: {self.lowvar_policy}")

    # ------------------------------ utilities ------------------------------

    # Convenience split for 'free' coupling
    def split_coefficients(self) -> Tuple[Union[np.ndarray, "torch.Tensor"], Union[np.ndarray, "torch.Tensor"]]:
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
        return 0 if self.coefficients_ is None else int(self.coefficients_.shape[1])
