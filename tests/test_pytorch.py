import torch
from symdisc.discovery.builders import getExtendedFeatureMatrix, getEquivariantResidualMatrix
from symdisc.vector_fields.euclidean import generate_euclidean_killing_fields

# Torch data
X = torch.randn(64, 3)

# Invariance: fake J_g (N, m, d) with m=2
Jg = torch.randn(64, 2, 3)

# Torch-aware vector fields: reuse NumPy ones (they accept tensors if they use tensor ops),
# or define small torch VFs inline:
kvs = []
for i in range(3):
    def make_T(i=i):
        def T(Xt):
            if Xt.ndim == 1:
                out = torch.zeros_like(Xt); out[i] = 1.0; return out
            out = torch.zeros_like(Xt); out[:, i] = 1.0; return out
        return T
    kvs.append(make_T())

A, info = getExtendedFeatureMatrix(X, Jg, kvs, backend="torch")
assert isinstance(A, torch.Tensor) and A.ndim == 2

# Equivariance: F=Id, J_F=I
F = lambda Xt: Xt
JF = lambda Xt: torch.eye(3).to(Xt).expand(Xt.shape[0], 3, 3)

M, einfo = getEquivariantResidualMatrix(X, F, JF, kvs, kvs, coupling='aligned', backend="torch")
assert isinstance(M, torch.Tensor) and M.ndim == 2
