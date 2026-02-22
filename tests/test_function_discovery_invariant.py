import numpy as np
from symdisc.discovery.function_invariance import FunctionDiscoveryInvariant
from symdisc.vector_fields import generate_euclidean_killing_fields_with_names

# Sample data in R^3
rng = np.random.default_rng(0)
t = rng.normal(size=2000)
X = np.column_stack([np.cos(t), np.sin(t), rng.normal(scale=0.1, size=t.shape)])

# Vector fields: pick R_0_1 only
fields, names = generate_euclidean_killing_fields_with_names(d=3, include_translations=False, include_rotations=True)
vf_map = dict(zip(names, fields))
vf_in = [vf_map["R_0_1"]]

# Features: polynomials up to degree 3 (no bias)
fd = FunctionDiscoveryInvariant(
    mode="polynomial",
    degree=3,
    include_bias=False,
    pca_method="pca",           # centered low-variance selection
    lowvar_policy="relative",
    rel_tol=1e-8,
)

fd.fit(X=X, vf_in=vf_in, normalize_rows=True)

# Evaluate discovered invariant functions on X
G = fd.transform(X)  # (N, r)
print("Discovered invariants G shape:", G.shape)

# Inspect correlation with known invariants [z, x^2+y^2]
Phi = fd._features(X)
W = fd.function_weights_  # (p, r)
# You can examine W columns to see which monomials they emphasize.


'''
import torch
from symdisc.discovery.function_invariance import FunctionDiscoveryInvariant
from symdisc.vector_fields import generate_euclidean_killing_fields

# Data
X = torch.stack([torch.cos(torch.randn(2000)), torch.sin(torch.randn(2000)), 0.1*torch.randn(2000)], dim=1)

# Use numpy polynomial features but build B/take SVD in torch via backend='torch'
vf_in = generate_euclidean_killing_fields(d=3, include_translations=False, include_rotations=True, backend='torch')

fd = FunctionDiscoveryInvariant(
    mode="polynomial",
    degree=3,
    include_bias=False,
    pca_method="svd",       # uncentered nullspace (torch SVD)
    lowvar_policy="relative",
    rel_tol=1e-8,
    backend="torch"         # ensure torch path in builder and SVD
)

fd.fit(X=X, vf_in=vf_in, normalize_rows=True)

'''
