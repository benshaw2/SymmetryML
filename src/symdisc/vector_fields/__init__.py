"""
Vector field utilities.

Currently includes:
- Euclidean Killing fields in R^d (translations and rotations),
  returned as batch-aware callables that accept (d,) or (N, d).

Import examples:
    from symdisc.vector_fields import generate_euclidean_killing_fields
    from symdisc.vector_fields.euclidean import generate_euclidean_killing_fields
"""

from .euclidean import (
    generate_euclidean_killing_fields,
    generate_euclidean_killing_fields_with_names,
)

__all__ = [
    "generate_euclidean_killing_fields",
    "generate_euclidean_killing_fields_with_names",
]
