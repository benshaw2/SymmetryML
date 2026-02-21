from .discovery.lse import LSE
from .discovery.invariance import discover_symmetry_coeffs
from .discovery.builders import getExtendedFeatureMatrix, getEquivariantResidualMatrix
#from .discovery.equivariance import EquivariantDiscovery  # <-- if file is equivariance.py, import accordingly
from .discovery.equivariance import EquivariantDiscovery  # (correct line; keep one of these)
from .vector_fields.euclidean import (
    generate_euclidean_killing_fields,
    generate_euclidean_killing_fields_with_names,
)

__all__ = [
    "LSE",
    "discover_symmetry_coeffs",
    "getExtendedFeatureMatrix",
    "getEquivariantResidualMatrix",
    "EquivariantDiscovery",
    "generate_euclidean_killing_fields",
    "generate_euclidean_killing_fields_with_names",
]

__version__ = "0.1.0"
