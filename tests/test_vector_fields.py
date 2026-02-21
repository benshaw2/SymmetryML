import numpy as np
from symdisc.vector_fields import generate_euclidean_killing_fields

def test_killing_fields_shapes():
    d = 3
    kvs = generate_euclidean_killing_fields(d)
    # Expected count: d translations + d*(d-1)/2 rotations
    expected = d + d*(d-1)//2
    assert len(kvs) == expected

    x = np.array([1.0, 2.0, 3.0])
    X = np.stack([x, 2*x], axis=0)

    for f in kvs:
        v = f(x)
        V = f(X)
        assert v.shape == (d,)
        assert V.shape == (2, d)
