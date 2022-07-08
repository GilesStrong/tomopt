import numpy as np

from tomopt.core import x0_from_mixture


def test_x0_from_mixture():
    props = x0_from_mixture([43.25 / 1.33, 42.7 / 3.52], [1.33, 3.52], [1, 3])
    assert np.abs(props["X0"] - 17.179) < 1e-3
    assert np.abs(props["density"] - 2.493) < 1e-3
