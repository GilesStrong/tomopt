from tomopt.core import LADLE_MATERIALS

import numpy as np


def test_ladle_materials():
    # Check that probabilities sum up to 1
    for material, properties in LADLE_MATERIALS.items():
        assert np.sum(list(properties["components"].values())) - 1 < 1e-5
