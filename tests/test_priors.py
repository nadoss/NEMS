import numpy as np

from nems import priors


def test_set_phi(modelspec):
    fir_coefficients = np.zeros((2, 10))
    fir_coefficients[:, 1] = 1
    expected = [
        {'coefficients': np.zeros((2, 18)).tolist()},
        {'level': [0]},
        {'coefficients': fir_coefficients.tolist()},
    ]

    modelspec_phi = priors.set_mean_phi(modelspec)
    actual = [m['phi'] for m in priors.set_mean_phi(modelspec)]
    assert actual == expected
