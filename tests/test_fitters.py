import numpy as np
from nems.fitters.mappers import simple_vector


def test_simple_vector_mapper(modelspec_with_phi):
    packer, unpacker = simple_vector(modelspec_with_phi)
    sigma = packer(modelspec_with_phi)

    # Add up and check that the number of entries in sigma matches the number
    # of expected coefficients given the modulespec.
    expected_n = 18*2 + 1 + 2*10
    assert len(sigma) == expected_n

    # Only one "tap" of each of the two FIR filters should be set to 1. All
    # other coefficients should be 0. Therefore, the sum of all coefficients
    # will be 2.
    assert np.sum(sigma) == 2

    # Now, try setting new values for the coefficients. See if they get set
    # properly.
    new_sigma = np.random.normal(size=expected_n)
    new_modelspec = unpacker(new_sigma)
    wc_phi_coefs = new_sigma[:18*2].reshape((2, 18))
    assert np.all(wc_phi_coefs == new_modelspec[0]['phi']['coefficients'])

    new_modelspec[0]['phi']['coefficients'][0, :] = 1
    assert np.all(wc_phi_coefs[0] != new_modelspec[0]['phi']['coefficients'][0])
    assert np.all(wc_phi_coefs[1] == new_modelspec[0]['phi']['coefficients'][1])
