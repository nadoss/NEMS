import pytest

from nems import initializers
from nems import priors


@pytest.fixture()
def modelspec():
    return initializers.from_keywords('wc18x2_lvl1_fir2x10')


@pytest.fixture()
def modelspec_with_phi(modelspec):
    return priors.set_mean_phi(modelspec)
