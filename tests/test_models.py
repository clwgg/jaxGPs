import pytest
import jax.numpy as jnp
import jax.random as random

from jaxGPs import GPR
from jaxGPs import ExponentialQuadratic


@pytest.fixture
def dummy_gpr():
    numpts = 10
    key = random.PRNGKey(0)
    x = random.uniform(key, shape=(numpts, 1)) * 10.0
    x = x.squeeze().sort().reshape(-1, 1)
    sigma_n = 0.1
    noise = random.normal(key, shape=(numpts, )) * sigma_n
    y = jnp.sin(x.squeeze()) + noise
    gpr = GPR(ExponentialQuadratic())
    gpr.update_data(x, y)
    return gpr


def test_copy(dummy_gpr):
    gpr = dummy_gpr
    gpr_copy = gpr.copy()
    assert id(gpr) != id(gpr_copy)
    for key in ['_param_tree', '_likelihood', 'kernel', 'mean', 'x', 'y']:
        assert id(gpr.__dict__[key]) != id(gpr_copy.__dict__[key])
