import pytest
from jax import jit
import jax.numpy as jnp
import jax.random as random

from jaxGPs import Exponential, ExponentialQuadratic


# ------------------------- ------------------------- #
def gen_x(dims=1):
    numpts = 20
    key = random.PRNGKey(0)
    x = random.uniform(key, shape=(numpts, dims), maxval=jnp.pi * 6)
    return x


# ------------------------- ------------------------- #
@pytest.mark.parametrize("jaxk", [Exponential, ExponentialQuadratic])
def test_jit(jaxk):
    x = gen_x()
    assert jnp.isclose(
        jaxk()(x),
        jit(jaxk())(x),
    ).all()


def test_product_kernel():
    x = gen_x()

    pK1 = Exponential()(x) * Exponential()(x)
    k = Exponential() * Exponential()
    pK2 = k(x)
    assert jnp.isclose(pK1, pK2).all()

    pK1 = Exponential()(x) * Exponential()(x) * Exponential()(x)
    k = Exponential() * Exponential() * Exponential()
    pK2 = k(x)
    assert jnp.isclose(pK1, pK2).all()


def test_additive_kernel():
    x = gen_x()

    aK1 = Exponential()(x) + Exponential()(x)
    k = Exponential() + Exponential()
    aK2 = k(x)
    assert jnp.isclose(aK1, aK2).all()

    aK1 = Exponential()(x) + Exponential()(x) + Exponential()(x)
    k = Exponential() + Exponential() + Exponential()
    aK2 = k(x)
    assert jnp.isclose(aK1, aK2).all()


def test_haversine_pairwise():
    from jaxGPs.jaxGPs import HaversineKernel
    x = gen_x(2)
    haversine = HaversineKernel._haversine
    dist = jnp.sqrt(HaversineKernel()._sqdist(x, x))
    for i, j in [(0, 1), (4, 7), (17, 12)]:
        dval = haversine(*x[i], *x[j])
        assert jnp.isclose(dist[i, j], dval)
        assert jnp.isclose(dist[j, i], dval)


def test_active_dims():
    x = gen_x(2)

    k1 = Exponential()(x[:, [0]])
    k2 = Exponential(active_dims=[0])(x)
    assert jnp.isclose(k1, k2).all()

    k1 = Exponential()(x[:, [1]])
    k2 = Exponential(active_dims=[1])(x)
    assert jnp.isclose(k1, k2).all()

    k1 = Exponential()(x[:, [0]])
    k2 = Exponential()(x)
    assert not jnp.isclose(k1, k2).all()
