import jax.numpy as jnp

from jaxGPs.jaxGPs import Transform, ParameterStore, Parameter


# ------------------------- Transform ------------------------- #
def test_parameter_transform():
    t = Transform()
    for x in jnp.linspace(-10, 10, 100):
        assert jnp.isclose(t.reverse(t.forward(x)), x)

# NOTE: Test naive vs. stable invert edge-cases!
# NOTE: Test potentially relevant forward edge cases as well!


# ------------------------- Parameter ------------------------- #


# ------------------------- ParameterStore ------------------------- #
class DummyKernel(ParameterStore):
    ls = Parameter()
    amp = Parameter()

    def __init__(self):
        super().__init__("kernel")
        self.ls = 1.0
        self.amp = 1.0


def test_parameter_set_indiv():
    pstore = DummyKernel()
    assert jnp.isclose(pstore.ls, 1.0)
    newval = 5.0
    pstore.ls = newval
    assert jnp.isclose(pstore.ls, newval)
    assert jnp.isclose(pstore.parameters()['kernel']['ls'], newval)
    pdc = pstore._param_tree['kernel']['ls']
    assert jnp.isclose(pdc.transform.forward(pdc.value), newval)

def TODO_test_parameter_updates():
    """
    NOTE: It would be great to add some tests for gpr.update_parameters()
    """
    raise NotImplementedError

