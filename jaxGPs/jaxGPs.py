from pprint import pprint
import numpy as np
from functools import partial
from weakref import ref as weakref
from copy import deepcopy
from abc import ABC, abstractmethod
from jax import value_and_grad
from jax import jit
from jax.example_libraries import optimizers
import jax.numpy as jnp
import jax.scipy as scipy
import jax.flatten_util as flatten_util
import jax.tree_util as tree_util
from jax.scipy.optimize import minimize
from scipy.optimize import minimize as scipy_minimize
from dataclasses import dataclass
from typing import Union, Optional
import jax
jax.config.update('jax_enable_x64', True)


# ------------------------- ------------------------- #
"""
NOTE: Some things this implementation is still missing
- Models
    - uncertain inputs!
- Kernels
    - change points (and chirp points?)
- Mean functions
    - Non-constant mean functions

Other (potential) GPR model features:
- caching for predictions (L, alpha)
- learning in batches over columns in Y
- parameter groups over columns in Y (e.g. allele frequency bins)
"""


# ------------------------- ------------------------- #
class Transform:
    @staticmethod
    def _softplus(x):
        return jnp.logaddexp(x, 0.)

    @staticmethod
    def _invert_softplus_naive(x):
        return jnp.log(jnp.exp(x) - 1.)

    @staticmethod
    def _invert_softplus_stable(x):
        return jnp.log(1 - jnp.exp(-x)) + x

    def _invert_softplus(self, x):
        return self._invert_softplus_stable(x)

    def forward(self, x):
        return self._softplus(x)

    def reverse(self, x):
        return self._invert_softplus(x)


class NoTransform:
    def forward(self, x):
        return jnp.array(x)

    def reverse(self, x):
        return jnp.array(x)


@dataclass
class ParameterDataClass:
    value: jnp.ndarray
    transform: Union[Transform, NoTransform]
    trainable: bool = True
_Parameter = ParameterDataClass


class ParameterDescriptor:
    def __init__(self, transform=True):
        if transform:
            self.t = Transform()
        else:
            self.t = NoTransform()

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, type):
        raw_val = obj._parameters[self.name].value
        return self.t.forward(raw_val)

    def __set__(self, obj, value):
        if self.name not in obj._parameters:
            obj._parameters[self.name] = _Parameter(
                value = self.t.reverse(value),
                transform = self.t,
            )
        else:
            obj._parameters[self.name].value = self.t.reverse(value)
Parameter = ParameterDescriptor


class ParameterStore:
    def __init__(self, name=None):
        self._head = True
        self._param_tree = {}
        if name is not None:
            self._param_tree[name] = {}
            self._parameters = self._param_tree[name]
            self._head = False

    def set_trainable(self, param, value):
        try:
            self._parameters[param].trainable = value
        except (KeyError, AttributeError) as error:
            raise ValueError(
                f"'{param}' is not a valid parameter of {self.__class__}"
            ) from None

    def update_parameters(self, new_values, is_transformed=True):
        if is_transformed:
            new_values = tree_util.tree_map(
                lambda v, t: t.transform.reverse(v),
                new_values,
                self._param_tree
            )
        new_val, new_tree = tree_util.tree_flatten(
            new_values, is_leaf=lambda x: x is None)
        param, treedef = tree_util.tree_flatten(
            self._param_tree)
        assert new_tree == treedef
        for new, p in zip(new_val, param):
            if new is not None:
                p.value = new

    def parameters(self):
        return tree_util.tree_map(
            lambda x: np.asarray(x.transform.forward(x.value)).tolist(),
            self._param_tree
        )

    def print_summary(self):
        info_dict = tree_util.tree_map(
            lambda x: (np.asarray(x.transform.forward(x.value)).tolist(),
                       "Trainable" if x.trainable else "NotTrainable"),
            self._param_tree)
        pprint(info_dict, width=1)

    def _get_raw_param_values(self, restrict=None):
        if restrict is None:
            return tree_util.tree_map(
                lambda x: x.value,
                self._param_tree
            )
        elif restrict == "train":
            return tree_util.tree_map(
                lambda x: x.value if x.trainable else None,
                self._param_tree
            )
        else:
            raise ValueError("`restrict` not in ['train', None]")

    def _register_child(self, child):
        if self._head:
            self._param_tree.update(child._param_tree)
            return

        for key in child._param_tree:
            curr = self._param_tree[key]
            if curr == {}:
                self._param_tree[key] = child._param_tree[key]
            elif type(curr) is list:
                self._param_tree[key] = curr + [child._param_tree[key]]
            else:
                self._param_tree[key] = [curr, child._param_tree[key]]


# ------------------------- ------------------------- #
class BaseKernel(ParameterStore, ABC):
    def __init__(self, active_dims: Optional[list] = None):
        super().__init__('kernel')
        self.adim = active_dims

    def __add__(self, other):
        return AdditiveKernel(self, other)

    def __mul__(self, other):
        return ProductKernel(self, other)

    @abstractmethod
    def _Kfun(self, X1, X2) -> jnp.ndarray:
        pass

    def __call__(self, X1, X2=None):
        X1 = jnp.array(X1)
        if X2 is None:
            X2 = X1
        else:
            X2 = jnp.array(X2)
        if self.adim is not None:
            return self._Kfun(X1[:, self.adim], X2[:, self.adim])
        return self._Kfun(X1, X2)


class CombinationKernel(BaseKernel):
    kernels: list

    def __init__(self, k1, k2):
        super().__init__()
        self.kernels = []
        self._extend_kernels(k1)
        self._extend_kernels(k2)

    def _extend_kernels(self, k):
        if isinstance(k, CombinationKernel):
            self.kernels.extend(k.kernels)
        else:
            self.kernels.extend([k])
        self._register_child(k)

    def _Kfun(self, X1, X2):
        raise NotImplementedError


class ProductKernel(CombinationKernel):
    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        return jnp.prod(
            jnp.stack([k(X1, X2) for k in self.kernels]),
            axis=0)


class AdditiveKernel(CombinationKernel):
    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        return jnp.sum(
            jnp.stack([k(X1, X2) for k in self.kernels]),
            axis=0)


class GaussMarkov(BaseKernel):
    """
    NOTE: This is currently only implemented for 1D inputs
    """
    amp = Parameter()

    def __init__(
            self,
            amp=1.0,
            active_dims: Optional[list] = None,
    ):
        super().__init__(active_dims=active_dims)
        self.amp = amp

    @staticmethod
    def _distfun(X1, X2):
        assert X1.shape[1] == X2.shape[1] == 1
        return jnp.minimum(X1, X2.T)

    def _Kfun(self, X1, X2):
        dist = self._distfun(X1, X2)
        return self.amp * dist


class EuclideanKernel(BaseKernel):
    ls = Parameter()
    amp = Parameter()

    def __init__(
            self,
            ls=1.0, amp=1.0,
            active_dims: Optional[list] = None,
    ):
        super().__init__(active_dims=active_dims)
        self.ls = ls
        self.amp = amp

    @staticmethod
    @jit
    def _sqdist(X1, X2):
        sqdist = jnp.sum((X1[:, jnp.newaxis, :] -
                          X2[jnp.newaxis, :, :]) ** 2, axis=-1)
        return jnp.maximum(sqdist, 1e-36)

    def _Kfun(self, X1, X2):
        sqdist = self._sqdist(X1, X2)
        raise NotImplementedError


class Exponential(EuclideanKernel):
    def _Kfun(self, X1, X2):
        sqdist = self._sqdist(X1, X2)
        return self.amp * jnp.exp(-0.5 * (jnp.sqrt(sqdist)/self.ls))


class ExponentialQuadratic(EuclideanKernel):
    def _Kfun(self, X1, X2):
        sqdist = self._sqdist(X1, X2)
        return self.amp * jnp.exp(-0.5 * (sqdist/self.ls**2))


class HaversineKernel(EuclideanKernel):
    @staticmethod
    @jit
    def _haversine(lon1, lat1, lon2, lat2):
        # convert decimal degrees to radians
        lon1 *= jnp.pi / 180
        lat1 *= jnp.pi / 180
        lon2 *= jnp.pi / 180
        lat2 *= jnp.pi / 180

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = jnp.sin(dlat/2)**2 + jnp.cos(lat1) * jnp.cos(lat2) * jnp.sin(dlon/2)**2
        c = 2 * jnp.arcsin(jnp.sqrt(a))
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles
        return c * r

    def _sqdist(self, X1, X2):
        assert X1.shape[1] == X2.shape[1] == 2
        gx2, gx1 = jnp.meshgrid(jnp.arange(len(X2)), jnp.arange(len(X1)))

        dist = self._haversine(
            *jnp.hsplit(X1.at[jnp.reshape(gx1,(-1,))].get(), 2),
            *jnp.hsplit(X2.at[jnp.reshape(gx2,(-1,))].get(), 2),
        )
        sqdist = jnp.reshape(dist**2, gx1.shape)
        return jnp.maximum(sqdist, 1e-36)


class HavExponential(HaversineKernel, Exponential):
    pass


class HavExponentialQuad(HaversineKernel, ExponentialQuadratic):
    pass


# ------------------------- ------------------------- #
class BaseMean(ParameterStore, ABC):
    def __init__(self):
        super().__init__('mean')

    @abstractmethod
    def __call__(self, X):
        pass


class ConstantMean(BaseMean):
    mean = Parameter(transform=False)

    def __init__(self, mean=0.0):
        super().__init__()
        self.mean = mean

    def __call__(self, X):
        return self.mean


# ------------------------- ------------------------- #
class GaussianLikelihood(ParameterStore):
    noise = Parameter()

    def __init__(self, noise=0.1):
        super().__init__('likelihood')
        self.noise = noise
        self.scaling = None

    def __call__(self):
        if self.scaling is None:
            return self.noise
        return self.noise / self.scaling


# ------------------------- ------------------------- #
class BaseModel(ParameterStore, ABC):
    def __init__(self):
        super().__init__()


class GPR(BaseModel):
    epsilon = 1e-6

    def __init__(self, kernel, mean=None, noise=0.1):
        super().__init__()
        likelihood = GaussianLikelihood(noise=noise)
        self._likelihood = likelihood
        self._register_child(likelihood)
        self.kernel = kernel
        self._register_child(kernel)
        if mean is None:
            mean = ConstantMean()
            mean.set_trainable('mean', False)
        self.mean = mean
        self._register_child(mean)
        self._init_data_vars()

    @property
    def noise(self):
        return self._likelihood

    @noise.setter
    def noise(self, value):
        self._likelihood.noise = value

    def _init_data_vars(self):
        self.x = None
        self.y = None
        self.n = None

    def update_data(self, x, y, noise_scaling=None):
        self.x = jnp.array(x)
        self.y = jnp.array(y)
        self.n = jnp.shape(x)[0]
        if noise_scaling is not None:
            noise_scaling = jnp.array(noise_scaling)
        self.noise.scaling = noise_scaling

    def copy(self):
        return deepcopy(self)

    def log_marginal_likelihood(self, split=False):
        assert self.n is not None

        log2pi = jnp.log(2. * 3.14159265)
        part1 = - 0.5 * self.yTa
        part2 = - jnp.sum(jnp.log(jnp.diag(self.L)))
        part3 = - 0.5 * self.n * log2pi

        if split:
            return part1, part2, part3
        else:
            return jnp.sum(part1 + part2 + part3)

    def predict_f(self, atx):
        kcross = self.kernel(self.x, atx)
        mu = jnp.dot(kcross.T, self.alpha)
        mu += self.mean(atx)

        v = scipy.linalg.solve_triangular(
            self.L, kcross, lower=True)
        Katx = self.kernel(atx)
        Katx += jnp.eye(Katx.shape[0]) * self.epsilon
        cov = Katx - jnp.dot(v.T, v)
        return mu, cov

    def predict_y(self, atx):
        mu, cov = self.predict_f(atx)
        diag_noise = jnp.eye(cov.shape[0]) * self.noise()
        return mu, cov + diag_noise

    @property
    def L(self):
        K = self.kernel(self.x)
        K += jnp.eye(self.n) * (self.noise() + self.epsilon)
        return scipy.linalg.cholesky(K, lower=True)

    @property
    def alpha(self):
        return scipy.linalg.solve_triangular(
            self.L.T, scipy.linalg.solve_triangular(
                self.L, self.y - self.mean(self.x), lower=True))

    @property
    def yTa(self):
        return jnp.sum(jnp.square(
            scipy.linalg.solve_triangular(
                self.L, self.y - self.mean(self.x), lower=True)), axis=0)

    def _get_objective(self, flattened=False):

        def log_m_likelihood_fun(train, unravel):
            if unravel is not None:
                train = unravel(train)
            self.update_parameters(train, is_transformed=False)
            lml = self.log_marginal_likelihood(split=False)
            return jnp.sum(lml)

        if flattened:
            return lambda train, unravel: - log_m_likelihood_fun(
                train, unravel)
        else:
            return lambda train: - log_m_likelihood_fun(
                train, None)

    def fit(self, lr=0.01, niter=100):
        """
        NOTE: Add convergence testing and progress output
        """
        objective = self._get_objective()
        params = self._get_raw_param_values("train")

        opt_init, opt_update, get_params = optimizers.adam(step_size=lr)
        opt_state = opt_init(params)

        @jit
        def step(step, opt_state):
            value, grads = value_and_grad(objective)(get_params(opt_state))
            opt_state = opt_update(step, grads, opt_state)
            return value, opt_state

        for i in range(niter):
            value, opt_state = step(i, opt_state)
        self.update_parameters(get_params(opt_state), is_transformed=False)

    def fit_jax_scipy(self):
#        method="l-bfgs-experimental-do-not-rely-on-this"
        method="bfgs"

        params, unravel = flatten_util.ravel_pytree( # type: ignore
            self._get_raw_param_values("train"))
        objective = partial(self._get_objective(flattened=True), unravel=unravel)

        res = minimize(fun=objective, x0=params, method=method)
        self.update_parameters(unravel(res.x), is_transformed=False)
        return res

    def fit_scipy(self, options=None):
        method = "L-BFGS-B"

        params, unravel = flatten_util.ravel_pytree( # type: ignore
            self._get_raw_param_values("train"))
        objective = partial(self._get_objective(flattened=True), unravel=unravel)

        def numpy_v_g(x):
            value, grad = value_and_grad(jit(objective))(x)
            return np.array(value), np.array(grad)

        res = scipy_minimize(fun=numpy_v_g,
                             x0=np.array(params),
                             jac=True,
                             method=method,
                             options=options)
        self.update_parameters(unravel(res.x), is_transformed=False)
        return res
