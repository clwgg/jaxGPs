import jax.numpy as jnp
import jax.random as random

from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib as mpl

from jaxGPs import GPR
from jaxGPs import ExponentialQuadratic

from scipy.stats import multivariate_normal


# --------------------------------------------------- #
# ----------------------       ---------------------- #
# --------------------------------------------------- #
def plot_posterior_fit(x, y, atx, mu, cov, ax):
    uncertainty = 1.96 * jnp.sqrt(jnp.diag(cov))
    ax.plot(x.squeeze(), y, ".")
    ax.plot(atx.squeeze(), mu, "-")
    ax.fill_between(atx.squeeze(),
                    mu + uncertainty,
                    mu - uncertainty,
                    alpha=0.1)


# --------------------------------------------------- #
# ----------------------       ---------------------- #
# --------------------------------------------------- #
def get_toy_data():
    numpts = 50
    key = random.PRNGKey(0)
    x = random.uniform(key, shape=(numpts, 1), maxval=jnp.pi * 6)
    x = jnp.sort(x, axis=0)
    sigma_n = 0.1
    noise = random.normal(key, shape=(numpts, )) * sigma_n
    y = jnp.sin(x.squeeze()) * jnp.linspace(1, 0, numpts)
    y += noise
    return x, y

def plot_toy_data():
    x, y = get_toy_data()
    fig = plt.figure(constrained_layout=True, figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(x.squeeze(), y, ".")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-1, 1)
    plt.savefig("./figures/toy_data.pdf", facecolor='None')
    plt.close()

def dist_toy_data():
    x, y = get_toy_data()
    fig = plt.figure(constrained_layout=True, figsize=(8, 4))
    ax = fig.add_subplot(121)
    ax.plot(x.squeeze(), x.squeeze(), ".")
    ax.set_title("x-by-x'")
    sqdist = jnp.sum((x[:, jnp.newaxis, :] -
                      x[jnp.newaxis, :, :]) ** 2, axis=-1)
    cw = mpl.cm.get_cmap( # type: ignore
        'coolwarm', 512)
    cw_hot = mpl.colors.ListedColormap( # type: ignore
        cw(jnp.linspace(0.5, 1.0, 256)))
    ax = fig.add_subplot(122)
    ax.set_title("d(x,x') distance matrix")
    ax.imshow(jnp.sqrt(sqdist), origin='lower', cmap=cw_hot, vmin=0, interpolation='none')
    plt.savefig("./figures/dist_toy_data.pdf", facecolor='None')
    plt.close()

def cov_toy_data():
    x, y = get_toy_data()
    sqdist = jnp.sum((x[:, jnp.newaxis, :] -
                      x[jnp.newaxis, :, :]) ** 2, axis=-1)
    K = ExponentialQuadratic()(x)
    fig = plt.figure(constrained_layout=True, figsize=(8, 4))
    ax = fig.add_subplot(121)
    ax.set_title("d(x,x') distance matrix")
    cw = mpl.cm.get_cmap( # type: ignore
        'coolwarm', 512)
    cw_hot = mpl.colors.ListedColormap( # type: ignore
        cw(jnp.linspace(0.5, 1.0, 256)))
    ax.imshow(jnp.sqrt(sqdist), origin='lower', cmap=cw_hot, vmin=0, interpolation='none')
    ax = fig.add_subplot(122)
    ax.set_title("k(x,x') covariance matrix")
    cw = mpl.cm.get_cmap( # type: ignore
        'coolwarm', 512)
    cw_cool = mpl.colors.ListedColormap( # type: ignore
        cw(jnp.linspace(0.5, 0.0, 256)))
    ax.imshow(K, origin='lower', cmap=cw_cool, vmin=0, interpolation='none')
    plt.savefig("./figures/cov_toy_data.pdf", facecolor='None')
    plt.close()

def fit_toy_data():
    x, y = get_toy_data()
    atx = jnp.linspace(x.min(), x.max(), 200)[:, jnp.newaxis]
    gpr = GPR(ExponentialQuadratic())
    gpr.update_data(x, y)
    gpr.fit_scipy()
    pprint(gpr.parameters())
    mu, cov = gpr.predict_f(atx)

    fig = plt.figure(constrained_layout=True, figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-1, 1)
    plot_posterior_fit(x, y, atx, mu, cov, ax)
    plt.savefig("./figures/fit_data.pdf", facecolor='None')
    plt.close()

def fit_toy_data_samples():
    x, y = get_toy_data()
    atx = jnp.linspace(x.min(), x.max(), 200)[:, jnp.newaxis]
    gpr = GPR(ExponentialQuadratic())
    gpr.update_data(x, y)
    gpr.fit_scipy()
    pprint(gpr.parameters())
    mu, cov = gpr.predict_f(atx)

    numsamples = 10
    mvn = multivariate_normal(mu, cov, seed=0)
    ysamples = mvn.rvs(size=numsamples)

    fig = plt.figure(constrained_layout=True, figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-1, 1)
    plot_posterior_fit(x, y, atx, mu, cov, ax)
    for i in range(numsamples):
        ax.plot(atx.squeeze(), ysamples[i,:], "-", label=i)
    plt.savefig("./figures/fit_data_samples.pdf", facecolor='None')
    plt.close()

def test_likelihood_split():
    x, y = get_toy_data()
    gpr = GPR(ExponentialQuadratic())
    gpr.update_data(x, y)
    atx = jnp.linspace(x.min(), x.max(), 200)[:, jnp.newaxis]
    gpr.kernel.amp = 0.2
    gpr.noise = 0.01

    split_lml = []
    ls_list = []
    for ls in jnp.geomspace(0.1, 10, 10):
        gpr.kernel.ls = ls
        split_lml.append(gpr.log_marginal_likelihood(split=True))
        ls_list.append(ls)

        fig = plt.figure(constrained_layout=True, figsize=(8, 4))
        ax = fig.add_subplot(121)
        mu, cov = gpr.predict_f(atx)
        plot_posterior_fit(x, y, atx, mu, cov, ax)
        ax.set_title(f"lengthscale: {ls:.02f}")

        ax = fig.add_subplot(122)
        ax.plot(ls_list, jnp.array(split_lml)[:, 0],
                linestyle="-", marker="o", color="red",
                label="Data Fit")
        ax.plot(ls_list, jnp.array(split_lml)[:, 1],
                linestyle="-", marker="o", color="blue",
                label="Capacity Control")
        ax.plot(ls_list, jnp.array(split_lml)[:, [0, 1]].sum(axis=1),
                linestyle="-", marker="o", color="purple")
        ax.legend()
        plt.savefig(f"./figures/RBF_fitls-{ls:.02f}.pdf", facecolor='None')
        plt.close()


# ------------------------- ------------------------- #
def compare_default_optimized():
    x, y = get_toy_data()

    atx = jnp.linspace(x.min(), x.max(), 200)[:, jnp.newaxis]
    gpr = GPR(ExponentialQuadratic())
#    gpr = GPR(Exponential())
    gpr.update_data(x, y)

    fig = plt.figure(constrained_layout=True, figsize=(8, 4))
    ax = fig.add_subplot(121)
    mu, cov = gpr.predict_f(atx)
    plot_posterior_fit(x, y, atx, mu, cov, ax)
    ax.set_title(f"{gpr.log_marginal_likelihood():.02f}")

    gpr.fit_scipy()
    pprint(gpr.parameters())

    ax = fig.add_subplot(122)
    mu, cov = gpr.predict_f(atx)
    plot_posterior_fit(x, y, atx, mu, cov, ax)
    ax.set_title(f"{gpr.log_marginal_likelihood():.02f}")

    plt.savefig(f"./figures/test_optim.pdf", facecolor='None')
    plt.close()


# ------------------------- ------------------------- #
if __name__ == "__main__":
    plot_toy_data()
#    dist_toy_data()
#    cov_toy_data()
    fit_toy_data()
    fit_toy_data_samples()
#    test_likelihood_split()
    compare_default_optimized()

