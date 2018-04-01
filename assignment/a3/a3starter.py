from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam

import autograd.scipy.stats.norm as norm
from autograd.scipy.misc import logsumexp
import autograd.scipy.stats.multivariate_normal as mvn

from data import load_mnist, plot_images, save_images
import matplotlib.pyplot as plt

# Load MNIST and Set Up Data
N = 300
D = 784
N_data, train_images, train_labels, test_images, test_labels = load_mnist()
train_images = np.round(train_images[0:N])
train_labels = train_labels[0:N]
test_images = np.round(test_images[0:10000])
# test_labels = np.round(test_images[0:10000]) a typo
test_labels = test_images[0:10000]

K = 10
prior_std = 10.0

# Choose two pixels and plot the K specific weights against eachother
contourK = 2
px1 = 392 # Middle Pixel
# px2 = px1 + 28*5 +1 # Middle Pixel + 5 rows down
px2 = px1+14 # Middle left-most edge

# Random initialization, with set seed for easier debugging
# Try changing the weighting of the initial randomization, default 0.01
init_params = (npr.RandomState(0).randn(K, D) * 0.01, npr.RandomState(1).randn(K, D) * 0.01)



def logistic_logprob(params, images, labels):
    # params is a block of S x K x D params
    # images is N x D
    # labels is N x K one-hot
    # return S logprobs, summing over N
    mul = np.einsum('skd,nd->snk', params, images)
    normalized = mul - logsumexp(mul, axis=-1, keepdims=True)
    return np.einsum('snk,nk->s', normalized, labels)

def diag_gaussian_log_density(x, mu, log_std):
    # assumes that mu and log_std are (S x K X D),
    # so we sum out the last two dimensions.
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=(-1, -2))

def sample_diag_gaussian(mean, log_std, num_samples, rs):
    return rs.randn(num_samples, *np.shape(mean)) * np.exp(log_std) + mean




def elbo_estimate(var_params, logprob, num_samples, rs):
    """Provides a stochastic estimate of the variational lower bound.
    var_params is (mean, log_std) of a Gaussian."""
    mean, log_std = var_params
    samples = #todo
    log_ps = #todo
    log_qs = #todo
    return  #todo # E_q(z|x)[log p(x,z) - log q(z|x)]

def logprob_given_data(params):
    data_logprob = #todo
    prior_logprob = #todo
    return data_logprob + prior_logprob

def objective(var_params, iter):
    return -elbo_estimate(var_params, logprob_given_data,
                          num_samples=50, rs=npr.RandomState(iter))


# Code for plotting the isocontours below
def logprob_given_two(params, two_params):
    N = two_params.shape[0]

    params_adjust = np.zeros((N, K, D))
    params_adjust[:, contourK, px1] = two_params[:, 0]
    params_adjust[:, contourK, px2] = two_params[:, 1]

    adjusted_params = params_adjust #+ params

    return logprob_given_data(adjusted_params)

# Set up plotting code
def plot_isocontours(ax, func, xlimits=[-10, 10], ylimits=[-10, 10], numticks=21, **kwargs):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
    Z = zs.reshape(X.shape)
    plt.contour(X, Y, Z, **kwargs)
    ax.set_yticks([])
    ax.set_xticks([])

def plot_posterior_contours(mean_params,logstd_params):
    plt.clf()
    logprob_adj = lambda two_params: logprob_given_two(mean_params, two_params)
    plot_isocontours(ax, logprob_adj, cmap='Blues')
    mean_2d = mean_params[contourK, [px1,px2]]
    logstd_2s = logstd_params[contourK, [px1,px2]]
    variational_contour = lambda x: mvn.logpdf(x, mean_2d, np.diag(np.exp(2*logstd_2s)))
    plot_isocontours(ax, variational_contour, cmap='Reds')
    plt.draw()
    plt.pause(10)

# Set up figure.
fig = plt.figure(figsize=(8,8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)

# Get gradient of objective using autograd.
objective_grad = grad(objective)

def print_perf(var_params, iter, gradient):
    mean_params, logstd_params = var_params
    print(".", end='')
    if iter % 30 == 0:
        save_images(mean_params, 'a3plotmean.png')
        save_images(logstd_params, 'a3plotsgd.png')
        sample = sample_diag_gaussian(mean_params, logstd_params, num_samples=1, rs=npr.RandomState(iter))
        save_images(sample[0, :, :], 'a3plotsample.png')

        ## uncomment for Question 2f)
        # plot_posterior_contours(mean_params,logstd_params)

        print(iter)
        print(objective(var_params,iter))

# The optimizers provided by autograd can optimize lists, tuples, or dicts of parameters.
# You may use these optimizers for Q4, but implement your own gradient descent optimizer for Q3!
optimized_params = adam(objective_grad, init_params, step_size=0.05, num_iters=10000, callback=print_perf)
