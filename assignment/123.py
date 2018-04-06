from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
from autograd.scipy.misc import logsumexp
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.optimizers import adam

from data import *


def black_box_variational_inference(logprob, D, num_samples):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557"""

    def unpack_params(params):
        # Variational dist is a diagonal Gaussian.
        mean, log_std = params[:D], params[D:]
        return mean, log_std

    def gaussian_entropy(log_std):
        return 0.5 * D * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    rs = npr.RandomState(0)

    def variational_objective(params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        mean, log_std = unpack_params(params)
        #print(mean.mean(), log_std.mean())
        samples = rs.randn(num_samples, D) * np.exp(log_std) + mean
        lower_bound = np.mean(logprob(samples, t))
        entropy = gaussian_entropy(log_std)
        #print('entropy ', entropy)
        #print('elbo ', lower_bound)
        return -(lower_bound + entropy)

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params


if __name__ == '__main__':
    # -------------- LOADING DATASET ------------------------
    # load the images
    npr.seed(0)
    _, train_images, train_labels, test_images, test_labels = load_mnist()

    rand_idx = np.arange(train_images.shape[0])
    npr.shuffle(rand_idx)

    train_images = train_images[rand_idx]
    train_labels = train_labels[rand_idx]

    # UNIFORM CLASS SAMPLE CODE
    # uniformly sample each class
    cls_labels = train_labels.argmax(axis=1)
    cls_images = [train_images[cls_labels == i] for i in range(10)]
    rand_cls = np.int32(npr.random(30) / 0.1)
    rand_idx = [npr.randint(cls_images[cls].shape[0]) for cls in rand_cls]

    train_images = np.vstack([cls_images[rand_cls[i]][rand_idx[i]] for i in range(30)])
    train_labels = np.vstack([train_labels[cls_labels == rand_cls[i]][rand_idx[i]] for i in range(30)])

    # binarize
    train_images = np.round(train_images)
    test_images = np.round(test_images)

    # -------------- LOADING DATASET ------------------------
    print('LOADED DATASET')

    # hyperparameters
    learning_rate = 0.01
    sigma_prior = 40.0
    samples = 1000
    train_iters = 100

    # TODO: Pass Sanity Check
    # Specify an inference problem by its unnormalized log-density.
    D = 784 * 10

    ''' Returns the log(p(t | x, w)p(w | sigma^2)) on the samples
        x is samples taken of shape (samples_size, 784)
    '''
    def log_density(w, t):
        w_reshape = w.T.reshape(784, 10, samples)
        w_squared = ((w_reshape / sigma_prior) ** 2) / 2.0

        z = np.tensordot(train_images, w_reshape, axes=1)
        sf_sum = logsumexp(z, axis=1, keepdims=True)
        # should be positive
        log_softmax = z - np.hstack([sf_sum for i in xrange(10)])
        expected = np.dstack([train_labels for i in xrange(samples)])
        thing = expected * log_softmax
        return thing.sum(axis=0).mean(axis=0) - w_squared.sum(axis=0).sum(axis=0)

    # Build variational objective.
    objective, gradient, unpack_params = \
        black_box_variational_inference(log_density, D, samples)


    def avg_pred_acc(W, X, t):
        # compute the log MAP estimation error
        W_reshape = W.T.reshape(784, 10, 100)

        z = np.tensordot(X, W_reshape, axes=1)
        sf_sum = logsumexp(z, axis=1, keepdims=True)

        softmax = z - np.hstack([sf_sum for i in xrange(10)])
        softmax_avg = softmax.mean(axis=2)

        return np.mean(np.argmax(softmax_avg, axis=1) == np.argmax(t, axis=1))


    def accuracy(X, t, trained_params):
        rs = npr.RandomState(2)
        mean, log_std = unpack_params(trained_params)
        # get 100 samples of w.
        sa = rs.randn(100, D) * np.exp(log_std) + mean
        # compute likelihood of dataset X, t with w drawn from posterior
        return avg_pred_acc(sa, X, t)


    ''' Callback function that is executed once training is finished
        params - the variational parameters  [means, log_std]
        t - iteration number
        g - ignore
    '''
    def callback(params, t, g):
        print("i {}, lower bound {}, test {}, train {} ".format(t, -objective(params, t), accuracy(test_images, test_labels, params), accuracy(train_images, train_labels, params)))

    print("Optimizing variational parameters...")
    init_mean    = 0 * np.ones(D)
    init_log_std = 0 * np.ones(D)
    init_var_params = np.concatenate([init_mean, init_log_std])
    variational_params = adam(gradient, init_var_params, step_size=learning_rate, num_iters=train_iters, callback=callback)


    # ---------------- STOCHASTIC VARIATIONAL INFERENCE DONE ---------------
    # now get Monte Carlo estimate p(t | x) over the test and training set

    print('TRAIN set accuracy: ', accuracy(train_images, train_labels, variational_params))
    print('TEST set accuracy: ', accuracy(test_images, test_labels, variational_params))

    means = variational_params[:D].reshape(784, 10).T
    std = np.exp(variational_params[D:]).reshape(784, 10).T

    save_images(means, 'svi_means_sigma_%.5f.png' % sigma_prior, ims_per_row=5)
    save_images(std, 'svi_std_sigma_%.5f.png' % sigma_prior, ims_per_row=5)

    rs = npr.RandomState(0)
    # get and save a single sample from the variational posterior
    mean, log_std = unpack_params(variational_params)
    sample = rs.randn(1, D) * np.exp(log_std) + mean
    save_images(sample.reshape(784, 10).T, 'svi_std_posterior_%.5f.png' % sigma_prior, ims_per_row=5)
