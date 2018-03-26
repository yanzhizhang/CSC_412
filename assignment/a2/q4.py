import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam
from autograd.scipy.misc import logsumexp
from autograd.scipy.special import expit as sigmoid

import os
import gzip
import struct
import array

import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve

from data import load_mnist, plot_images, save_images

# Load MNIST and Set Up Data
N_data, train_images, train_labels, test_images, test_labels = load_mnist()
train_images = np.round(train_images[0:10000])
train_labels = train_labels[0:10000]
test_images = np.round(test_images[0:10000])

# Starter Code for 4d
# A correct solution here only requires you to correctly write the neglogprob!
# Because this setup is numerically finicky
# the default parameterization I've given should give results if neglogprob is correct.
K = 30
D = 784

# Random initialization, with set seed for easier debugging
# Try changing the weighting of the initial randomization, default 0.01
init_params = npr.RandomState(0).randn(K, D) * 0.01

# Implemented batching for you
batch_size = 10
num_batches = int(np.ceil(len(train_images) / batch_size))
def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx+1) * batch_size)

# This is numerically stable code to for the log of a bernoulli density
# In particular, notice that we're keeping everything as log, and using logaddexp
# We never want to take things out of log space for stability
def bernoulli_log_density(targets, unnormalized_logprobs):
    t2 = targets * 2 - 1
    # Now t2 is -1 or 1, which makes the following form nice
    label_probabilities = -np.logaddexp(0, -unnormalized_logprobs*t2)
    return np.sum(label_probabilities, axis=-1)   # Sum across pixels.

def batched_loss(params, iter):
    data_idx = batch_indices(iter)
    return neglogprob(params, train_images[data_idx, :])

def neglogprob(params, data):
    # Implement this as the solution for 4c!
    params = sigmoid(params)
    result = np.exp(np.dot(data, np.log(params).T) + np.dot(1-data, np.log(1-params).T))
    result = np.mean(result, axis=1)
    result = np.log(result)
    return -np.mean(result)

# Get gradient of objective using autograd.
objective_grad = grad(batched_loss)

def print_perf(params, iter, gradient):
    if iter % 30 == 0:
        save_images(sigmoid(params), 'q4plot.png')
        print(batched_loss(params, iter))

def plot_bottom_half(xs, theta):
    result = np.zeros(xs.shape)
    for i in range(xs.shape[0]):
        x = xs[i]
        theta_top_temp = theta[:, :int(theta.shape[1]/2)]
        theta_bottom_temp = theta[:, int(theta.shape[1]/2):]
        x_top_temp = x[:int(theta.shape[1]/2)]
        x_bottom_temp = x[int(theta.shape[1]/2):]
        x_top_temp = np.full(theta_top_temp.shape, x_top_temp)
        first = theta_top_temp**x_top_temp * (1-theta_top_temp) ** (1-x_top_temp)
        first = np.prod(first, axis=1)
        first = np.full(theta_bottom_temp.T.shape, first).T
        x_bottom_temp = np.full(theta_bottom_temp.shape, x_bottom_temp)
        second = theta_bottom_temp
        second = np.sum(first * second, axis=0)
        first = np.sum(first, axis = 0)
        result[i, int(theta.shape[1]/2):] = second / first
        result[i, :int(theta.shape[1]/2)] = x[:int(theta.shape[1]/2)]
    return result

if __name__ == '__main__':
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()

    optimized_params = adam(objective_grad, init_params, step_size=0.2, num_iters=10000, callback=print_perf)
    optimized_params = sigmoid(optimized_params)

    save_images(optimized_params, '4_c.jpg')

    picked_images = train_images[np.random.permutation(train_images.shape[0])[:20], :]
    images = plot_bottom_half(picked_images, optimized_params)
    save_images(images, '4_d.jpg')
