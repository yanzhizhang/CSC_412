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
    # unnormalized_logprobs are in R
    # Targets must be 0 or 1
    t2 = targets * 2 - 1
    # Now t2 is -1 or 1, which makes the following form nice
    label_probabilities = -np.logaddexp(0, -unnormalized_logprobs*t2)
    return np.sum(label_probabilities, axis=-1)   # Sum across pixels.

def batched_loss(params, iter):
    data_idx = batch_indices(iter)
    return neglogprob(params, train_images[data_idx, :])

def neglogprob(params, data):
    # Implement this as the solution for 4c!
    for c in range(K):
        numerator =
        sum_inner = 0.0
        for d in range(D):
            sum =np.add(np.divide((data[c,d]+params[c,d]) , (params[c,d]*(1-params[c,d]))), sum_inner)
    return

# Get gradient of objective using autograd.
objective_grad = grad(batched_loss)

def print_perf(params, iter, gradient):
    if iter % 30 == 0:
        save_images(sigmoid(params), 'q4plot.png')
        print(batched_loss(params, iter))


def q4d():
	theta = pickle.load(open('theta2.p', 'rb'))
	pi = pickle.load(open('pi2.p', 'rb'))

	for i in range(20):
		print(i)
		image = train_images[i]
		top = image[:392]
		bottom = []


		for d in range(392,784):
			# Calculate p(x = 0| x_top , theta) and p(x = 1| x_top, theta)
			p_0 = 0
			p_1 = 0

			for k in range(K):
				product_1 = theta[k, d]
				product_0 = (1 - theta[k, d])

				product = 1
				# This is p(x_t | k, theta)
				for d_prime in range(0, 392):
					product = product * math.pow(theta[k, d_prime], top[d_prime]) * math.pow((1 - theta[k, d_prime]), (1 - top[d_prime]))

				p_1 = p_1 + (product_1 * product)
				p_0 = p_0 + (product_0 * product)


			# x_d = 1 if p(x = 1 | x_top, theta) >= p(x = 0 | x_top, theta)
			print('p_0: ' + str(p_0))
			print('p_1: ' + str(p_1))
			x_d = 1 if p_1 >= p_0 else 0
			bottom.append(x_d)

		img = top.tolist()  + bottom
		print('=============================')
		print(img)
		img = np.asarray(img)
		print(img.shape)
		data.save_images(img.reshape((1, 784)), str(i) + 'b.jpg')

if __name__ == '__main__':
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    auto_gd(train_images, train_labels)

    optimized_params = adam(objective_grad, init_params, step_size=0.2, num_iters=10000, callback=print_perf)
