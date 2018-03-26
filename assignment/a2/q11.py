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

def get_digits_by_label(train_images, labels, query_label):

    digits =[]
    for i in range(train_images.shape[0]):
        if labels[i][query_label] == 1:
            digits.append(train_images[i])
    return digits

def compute_means(train_images, train_labels):
    np.where(train_images>0.5, 1, 0)
    means = []
    for i in range(0, 10):
        i_digits = get_digits_by_label(train_images, train_labels, i)
        means.append(np.mean(i_digits, axis=0))

    save_images(np.array(means), "1_c.jpg")
    return np.array(means)

def avg_log_likelihood(data, labels, theta):
    '''
    1/N*sum(logP(c|x, theta, pi))
    '''
    pi = 0.1
    c, d = theta.shape
    average = 0
    for i in range(0, len(data)):
        sum_c = log_bernoulli_prod(data[i], theta, pi)
        target = sum_c[np.argmax(labels[i])]

        average += target
    average /= len(data)
    return average

def log_bernoulli_prod(flat_data, theta, pi):
    '''
    return log(p(c|x)) = log(p(c,x)/sum_c(p(c,x))
    :param flat_data:
    :param theta:
    :param pi:
    :return:
    '''
    c, d = theta.shape
    P_n = np.where(flat_data > 0.5, theta, np.ones((c, d)) - theta)
    sum_c = np.sum(np.log(P_n), axis=1)
    sum_c += np.log(pi)
    P_x = logsumexp(sum_c)
    sum_c -= P_x
    return sum_c


if __name__ == '__main__':
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    means = compute_means(train_images, train_labels)
    avg_log_likelihood(test_images, test_labels, means)
