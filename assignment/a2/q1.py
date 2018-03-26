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

def compute_means(train_images, train_labels, save_image):
    np.where(train_images>0.5, 1, 0)
    means = []
    for i in range(0, 10):
        i_digits = get_digits_by_label(train_images, train_labels, i)
        means.append(np.mean(i_digits, axis=0))
    if save_image:
        save_images(np.array(means), "1_c.jpg")
    return np.array(means)

def log_bernoulli_prod(flat_data, theta):
    c, d = theta.shape
    P_theta = np.where(flat_data > 0.5, theta, np.ones((c, d)) - theta)
    sum_c = np.sum(np.log(P_theta), axis=1)
    sum_c += np.log(0.1)
    logsumexp_temp = logsumexp(sum_c)
    sum_c -= logsumexp_temp
    return sum_c

def avg_log_likelihood(data, labels, theta):
    c, d = theta.shape
    average = 0
    for i in range(len(data)):
        sum_c = log_bernoulli_prod(data[i], theta)
        target = sum_c[np.argmax(labels[i])]
        average += target
    average /= len(data)
    return average

def prediction_accuracy(data, labels, theta):
    accuracy = 0
    for i in range(len(data)):
        prob_arr = log_bernoulli_prod(data[i], theta)
        pred = np.argmax(prob_arr)
        target = np.argmax(labels[i])
        if pred == target:
            accuracy += 1
    return np.divide(accuracy,len(data))


if __name__ == '__main__':
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    save_image = 0
    means = compute_means(train_images, train_labels, save_image)

    # print(avg_log_likelihood(train_images, test_labels, means))
    # print(avg_log_likelihood(test_images, test_labels, means))

    print (prediction_accuracy(train_images, train_labels, means))
    print (prediction_accuracy(test_images, test_labels, means))
