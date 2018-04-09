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
import scipy.misc

SIGMA = 10
lr = 1e-3
ITERATION = 100

def training_loop(train_images, train_labels, save_image):
    weights = np.zeros((784, 10))

    for i in range (0, ITERATION):
        prob=np.matmul(train_images,weights)
        prob_sum = np.array([scipy.misc.logsumexp(prob,axis=1)]).T
        softmax_w = -np.exp(prob-prob_sum)
        grad = np.matmul(softmax_w.T,train_images).T + np.matmul(train_images.T,train_labels)
        grad -= np.divide(weights, np.power(SIGMA,2))
        weights+=lr*grad

    if save_image:
        save_images(weights.T, "1_c.jpg")
    return weights.T

def avg_log_likelihood(data, labels, weights):
    bot = np.matmul(data,weights.T)
    top = np.multiply(bot, labels)
    top = np.sum(np.exp(top), axis =1 )
    bot = np.sum(np.exp(bot), axis =1 )
    return np.mean(np.log(np.divide(top, bot)))

def prediction_accuracy(data, labels, weights):
    prob=np.matmul(data,weights.T)
    correct=np.equal(np.argmax(prob,axis=1),np.argmax(labels,axis=1))
    return np.divide(np.sum(correct),len(labels))


if __name__ == '__main__':
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()

    train_images = train_images[0:300]
    train_images = np.round(train_images)
    train_labels = train_labels[0:300]
    save_image = True

    weights = training_loop(train_images, train_labels, save_image)

    print("Training set's average predictive log-likelihood per data point: %f" % avg_log_likelihood(train_images, train_labels, weights))
    print("Test set's average predictive log-likelihood per data point: %f" %  avg_log_likelihood(test_images, test_labels, weights))
    print("Training set's prediction accuracy: %f" % prediction_accuracy(train_images, train_labels, weights))
    print("Test set's prediction accuracy: %f" % prediction_accuracy(test_images, test_labels, weights))
