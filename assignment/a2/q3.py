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


def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def softmax_loss(weights, inputs, labels):
    num = inputs.shape[0]
    scores = np.dot(inputs, weights)
    prob = softmax(scores)
    loss = (-1/num) * np.sum(labels*np.log(prob))
    grad = (-1 /num) * np.dot(inputs.T, (labels - prob))
    return loss, grad

def auto_gd(train_images, train_labels, save_image):
    weights = np.zeros((784, 10))
    lr = 0.8
    for i in range (0, 10):
        loss, grad = softmax_loss(weights, train_images, train_labels)
        weights -= lr*grad

    if save_image:
        save_images(weights.T, "3_c.jpg")
    return weights.T

def avg_log_likelihood(data, labels, weights):
    average = []
    for i in range(len(data)):
        denomenator = np.multiply(weights , data[i])
        nonminator = np.dot(labels[i],denomenator)
        prob_i = np.divide(nonminator, logsumexp(denomenator))
        average.append(prob_i)
    return np.mean(np.array(average))


if __name__ == '__main__':
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()

    '''
    3c
    '''
    save_image = True
    weights = auto_gd(train_images, train_labels, save_image)
    '''
    3d
    '''
    print(avg_log_likelihood(test_images, test_labels, weights))
