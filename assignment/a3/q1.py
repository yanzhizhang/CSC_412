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

SIGMA = 9

def softmax(z):
    z -= np.max(z)
    return (np.exp(z).T / np.sum(np.exp(z), axis=1)).T

def softmax_grad(weights, train_images, train_labels):
    scores = np.dot(train_images, weights)
    prob = softmax(scores)
    grad = (-1 /train_images.shape[0]) * np.dot(train_images.T, (train_labels - prob))
    prior = 1/SIGMA
    grad += weights*prior
    return grad

def auto_gd(train_images, train_labels, save_image):
    weights = np.zeros((784, 10))
    lr = 0.7
    for i in range (0, 100):
        weights -= lr*softmax_grad(weights, train_images, train_labels)

    if save_image:
        save_images(weights.T, "1_c.jpg")
    return weights.T

def avg_log_likelihood(data, labels, weights):
    average = []
    for i in range(len(data)):
        denomenator = []
        for j in weights:
            denomenator.append(np.dot(j,data[i]))
        nonminator = np.exp(np.dot(labels[i],denomenator))
        prob_i = np.divide(nonminator, logsumexp(np.array(denomenator)))
        average.append(prob_i)
    return -np.log(np.mean(np.array(average)))

def prediction_accuracy(data, labels, theta):
    accuracy = 0
    for i in range(len(data)):
        denomenator = []
        for j in weights:
            denomenator.append(np.dot(j,data[i]))
        prediction = np.argmax(denomenator)
        target = np.argmax(labels[i])
        if prediction == target:
            accuracy += 1
    return np.divide(accuracy,len(data))


if __name__ == '__main__':
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()

    train_images = train_images[0:300]
    train_labels = train_labels[0:300]

    save_image = True
    weights = auto_gd(train_images, train_labels, save_image)


    print("Training set's average predictive log-likelihood per data point: %f" % avg_log_likelihood(train_images, train_labels, weights))
    print("Test set's average predictive log-likelihood per data point: %f" %  avg_log_likelihood(test_images, test_labels, weights))
    print("Training set's prediction accuracy: %f" % prediction_accuracy(train_images, train_labels, weights))
    print("Test set's prediction accuracy: %f" % prediction_accuracy(test_images, test_labels, weights))
