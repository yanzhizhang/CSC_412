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
        means.append(np.mean(i_digits, axis=0).reshape(28,28))

    all_concat = np.concatenate(means, 1)

    # plt.imshow(means, cmap='gray')
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def avg_pre_log(train_images, train_labels):
    np.where(train_images>0.5, 1, 0)
    means = []
    for i in range(0, 10):
        i_digits = get_digits_by_label(train_images, train_labels, i)
        means.append(np.mean(i_digits, axis=0).reshape(28,28))




if __name__ == '__main__':
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    compute_means(train_images, train_labels)
