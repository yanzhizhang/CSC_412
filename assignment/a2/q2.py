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

def sample_image(train_images, train_labels):
    np.where(train_images>0.5, 1, 0)
    generated_data = []
    means = []
    for i in range(0, 10):
        i_digits = get_digits_by_label(train_images, train_labels, i)
        means.append(np.mean(i_digits, axis=0))
        temp_generated_data = np.random.rand(784) - means[i]
        np.where(temp_generated_data>0, 1, 0)
        generated_data.append(temp_generated_data)

    save_images(np.array(generated_data), "2_c" + '.jpg')

def generate_botttom(train_images, train_labels):
    np.where(train_images>0.5, 1, 0)
    means = []
    for i in range(0, 10):
        i_digits = get_digits_by_label(train_images, train_labels, i)
        means.append(np.mean(i_digits, axis=0).reshape(28,28))

if __name__ == '__main__':
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    sample_image(train_images, train_labels)
    generate_botttom(train_images, train_labels)
