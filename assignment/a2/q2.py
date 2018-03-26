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

'''
q2 part c
'''
def get_digits_by_label(train_images, labels, query_label):
    digits =[]
    for i in range(train_images.shape[0]):
        if labels[i][query_label] == 1:
            digits.append(train_images[i])
    return digits

def generate_image(train_images, train_labels,save_image):
    np.where(train_images>0.5, 1, 0)
    generated_data = []
    means = []
    for i in range(0, 10):
        i_digits = get_digits_by_label(train_images, train_labels, i)
        means.append(np.mean(i_digits, axis=0))
        temp_generated_data = np.random.rand(784) - means[i]
        generated_data.append(np.where(temp_generated_data>0, 0, 1))
    if save_image:
        save_images(np.array(generated_data), "2_c.jpg")
    return np.array(means)


'''
q2 part f
'''
def log_bernoulli_prod_top(data_i, theta, pi):
    c, d = theta.shape
    P_n = np.where(data_i > 0.5, theta, np.ones((c, d)) - theta)
    P_n = P_n[:392]
    sum_c = np.sum(np.log(P_n), axis=1) + np.log(pi)
    temp = logsumexp(sum_c)
    sum_c -= temp
    return sum_c

def plot_bottom_half(data, labels, theta, size, save_image):
    result_arr = []
    theta_b = theta[:, 392:]
    c, d = theta_b.shape
    pi = 0.1
    for i in range(0, size):
        top_half = data[i, :392]
        bottom_half = data[i, 392:]
        P_cx = np.exp(log_bernoulli_prod_top(data[i], theta, pi))
        img_arr = []
        for n in range(392, 784):
            temp_sum = 0
            for class_j in range(0, c):
                prob = [1 - theta[class_j][n], theta[class_j][n]]
                choice = [0, 1]
                temp_sample = np.random.choice(choice, 1, p=prob)
                temp_sum += temp_sample * P_cx[class_j]
            img_arr.append(temp_sum)
        bottom_half = np.asarray(img_arr).reshape(392, )
        result = np.where(np.concatenate((top_half, bottom_half))>0.5, 1, 0)
        result_arr.append(result)

    if save_image:
        save_images(np.asarray(result_arr), "2_f.jpg")
    return np.asarray(result_arr)


if __name__ == '__main__':
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    save_image = 1
    theta = generate_image(train_images, train_labels,save_image)

    save_image = 1
    plot_bottom_half(test_images, test_labels, theta, 20, save_image)
