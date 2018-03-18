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
def log_bernoulli_prod_top(flat_data, theta, pi):
    '''
    return log(p(c|x)) = log(p(c,x)/sum_c(p(c,x))
    :param flat_data:
    :param theta:
    :param pi:
    :return:
    '''
    c, d = theta.shape
    P_n = np.where(flat_data > 0.5, theta, np.ones((c, d)) - theta)
    P_n = P_n[:392]
    sum_c = np.sum(np.log(P_n), axis=1)
    sum_c += np.log(pi)
    #sum_c = np.multiply(pi, sum_c)
    # marginalizing all X's
    P_x = logsumexp(sum_c)
    sum_c -= P_x
    return sum_c

def comb_marginal_pixels(data, labels, theta, size):
    result_arr = []
    theta_b = theta[:, 392:]
    c, d = theta_b.shape
    pi = 0.1
    for i in range(0, size):
        top_half = data[i, :392]
        bottom_half = data[i, 392:]
        #generating class probability given top data
        P_cx = np.exp(log_bernoulli_prod_top(data[i], theta, pi))

        img_arr = []
        for n in range(392, 784):
            total = 0
            for cl in range(0, c):
                if theta[cl][n]>1 or theta[cl][n]< 0:
                    print (theta[cl][n])
                    print(cl,n)
                prob = [1 - theta[cl][n], theta[cl][n]]
                choice = [0, 1]
                c_sample = np.random.choice(choice, 1, p=prob)
                total +=c_sample*P_cx[cl]
            img_arr.append(total)

        P_pixel = np.asarray(img_arr).reshape(392, )
        bottom_half = P_pixel
        # pre_sum = np.multiply(P_pixel, np.broadcast(P_cx, axis=0))
        #bottom_half = np.sum(pre_sum, axis=0)
        result = np.concatenate((top_half, bottom_half))
        result = np.round(result)
        result_arr.append(result)
    save_images(np.asarray(result_arr), "2_f.jpg")

def comb_marginal_pixels_gray(data, labels, theta, size):
    result_arr = []
    theta_b = theta[:, 392:]
    c, d = theta_b.shape
    pi = 0.1
    for i in range(0, size):
        top_half = data[i, :392]
        bottom_half = data[i, 392:]
        #generating class probability given top data
        P_cx = np.exp(log_bernoulli_prod_top(data[i], theta, pi))

        img_arr = []
        for n in range(392, 784):
            total = 0
            for cl in range(0, c):
                # prob = [1 - theta[cl][n], theta[cl][n]]
                # choice = [0, 1]
                # c_sample = np.random.choice(choice, 1, p=prob)
                total +=theta[cl][n]*P_cx[cl]
            img_arr.append(total)

        P_pixel = np.asarray(img_arr).reshape(392, )
        bottom_half = P_pixel
        # pre_sum = np.multiply(P_pixel, np.broadcast(P_cx, axis=0))
        #bottom_half = np.sum(pre_sum, axis=0)
        result = np.concatenate((top_half, bottom_half))
        result_arr.append(result)
    test.plot_images(np.asarray(result_arr), plt)
    plt.show()
    plt.savefig('pic_2.png')


if __name__ == '__main__':
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    save_image = 1
    theta = generate_image(train_images, train_labels,save_image)

    comb_marginal_pixels(test_images, test_labels, theta, 20)
    # comb_marginal_pixels_gray(test_images, test_labels, theta, 20)
