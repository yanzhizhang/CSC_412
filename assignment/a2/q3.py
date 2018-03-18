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


def softmax(x):
    return x - logsumexp(x, axis = 1)[: , None]

# w is a weight matrix of 10 by 784， x is an image of N by 784
def logistic_predictions(w, x):
    temp = np.dot(x, w)
    # print(temp.shape)
    return softmax(temp)

# This is -log likelihood
def training_loss(w,x,target):
	preds = logistic_predictions(w, x)
	# Cross entropy
	label_probabilities = target * preds
	return -np.sum(label_probabilities)

def auto_gd(train_images, train_labels):
    # Define a function that returns gradients of training loss using autograd.
    training_gradient_fun = grad(training_loss)
    # Optimize weights using gradient descent.
    weights = np.zeros((784, 10))
    for i in range(100):
        weights -= training_gradient_fun(weights,train_images,train_labels) * 0.3
    # print("Trained loss:", training_loss(weights,train_images,train_labels))
    # Plot w
    w = []
    for i in range(10):
        w_i = weights[:, i]
        w.append(w_i)
    save_images(np.array(w), "q3_c" + '.jpg')

# def q3e():
# 	w = pickle.load(open('weights.p', 'rb'))
# 	train_prediction = logistic_predictions(w, train_images)
# 	l_train = np.average(np.sum(train_prediction, axis = 1))
# 	# print(train_prediction[0])
# 	train_prediction = np.argmax(train_prediction, axis = 1)
# 	# print(train_prediction[0])
#
# 	test_prediction = np.dot(test_images, w)
# 	l_test = np.average(np.sum(test_prediction, axis = 1))
# 	test_prediction = np.argmax(test_prediction, axis = 1)
#
# 	num_correct = 0
# 	for i, prediction in enumerate(train_prediction):
# 		print('Predict: ' + str(prediction) + ' label: ' + str(train_labels[i]))
# 		if (train_labels[i][prediction] == 1):
# 			num_correct = num_correct + 1
# 	train_accuracy = float(num_correct) / len(train_labels)
#
# 	num_correct = 0
# 	for i, prediction in enumerate(test_prediction):
# 		if (test_labels[i][prediction] == 1):
# 			num_correct = num_correct + 1
# 	test_accuracy = float(num_correct) / len(test_labels)
#
# 	print(train_accuracy)
# 	print(test_accuracy)
# 	print(l_train)
# 	print(l_test)

if __name__ == '__main__':
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    auto_gd(train_images, train_labels)
