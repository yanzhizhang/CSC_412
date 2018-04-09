import numpy as np
from data import *
import matplotlib.pyplot as plt
import matplotlib.image
import scipy.misc
N_data, train_images, train_labels, test_images, test_labels=load_mnist()

N_data = 300
train_images = train_images[0:N_data]
train_images = np.round(train_images)
train_labels = train_labels[0:N_data]
N = np.sum(train_labels,axis=0)#how many examples in each class in training
T=X=[[] for i in range(10)]
number=np.array([i for i in range (10)])
train_label_number = np.dot(train_labels,number)
test_label_number = np.dot(test_labels,number)
for i in range(N_data):
    X[train_label_number[i]].append(train_images[i])

X=np.array(X)
dw=w=np.zeros([784,10])



max_it = 300+1
lr=1e-4
'''
lr=1e-4
300
m= 3.9
omega^2= 49.4024491055
average log liklihood in training:
-0.0106097814187
accuracy in training:
1.0
average log liklihood in test:
-0.725289736701
accuracy in test:
0.7822
'''
def logsumexp(a):
    return np.log(np.sum(np.exp(a),axis=1))

X=train_images
T=train_labels
omega_2 = 0.001
for m in np.arange(-5,5,0.1):
    omega_2 = np.e**(m)
    for it in range(1,max_it):
        Xw=np.matmul(X,w)

        de = np.array([scipy.misc.logsumexp(Xw,axis=1)]).T
        dw = -np.exp(Xw-de)
        dw = np.matmul(dw.T,X).T
        dw += np.matmul(X.T,T)
        dw -= w/omega_2
        w+=lr*dw

        Xw=np.matmul(train_images,w)
    print('m=',m)
    print('omega^2=',omega_2)
    print("average log liklihood in training:")
    XwLc = np.sum(Xw*train_labels,axis=1)
    #only the probability of X = c (X is c)
    print(np.sum(XwLc-scipy.misc.logsumexp(Xw,axis=1))/N_data)

    print("accuracy in training:")
    probabilities=Xw
    #all the probabilities fo X is 0, x is 1 .... 
    correct=np.equal(np.argmax(probabilities,axis=1),train_label_number)
    accuracy=np.sum(correct)/N_data
    print(accuracy)


    Xw=np.matmul(test_images,w)
    probabilities=Xw
    print("average log liklihood in test:")
    XwLc = np.sum(Xw*test_labels,axis=1)
    #only the probability of X = c (X is c)
    print(np.sum(XwLc-scipy.misc.logsumexp(Xw,axis=1))/10000)

    print("accuracy in test:")
    #all the probabilities fo X is 0, x is 1 .... 
    correct=np.equal(np.argmax(probabilities,axis=1),test_label_number)
    accuracy=np.sum(correct)/10000
    print(accuracy)

    #T=np.array([train_images[0]])
    save_images(w.T,"1c,omega^2="+str(omega_2)+".png")
