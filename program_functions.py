import os
os.chdir('C:/Users/tomas/OneDrive/Documents/Studies/PW-IAD/MGU/projekt1-implementacja_backpropagation/MGUProjekt1')
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from network import Layer
from network import Network

def data_read_classification(name,size):
    train = pd.read_csv('C:/Users/tomas/OneDrive/Documents/Studies/PW-IAD/MGU/projekt1-implementacja_backpropagation/Classification/data.'+name+'.train.'+str(size)+'.csv')
    test = pd.read_csv('C:/Users/tomas/OneDrive/Documents/Studies/PW-IAD/MGU/projekt1-implementacja_backpropagation/Classification/data.'+name+'.test.'+str(size)+'.csv')
    train_features = np.array(train[['x','y']])
    train_labels = np.array(train['cls'])
    test_features = np.array(test[['x','y']])
    test_labels = np.array(test['cls'])

    # One Hot Encoder
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)

    train_labels_encoded = label_encoder.fit_transform(train_labels)
    train_labels_encoded = train_labels_encoded.reshape(len(train_labels_encoded), 1)
    train_labels_encoded = onehot_encoder.fit_transform(train_labels_encoded)

    test_labels_encoded = label_encoder.fit_transform(test_labels)
    test_labels_encoded = test_labels_encoded.reshape(len(test_labels_encoded), 1)
    test_labels_encoded = onehot_encoder.fit_transform(test_labels_encoded)

    return [train_features, train_labels_encoded, test_features, test_labels_encoded, train_labels - 1, test_labels - 1]

def data_read_regression(name,size):
    train = pd.read_csv('C:/Users/tomas/OneDrive/Documents/Studies/PW-IAD/MGU/projekt1-implementacja_backpropagation/Regression/data.'+name+'.train.'+str(size)+'.csv')
    test = pd.read_csv('C:/Users/tomas/OneDrive/Documents/Studies/PW-IAD/MGU/projekt1-implementacja_backpropagation/Regression/data.'+name+'.test.'+str(size)+'.csv')
    train_features = np.array(train['x'])
    train_labels = np.array(train['y'])
    test_features = np.array(test['x'])
    test_labels = np.array(test['y'])

    return [train_features, train_labels, test_features, test_labels]

def plot_errors(errors):
    plt.plot(errors[0])
    plt.plot(errors[1])
    plt.legend(['training set error', 'test set error'], loc='upper right')
    plt.title('Errors')
    plt.show()

def plot_regression(network, data):
    """
    Arguments:
    - network must be an object of class Network
    - data must be data given by function: data_read_regression
    """
    plt.scatter(data[2], network.fit(data[2]), s=3, facecolors='none', edgecolors='r', zorder = 10)
    plt.plot(np.sort(data[0]), data[1][np.argsort(data[0])],linewidth = 7, zorder = 5)
    plt.legend(['function','approximation'], loc ="upper left")
    plt.title('Function from training set and approximation')
    plt.show()

def plot_classification(network, data):
    """
    Arguments:
    - network must be an object of class Network
    - data must be data given by function: data_read_regression
    """
    # Plot training set
    plt.scatter(data[0][:,0], data[0][:,1] , c = data[4])
    plt.title('Training set')
    plt.show()

    # Plot test set
    plt.scatter(data[2][:,0], data[2][:,1], c = network.fit(data[2]).reshape(data[2].shape[0]))
    plt.title('Test set')
    plt.show()

