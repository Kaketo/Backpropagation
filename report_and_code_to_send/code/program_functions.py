import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from network import Layer
from network import Network

def data_read_classification(name,size):
    train = pd.read_csv('data.'+name+'.train.'+str(size)+'.csv')
    test = pd.read_csv('data.'+name+'.test.'+str(size)+'.csv')
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
    train = pd.read_csv('data.'+name+'.train.'+str(size)+'.csv')
    test = pd.read_csv('data.'+name+'.test.'+str(size)+'.csv')
    train_features = np.array(train['x'])
    train_labels = np.array(train['y'])
    test_features = np.array(test['x'])
    test_labels = np.array(test['y'])

    return [train_features, train_labels, test_features, test_labels]

def plot_errors(errors):
    plt.plot(errors[0])
    plt.plot(errors[1])
    plt.legend(['training set error', 'test set error'], loc='upper right')
    plt.title('Training set and test set errors')
    plt.xlabel('Iterations (epochs)')
    plt.ylabel('Error')
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

def plot_classification_mesh(network, data):
    """
    Arguments:
    - network must be an object of class Network
    - data must be data given by function: data_read_classification
    """
    min_x = min(min(data[0][:,0]), min(data[0][:,1]))
    max_x = max(max(data[0][:,0]), max(data[0][:,1]))
    density = (max_x - min_x) / 1000
    x1 = np.arange(min_x, max_x, density)
    y1 = np.arange(min_x, max_x, density)
    x1 = np.repeat(x1,50)
    y1 = np.tile(y1,50)
    data[2] = np.c_[x1,y1]
    
    # Plot test set
    plt.scatter(data[2][:,0], data[2][:,1], cmap = 'Dark2',s = 35, alpha = 0.01, c = network.fit(data[2]).reshape(data[2].shape[0]))
    # Plot training set
    plt.scatter(data[0][:,0], data[0][:,1], cmap = 'Dark2', c = data[4])
    plt.title('Training and test set')
    plt.show()

def test_lr(network, x, lrate , momentum = 0.9, iterations = 100):
    """
    Arguments:
    - network must be an object of class Network
    - data must be data given by function: data_read_classification
    - lrate must be np array 
    """
    errors = np.zeros(len(lrate))
    for i in range(len(lrate)):
        brain = Network(learning_rate = lrate[i], momentum_rate = momentum, iterations = iterations)
        for j in range(len(network.layers)):
            brain.add(Layer(network.layers[j].inputs_neurons, network.layers[j].output_neurons, network.layers[j].activation_func_name))
        all_errors = brain.train_and_evaluate(x[0],x[1],x[2],x[3])
        errors[i] = all_errors[0][iterations - 1]
    plt.plot(sorted(lrate), errors)
    plt.xlabel('Learning Rate')
    plt.ylabel('Error of network after ' + str(iterations) + ' iterations')
    plt.show()

def weights_norms_plot(errors):
    matrix_norm = []
    for i in range(len(errors[2])):
        matrix_norm_iter = []
        for j in range(len(errors[2][i])):
            matrix_norm_iter.append(np.linalg.norm(errors[2][i][j]))
        matrix_norm.append(matrix_norm_iter)
    
    for i in range(len(errors[2][0])):
        plt.plot(range(len(errors[2])),np.array(matrix_norm)[:,i], label = 'layer ' + str(i))
    plt.xlabel('Iterations (epochs)')
    plt.ylabel('Frobenius norm')
    plt.title('Frobenius norm of all weights in all layers')
    plt.legend()
    plt.show()