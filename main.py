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


### Regression
x = data_read_regression('linear',100)
brain=Network(learning_rate = 0.0001, momentum_rate = 0.8, iterations = 1000)
brain.add(Layer(1,5,'sigmoid'))
brain.add(Layer(5,50,'sigmoid'))
brain.add(Layer(50,1,'linear'))
brain.train_once(x[0],x[1],x[2],x[3])
errors = brain.train_and_evaluate(x[0],x[1],x[2],x[3])

# Plot errors
plt.plot(errors[0])
plt.plot(errors[1])
plt.legend(['training set error', 'test set error'], loc='upper right')
plt.title('Errors')
plt.show()

# Plot regression
plt.scatter(x[2], brain.fit(x[2]), s=3, facecolors='none', edgecolors='r', zorder = 10)
plt.plot(np.sort(x[0]), x[1][np.argsort(x[0])],linewidth = 7, zorder = 5)
plt.legend(['function','approximation'], loc ="upper left")
plt.title('Function from training set and approximation')
plt.show()


#### Classification
x = data_read_classification('circles',10000)
brain = Network(learning_rate = 0.001, momentum_rate = 0.8, iterations = 500)
brain.add(Layer(2,10,'sigmoid'))
brain.add(Layer(10,100,'sigmoid'))
brain.add(Layer(100,50,'sigmoid'))
brain.add(Layer(50,4,'sigmoid'))
errors = brain.train_and_evaluate(x[0],x[1],x[2],x[3])

# Plot errors
plt.plot(errors[0])
plt.plot(errors[1])
plt.legend(['training set error', 'test set error'], loc='upper right')
plt.title('Errors')
plt.show()

# Plot training set
plt.scatter(x[0][:,0], x[0][:,1] , c = x[4])
plt.title('Training set')
plt.show()

# Plot test set
plt.scatter(x[2][:,0], x[2][:,1], c = brain.fit(x[2]).reshape(x[2].shape[0]))
plt.title('Test set')
plt.show()
