import numpy as np 

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def linear(x):
    return x
def linear_prime(x):
    return 1

def softmax(x):
    return np.exp(x)/sum(np.exp(x))
def softmax_prime(x):
    return np.exp(np.sum(x)) / np.sum(np.exp(x)) ** 2

ACTIVATIONS = {
    "sigmoid": sigmoid,
    "linear": linear,
    "softmax": softmax
}
ACTIVATIONS_DERIVATIVES = {
    "sigmoid": sigmoid_prime,
    "linear": linear_prime,
    "softmax": softmax_prime
}
