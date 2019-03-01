import numpy as np 

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def linear(x):
    return x
def linear_prime(x):
    return 1

ACTIVATIONS = {
    "sigmoid": sigmoid,
    "linear": linear
}
ACTIVATIONS_DERIVATIVES = {
    "sigmoid": sigmoid_prime,
    "linear": linear_prime
}
