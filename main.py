import numpy as np
import random

class Network():

    def __init__(self, size, activation_func, activation_func_deriv, iterations, learning_rate, inertia_rate, problem):
        """
        Arguments:
        ``size`` is a list, each element of ``size`` is number of neurons in each layer
        ``activation_func`` is a function of activation for neuron
        ``activation_func_deriv`` is a function that is derivative of activation function
        ``iterations`` is an integer
        ``learning_rate`` is a float between 0 and 1
        ``inertia_rate`` is a ...
        ``problem`` is a string 'classification' or 'regression'
        """
        # Check types of arguments
        assert isinstance(size,list) and callable(activation_func) and callable(activation_func_deriv)
        assert isinstance(iterations,int) and isinstance(learning_rate,float) # dodac dla inertia
        assert isinstance(problem,str)
        
        # Setting up constant values
        self.learning_rate = learning_rate
        self.inertia_rate = inertia_rate
        if problem == 'regression':
            self.problem = 'regression'
        elif problem == 'classification':
            self.problem = 'classification'
        else:
            raise Exception('Problem must be string: classification or regression')

        # Computes number of layers in neural network
        self.num_layers = len(size)
        self.size = size

         # Computes list of biases arrays (num_layers - 1, 1) for each layer (of course without input layer)
        self.biases = []
        for i in range(self.num_layers - 1):
            self.biases.append(np.random.randn(size[i + 1], 1))
        
        # Computes list of weights arrays(matrixes) (each column is weigths outgoing from one neuron)
        self.weights = []
        for i in range(self.num_layers - 1):
            self.weights.append(np.random.randn(size[i + 1], size[i]))

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

brain = Network([1,2,3],sigmoid,sigmoid_prime,1000,0.1,0.1,'regression')
