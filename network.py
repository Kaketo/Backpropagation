import os
os.chdir('C:/Users/tomas/OneDrive/Documents/Studies/PW-IAD/MGU/projekt1-implementacja_backpropagation/MGUProjekt1')
import numpy as np
import pandas as pd
import random
import activations

class Layer():
    def __init__(self, inputs, outputs, activation_func):
        assert isinstance(inputs,int)
        assert isinstance(outputs,int)
        
        self.inputs_neurons = inputs
        self.output_neurons = outputs
        self.activation_func = activations.ACTIVATIONS[activation_func]
        self.activation_func_derivative = activations.ACTIVATIONS_DERIVATIVES[activation_func]

        self.weights = np.random.randn(outputs, inputs)
        self.biases = np.random.randn(outputs,1)

        self.outputs = [None]
        self.activations = [None]
        self.gradient_weights = [None]
        self.gradient_biases = [None]

class Network():
    def __init__(self, learning_rate, momentum_rate, iterations):
        self.layers = []
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.iterations = iterations
    
    def add(self, layer):
        assert isinstance(layer, Layer)
        self.layers.append(layer)

    def feedforward(self, x):
        """
        Return the output of the network if ``x`` is input.
        """
        if isinstance(x,float) or isinstance(x,int):
            x = np.array(x)
        else:
            x = np.array(x).reshape(len(x),1)
        activation = x
        for i in range(len(self.layers)):
            activation = self.layers[i].activation_func(np.dot(self.layers[i].weights, activation) + self.layers[i].biases)
        return activation

    def backpropagation(self, x, y):
        """
        Arguments:
        x - inputs to first layer of network (list [1,2,3,...] or numpy array, size = number of neurons in first layer) 
        y - desired output from network (list [1,2,3,...] or numpy array, size = number of neurons in last layer)
        Return a list [gradient_biases, gradient_weights] 
        """
        x = np.array(x).reshape(self.layers[0].inputs_neurons,1)
        y = np.array(y).reshape(self.layers[-1].output_neurons,1)

        # Feedforward
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].outputs = np.dot(self.layers[i].weights, x) + self.layers[i].biases
                self.layers[i].activations = self.layers[i].activation_func(self.layers[i].outputs)
            else:
                self.layers[i].outputs = np.dot(self.layers[i].weights, self.layers[i-1].activations) + self.layers[i].biases
                self.layers[i].activations = self.layers[i].activation_func(self.layers[i].outputs)

        # First step - calculate error within the output of network and wanted answer 
        # Cost function E = 1/2 sum(y_est - y)^2
        delta = (self.layers[-1].activations - y) * self.layers[-1].activation_func_derivative(self.layers[-1].outputs)
        # Second step - calculate gradient of biases and weights for last layer
        self.layers[-1].gradient_biases = delta
        self.layers[-1].gradient_weights = np.dot(delta, self.layers[-2].activations.reshape(1,self.layers[-2].activations.shape[0]))
        # Third step - calculate gradient for all other layers TO DO
        for i in range(2, len(self.layers)+1):    
            activ_deriv = self.layers[-i].activation_func_derivative(self.layers[-i].outputs)
            delta = np.dot(self.layers[-i+1].weights.reshape(self.layers[-i+1].weights.shape[1],self.layers[-i+1].weights.shape[0]), delta) * activ_deriv
            self.layers[-i].gradient_biases = delta

            if i+1 > len(self.layers):
                self.layers[-i].gradient_weights = np.dot(delta, x.transpose())
            else:
                 self.layers[-i].gradient_weights = np.dot(delta, self.layers[-i-1].activations.transpose())
    
    def train_once(self, x, y, x_test, y_test):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a training data set.
        Arguments:
        x - numpy array of values that go to input of the network, size of each row  must be equal to number of neurons in input layer
        y - numpy array of values that we desire to be the output of network, size of each row mus be equal to number of neurons in output layer
        """
        # Use backpropagation to update weights and biases
        for i in range(len(x)):
            self.backpropagation(x[i], y[i])
            prev_delta_gradient_bias = 0
            prev_delta_gradient_weight = 0
            for j in range(len(self.layers)):
                self.layers[j].weights = self.layers[j].weights - self.learning_rate * self.layers[j].gradient_weights + self.momentum_rate * prev_delta_gradient_weight
                self.layers[j].biases = self.layers[j].biases - self.learning_rate * self.layers[j].gradient_biases + self.momentum_rate * prev_delta_gradient_bias
                prev_delta_gradient_weights = - self.learning_rate * self.layers[j].gradient_weights + self.momentum_rate * prev_delta_gradient_weight
                prev_delta_gradient_biases = - self.learning_rate * self.layers[j].gradient_biases + self.momentum_rate * prev_delta_gradient_bias


        # Calculate error of training and test set
        train_predictions = np.zeros([y.shape[0], self.layers[-1].biases.shape[0]])
        test_predictions = np.zeros([y_test.shape[0], self.layers[-1].biases.shape[0]])

        for m in range(len(x)):
            train_predictions[m] = self.feedforward(x[m]).transpose()
        for n in range(len(x_test)):
            test_predictions[n] = self.feedforward(x_test[n]).transpose()
        
        if test_predictions.shape[1] > 1:
            train_error = sum(sum(pow(y - train_predictions,2))) / (x.shape[0] * train_predictions.shape[1])
            test_error = sum(sum(pow(y_test - test_predictions,2))) / (x.shape[0] * test_predictions.shape[1])
        else:
            train_error = sum(pow(y.reshape(train_predictions.shape[0],1)-train_predictions,2)) / (x.shape[0] * train_predictions.shape[1])
            test_error = sum(pow(y_test.reshape(test_predictions.shape[0],1)-test_predictions,2)) / (x_test.shape[0] * test_predictions.shape[1])
        # Return errors
        return [train_error, test_error]

    def train_and_evaluate(self, x, y, x_test, y_test):
        training_set_error = np.zeros(self.iterations)
        test_set_error = np.zeros(self.iterations)
        # Network training and errors calculation
        for i in range(self.iterations):
            [training_set_error[i], test_set_error[i]] = self.train_once(x, y, x_test, y_test)
        return [training_set_error, test_set_error]

    def fit(self, input_vector):
        """
        Function to make a prediction on trained network where ``input_vector`` is np array

        If last layer has one neuron function produces regression problem
        If last layer has two or more neurons functions produces classification problem
        """
        output_vector = np.zeros((input_vector.shape[0],1))
        for i in range(input_vector.shape[0]):
            x_one_row = self.feedforward(input_vector[i])
            if self.layers[-1].biases.shape[0] > 1:
                output_vector[i] = np.argmax(x_one_row)
            else:
                output_vector[i] = x_one_row
        return output_vector.reshape(output_vector.shape[0],1)