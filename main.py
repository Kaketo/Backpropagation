import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Network():

    def __init__(self, size, activation_func, activation_func_deriv, iterations, learning_rate, momentum_rate, problem):
        """
        Arguments:
        ``size`` is a list, each element of ``size`` is number of neurons in each layer
        ``activation_func`` is a function of activation for neuron
        ``activation_func_deriv`` is a function that is derivative of activation function
        ``iterations`` is an integer
        ``learning_rate`` is a float between 0 and 1
        ``momentum_rate`` is a ...
        ``problem`` is a string 'classification' or 'regression'
        """
        # Check types of arguments
        assert isinstance(size,list) and callable(activation_func) and callable(activation_func_deriv)
        assert isinstance(iterations,int) and isinstance(learning_rate,float) and isinstance(momentum_rate,float)
        assert isinstance(problem,str)
        
        # Set up constant values
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.iterations = iterations
        if problem == 'regression':
            self.problem = 'regression'
        elif problem == 'classification':
            self.problem = 'classification'
        else:
            raise Exception('Problem must be string: classification or regression')
        
        # Set up activation function and its derivative
        self.activation_func = activation_func
        self.activation_func_deriv = activation_func_deriv

        # Set up number of layers in neural network
        self.num_layers = len(size)
        self.size = size

        # Computes list of biases arrays (num_layers - 1, 1) for each layer except input layer
        # Starting biases values are generated with normal distribution N(0,1)
        self.biases = []
        for i in range(self.num_layers - 1):
            self.biases.append(np.random.randn(size[i + 1], 1))
        
        # Computes list of weights arrays(matrixes) (each column is weigths outgoing from one neuron)
        # Starting weights values are generated with normal distribution N(0,1)
        self.weights = []
        for i in range(self.num_layers - 1):
            self.weights.append(np.random.randn(size[i + 1], size[i]))
    

    def feedforward(self, input_vector):
        """
        Return the output of the network if ``input_vector`` is input.
        """
        input_vector = np.array(input_vector).reshape(len(input_vector),1)
        for i in range(len(self.weights)):
            input_vector = self.activation_func(np.dot(self.weights[i], input_vector) + self.biases[i])
        return input_vector

    def backprop(self, x, y):
        """
        Arguments:
        x - inputs to first layer of network (list [1,2,3,...], size = number of neurons in first layer) 
        y - desired output from network (list [1,2,3,...], size = number of neurons in last layer)
        Return a list [gradient_biases, gradient_weights] 
        """
        # Make data ready
        # Change inputs to numpy arrrays of size [n,1]
        x = np.array(x).reshape(self.size[0],1)
        y = np.array(y).reshape(self.size[-1],1)
        # Create empty arrays with the same shape as weights and biases
        gradient_biases = [np.zeros(b.shape) for b in self.biases]
        gradient_weights = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        activation = x
        # List to store activations of all layers
        activations = [activation] 
        # List to store outputs of all layers
        outputs = [] 
        for i in range(len(self.weights)):
            # output of actual layer
            curr_output = np.dot(self.weights[i], activation) + self.biases[i]
            outputs.append(curr_output)
            activation = self.activation_func(curr_output)
            activations.append(activation)
            
        # first step - calculate error within the output of network and wanted answer 
        delta = (activations[-1] - y) * self.activation_func_deriv(outputs[-1])
        # second step - calculate gradient of biases and weights for last layer
        gradient_biases[-1] = delta
        gradient_weights[-1] = np.dot(delta, activations[-2].transpose())
        # third step - calculate gradient for all other layers
        for i in range(2, self.num_layers):
            z = outputs[-i]
            sp = self.activation_func_deriv(z)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sp
            gradient_biases[-i] = delta
            gradient_weights[-i] = np.dot(delta, activations[-i-1].transpose())
        return [gradient_biases, gradient_weights]
    
    def train_once(self, x, y, x_test, y_test):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a training data set.
        Arguments:
        x - numpy array of values that go to input of the network, size of each row  must be equal to number of neurons in input layer
        y - numpy array of values that we desire to be the output of network, size of each row mus be equal to number of neurons in output layer
        """

        gradient_biases = [np.zeros(b.shape) for b in self.biases]
        gradient_weights = [np.zeros(w.shape) for w in self.weights]

        # Calculate change in gradients of every layer using backprop() function
        prev_delta_gradient_biases = [np.zeros(b.shape) for b in self.biases]
        prev_delta_gradient_weights = [np.zeros(w.shape) for w in self.weights]
        for m in range(len(x)):
            [delta_gradient_biases, delta_gradient_weights] = self.backprop(x[m], y[m])
            for k in range(len(self.weights)):
                gradient_biases[k] = gradient_biases[k] + delta_gradient_biases[k]
                gradient_weights[k] = gradient_weights[k] + delta_gradient_weights[k] 
                self.weights[k] = self.weights[k] - self.learning_rate * gradient_weights[k] + self.momentum_rate * prev_delta_gradient_weights[k]
                self.biases[k] = self.biases[k] - self.learning_rate * gradient_biases[k] + self.momentum_rate * prev_delta_gradient_biases[k]
                prev_delta_gradient_weights[k] = - self.learning_rate * gradient_weights[k] + self.momentum_rate * prev_delta_gradient_weights[k]
                prev_delta_gradient_biases[k] = - self.learning_rate * gradient_biases[k] + self.momentum_rate * prev_delta_gradient_biases[k]
        
        # Calculate error of training and test set
        train_predictions = np.zeros(y.shape[0])
        test_predictions = np.zeros(y_test.shape[0])
        for m in range(len(x)):
            train_predictions[m] = self.feedforward(x[m])
        for n in range(len(x_test)):
            test_predictions[n] = self.feedforward(x_test[n])

        # Return error  
        return [sum(abs(y - train_predictions)) / len(x), sum(abs(y_test - test_predictions)) / len(x_test)]

    
    def train_and_evaluate(self, x, y):
        train_features, test_features, train_labels, test_labels = train_test_split(x, y, test_size = 0.20, random_state = 0)
        
        training_set_error = np.zeros(self.iterations)
        test_set_error = np.zeros(self.iterations)
        # Network training and errors calculation
        for i in range(self.iterations):
            [training_set_error[i], test_set_error[i]] = self.train_once(train_features, train_labels, test_features, test_labels)
        return [training_set_error, test_set_error]

        
            

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

### TESTY
# Load data
simple = pd.read_csv('C:/Users/tomas/OneDrive/Documents/Studies/PW-IAD/MGU/projekt1-implementacja_backpropagation/Classification/data.simple.test.100.csv')
simple_np = np.array(simple[['x','y']])
simple_np_ans = np.array(simple['cls'])


brain = Network([2,2,1],sigmoid,sigmoid_prime,200,0.01,0.1,'classification')
errors = brain.train_and_evaluate(simple_np,simple_np_ans-1)
brain.feedforward(simple_np[0])
simple_np_ans[0] - 1
brain.feedforward(simple_np[1])
simple_np_ans[1] - 1
plt.plot(errors[0])
plt.plot(errors[1])
plt.legend(['training set error', 'test set error'], loc='upper left')
plt.show()



# brain.feedforward([1,1])
# brain.feedforward([0,0])
# brain.weights
# brain.biases


# Test for XOR
xor = Network([2,2,1],sigmoid,sigmoid_prime,1000,0.1,0.1,'classification')
errors = xor.train_and_evaluate(np.array([[0,1],[1,0],[1,1],[0,0]]), np.array([[1],[1],[0],[0]]))
plt.plot(errors)
plt.show()

xor.feedforward([0,1])
xor.feedforward([1,0])
xor.feedforward([1,1])
xor.feedforward([0,0])
xor.weights
xor.biases