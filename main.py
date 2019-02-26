import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

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
        # Version for classification
        if self.problem == 'classification':
            input_vector = np.array(input_vector).reshape(len(input_vector),1)
            for i in range(len(self.weights)):
                input_vector = self.activation_func(np.dot(self.weights[i], input_vector) + self.biases[i])
            return input_vector
        
        # Version for regression
        else:
            input_vector = np.array(input_vector)
            for i in range(len(self.weights)):
                if i != (len(self.weights)-1):
                    input_vector = self.activation_func(np.dot(self.weights[i], input_vector).reshape(brain_r.biases[i].shape[0],1) + self.biases[i])
                else:
                    input_vector = np.dot(self.weights[i], input_vector).reshape(brain_r.biases[i].shape[0],1) + self.biases[i]
            return input_vector

    def backprop(self, x, y):
        """
        Arguments:
        x - inputs to first layer of network (list [1,2,3,...], size = number of neurons in first layer) 
        y - desired output from network (list [1,2,3,...], size = number of neurons in last layer)
        Return a list [gradient_biases, gradient_weights] 
        """
        if self.problem == 'classification':
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
        else:
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
                if i != (len(self.weights)-1):
                    activation = self.activation_func(curr_output)
                    activations.append(activation)
                else:
                    activation = curr_output
                    activations.append(activation)

            # first step - calculate error within the output of network and wanted answer 
            delta = (activations[-1] - y)
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
        if self.problem == 'classification':
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
            train_predictions = np.zeros([y.shape[0],y.shape[1]])
            test_predictions = np.zeros([y_test.shape[0], y_test.shape[1]])
            for m in range(len(x)):
                train_predictions[m] = self.feedforward(x[m]).transpose()
            for n in range(len(x_test)):
                test_predictions[n] = self.feedforward(x_test[n]).transpose()

            # Return error  
            return [sum(sum(abs(y - train_predictions))) / (len(x) * y.shape[1]), sum(sum(abs(y_test - test_predictions))) / (len(x_test)*y.shape[1])]
        else:
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
            train_predictions = np.zeros([y.shape[0]])
            test_predictions = np.zeros([y_test.shape[0]])
            for m in range(len(x)):
                train_predictions[m] = self.feedforward(x[m])
            for n in range(len(x_test)):
                test_predictions[n] = self.feedforward(x_test[n])

            # Return error  
            return [sum(abs(y - train_predictions)) / (len(x)), sum(abs(y_test - test_predictions)) / (len(x_test))]
            
    
    def train_and_evaluate(self, x, y, x_test, y_test):
        #train_features, test_features, train_labels, test_labels = train_test_split(x, y, test_size = 0.20, random_state = 0)
        
        training_set_error = np.zeros(self.iterations)
        test_set_error = np.zeros(self.iterations)
        # Network training and errors calculation
        for i in range(self.iterations):
            [training_set_error[i], test_set_error[i]] = self.train_once(x, y, x_test, y_test)
        return [training_set_error, test_set_error]

    def fit(self, input_vector):
        """
        Function to make a prediction on trained network where ``input_vector`` is 
        """
        # Version for classification
        output_vector = np.zeros(input_vector.shape[0])
        for k in range(input_vector.shape[0]):
            x_one_row = input_vector[k].reshape(input_vector.shape[1],1)
            for i in range(len(self.weights)):
                x_one_row = self.activation_func(np.dot(self.weights[i], x_one_row) + self.biases[i])
            output_vector[k] = np.argmax(x_one_row)
        return output_vector.reshape(output_vector.shape[0],1)
        

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

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

    return [train_features, train_labels_encoded, test_features, test_labels_encoded]

def data_read_regression(name,size):
    train = pd.read_csv('C:/Users/tomas/OneDrive/Documents/Studies/PW-IAD/MGU/projekt1-implementacja_backpropagation/Regression/data.'+name+'.train.'+str(size)+'.csv')
    test = pd.read_csv('C:/Users/tomas/OneDrive/Documents/Studies/PW-IAD/MGU/projekt1-implementacja_backpropagation/Regression/data.'+name+'.test.'+str(size)+'.csv')
    train_features = np.array(train['x'])
    train_labels = np.array(train['y'])
    test_features = np.array(test['x'])
    test_labels = np.array(test['y'])

    return [train_features, train_labels, test_features, test_labels]

### TESTY
# Classification
#dt = data_read_classification('simple',1000)
dt = data_read_regression('multimodal',100)

brain_r = Network([1,10,50,1],sigmoid,sigmoid_prime,1000,0.000001,0.001,'regression')
#brain_r.feedforward([1])
#brain_r.backprop([1],[2])
errors = brain_r.train_and_evaluate(dt[0],dt[1],dt[2],dt[3])
plt.plot(errors[0])
plt.plot(errors[1])
plt.legend(['training set error', 'test set error'], loc='upper left')
plt.show()
brain_r.feedforward([-1])
#brain_r.feedforward([0])

# brain = Network([2,2,2],sigmoid,sigmoid_prime,1000,0.0001,0.001,'classification')
# errors = brain.train_and_evaluate(dt[0],dt[1],dt[2],dt[3])
# plt.plot(errors[0])
# plt.plot(errors[1])
# plt.legend(['training set error', 'test set error'], loc='upper left')
# plt.show()