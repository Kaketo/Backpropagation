import os
os.chdir('C:/Users/tomas/OneDrive/Documents/Studies/PW-IAD/MGU/projekt1-implementacja_backpropagation/MGUProjekt1')
from network import Layer
from network import Network
from program_functions import data_read_classification
from program_functions import data_read_regression
from program_functions import plot_classification
from program_functions import plot_regression
from program_functions import plot_errors

### Regression
x = data_read_regression('linear',100)
brain = Network(learning_rate = 0.0001, momentum_rate = 0.8, iterations = 100)
brain.add(Layer(1,5,'sigmoid'))
brain.add(Layer(5,50,'sigmoid'))
brain.add(Layer(50,1,'linear'))
errors = brain.train_and_evaluate(x[0],x[1],x[2],x[3])
errors = brain.train_mini_batch_and_evaluate(x[0],x[1],x[2],x[3],10)
plot_errors(errors)
plot_regression(brain, x)


#### Classification
x = data_read_classification('simple',100)
brain = Network(learning_rate = 0.01, momentum_rate = 0.8, iterations = 1000)
brain.add(Layer(2,10,'sigmoid'))
brain.add(Layer(10,100,'sigmoid'))
brain.add(Layer(100,50,'sigmoid'))
brain.add(Layer(50,2,'sigmoid'))
errors = brain.train_and_evaluate(x[0],x[1],x[2],x[3])
plot_errors(errors)
plot_classification(brain, x)

