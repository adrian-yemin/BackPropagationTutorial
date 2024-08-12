from Network import Network
from SigmoidActivation import SigmoidActivation
import numpy
from BackPropagationMethods import BackPropagationMethods
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('/Users/adrian/Downloads/iris2.csv')

X = data.drop(['Setosa', "Versicolor", "Virginica"], axis=1)
y = data.drop(['sepal.length', 'sepal.width', 'petal.length', 'petal.width'], axis=1)

X = X.to_numpy()
y = y.to_numpy()
y_list = y.tolist()
X_list = X.tolist()
for i in range(len(X_list)):
    X_list[i] = numpy.array(X_list[i])
    X_list[i] = numpy.reshape(X_list[i], (4, 1))

X_train, X_test, y_train, y_test = train_test_split(X_list, y_list, test_size=0.2, random_state=41)

# First training sample
inputS1 = numpy.zeros(shape=(2, 1))  # Allocate memory for the input
inputS1[0, 0] = 0.3  # Store the value of x0 for the first training sample
inputS1[1, 0] = 0.5  # Store the value of x1 for the first training sample
inputS1 = [0.3, 0.5]
outputS1 = [0.1]  # Target output for the first sample

# Second sample
inputS2 = numpy.zeros(shape=(2, 1))  # Allocate memory for the input
inputS2[0, 0] = 0.5  # Store the value of x0 for the second training sample
inputS2[1, 0] = 0.9  # Store the value of x1 for the second training sample
inputS2 = [0.5, 0.9]
outputS2 = [0.2]  # Target output for the second sample
# Store the input and output for each sample in an array
input = [inputS1, inputS2]  # This is the array that holds all the training samples input data
output = [outputS1, outputS2]  # This is the array that holds all the training samples output(target) data

# Network and associated parameters
learningRate = 0.01
numNeuronsPerLayer = [8, 9, 3]
actFunctions = [SigmoidActivation(), SigmoidActivation(), SigmoidActivation()]
n1 = Network(4, numNeuronsPerLayer, actFunctions)

backprop = BackPropagationMethods()
backprop.BatchGradientDescent(n1, X_train, y_train, learningRate, 100)
# print(n1.Output(inputS2))


