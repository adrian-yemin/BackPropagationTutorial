from Network import Network
from SigmoidActivation import SigmoidActivation
import numpy
from BackPropagationMethods import BackPropagationMethods

# First training sample
inputS1 = numpy.zeros(shape=(2, 1))  # Allocate memory for the input
inputS1[0, 0] = 0.3  # Store the value of x0 for the first training sample
inputS1[1, 0] = 0.5  # Store the value of x1 for the first training sample
# inputS1 = [0.3, 0.5]
outputS1 = [0.1]  # Target output for the first sample
# Second sample
inputS2 = numpy.zeros(shape=(2, 1))  # Allocate memory for the input
inputS2[0, 0] = 0.5  # Store the value of x0 for the second training sample
inputS2[1, 0] = 0.9  # Store the value of x1 for the second training sample
# inputS2 = [0.5, 0.9]
outputS2 = [0.2]  # Target output for the second sample
# Store the input and output for each sample in an array
input = [inputS1, inputS2]  # This is the array that holds all the training samples input data
output = [outputS1, outputS2]  # This is the array that holds all the training samples output(target) data
# Network and associated parameters
learningRate = 0.5
numNeuronsPerLayer = [3, 2, 1]
actFunctions = [SigmoidActivation(), SigmoidActivation(), SigmoidActivation()]
n1 = Network(2, numNeuronsPerLayer, actFunctions)
# Set dummy weights for testing
# first layer
n1.Layers[0].NeuronArray[0].Weights[0] = 0.2
n1.Layers[0].NeuronArray[0].Weights[1] = 0.3
n1.Layers[0].NeuronArray[0].Bias = 0.0
n1.Layers[0].NeuronArray[1].Weights[0] = 0.1
n1.Layers[0].NeuronArray[1].Weights[1] = 0.25
n1.Layers[0].NeuronArray[1].Bias = 0.0
n1.Layers[0].NeuronArray[2].Weights[0] = 0.5
n1.Layers[0].NeuronArray[2].Weights[1] = 0.6
n1.Layers[0].NeuronArray[2].Bias = 0.0
# second layer
n1.Layers[1].NeuronArray[0].Weights[0] = 0.8
n1.Layers[1].NeuronArray[0].Weights[1] = 0.1
n1.Layers[1].NeuronArray[0].Weights[2] = 0.7
n1.Layers[1].NeuronArray[0].Bias = 0.0
n1.Layers[1].NeuronArray[1].Weights[0] = 0.2
n1.Layers[1].NeuronArray[1].Weights[1] = 0.1
n1.Layers[1].NeuronArray[1].Weights[2] = 0.3
n1.Layers[1].NeuronArray[1].Bias = 0.0
# Output layer x
n1.Layers[2].NeuronArray[0].Weights[0] = 0.5
n1.Layers[2].NeuronArray[0].Weights[1] = 0.3
n1.Layers[2].NeuronArray[0].Bias = 0.0

backprop = BackPropagationMethods()
backprop.BatchGradientDescent(n1, input, output, learningRate, 1)
print(n1.Output(inputS2))
