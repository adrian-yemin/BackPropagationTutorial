import numpy
import copy
from SigmoidActivation import SigmoidActivation
import math


class Neuron():
    def __init__(self, numInputs, actFunction):
        self.Weights = numpy.empty(numInputs, dtype=float)  # Define the weights for the neuron
        self.WeightNum = len(self.Weights)  # Variable to check the number of connections(weights) the neuron has
        self.Bias = 0.0
        self.ActivationFunction = copy.deepcopy(actFunction)
        self.A = 0.0  # Current output A=f(z)
        self.APrime = 0.0  # Derivative of the activation function, APrime=f'(z)

    # Output of the neuron
    def Output(self, input):
        z = 0.0
        self.APrime = 0.0
        for i in range(0, self.WeightNum):  # Sum the weighted inputs
            z += input[i] * self.Weights[i]
        z += self.Bias
        self.A = self.ActivationFunction.Output(z)  # Apply the activation function
        self.APrime = self.ActivationFunction.OutputPrime(z)  # Compute derivative
        return self.A

    def RandomizeWeights(self, fanOut, randGenerator):
        r = 4 * math.sqrt(6 / (self.WeightNum + fanOut))
        for i in range(0, self.WeightNum):
            self.Weights[i] = randGenerator.uniform(-r, r)

    def RandomizeBias(self, randGenerator):
        randNumber = randGenerator.uniform(0, 1)
        self.Bias = randNumber
