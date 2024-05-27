# Layer class for the network
from Neuron import Neuron
import numpy
import copy
from SigmoidActivation import SigmoidActivation


class Layer():
    # Default constructor for the layer class
    def __init__(self, inputsPerNeuron, numNeurons, actFunction):
        self.NumNeurons = numNeurons  # Total number of neurons in the layer
        self.InputsPerNeuron = inputsPerNeuron  # Number of inputs each neuron should take in
        self.LayerOutput = numpy.zeros(shape=(numNeurons, 1))
        self.LayerOutputPrime = numpy.zeros(
            shape=(numNeurons, 1))  # This stores the derivatives of the activation functions
        self.WMatrix = numpy.zeros(
            shape=(numNeurons, inputsPerNeuron))  # First index is the neuron, second index is the weight for neurons
        self.BMatrix = numpy.zeros(shape=(numNeurons, 1))
        self.NeuronArray = []
        for i in range(0, self.NumNeurons):
            self.NeuronArray.append(Neuron(inputsPerNeuron, actFunction))

    # Gives the output of the layer as an array(each element corresponds to the output of one neuron)
    def Output(self, inputArray):
        if inputArray is None:  # In this case we assume a forward pass has already been done
            for i in range(0, self.NumNeurons):
                self.LayerOutput[i] = self.NeuronArray[i].Output(inputArray)  # Return the stored output
        else:  # Forward pass has not been done so need to feed input through the layer
            for i in range(0, self.NumNeurons):
                self.LayerOutput[i] = self.NeuronArray[i].Output(inputArray)
        return self.LayerOutput

    # Assume a forward pass has already been done
    def OutputPrime(self):
        for i in range(0, self.NumNeurons):
            self.LayerOutputPrime[i] = self.NeuronArray[i].APrime
        return self.LayerOutputPrime

    # Get the W Matrix, also used for updating it
    def GetWMatrix(self):
        for i in range(0, self.NumNeurons):
            for j in range(0, self.InputsPerNeuron):
                self.WMatrix[i, j] = self.NeuronArray[i].Weights[j]
        return self.WMatrix

    # Set the W Matrix
    def SetWMatrix(self, updatedWMatrix):
        self.WMatrix = copy.deepcopy(updatedWMatrix)
        for i in range(0, self.NumNeurons):
            updatedWeightsForNeuron = updatedWMatrix[i, :]  # Take out the column to get the weights for the neuron
            self.NeuronArray[i].Weights = updatedWeightsForNeuron

    # Get the B Matrix, also used for updating it
    def GetBMatrix(self):
        for i in range(0, self.NumNeurons):
            self.BMatrix[i, 0] = self.NeuronArray[i].Bias
        return self.BMatrix

    # Set the B Matrix
    def SetBMatrix(self, updatedBMatrix):
        self.BMatrix = copy.deepcopy(updatedBMatrix)
        for i in range(0, self.NumNeurons):
            self.NeuronArray[i].Bias = updatedBMatrix[i, 0]

    def RandomizeLayerWeights(self, fanOut, randGenerator):
        for i in range(0, self.NumNeurons):
            self.NeuronArray[i].RandomizeWeights(fanOut, randGenerator)
        self.GetWMatrix()  # Updates the WMatrix variable in the Layer class with the newly randomized weights

    def RandomizeLayerBias(self, randGenerator):
        for i in range(0, self.NumNeurons):
            self.NeuronArray[i].RandomizeBias(randGenerator)
        self.GetBMatrix()
