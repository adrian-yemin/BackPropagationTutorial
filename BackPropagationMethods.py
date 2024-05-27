import numpy


# Has the methods to train a network using backpropagation
class BackPropagationMethods():
    # Does batch backpropagation training
    @staticmethod
    def GradientDescent(net, input, y, learningRate):
        output = net.Output(input)
        deltas = BackPropagationMethods.ComputeAllDeltas(net, output, y)
        BackPropagationMethods.UpdateWeights(net, learningRate, input, deltas)
        BackPropagationMethods.UpdateBiases(net, learningRate, deltas)

    # Iterative method for actual use in application
    @staticmethod
    def StochasticGradientDescent(net, inputs, targets, learningRate, numIterations):
        trainingSampleNum = len(inputs)  # Determine how many samples are in the training set
        for currentIteration in range(0, numIterations):  # Train for a fixed number of iterations
            for currentSample in range(0, trainingSampleNum):  # Train for each sample
                currentInput = inputs[currentSample]  # Get the current training input to the network
                currentTarget = targets[currentSample]  # Get the target (ideal) output of the network
                BackPropagationMethods.GradientDescent(net, currentInput, currentTarget, learningRate)

    @staticmethod
    def BatchGradientDescent(net, inputs, targets, learningRate, numIterations):
        trainingSampleNum = len(inputs)  # Determine how many samples are in the training set
        for currentIteration in range(0, numIterations):  # Train for a fixed number of iterations
            holderJW = BackPropagationMethods.GenerateEmptyJWHolder(net)
            holderJB = BackPropagationMethods.GenerateEmptyJBHolder(net)
            for currentSample in range(0, trainingSampleNum):  # Train for each sample
                currentInput = inputs[currentSample]  # Get the current training input to the network
                currentTarget = targets[currentSample]  # Get the target (ideal) output of the network
                output = net.Output(currentInput)
                deltas = BackPropagationMethods.ComputeAllDeltas(net, output, currentTarget)
                # TO DO: Compute weight partial derivative for the current sample and store it using the SumHolders
                # method
                currentJW = BackPropagationMethods.GenerateEmptyJWHolder(net)
                partials = BackPropagationMethods.CalculateWeights(net, currentInput, deltas, currentJW)
                BackPropagationMethods.SumHolders(partials, holderJW)
                # TO DO: Compute bias partial derivative for the current sample and store it using the SumHolders method
                currentJB = deltas
                BackPropagationMethods.SumHolders(currentJB, holderJB)
            BackPropagationMethods.ApplyAverageToHolders(trainingSampleNum, holderJW)
            BackPropagationMethods.ApplyAverageToHolders(trainingSampleNum, holderJB)
            # TO DO: Apply holderJW to update the weights of the network
            # TO DO: Apply holderJB to update the biases of the network
            for i in range(0, net.LayerNum):
                currentLayer = net.Layers[i]
                for j in range(0, currentLayer.NumNeurons):
                    currentNeuron = currentLayer.NeuronArray[j]
                    currentNeuron.Bias -= holderJB[i][j] * learningRate
                    for k in range(0, currentNeuron.WeightNum):
                        currentNeuron.Weights[k] -= holderJW[i][j][k] * learningRate

    # Generates an empty holder with the correct size numpy matrices for weights for batch gradient descent to use
    @staticmethod
    def GenerateEmptyJWHolder(net):
        EmptyJWHolder = []
        for i in range(net.LayerNum):
            matrixJWForLayer = numpy.zeros(shape=(net.Layers[i].NumNeurons, net.Layers[i].InputsPerNeuron))
            # matrixJWForLayer = numpy.matrix.transpose(matrixJWForLayer)  # Transpose so that it is in the same form
            # as the matrix returned by ComputeJW
            EmptyJWHolder.append(matrixJWForLayer)
        return EmptyJWHolder

    # Generates an empty holder with the correct size numpy matrices for biases for batch gradient descent to use
    @staticmethod
    def GenerateEmptyJBHolder(net):
        EmptyJBHolder = []
        for i in reversed(range(net.LayerNum)):
            matrixJBForLayer = numpy.zeros(shape=(net.Layers[i].NumNeurons, 1))
            EmptyJBHolder.append(matrixJBForLayer)
        return EmptyJBHolder

    # Multiplies the summed deltas by a 1/m term
    def ApplyAverageToHolders(trainingSampleNum, holderJ):
        averagingTerm = 1 / trainingSampleNum
        for i in range(0, len(holderJ)):
            holderJ[i] *= averagingTerm

    # Adds the matrix of JW or JB together for batch gradient descent
    def SumHolders(currentJW, holderJW):
        for i in range(0, len(currentJW)):
            holderJW[i] = numpy.add(currentJW[i], holderJW[i])

    # Get the delta for each neuron in each layer
    # Returns delta, the first index is the layer number (last to first) and the second index is the neuron number
    @staticmethod
    def ComputeAllDeltas(net, output, y):
        # Memory for the delta solution
        delta = []
        # Compute delta
        deltaIndex = 0
        for i in reversed(range(net.LayerNum)):
            # Output layer is handled differently than the rest
            if i == (net.LayerNum - 1):
                aPrime = net.Layers[net.LayerNum - 1].OutputPrime()
                delta.append(-(y - output) * aPrime)
                # Don't update the delta index for the first time
            else:
                currentWeights = net.Layers[i + 1].GetWMatrix()  # Get weights from the i+1 layer (one layer ahead)
                aPrime = net.Layers[i].OutputPrime()  # get f'(z) from the ith layer (current layer)
                currentDelta = numpy.multiply(numpy.dot(numpy.matrix.transpose(currentWeights), delta[deltaIndex]),
                                              aPrime)
                delta.append(currentDelta)
                deltaIndex += 1
        delta.reverse()
        return delta

    @staticmethod
    def UpdateWeights(net, learningRate, input, deltas):
        for k in range(net.LayerNum - 1, -1, -1):
            currentLayer = net.Layers[k]
            previousLayer = net.Layers[k - 1] if k > 0 else None
            for j in range(0, currentLayer.NumNeurons):
                currentNeuron = currentLayer.NeuronArray[j]
                for i in range(0, currentNeuron.WeightNum):
                    output = previousLayer.NeuronArray[i].A if k > 0 else input[i]
                    currentNeuron.Weights[i] -= deltas[k][j][0] * output * learningRate

    @staticmethod
    def CalculateWeights(net, input, deltas, currentJW):
        for k in reversed(range(net.LayerNum)):
            currentLayer = net.Layers[k]
            previousLayer = net.Layers[k - 1] if k > 0 else None
            for j in range(0, currentLayer.NumNeurons):
                currentNeuron = currentLayer.NeuronArray[j]
                for i in range(0, currentNeuron.WeightNum):
                    output = previousLayer.NeuronArray[i].A if k > 0 else input[i][0]
                    currentJW[k][j][i] = deltas[k][j][0] * output
        return currentJW

    @staticmethod
    def UpdateBiases(net, learningRate, deltas):
        for k in range(0, net.LayerNum):
            currentLayer = net.Layers[k]
            for j in range(0, currentLayer.NumNeurons):
                currentNeuron = currentLayer.NeuronArray[j]
                currentNeuron.Bias -= deltas[k][j] * learningRate