# This is the abstract class that all activation functions should inherit from
from abc import ABCMeta, abstractmethod  # ABC=Abstract Base Class


class AbstractActivationFunction(metaclass=ABCMeta):
    def Output(self, x):  # Output of activation function from input x
        raise NotImplemented("The activation function your are using does not implement an Output method.")

    def OutputPrime(self, x):  # Output of the derivative of the activation function from input x
        raise NotImplemented(f"The activation function your are using does not implement an OutputPrime method.")
