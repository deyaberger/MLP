import numpy as np

def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return (a)

class Layer:
    def __init__(self, input_size, output_size, activation_function):
        self.w = self.xavier_init(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.activation = self.activations(activation_function)
    
    def activations(self, activation_function):
        activation = None
        if activation_function == "sigmoid":
            activation = sigmoid
        return (activation)
        
    def xavier_init(self, input_size, output_size):
        raise NotImplementedError
    
    def add_bias_units(self, X):
        raise NotImplementedError
    
    def forward(self, X):
        X = self.add_bias_units(X)
        self.z = np.matmul(X, self.w)
        self.a = self.activation(self.z)
    
        