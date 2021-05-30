import numpy as np
from numpy.random import rand
from math import sqrt

def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return (a)

def sigmoid_derivative(a, z):
    da = (a) * (1 - a)
    _, n = da.shape
    da = np.einsum('ij,jk->ijk' , da, np.eye(n, n))
    return da
 

class Layer:
    def __init__(self, units, activation, input_size = None):
        self.input_size = input_size
        self.units = units
        self.w = None
        self.b = self.xavier_init(1, units)
        if self.input_size != None:
            self.w = self.xavier_init(input_size, units)
        self.activation, self.activation_derivative = self.activations(activation)

    
    def activations(self, activation):
        a = None
        if activation == "sigmoid":
            a, d_a = sigmoid, sigmoid_derivative
        return (a, d_a)

        
    def xavier_init(self, input_size, units):
        lower, upper = -(1.0 / sqrt(input_size)), (1.0 / sqrt(input_size))
        weights = rand(input_size, units)
        weights = weights * (upper - lower) + lower
        return (weights)
    
    
    def forward(self, X):
        self.X = X
        self.z = np.matmul(X, self.w) + self.b
        self.a = self.activation(self.z)
        return (self.a)
    
    
    def backwards(self, djda):
        dadz = self.activation_derivative(self.a, self.z)
        djdz = np.einsum('ik,ikj->ij', djda, dadz)
        print(f"{djdz.shape = }, {self.X.shape = }")
        self.djdw = np.matmul(self.X.T, djdz)
        self.djdb = djdz
        print(f"{self.w.shape = }, {djdz.shape = }")
        djdx = np.matmul(self.w, djdz.T) ### TODO :: >>>???
        return (djdx)
        
    
    def __str__(self):
        return(f"activation = {self.activation.__name__}\n{self.input_size = }, {self.units = }\n{self.w.shape = }, {self.z.shape = }, {self.a.shape = }")
