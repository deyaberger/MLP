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

def softmax(z):
    z = z - z.max(axis = 1, keepdims=True)
    e = np.exp(z)
    s = np.sum(e, axis = 1, keepdims=True)
    return (e / s)

def softmax_derivative(a, z):
    m, n = a.shape # m = nb examples, n = nb features

    # First we create for each example feature vector, it's outer product with itself:
    # ( p1^2  p1*p2  p1*p3 .... )
    # ( p2*p1 p2^2   p2*p3 .... )
    # ( ...                     )
    tensor1 = np.einsum('ij,ik->ijk', a, a)  # (m, n, n)

    # Second we need to create an (n,n) identity of the feature vector
    # ( p1  0  0  ...  )
    # ( 0   p2 0  ...  )
    # ( ...            )
    tensor2 = np.einsum('ij,jk->ijk', a, np.eye(n, n))  # (m, n, n)

    # Then we need to subtract the first tensor from the second
    # ( p1 - p1^2   -p1*p2   -p1*p3  ... )
    # ( -p1*p2     p2 - p2^2   -p2*p3 ...)
    # ( ...                              )
    da = tensor2 - tensor1
    return da


class Layer:
    def __init__(self, units, activation, input_size = None):
        self.input_size = input_size
        self.units = units
        self.w = None
        self.vw = None
        self.b = self.xavier_init(1, units)
        self.vb = np.zeros((1, units))
        if self.input_size != None:
            self.w = self.xavier_init(input_size, units)
            self.vw = np.zeros((input_size, units))
        self.activation, self.activation_derivative = self.activations(activation)

    
    def activations(self, activation):
        a = None
        if activation == "sigmoid":
            a, d_a = sigmoid, sigmoid_derivative
        if activation == "softmax":
            a, d_a = softmax, softmax_derivative
        return (a, d_a)

        
    def xavier_init(self, input_size, units):
        lower, upper = -(1.0 / sqrt(input_size)), (1.0 / sqrt(input_size))
        np.random.seed(70) ### ! To be changed eventualy if you want to improve results
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
        self.djdw = np.matmul(self.X.T, djdz)
        self.djdb = djdz
        djdx = np.matmul(djdz, self.w.T)
        return (djdx)
        
    
    def __str__(self):
        return(f"activation = {self.activation.__name__}\n{self.input_size = }, {self.units = }\n{self.w.shape = }, {self.z.shape = }, {self.a.shape = }")
