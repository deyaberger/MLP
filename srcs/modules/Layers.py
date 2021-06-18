import numpy as np
from utils import xavier_init, get_activation, add_bias_units

class Layer:
    def __init__(self, units, activation, input_size = None):
        self.input_size = input_size
        self.units = units
        self.w = None
        self.vw = 0
        self.b = xavier_init(1, units)
        self.vb = np.zeros((1, units))
        if self.input_size != None:
            self.w = xavier_init(input_size, units)
            self.vw = np.zeros((input_size, units))
        self.activation, self.activation_derivative = get_activation(activation)

    
    def forward(self, X):
        self.X = X
        self.z = np.matmul(self.X, self.w) + self.b
        self.a = self.activation(self.z)
        return (self.a)
    
    
    def backwards(self, djda):
        dadz = self.activation_derivative(self.a, self.z)
        djdz = np.einsum('ik,ikj->ij', djda, dadz)
        self.djdw = np.matmul(self.X.T, djdz)
        self.djdb = np.mean(djdz, axis = 0)
        djdx = np.matmul(djdz, self.w.T)
        return (djdx)
        
    
    def __str__(self):
        return(f"activation = {self.activation.__name__}\n{self.input_size = }, {self.units = }\n{self.w.shape = }")
