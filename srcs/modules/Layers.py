try:
    import numpy as np
    from utils import xavier_init, get_activation
except ModuleNotFoundError as e:
    import sys
    print(f"{e}\nPlease run 'pip install -r requirements.txt'")
    sys.exit()


class Layer:
    '''
    Class creating Layers of different input and output sizes for our model,
    with their weights, bias and activation function.
    '''
    def __init__(self, units, activation = None, input_size = None):
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
        '''
        Calculating the weighted sum : (our inputs X) * (our weights) + (our biases)
        "a" is the output of each layer
        '''
        self.X = X
        self.z = np.matmul(self.X, self.w) + self.b
        self.a = self.activation(self.z)
        return (self.a)
    
    
    def backwards(self, djda):
        '''
        Calculating the derivative of :
        (dj / da) = loss_derivative
        (dx/dw) * (dz/db) * (dz/dw) * (da/dz) * (dj/da)
        '''
        dadz = self.activation_derivative(self.a)
        djdz = np.einsum('ik,ikj->ij', djda, dadz)
        self.djdw = np.matmul(self.X.T, djdz)
        self.djdb = np.mean(djdz, axis = 0)
        djdx = np.matmul(djdz, self.w.T)
        return (djdx)
        
    
    def __str__(self):
        return(f"activation = {self.activation.__name__}\n{self.input_size = }, {self.units = }\n{self.w.shape = }")
