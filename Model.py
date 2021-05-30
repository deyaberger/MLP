import pickle
from Layers import Layer
import numpy as np
import sys

epsilon = 0.000001
epochs = 100

def crossentropy(a, y):
    log_a = np.log(a)
    temp_loss = np.sum((log_a * y), axis = 1, keepdims = True)
    loss = np.mean(temp_loss) * -1.0
    return (loss)

def crossentropy_derivative(a, y):
    d_cross = -1.0 * (y / (a + epsilon))
    d_cross = d_cross / y.shape[0]
    return (d_cross)
    

class Model:
    def __init__(self, layers_list):
        self.layers = layers_list
        last_layer = self.layers[0]
        for i, l in enumerate(self.layers):
            if l.input_size == None:
                l.input_size = last_layer.units
                l.w = l.xavier_init(l.input_size, l.units)
                last_layer = l
            
    def feed_forward(self, X):
        for l in self.layers:
            X = l.forward(X)
        return (X)
    
    def get_loss(self, loss_name):
        loss_function = None
        if loss_name == "crossentropy":
            loss_function, loss_function_derivative = crossentropy, crossentropy_derivative
        return loss_function, loss_function_derivative

    def optimizer_function(self, optimizer_name):
        raise NotImplementedError
    
    def compile(self, loss, optimizer = None):
        self.loss_function, self.loss_function_derivative = get_loss(loss)
        # self.optimizer = optimizer_function(optimizer)
    
    def fit(self, X, y):
        for e in epochs:
            a = self.feed_forward(X)
            print(round(self.loss_function(a, y), 2))
            djda = self.loss_function_derivative(a, y)
        
            

if __name__ == "__main__":
    with open("matrix.pkl", "rb") as f:
        info = pickle.load(f)
    X = info["X"]
    H = info["y_predicted"]
    y = info["y"]
    
    print(f"AT FIRST: {X.shape = }")
    model = Model([
        Layer(units = 4, activation = "sigmoid", input_size = X.shape[1]),
        Layer(units = 1, activation = "sigmoid")
    ])
    model.compile(loss = "crossentropy")
    model.fit(X, y)