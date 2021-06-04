import pickle
from Layers import Layer
import numpy as np
import sys

epsilon = 0.000001
epochs = 1000
lr = 0.01
momentum = 0.9

def get_loss(loss_name):
    loss_function = None
    if loss_name == "crossentropy":
        loss_function, loss_function_derivative = crossentropy, crossentropy_derivative
    return loss_function, loss_function_derivative

def optimizer_function(optimizer_name):
    optimizer_function = None
    if optimizer_name == "gradient_descent":
        optimizer_function = gradient_descent
    if optimizer_name == "nag_gradient_descent":
        optimizer_function = gradient_descent
    return (optimizer_function)
        
def crossentropy(a, y):
    log_a = np.log(a)
    temp_loss = np.sum((log_a * y), axis = 1, keepdims = True)
    loss = np.mean(temp_loss) * -1.0
    return (loss)

def crossentropy_derivative(a, y):
    d_cross = -1.0 * (y / (a + epsilon))
    d_cross = d_cross / y.shape[0]
    # print(f"{d_cross.shape =}")
    return (d_cross)

def gradient_descent(layer):
    layer.w = layer.w - (lr * layer.djdw)
    layer.b = layer.b - (lr * layer.djdb)

def nag_gradient_descent(layer, batch_size = 100):
    

    

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
    
    
    def compile(self, loss, optimizer): 
        self.loss_function, self.loss_function_derivative = get_loss(loss)
        self.optimizer = optimizer_function(optimizer)
        
    def backpropagation(self, djda):
        for layer in reversed(self.layers):
            djda = layer.backwards(djda)
    
    def improve(self):
        for layer in self.layers:
            self.optimizer(layer) 
    
    def fit(self, X, y):
        for e in range(epochs):
            a = self.feed_forward(X)
            if e == 0 or e == epochs - 1:
                print("LOSS:", round(self.loss_function(a, y), 2))
            djda = self.loss_function_derivative(a, y)
            self.backpropagation(djda)
            self.improve()
                
        
            

if __name__ == "__main__":
    with open("matrix.pkl", "rb") as f:
        info = pickle.load(f)
    X = info["X"]
    H = info["y_predicted"]
    y = info["y"]
    
    model = Model([
        Layer(units = 4, activation = "softmax", input_size = X.shape[1])#,
        # Layer(units = 4, activation = "sigmoid")
    ])
    model.compile(loss = "crossentropy", optimizer = "gradient_descent")
    model.fit(X, y)
    yhat = model.layers[-1].a
    yhatmax = (yhat == yhat.max(axis=1, keepdims = True)).astype(int)
    Hmax = (H == H.max(axis=1, keepdims = True)).astype(int)
    ymax = (y == y.max(axis=1, keepdims = True)).astype(int)
    err = 0
    for yhat, y in zip(yhatmax, Hmax):
        if (np.argmax(yhat) != np.argmax(y)):
            err += 1
    success = round(((1 - (err / yhatmax.shape[0])) * 100), 2)
    print(f"success =  {success}%")
        