from utils import optimizer_function, get_loss, conf
import numpy as np

class Model:
    def __init__(self, layers_list):
        self.layers = layers_list
        last_layer = self.layers[0]
        for l in self.layers:
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
        for e in range(conf.epochs):
            a = self.feed_forward(X)
            loss = self.loss_function(a, y)
            val_loss = self.loss_function(a, y) ### TODO: change into test set
            djda = self.loss_function_derivative(a, y)
            self.backpropagation(djda)
            self.improve()
            print(f"epoch {e}/{conf.epochs} - loss: {round(loss, 4)} - val_loss: {round(val_loss, 4)}")
        return (a)
            # if e == 0 or e == conf.epochs - 1:
            # 	score.evaluate(e, a, y)
            # 	print("LOSS:", round(self.loss_function(a, y), 2))
            # 	print(f"*average F1_score = {round(np.mean(score.F1_score), 3)}\n*average accuracy = {round(np.mean(score.accuracy), 3)}\n")
                