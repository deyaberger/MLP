from utils import optimizer_function, get_loss, conf, save_json
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
        print(self.optimizer.__name__)
        print(self.loss_function.__name__)
        
        
    def backpropagation(self, djda):
        for layer in reversed(self.layers):
            djda = layer.backwards(djda)
    
    def improve(self):
        for layer in self.layers:
            self.optimizer(layer) 
    
    def fit(self, X, y, score):
        for e in range(conf.epochs):
            a = self.feed_forward(X)
            loss = self.loss_function(a, y)
            val_loss = 0 # self.loss_function(a, y) ### TODO: change into test set
            djda = self.loss_function_derivative(a, y)
            self.backpropagation(djda)
            self.improve()
            score.evaluation(y, a)
            score.keep_track(loss, val_loss)
            # print(f"\nepoch {e + 1}/{conf.epochs} - loss: {round(loss, 4)} - val_loss: {round(val_loss, 4)}")
            # print(score)
        return (a)


    def save_architecture(self, name):
        infos = {"optimizer" : self.optimizer.__name__, "loss" : self.loss_function.__name__, "layers" : []}
        for l in self.layers:
            layer = {"activation" : l.activation.__name__, "input_size" : l.input_size, "output_size": l.units}
            infos["layers"].append(layer)
        save_json(f"{name}.json", infos)
    
    
    def save_weights(self, name):
        for i, l in enumerate(self.layers):
            np.savetxt(f"{name}_{i}.csv", l.w, delimiter=",", fmt="%s")
    
        
        
    
        # np.savetxt(name, self.optimizer, delimiter=",", fmt="%s")
                