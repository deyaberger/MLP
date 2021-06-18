from utils import xavier_init, optimizer_function, get_loss, conf, save_json
import numpy as np
import time
import pickle


class Model:
    def __init__(self, layers_list):
        self.layers = layers_list
        last_layer = self.layers[0]
        for l in self.layers:
            if l.input_size == None:
                l.input_size = last_layer.units
                l.w = xavier_init(l.input_size, l.units)
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
    
    def overfitting(self, history):
        if len(history) < 2:
            return (False)
        if history[-1][1] > history[-2][1]:
            return (True)
        return (False)

    def fit(self, X, y, score):
        past = []
        for e in range(conf.epochs):
            a = self.feed_forward(X)
            loss = self.loss_function(a, y)
            djda = self.loss_function_derivative(a, y)
            self.backpropagation(djda)
            self.improve()
            val_a = self.feed_forward(score.X)
            val_loss = self.loss_function(val_a, score.y) ### TODO: change into test set
            score.evaluation(val_a)
            score.keep_track(loss, val_loss)
            if conf.early_stop == True and self.overfitting(score.history) == True:
                print(f"Stoping training loop at epoch {e} to avoid Overfitting (Validation loss has started to increase)")
                break
            print(f"\nepoch {e + 1}/{conf.epochs} - loss: {round(loss, 4)} - val_loss: {round(val_loss, 4)}")
            print(score)
        return (a)


    def save_architecture(self, name):
        infos = {"optimizer" : self.optimizer.__name__, "loss" : self.loss_function.__name__, "layers" : []}
        for l in self.layers:
            layer = {"activation" : l.activation.__name__, "input_size" : l.input_size, "output_size": l.units}
            infos["layers"].append(layer)
        save_json(f"{name}.json", infos)
    
    
    def save_weights(self, name):
        weights = []
        for i, l in enumerate(self.layers):
            weights.append(l.w)
        with open(f"{name}.pkl", "wb") as f:
            pickle.dump(weights, f)    
        