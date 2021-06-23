try:
    from utils import conf, xavier_init, optimizer_function, get_loss, conf, save_json, mini_batch
    import pickle
    from termcolor import cprint
except ModuleNotFoundError as e:
    import sys
    print(f"{e}\nPlease run 'pip install -r requirements.txt'")
    sys.exit()


class Model:
    '''
    Creation of a model with its layers and its loss and optimizer functions
    '''
    def __init__(self, layers_list):
        self.layers = layers_list
        last_layer = self.layers[0]
        for l in self.layers:
            if l.input_size == None:
                l.input_size = last_layer.units
                l.w = xavier_init(l.input_size, l.units)
                last_layer = l
            
    def feed_forward(self, X):
        '''
        Feed forward : passing the information forward in the neural network:
        through the input nodes then through the hidden layers and finally through the output nodes
        '''
        for l in self.layers:
            X = l.forward(X)
        return (X)
    
    
    def compile(self, loss, optimizer):
        self.loss_function, self.loss_function_derivative = get_loss(loss)
        self.optimizer = optimizer_function(optimizer)
        
        
    def backpropagation(self, djda):
        '''
        Calculating the derivative of the loss according to the weights of each layers
        '''
        for layer in reversed(self.layers):
            djda = layer.backwards(djda)
    
    
    def improve(self):
        '''
        Computing gradient descent or other form of optimizer 
        '''
        for layer in self.layers:
            self.optimizer(layer)
    
    def overfitting(self, history):
        '''
        If loss keeps decreasing but validation loss starts increasing it means we might overfit our model
        '''
        if len(history) < 2:
            return (False)
        if history[-1][conf.eval["val_loss"]] > history[-2][conf.eval["val_loss"]]:
            return (True)
        return (False)

    def fit(self, X, y, score):
        for e in range(self.args.epochs):
            if self.args.batch == True:
                X, y = mini_batch(X, y, conf.batch_size)
            a = self.feed_forward(X)
            loss = self.loss_function(a, y)
            djda = self.loss_function_derivative(a, y)
            self.backpropagation(djda)
            self.improve()
            val_a = self.feed_forward(score.X)
            val_loss = self.loss_function(val_a, score.y)
            score.evaluation(val_a)
            score.keep_track(loss, val_loss)
            if self.args.early_stop == True and self.overfitting(score.history) == True:
                cprint(f"\nStoping training loop at epoch {e} to avoid Overfitting (Validation loss has started to increase)", "yellow")
                break
            print(f"\nepoch {e + 1}/{self.args.epochs}:\n{score}")


    def save_topology(self, name):
        '''
        Saving our achitecture in a json file (easy to read)
        '''
        infos = {"optimizer" : self.optimizer.__name__, "loss" : self.loss_function.__name__, "layers" : []}
        for l in self.layers:
            layer = {"activation" : l.activation.__name__, "input_size" : l.input_size, "output_size": l.units}
            infos["layers"].append(layer)
        save_json(f"{name}.json", infos)
    
    
    def save_weights(self, name):
        '''
        Saving our weights in a pickle file (easy to load)
        '''
        weights = []
        for i, l in enumerate(self.layers):
            weights.append(l.w)
        with open(f"{name}.pkl", "wb") as f:
            pickle.dump(weights, f)    
        