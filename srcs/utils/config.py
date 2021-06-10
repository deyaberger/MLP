class Config:
    def __init__(self):
        self.epsilon = 0.000001
        self.epochs = 1000
        self.lr = 0.1
        self.mm = 0.9
        self.loss = 'crossentropy'
        self.optimizer = 'gradient_descent'
        self.model_name = "model_dd"
        self.weights_name = "weights_dd"
        
conf = Config()