class Config:
    def __init__(self):
        self.epsilon = 0.000001
        self.epochs = 1000
        self.lr = 0.01
        self.mm = 0.9
        self.loss = 'crossentropy'
        self.optimizer = 'gradient_descent'
        
conf = Config()