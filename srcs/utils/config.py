class Config:
    def __init__(self):
        self.datafile = "../datasets/data.csv"
        self.epsilon = 0.000001
        self.epochs = 4000
        self.lr = 0.1
        self.mm = 0.9
        self.loss = 'crossentropy'
        self.optimizer = 'gradient_descent'
        self.model_folder = '../models/'
        self.model_prefix = "model_"
        self.weights_prefix = "weights_"
        self.eval_prefix = "eval_"
        self.name = "classic"
        self.model_path = f"{self.model_folder}{self.model_prefix}{self.name}"
        self.weights_path = f"{self.model_folder}{self.weights_prefix}{self.name}"
        self.eval_path = f"{self.model_folder}{self.eval_prefix}{self.name}"
        self.graph = False
        self.verbose = True
        self.eval = {"loss" : 0, "val_loss" : 1, "mean_sensitivity" : 2, "mean_specificity" : 3, "mean_precision" : 4, "mean_f1" : 5}
        self.train_size = 0.7
conf = Config()