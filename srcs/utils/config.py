class Config:
    '''
    Things that can be changed:
    epochs (int not too big), lr (learning rate) (from 0.001 to 1), mm (momentum coef) (keep it close to 1), train size (from 0.1 to 0.9)
    '''
    def __init__(self):
        self.lr = 0.1
        self.mm = 0.9
        self.train_size = 0.7
        self.batch_size = 30
        ### Avoid changing the following vairables:
        self.model_folder = '../models/'
        self.topo_prefix = "topo_"
        self.weights_prefix = "weights_"
        self.eval_prefix = "eval_"
        self.model_path = f"{self.model_folder}{self.topo_prefix}"
        self.weights_path = f"{self.model_folder}{self.weights_prefix}"
        self.eval_path = f"{self.model_folder}{self.eval_prefix}"
        self.eval = {"loss" : 0, "val_loss" : 1, "mean_sensitivity" : 2, "mean_specificity" : 3, "mean_precision" : 4, "mean_f1" : 5}
        self.epsilon = 0.000001
        self.datafile = "../datasets/data.csv"
conf = Config()