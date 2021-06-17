import numpy as np
np.seterr(divide='ignore', invalid='ignore') #TODO : maybe change it at the end

class ModelEvaluation:
    def __init__(self, X, y):
        self.history = []
        self.X = X
        self.y = y
        
    def evaluation(self, yhat):
        self.eval = self.evaluate_classifier(self.y, yhat)
        self.get_all_metrics()
        self.get_mean_metrics()
    
        
    def evaluate_classifier(self, yhat):
        yhatmax = (yhat == yhat.max(axis=1, keepdims = True)).astype(int)
        eval = []
        for col in range(self.y.shape[1]):
            yy = self.y[:, col]
            yyhatmax = yhatmax[:, col]
            e = 2 * yy + yyhatmax
            tp = (e == 3).astype(int).sum()
            tn = (e == 0).astype(int).sum()
            fn = (e == 2).astype(int).sum()
            fp = (e == 1).astype(int).sum()
            eval.append([tp, fp, tn, fn])
        eval = np.array(eval)
        return (eval)
    
    
    def calculate_metrics(self, tp, fp, tn, fn):
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        f1 = 2.0 * (sensitivity * precision) / (sensitivity + precision)
        return (sensitivity, specificity, precision, f1)
    
    
    def get_all_metrics(self):
        self.global_sensitivity, self.global_specificity, self.global_precision, self.global_f1 = self.calculate_metrics(self.eval[:, 0], self.eval[:, 1], self.eval[:, 2], self.eval[:, 3])

    
    def get_mean_metrics(self):
        self.mean_sensitivity = np.mean(self.global_sensitivity)
        self.mean_specificity = np.mean(self.global_specificity)
        self.mean_precision = np.mean(self.global_precision)
        self.mean_f1 = np.mean(self.global_f1)
    
    
    def keep_track(self, loss, val_loss):
        self.history.append([loss, val_loss, self.mean_sensitivity, self.mean_specificity, self.mean_precision, self.mean_f1])
    
    
    def save(self, name):
        np.savetxt(f"{name}.csv", self.history, delimiter=",", fmt="%s")
    
    
    def __str__(self):
        all = f"global_sensitivity = {self.global_sensitivity}\nglobal_specificity = {self.global_specificity}\nglobal_precision = {self.global_precision}\nglobal_f1 = {self.global_f1}"
        mean = f"mean_sensitivity = {round(self.mean_sensitivity, 4)}, mean_specificity = {round(self.mean_specificity, 4)}, mean_precision = {round(self.mean_precision, 4)}, mean_f1 = {round(self.mean_f1, 4)}"
        all_mean = f"MEAN:\n{mean}\n\nALL:\n{all}"
        return(mean)
    