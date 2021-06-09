import numpy as np

class ModelEvaluation:
    def __init__(self):
        self.history = []
        
    def evaluation(self, y, yhat):
        self.eval = self.evaluate_classifier(y, yhat)
        self.get_all_metrics()
        self.get_mean_metrics()
    
        
    def evaluate_classifier(self, y, yhat):
        yhatmax = (yhat == yhat.max(axis=1, keepdims = True)).astype(int)
        eval = []
        for col in range(y.shape[1]):
            yy = y[:, col]
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
        self.global_sensitivity, self.global_specificity, self.global_precision, self.gloabl_f1 = self.calculate_metrics(self.eval[:, 0], self.eval[:, 1], self.eval[:, 2], self.eval[:, 3])

    
    def get_mean_metrics(self):
        self.mean_sensitivity = np.mean(self.global_sensitivity)
        self.mean_specificity = np.mean(self.global_specificity)
        self.mean_precision = np.mean(self.global_precision)
        self.mean_f1 = np.mean(self.gloabl_f1)
    
    