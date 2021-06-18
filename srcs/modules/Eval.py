import numpy as np
np.seterr(divide='ignore', invalid='ignore') #TODO : maybe change it at the end
from termcolor import cprint, colored

class ModelEvaluation:
    def __init__(self, X, y):
        self.history = []
        self.X = X
        self.y = y
        
    def evaluation(self, yhat):
        self.eval = self.evaluate_classifier(yhat)
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
    
    def percentage(self, before, after):
        res = (1 - (after / before)) * 100
        res = res * -1
        return (round(res, 2))
    

    def calc_diff(self, before, after):
        diff_loss = self.percentage(before[0], after[0])
        diff_val_loss = self.percentage(before[1], after[1])
        return (diff_loss, diff_val_loss)
 

    def print_summary(self):
        if len(self.history) < 2:
            return (False)
        diff_loss, diff_val_loss = self.calc_diff(self.history[0], self.history[-1])
        start_loss = f"loss: {round(self.history[0][0], 4)} - val_loss: {round(self.history[0][1], 4)}"
        end_loss = f"loss: {round(self.history[-1][0], 4)} ({diff_loss}%) - val_loss: {round(self.history[-1][1], 4)} ({diff_val_loss}%)"
        if diff_loss < -90 and diff_val_loss < -90:
            end_loss = f"loss: {round(self.history[-1][0], 4)} " + colored(f"({diff_loss}%)", "green") + f" - val_loss: {round(self.history[-1][1], 4)} " + colored(f"({diff_val_loss}%)", "green")
        start_mean = f"mean_sensitivity = {round(self.history[0][2], 4)}, mean_specificity = {round(self.history[0][3], 4)}, mean_precision = {round(self.history[0][4], 4)}, mean_f1 = {round(self.history[0][5], 4)}"
        end_mean = f"mean_sensitivity = {round(self.history[-1][2], 4)}, mean_specificity = {round(self.history[-1][3], 4)}, mean_precision = {round(self.history[-1][4], 4)}, mean_f1 = {round(self.history[-1][5], 4)}"
        cprint("\n\t\t\tSUMMARY:\n", "blue")
        cprint("**\tBefore Training:\t**\n", "blue")
        print(f"{start_loss}\n{start_mean}\n\n")
        cprint("\n**\tAfter Training:\t**\n", "blue")
        print(f"{end_loss}\n{end_mean}\n")

    
    def __str__(self):
        losses = f"loss : {round(self.history[0][0], 4)}, validation_loss : {round(self.history[0][1], 4)}"
        mean = f"mean_sensitivity = {round(self.mean_sensitivity, 4)}, mean_specificity = {round(self.mean_specificity, 4)}, mean_precision = {round(self.mean_precision, 4)}, mean_f1 = {round(self.mean_f1, 4)}"
        return(f"{losses} {mean}")
    