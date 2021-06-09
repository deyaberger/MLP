import numpy as np

class ModelEvaluation:
    def __init__(self):
        ''' Validation metrics:
        True positive: correct prediction for a student to be part of a certain house in Hogwart
        True negative: correct prediction for a student to NOT be part of a certain house in Hogwart
        False positive: incorrect prediction for a student to be part of a certain house in Hogwart
        False negative: incorrect prediction for a student to NOT be part of a certain house in Hogwart
        '''
        self.precision_total, self.sensitivity_total, self.accuracy_total, self.F1_score_total = [], [], [], []
        self.init_metrics()
    
    def init_metrics(self):
        self.true_positive = np.zeros((4, 1))
        self.true_negative = np.zeros((4, 1))
        self.false_negative = np.zeros((4, 1))
        self.false_positive = np.zeros((4, 1))

    def calculate_kpi(self):
        ''' Commonly use KPIs:
        Precision: expresses the proportion of the data points our model says was relevant actually were relevant.
        Sensitivity: (also called redcall) expresses the ability to find all relevant instances in a dataset.
        Accuracy: proportion of correct predictions over total predictions.
        F1 Score: the harmonic mean of the model s precision and recall.
        '''
        self.precision = self.my_divide(self.true_positive, (self.true_positive + self.false_positive))
        self.sensitivity = self.my_divide(self.true_positive, (self.true_positive + self.false_negative))
        self.accuracy = self.my_divide((self.true_positive + self.true_negative), (self.true_positive + self.true_negative + self.false_positive + self.false_negative))
        self.F1_score = (2 * (self.my_divide((self.precision * self.sensitivity), (self.precision + self.sensitivity))))
    
    def save_evolution(self):
        ''' Saving all values of our kpis during each call to the evaluate function, to display them in the graphs '''
        self.precision_total.append(round(np.mean(self.precision), 3))
        self.sensitivity_total.append(round(np.mean(self.sensitivity), 3))
        self.accuracy_total.append(round(np.mean(self.accuracy), 3))
        self.F1_score_total.append(round(np.mean(self.F1_score), 3))

    def evaluate(self, epochs, yhat, y):
        '''
        Calculating different type of KPIs to evaluate the performance of our model and its evolution
        '''
        self.init_metrics()
        for i in range(len(yhat)):
            predicted_house, real_house = np.argmax(yhat[i]), np.argmax(y[i])
            if predicted_house == real_house:
                self.true_positive[predicted_house] += 1
                for index, nb in enumerate(yhat[i]):
                    if index != i:
                        self.true_negative[index] += 1
            else:
                self.false_positive[predicted_house] += 1
                self.false_negative[real_house] += 1
        self.calculate_kpi()
        self.save_evolution()

    def my_divide(self, a, b):
        result = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        return(result)