import pickle
from Layers import Layer
import numpy as np
import sys

epsilon = 0.000001
epochs = 1000
lr = 0.01
Mom = 0.9

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
		# self.save_evolution()

	def my_divide(self, a, b):
		result = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
		return(result)

	def show_graph(self, name, x, y, fig):
		plt.figure(fig)
		plt.title(f"{name} evolution")
		plt.xlabel("iterations")
		plt.ylabel(name)
		plt.plot(x, y)
		plt.pause(0.001)

def get_loss(loss_name):
	loss_function = None
	if loss_name == "crossentropy":
		loss_function, loss_function_derivative = crossentropy, crossentropy_derivative
	return loss_function, loss_function_derivative

def optimizer_function(optimizer_name):
	optimizer_function = None
	if optimizer_name == "gradient_descent":
		optimizer_function = gradient_descent
	elif optimizer_name == "momentum":
		optimizer_function = momentum
	return (optimizer_function)
		
def crossentropy(a, y):
	log_a = np.log(a)
	temp_loss = np.sum((log_a * y), axis = 1, keepdims = True)
	loss = np.mean(temp_loss) * -1.0
	return (loss)

def crossentropy_derivative(a, y):
	d_cross = -1.0 * (y / (a + epsilon))
	d_cross = d_cross / y.shape[0]
	return (d_cross)


def gradient_descent(layer):
	layer.w = layer.w - (lr * layer.djdw)
	layer.b = layer.b - (lr * layer.djdb)


def momentum(layer):
	layer.vw = Mom * (layer.vw) + (1 - Mom) * layer.djdw
	layer.vb = Mom * (layer.vb) + (1 - Mom) * layer.djdb
	layer.w = layer.w - (lr * layer.vw)
	layer.b = layer.b - (lr * layer.vb)

	

class Model:
	def __init__(self, layers_list):
		self.layers = layers_list
		last_layer = self.layers[0]
		for i, l in enumerate(self.layers):
			if l.input_size == None:
				l.input_size = last_layer.units
				l.w = l.xavier_init(l.input_size, l.units)
				last_layer = l
			
	def feed_forward(self, X, i):
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
	
	def fit(self, X, y, score):
		for e in range(epochs):
			a = self.feed_forward(X, e)
			djda = self.loss_function_derivative(a, y)
			self.backpropagation(djda)
			self.improve()
			if e == 0 or e == epochs - 1:
				score.evaluate(e, a, y)
				print("LOSS:", round(self.loss_function(a, y), 2))
				print(f"*average F1_score = {round(np.mean(score.F1_score), 3)}\n*average accuracy = {round(np.mean(score.accuracy), 3)}\n")
				
			

if __name__ == "__main__":
	with open("matrix.pkl", "rb") as f:
		info = pickle.load(f)
	X = info["X"]
	H = info["y_predicted"]
	y = info["y"]
	
	model = Model([
		Layer(units = 4, activation = "softmax", input_size = X.shape[1])#,
		# Layer(units = 4, activation = "sigmoid")
	])
	model.compile(loss = "crossentropy", optimizer = "gradient_descent")
	score = ModelEvaluation()
	model.fit(X, y, score)
	yhat = model.layers[-1].a
	yhatmax = (yhat == yhat.max(axis=1, keepdims = True)).astype(int)
	Hmax = (H == H.max(axis=1, keepdims = True)).astype(int)
	ymax = (y == y.max(axis=1, keepdims = True)).astype(int)
	err = 0
	for yhat, y in zip(yhatmax, Hmax):
		if (np.argmax(yhat) != np.argmax(y)):
			err += 1
	success = round(((1 - (err / yhatmax.shape[0])) * 100), 2)
	print(f"success =  {success}%")
		