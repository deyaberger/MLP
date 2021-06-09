from utils import optimizer_function, get_loss, conf
import numpy as np

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
		for e in range(conf.epochs):
			a = self.feed_forward(X, e)
			djda = self.loss_function_derivative(a, y)
			self.backpropagation(djda)
			self.improve()
			if e == 0 or e == conf.epochs - 1:
				score.evaluate(e, a, y)
				print("LOSS:", round(self.loss_function(a, y), 2))
				print(f"*average F1_score = {round(np.mean(score.F1_score), 3)}\n*average accuracy = {round(np.mean(score.accuracy), 3)}\n")
				