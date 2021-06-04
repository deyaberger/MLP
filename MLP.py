import pandas as pd
import numpy as np

class LogisticRegression:
	
	def __init__(self):
		self.read_csv(self.args.datafile, train)
		self.activation = self.sigmoid
	
	def check_input(self, df, features, train):
		if features == []:
			display_error("Missing features for our training")
		if train == True and (self.y.any() == False or sum(pd.isnull(df["Hogwarts House"]))):
			display_error("Missing data in the Hogwarts House column")
		if self.y.shape[0] != self.X.shape[0]:
			display_error("Error in dataset, matrixes cannot be used for our logistic regression")
		try:
			np.isnan(self.X.astype(np.float))
		except ValueError as e:
			display_error(e)
   

	def read_csv(self, datafile, train = True):
		if self.args.verbose == 1:
			print("- Reading CSV file -\n")
		df = pd.read_csv(datafile)
		df.fillna(df.median(), inplace = True)
		self.features = []
		if train == True:
			for i in self.args.features:
				self.features.append(df.columns[i + 6])
		else:
			self.features = self.args.features
		if self.args.verbose == 1:
			print(f'- Features used for our Logistic Regression training :\n{self.features}\n')
		self.X = df[self.features].to_numpy()
		one_hot_encoding = pd.get_dummies(df["Hogwarts House"], drop_first = False)
		self.houses = list(one_hot_encoding.columns)
		self.y = one_hot_encoding.to_numpy()
		self.check_input(df, self.features, train)
	
	def feature_scale_normalise(self):
		if self.args.verbose == 1:
			print("- Feature Scaling our data -\n")
		self.scaler = StandardScaler()
		self.scaler.fit(self.X)
		self.X = self.scaler.transform(self.X)

	def add_bias_units(self):
		if self.args.verbose == 1:
			print("- Adding bias units to the Matrix of data -\n")
		bias_units = np.ones((self.X.shape[0], 1))
		self.X = np.concatenate((bias_units, self.X), axis = 1)
	
	def split_data(self):
		if self.args.verbose == 1:
			print(f"- Splitting our data into a training and a testing set _ training set = [{round(self.args.train_size * 100)}%], test_set = [{round((1 - self.args.train_size) * 100)}%] -\n")
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=self.args.train_size, random_state=42)
	
	def init_weights(self):
		if self.args.verbose == 1:
			print("- Initializing all our weights (thetas) to 0 -\n")
		self.thetas = np.zeros((self.X_train.shape[1], self.y_train.shape[1]))
	

	def sigmoid(self, z):
		ret = 1 / (1 + np.exp(-z))
		return(ret)
	
	def softmax(self, z):
		ret = np.exp(z) / sum(np.exp(z))
		return(ret)

	def hypothesis(self, X):
		z = np.matmul(X, self.thetas)
		self.H = self.activation(z)
	
	def compute_loss_gradient(self, X, y):
		error = self.H - y
		self.loss_gradient = np.matmul(X.T, error) / len(X)
	
	def gradient_descent(self):
		self.thetas = self.thetas - (self.args.learning_rate * self.loss_gradient)
	
	def predict(self, y, i):
		predicted_house = np.argmax(self.H[i])
		real_house = np.argmax(y[i])
		return (predicted_house, real_house)

	
	def calculate_cost(self, X, y):
		ylogh = y * np.log(self.H)
		losses = np.sum(ylogh, axis = 1, keepdims = True)
		self.cost = -1.0 * (np.mean(losses))
		self.total_cost.append(self.cost)

	
	def fit(self, score):
		self.total_cost = []
		if self.args.verbose == 1:
			print(f"- Fitting our model to minimize our cost and find the best values for out thetas: -")
			print(f"nb of iterations = [{self.args.epochs}]\nactivation function = [{self.args.activation}]\nstochastic gradient descent = [{self.args.stochastic}]\n")
		for i in range(self.args.epochs):
			score.evaluate(self, i, self.X_test, self.y_test)
			X, y = self.X_train, self.y_train
			if self.args.stochastic == True:
				X, y = self.choose_stochastic_batch(X, y)
			self.hypothesis(X)
			self.calculate_cost(X, y)
			self.compute_loss_gradient(X, y)
			self.gradient_descent()
			if i == 0 and self.args.verbose == 1:
				print(f"--> Before training:\nCost = {round(self.cost, 3)}\n*average F1_score = {round(np.mean(score.F1_score), 3)}\n*average accuracy = {round(np.mean(score.accuracy), 3)}\n")

		if self.args.verbose == 1:
			print(f"--> After training:\nCost = {round(self.cost, 3)}\n*average F1_score = {round(np.mean(score.F1_score), 3)}\n*average accuracy = {round(np.mean(score.accuracy), 3)}\n")

