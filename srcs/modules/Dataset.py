import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import conf

class Dataset:
	def __init__(self, datafile):
		df = self.read_specific_csv(datafile)
		self.from_df_to_matrix(df)
	
	def read_specific_csv(self, datafile):
		df = pd.read_csv(datafile, header = None)
		if df.empty == True:
			print("Error, empty dataset")
		self.features = [f"feature {i}" for i in range(df.shape[1] - 2)]
		self.y_name = "diagnosis"
		columns = ["id", self.y_name]
		columns.extend(self.features)
		df.columns = columns
		return (df)

	def from_df_to_matrix(self, df):
		df[self.features] = df[self.features].fillna(df[self.features].median())
		self.X = df[self.features].to_numpy()
		one_hot_encoding = pd.get_dummies(df[self.y_name], drop_first = False)
		self.y_classification = list(one_hot_encoding.columns)
		self.y = one_hot_encoding.to_numpy()
	
	
	def feature_scale_normalise(self):
		self.scaler = StandardScaler()
		self.scaler.fit(self.X)
		self.X = self.scaler.transform(self.X)


	def split_data(self):
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=conf.train_size, random_state=42)

	
	def __str__(self):
		return (f"{self.X_train.shape = }, {self.X_test.shape = }, {self.y_train.shape = }, {self.y_test.shape = }")
	