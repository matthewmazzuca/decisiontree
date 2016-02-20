import numpy as np
import urllib

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

class Learning:

	def __init__(self,training):
		self.training = training
		self.X, self.y = self.init_data()
		self.model = self.get_model()

	def init_data(self):
		raw = urllib.urlopen(self.training)
		dataset = np.loadtxt(raw, delimiter=",")
		X = dataset[:,0:7]
		y = dataset[:,8]
		return X, y

	def get_model(self):
		model = DecisionTreeClassifier(criterion='entropy', presort=True)
		model.fit(self.X, self.y)
		return model

	def num_features(self):
		print self.model.n_features_
		return self.model.n_features_

	def tree_structure(self):
		print model.tree_
		return model.tree_