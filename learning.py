import numpy as np
import urllib

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import tree
import os

class Learning:

	def __init__(self,training):
		self.training = training
		self.X, self.y = self.init_data()
		self.model = self.get_model()
		self.tree = self.tree_structure()
		self.num_nodes = self.num_nodes()
		self.children_right, self.children_left = self.children()
		self.feature = self.feature()
		self.threshold = self.threshold()

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
		return self.model.n_features_



	# tree properties
	def tree_structure(self):
		return self.model.tree_

	def num_nodes(self):
		return self.tree.node_count

	def children(self):
		children_left = self.tree.children_left
		children_right = self.tree.children_right
		return children_right, children_left

	def feature(self):
		return self.tree.feature

	def threshold(self):
		return self.tree.threshold

	def tree_print(self, *args, **kwargs):
		node_depth = np.zeros(shape=self.num_nodes)
		is_leaves = np.zeros(shape=self.num_nodes, dtype=bool)
		stack = [(0, -1)]  # seed is the root node id and its parent depth
		while len(stack) > 0:
		    node_id, parent_depth = stack.pop()
		    node_depth[node_id] = parent_depth + 1

		    # If we have a test node
		    if (self.children_left[node_id] != self.children_right[node_id]):
		        stack.append((self.children_left[node_id], parent_depth + 1))
		        stack.append((self.children_right[node_id], parent_depth + 1))
		    else:
		        is_leaves[node_id] = True

		print("The binary tree structure has %s nodes and has "
		      "the following tree structure:"
		      % self.num_nodes)
		for i in range(self.num_nodes):
		    if is_leaves[i]:
		        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
		    else:
		        print("%snode=%s test node: go to node %s if X[:, %s] <= %ss else to "
		              "node %s."
		              % (node_depth[i] * "\t",
		                 i,
		                 self.children_left[i],
		                 self.feature[i],
		                 self.threshold[i],
		                 self.children_right[i],
		                 ))
		return

	def test(self):
		X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=0)
		
		leave_id = self.model.apply(X_test)
		data = {}
		print X_test
		for i in leave_id:
			data[leave_id[i]] = X_test[i]

		print data

	def produce_image(self):
		tree.export_graphviz(self.model, out_file='tree.dot')
		os.system("dot -Tpng tree.dot -o tree.png")
		os.system("open tree.png")
		return


