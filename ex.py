# from sklearn.datasets import load_iris
# from sklearn import tree
# from sklearn.externals.six import StringIO
# from IPython.display import Image, display
# # from os import system
# # import pydot

# X = [[0,0], [1,1]]
# Y = [0,1]
# clf = tree.DecisionTreeClassifier()
# iris = load_iris()
# clf = clf.fit(X,Y)
# clf = clf.fit(iris.data, iris.target)
# tree.export_graphviz(clf, out_file='tree.dot')

# system("dot -Tpng /path/tree.dot -o path/tree.png")

# print clf.predict([[2., 2.]])
# print clf.predict_proba([[2., 2

# dot_data = StringIO()  
# tree.export_graphviz(clf, out_file=dot_data,  
#                          feature_names=iris.feature_names,  
#                          class_names=iris.target_names,  
#                          filled=True, rounded=True,  
#                          special_characters=True)  
# graph = pydot.graph_from_dot_data(dot_data.getvalue())  
# display(Image(graph.create_png()))
# img.show()
# tree.plot(clf)

# Decision Tree Classifier
# from sklearn import datasets
# from sklearn import metrics
# from sklearn.tree import DecisionTreeClassifier
# # load the iris datasets
# dataset = datasets.load_iris()
# # fit a CART model to the data
# model = DecisionTreeClassifier()
# model.fit(dataset.data, dataset.target)
# print(model)
# # make predictions
# expected = dataset.target
# predicted = model.predict(dataset.data)
# # summarize the fit of the model
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))

# not really important for boolean operation
# from sklearn import preprocessing
# # normalize the data attributes
# normalized_X = preprocessing.normalize(X)
# # standardize the data attributes
# standardized_X = preprocessing.scale(X)

# print normalized_X
# print standardized_X

import numpy as np
import urllib
# url with dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# download the file
raw_data = urllib.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
# separate the data from the target attributes
# take colums and put in array
X = dataset[:,0:7]
y = dataset[:,8]

from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, y)
# display the relative importance of each attribute
print(model.feature_importances_)

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


