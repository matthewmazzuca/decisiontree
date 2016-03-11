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

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from learning import Learning

# url with dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# download the file
raw_data = urllib.urlopen(url)
# print ra
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
# d2 = np.loadtxt('final_training.txt', dtype=str, delimiter="\t")
# print d2[1]
# with open('final_training.txt', 'r') as f:

# 		for row in f:

# 			rows = row.split('\r')
# 			for i in rows:
# 				currentrow =  i.split('\t')
# 				d2.append(currentrow)

# f.close()


# d2 = np.loadtxt(, dtype=str, delimiter='\t', comments=None)
# separate the data from the target attributes
# take colums and put in array
# Y2 = d2[:,0]
# X2 = d2[:,1:,]

# X2 = X2.astype(int)
# print X2
# print Y2

# X2 = np.delete(X2, 0, 0)
# Y2 = np.delete(Y2, 0)

# print len(X2)
# print len(Y2)

# print X2
# print Y2

# X = dataset[:,0:7]
# print X[1]
# y = dataset[:,8]
# print y[1]


learning = Learning('final_training3.txt')


# reshaped = [learning.X[1]]
# for item in learning.y:
# 	print item
# print learning.X[30]
# for item in learning.target:
# 	print item

print learning.get_question(0)
# for item in range(len(learning.X[30])):
# 	if learning.X[30][item] == 1:
# 		print learning.target[item]
# learning.tree_print()
# learning.treeToJson()
# learning.produce_image()
# print learning.predict(reshaped)
# print learning.predict([learning.X[30]])

# print "your diagnosis:", '\t', "you have acid reflux"
# metrics



# fit a CART model to the data

# make predictions
# summarize the fit of the model
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))


# from sklearn import metrics
# from sklearn.ensemble import ExtraTreesClassifier
# model = ExtraTreesClassifier()
# model.fit(X, y)
# # display the relative importance of each attribute
# print(model.feature_importances_)


# ############feature identifications
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# # create the RFE model and select 3 attributes
# rfe = RFE(model, 3)
# rfe = rfe.fit(X, y)
# # summarize the selection of the attributes
# print(rfe.support_)
# print(rfe.ranking_)




