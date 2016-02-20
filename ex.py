from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
from IPython.display import Image, display
# from os import system
# import pydot

X = [[0,0], [1,1]]
Y = [0,1]
clf = tree.DecisionTreeClassifier()
iris = load_iris()
clf = clf.fit(X,Y)
clf = clf.fit(iris.data, iris.target)
tree.export_graphviz(clf, out_file='tree.dot')

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

