from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

clf_tree = tree.DecisionTreeClassifier()
clf = svm.SVC()
clf1 = KNeighborsClassifier()
clf2 = Perceptron()

clf_tree = clf_tree.fit(X,Y)
clf = clf.fit(X,Y)
clf1 = clf1.fit(X,Y)
clf2 = clf2.fit(X,Y)

prediction_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y, prediction_tree)*100
print('Accuracy for DecisionTree: {}'.format(acc_tree))

prediction = clf.predict(X)
acc = accuracy_score(Y,prediction)*100
print('Accuracy for DecisionTree: {}'.format(acc))

prediction1 = clf1.predict(X)
acc1 = accuracy_score(Y,prediction1)*100
print('Accuracy for DecisionTree: {}'.format(acc1))

prediction2 = clf2.predict(X)
acc2 = accuracy_score(Y,prediction2)*100
print('Accuracy for DecisionTree: {}'.format(acc2))
