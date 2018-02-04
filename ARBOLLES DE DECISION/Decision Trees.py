from sklearn import tree
X = [[0, 0], [1, 1],[0,1]]
Y = [0, 1,0]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
print(clf.predict([[3,0]]))	