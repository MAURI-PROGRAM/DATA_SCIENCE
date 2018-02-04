import numpy as numpy
import pandas as pd


from sklearn.tree import DecisionTreeClassifier

df=pd.read_csv('data.csv')
print(df.head)
x_train=df.loc[:,'buying':'safety']
y_train=df.loc[:,'values']
tree=DecisionTreeClassifier(max_leaf_nodes=3,random_state=0)
tree.fit(x_train,y_train)

prediction =tree.predict([[4,3,2,1,2,3]])
print(prediction)