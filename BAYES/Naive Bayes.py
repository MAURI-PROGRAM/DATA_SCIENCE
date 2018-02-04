import numpy as numpy
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB

df=pd.read_csv('data.csv')
print(df.head)

plt.xlabel('Feature')
plt.ylabel('Survived')
X=df.loc[:,'Age']
Y=df.loc[:,'Survived']
plt.scatter(X,Y,color='blue',label='Year')

X=df.loc[:,'Nodes']
Y=df.loc[:,'Survived']
plt.scatter(X,Y,color='red',label='Year')

plt.legend(loc=4,prop={'size':7})
plt.show()

X=df.loc[:,'Age':'Nodes']
Y=df.loc[:,'Survived']

clf=GaussianNB()
clf.fit(X,Y)
prediction=clf.predict([[12,70,12],[13,20,13]])
print(prediction)