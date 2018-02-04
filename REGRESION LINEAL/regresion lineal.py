import numpy as np
import pandas as pd

from matplotlib import pyplot as plt 
from sklearn.linear_model import LinearRegression

df=pd.read_csv('data.csv')

print(df.head())

x_train=df['Father'].values[:,np.newaxis]
y_train=df['Son'].values

lm=LinearRegression()
lm.fit(x_train,y_train)

x_test=[[72.8],[61.1],[67.4],[70.2],[59.2],[75.6]]

predictions=lm.predict(x_test)

print(predictions)

plt.scatter(x_train,y_train,color='b')
plt.plot(x_test,predictions,color='black',linewidth=3)

plt.xlabel('Father height in inches')
plt.ylabel('Son height in inches')
plt.show()
