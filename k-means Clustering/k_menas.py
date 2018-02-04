import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('data.csv')

print(df.head())

k=2
Kmeans=KMeans(n_clusters=k)
Kmeans=Kmeans.fit(df)

labels=Kmeans.labels_

centroids=Kmeans.cluster_centers_

x_test=[[4.6,67],[2.885,61],[1.66,90],[5.6,54],[2.6,80]]

predictions=Kmeans.predict(x_test)
print(predictions)

colors=['blue','red','green','black']

y=0

for x in labels:
	plt.scatter(df.iloc[y,0],df.iloc[y,1],color=colors[x])
	y+=1

for x in range(k):
	lines=plt.plot(centroids[x,0],centroids[x,1],'kx')
	plt.setp(lines,ms=15.0)
	plt.setp(lines,mew=2.0)

title=('N.o de grupos={}').format(k)
plt.title=title
plt.xlabel('erupciones en (min)')
plt.ylabel('esperar (min)')
plt.show()
