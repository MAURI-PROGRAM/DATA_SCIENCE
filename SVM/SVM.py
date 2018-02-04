import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.datasets import make_circles

df,value= make_circles(n_samples=500,noise=.05,factor=.5)

plt.scatter(df[:,0],df[:,1],c=value)
plt.show()


x=df[:,0]
y=df[:,1]
z=x**2+y**2

kernals=['linear','poly','rbf']
training_set=np.c_[x,y]

for kernal in kernals:
	clf=svm.SVC(kernel=kernal,gamma=2)
	clf.fit(training_set,value)
	prediction=clf.predict([[-0.4,-0.4]])
	print(prediction)

	X=training_set
	y=value
	X0=X[np.where(y==0)]
	X1=X[np.where(y==1)]
	plt.figure()

	x_min=X[:,0].min()
	x_max=X[:,0].max()
	y_min=X[:,1].min()
	y_max=X[:,1].max()

	XX,YY=np.mgrid[x_min:x_max:200j,y_min:y_max:200j]
	Z=clf.decision_function(np.c_[XX.ravel(),YY.ravel()])
	Z=Z.reshape(XX.shape)
	plt.pcolormesh(XX,YY,Z>0,cmap=plt.cm.Paired)
	plt.contour(XX,YY,Z,colors=['k','k','k'],linestyle=['--','-','--'],levels=[-.5,0,.5])
	plt.scatter(X0[:,0],X0[:,1],c='r',s=50)
	plt.scatter(X1[:,0],X1[:,1],c='b',s=50)
	title=('SVC with {} Kernal').format(kernal)
	plt.title(title)
	plt.show()
