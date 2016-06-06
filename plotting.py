import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot3d(fun):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x = np.arange(-5.0, 12.0, 0.05)
	y = np.arange(-5.0, 12.0, 0.05)
	X, Y = np.meshgrid(x, y)
	zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
	Z = zs.reshape(X.shape)

	ax.plot_surface(X, Y, Z)

	ax.set_xlabel('X1 Label')
	ax.set_ylabel('X2 Label')
	ax.set_zlabel('Z Label')

	plt.show()

def plotcontour(fun):	
	x = np.arange(-5.0, 12.0, 0.05)
	y = np.arange(-5.0, 12.0, 0.05)
	X, Y = np.meshgrid(x, y)
	zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
	Z = zs.reshape(X.shape)
	
	plt.figure()
	CS = plt.contour(X, Y, Z)
	
	plt.clabel(CS, inline=1, fontsize=10)		
	plt.show()