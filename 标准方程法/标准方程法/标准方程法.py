import numpy as np
import matplotlib.pyplot as plt
from  mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as LA

list = np.genfromtxt("data.csv",delimiter = ',')
old_X = list[:,0:2]
Y = list[:,2,np.newaxis]
ONES = np.ones([20,1])
X = np.append(ONES,old_X,axis = 1)

r,c = list.shape

def cal_w(X,Y):
    W = LA.inv(X.T.dot(X)).dot(X.T).dot(Y)
    return W

def h(x1,x2,th0,th1,th2):
    return (th0 + th1 * x1 + th2 * x2)

def cal_error(list,th0,th1,th2):
    sigma = 0
    for item in list:
        x1 = item[0]
        x2 = item[1]
        y = item[2]
        sigma += (h(x1,x2,th0,th1,th2) - y)**2
    error = sigma / r
    return error

W = cal_w(X,Y)

th0 = W[0][0]
th1 = W[1][0]
th2 = W[2][0]

new_x1 = list[:,0]
new_x2 = list[:,1]
new_x1,new_x2 = np.meshgrid(new_x1,new_x2)
new_y = th0 + th1 * new_x1 + th2 * new_x2

print("error : {0}".format(cal_error(list,th0,th1,th2)))

fig = plt.figure()
ax = fig.add_subplot(111,projection = "3d")
ax.scatter(list[:,0],list[:,1],list[:,2],c = 'r',marker = 'o')
ax.plot_surface(new_x1,new_x2,new_y)
plt.show()

