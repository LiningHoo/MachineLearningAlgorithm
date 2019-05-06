import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

list = np.genfromtxt("data.csv",delimiter = ',',encoding= "utf-8")

r,c = list.shape

x1 = list[:,0]
x2 = list[:,1]
y = list[:,2]


th0 = 0
th1 = 0
th2 = 0


alpha = 0.01



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

print("before\nth0 = {0},th1 = {1},th2 = {2}".format(th0,th1,th2))

print("error: " + str(cal_error(list,th0,th1,th2)))

for i in range(10000):
    sigma0 = 0
    sigma1 = 0
    sigma2 = 0

    for item in list:
        x1 = item[0]
        x2 = item[1]
        y = item[2]

        sigma0 = (h(x1,x2,th0,th1,th2) - y) / r * alpha
        sigma1 = (h(x1,x2,th0,th1,th2) - y) / r * x1 * alpha
        sigma2 = (h(x1,x2,th0,th1,th2) - y) / r * x2 * alpha

        th0 -= sigma0
        th1 -= sigma1
        th2 -= sigma2

new_x1 = list[:,0]
new_x2 = list[:,1]
new_x1,new_x2 = np.meshgrid(new_x1,new_x2)
new_y = th0 + th1 * new_x1 + th2 * new_x2

print("\nafter\nth0 = {0},th1 = {1},th2 = {2}".format(th0,th1,th2))

print("error: " + str(cal_error(list,th0,th1,th2)))

fig = plt.figure()
ax = fig.add_subplot(111,projection = "3d")
ax.scatter(list[:,0],list[:,1],list[:,2],c = 'r')
ax.plot_surface(new_x1,new_x2,new_y)
plt.show()

