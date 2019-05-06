import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

list = np.genfromtxt("LogiReg_data.txt",delimiter = ',')
x1 = list[:,0,np.newaxis]
x2 = list[:,1,np.newaxis]
y = list[:,2,np.newaxis]
m,n = list.shape
xs = np.append(np.ones([m,1]),list[:,:2],axis = 1)
list = np.append(np.ones([m,1]),list,axis = 1)
ths = np.array([0.0,0.0,0.0])
ths = ths[:,np.newaxis]
basedata = list[:,:]
delta = np.array([None,None,None])
alpha = 10

for data in basedata:
    if data[3] == 1:
        plt.scatter(data[1],data[2],c = 'g',marker = 'o')
    else:
        plt.scatter(data[1],data[2],c = 'r',marker = 'x')

def g(x):
     if x > 0:
        return 1.0 / (1.0 + np.exp(-x))
     else:
        return (np.exp(x)) / (1 + np.exp(x))

def z(x):
    return ths.dot(x.T)

def h(z):
    return g(z)

for i in range(50):
    #gs = []
    #x_T = xs.T
    #z = xs.dot(ths)
    #for item in z:
     #  gs.append(g(item))
    #gs = np.array(gs)
    #print(xs.shape)
    #print(x_T.shape)
    #print(gs.shape)
    #print((gs - y).shape)
    sigma = xs.T.dot(g(xs.dot(ths)) - y)
    ths -= alpha * sigma


x = np.arange(min(x1),max(x1),0.1)
new_th0 = -(ths[0] / ths[2])
new_th1 = -(ths[1] / ths[2])
y_plot = new_th0 + new_th1 * x

print(ths)

plt.xlabel("x1")
plt.ylabel("x2")
plt.plot(x,y_plot,'b')
plt.show()


