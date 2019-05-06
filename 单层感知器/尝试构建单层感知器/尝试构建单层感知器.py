import numpy as np
import matplotlib.pyplot as plt

list = np.genfromtxt("data.csv",delimiter = ',')
X = np.append(np.ones([list.shape[0],1]),list[:,:2],axis = 1)
Y = list[:,2,np.newaxis]
W = np.array([0.0,0.0,0.0])
W = W[:,np.newaxis]

lr = 0.11

for i in range(500):
    O = X.dot(W)
    delta_w = (X.T).dot(Y - O) / list.shape[0]
    W += delta_w * lr
    if (O == Y).all():
        break

W = W.ravel()
b = -W[0] / W[2]
k = -W[1] / W[2]

for data in list:
    if data[2] == 1:
        plt.scatter(data[0],data[1],c = 'g',marker = 'o')
    else:
        plt.scatter(data[0],data[1],c = 'r',marker = 'x')
x = np.arange(0,5,0.01)
y = k * x + b
plt.plot(x,y,c = 'b')
plt.show()

