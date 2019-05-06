import numpy as np

statistic = np.genfromtxt("data.csv",delimiter = ',')
X = statistic[:,0:2]
X = np.append(np.ones(shape = [X.shape[0],1]),X,axis = 1)
Y = statistic[:,2,np.newaxis]
Y = Y.T

V = (np.random.random(size = (3,4)) - 0.5) * 2
W = (np.random.random(size = (4,1)) - 0.5) * 2

lr = 0.11

print(Y)

def sigmoid(x):
    if (x >= 0).all():
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))

def dsigmoid(x):
    return x * (1 - x)

for i in range(200000):
    L1 = sigmoid(np.dot(X,V))
    L2 = sigmoid(np.dot(L1,W))

    L2_delta = (Y.T - L2) * dsigmoid(L2)
    L1_delta = L2_delta.dot(W.T) * dsigmoid(L1)

    W_C = lr * L1.T.dot(L2_delta)
    V_C = lr * X.T.dot(L1_delta)

    W += W_C
    V += V_C

    if i % 1000 == 0:
        print("Error: {0}".format(np.mean(np.abs(Y.T - L2))))

