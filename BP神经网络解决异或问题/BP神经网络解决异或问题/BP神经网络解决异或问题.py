import numpy as np

X = np.array([[1,0,0],
               [1,0,1],
               [1,1,0],
               [1,1,1]])

Y = np.array([[0,1,1,0]])

V = np.random.random(size = (3,4))
W = np.random.random(size = (4,1))

def sigmoid(x):
    if (x >= 0).all():
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))

def dsigmoid(x):
    return x * (1 - x)

lr = 0.11

for i in range(20000):
    L1 = sigmoid(np.dot(X,V))
    L2 = sigmoid(np.dot(L1,W))

    L2_delta = (Y.T - L2) * (dsigmoid(L2))
    L1_delta = L2_delta.dot(W.T) * (dsigmoid(L1))

    W += lr * (L1.T).dot(L2_delta)
    V += lr * (X.T).dot(L1_delta)

    if i % 500 == 0:
        print("Error: {0}".format(np.mean(Y.T - L2)))
