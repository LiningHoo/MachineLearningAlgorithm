import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

list = np.genfromtxt(r"longley.csv",delimiter = ',')
x_data = list[1:17,2:]
y_data = list[1:17,1,np.newaxis]
r,c = x_data.shape
x_data = np.append(np.ones([r,1]),x_data,axis = 1)

def cal_ws(x,y,lam):
    ws = LA.inv(x.T.dot(x) + lam * np.eye(x.T.dot(x).shape[0])).dot(x.T).dot(y)
    return ws

def cal_error(x,y,lam,ws):
    sigma0 = 0
    sigma1 = 0
    for i in range(x.shape[0]):
        sigma0 += (x[i].dot(ws) - y[i])**2
    for w in ws:
        sigma1 += w**2
    error = (sigma0 + lam * sigma1) / (2 * x.shape[0])
    return error


lambdas = np.arange(0.001,1.001,0.001)
errors = []
W = []
best_lam = None
for lam in lambdas:
    ws = cal_ws(x_data,y_data,lam)
    W.append(ws)
    error = cal_error(x_data,y_data,lam,ws)
    errors.append(error)
min_error = min(errors)
for index in range(len(errors)):
    if errors[index] == min_error:
        best_lam = lambdas[index]
        best_ws = W[index]

print("The best lambda is {0} . ".format(best_lam))
print("\n")
for i in range(len(best_ws)):
    print("w{0} = {1} ".format(i,best_ws[i][0]))

plt.plot(lambdas,errors,'b')
plt.show()