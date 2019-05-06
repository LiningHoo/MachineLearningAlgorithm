import matplotlib.pyplot as plt
import numpy as np

list = np.genfromtxt("data.csv",delimiter= ',')
r,c = list.shape

x = []
y = []

for item in list:
    x.append(item[0])
    y.append(item[1])

x_left_num = min(x)
x_right_num = max(x)


alpha = 0.01

th0 = 0
th1 = 0

def h(x,th0,th1):
    return (th0 + th1 * x)

def cal_error(list,th0,th1):
    sigma = 0
    for item in list:
        x = item[0]
        y = item[1]
        sigma += (h(x,th0,th1) - y)**2 
    error = sigma / r
    return error

for i in range(5000):
    for item in list:
        sigma0 = (h(item[0],th0,th1) - item[1]) / r * alpha
        sigma1 = (h(item[0],th0,th1) - item[1]) * item[0] / r * alpha

        th0 -= sigma0
        th1 -= sigma1

x_new = np.arange(x_left_num,x_right_num,0.01)
y_new = []
for item in x_new:
    y_new.append(th0 + th1 * item)

print("error: " + str(cal_error(list,th0,th1)))

plt.scatter(x,y,c = 'g')
plt.plot(x_new,y_new,c ='b')
plt.show()

