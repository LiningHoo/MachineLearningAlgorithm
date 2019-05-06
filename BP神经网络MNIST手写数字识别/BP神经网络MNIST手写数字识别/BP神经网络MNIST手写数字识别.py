from sklearn.datasets import fetch_mldata
import numpy as np
import random
from numba import jit 


# 层类
class Layer():
    delta = None
    net = None
    W_C = None
    output = None
    x = None
    x_test = None
    y_test = None
    net_test = None
    output_test = None
    def __init__(self,inputs,ouputs):
        self.W = (np.random.random(size = (inputs,ouputs)) - 0.5) * 2

# 激活函数
@jit
def activation_func(x):
    if (x >= 0).all():
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))
# 激活函数的导数
@jit
def dactivation_func(x):
    return x * (1 - x)

# 前向传播
@jit
def forward_propagation(layers):
    length = len(layers)
    layers[0].x = X
    layers[0].net = np.dot(X,layers[0].W)
    layers[0].output = activation_func(layers[0].net)
    for index in range(1,length):
        layers[index].x = layers[index - 1].output
        layers[index].net = np.dot(layers[index].x,layers[index].W)
        layers[index].output = activation_func(layers[index].net)

# 反向传播
@jit
def back_propagation(layers):
    length = len(layers)
    layers[length - 1].delta = (Y.T - layers[length - 1].output) * dactivation_func(layers[length - 1].output)
    layers[length - 1].W_C = lr * (layers[length - 1].x.T).dot(layers[length - 1].delta)
    layers[length - 1].W += layers[length - 1].W_C
    for index in range(length - 2,-1,-1):
        layers[index].delta = layers[index + 1].delta.dot(layers[index + 1].W.T) * dactivation_func(layers[index].output)
        layers[index].W_C = lr * (layers[index].x.T).dot(layers[index].delta)
        layers[index].W += layers[index].W_C

# 生成one-hot编码
def new_Y(Y):
    new_Ys = []
    for y in Y:
        zeros = [0 for i in range(10)]
        zeros[int(y)] = 1.0
        new_Ys.append(zeros)
    new_Ys = np.array(new_Ys)
    return new_Ys

# 归一化处理
def normalizing_0_to_1(Mat):
    for line in Mat:
        line /= 255
    return Mat

threshold = 0.3
def save_Ws(Accuracy,layers):
    global threshold
    if Accuracy > threshold:
        file = open(r"./Appropriate Ws.out","a+")
        # file.writelines("Accuracy " + str(Accuracy) + "\n")
        for layer in layers:
            file.write("layer[" + str(layers.index(layer)) + "]'s Ws:\n")
            for line in layer.W:
                for w in line:
                    file.write(str(w) + " ")    
            file.write("\n")
        file.close()
        threshold = threshold + 0.05

Accuracy = 0.0
# 验证函数
@jit
def tester(layers,X_test,Y_test):
    global Accuracy
    global global_step
    correct = 0
    length = len(layers)
    layers[0].x_test = X_test
    layers[0].net_test = np.dot(layers[0].x_test,layers[0].W)
    layers[0].output_test = activation_func(layers[0].net_test)
    for index in range(1,length):
        layers[index].x_test = layers[index - 1].output_test
        layers[index].net_test = np.dot(layers[index].x_test,layers[index].W)
        layers[index].output_test = activation_func(layers[index].net_test)
    output_len = layers[length - 1].output_test.shape[0]
    index = 0
    while(index < output_len):
        if np.argmax(layers[length - 1].output_test[index]) == np.argmax((Y_test.T)[index]):
            correct += 1
        index += 1
    if correct / layers[length - 1].output_test.shape[0] > Accuracy:
        save_Ws(correct / layers[length - 1].output_test.shape[0],layers)
        #print("Error: {0}".format(np.mean(np.abs(Output.output - Y.T))),end="   ")
        #print("Accuracy: {0}%".format(round(((correct / layers[length - 1].output_test.shape[0]) * 100),2)),end="   ")
        #print("global_step = {0}".format(global_step))
        #Accuracy = correct / layers[length - 1].output_test.shape[0]
    print("Accuracy: {0}%".format(round(((correct / layers[length - 1].output_test.shape[0]) * 100),2)))
    return None



# mnist = fetch_mldata('MNIST Original',data_home = r'./datasets')
# X = mnist['data']
# Y = mnist['target']
statistics = np.genfromtxt("train.csv",delimiter = ',')
test_statistics = np.genfromtxt("test.csv",delimiter = ',')
X_get = statistics[:,1:]
Y_get = statistics[:,0]
X_test_get = test_statistics[:,1:]
Y_test_get = test_statistics[:,0]
X_get = X_get[:,:]
Y_gte = Y_get[:]
Y_get = new_Y(Y_get) # 转换为one hot编码
Y_test_get = new_Y(Y_test_get)
Y_get = Y_get.T
Y_test_get = Y_test_get.T
# 特征矩阵添加偏置值
X_get = np.append(np.ones(shape = [X_get.shape[0],1]),X_get,axis = 1)
X_test_get = np.append(np.ones(shape = [X_test_get.shape[0],1]),X_test_get,axis = 1)
# 归一化处理数据
X_get = normalizing_0_to_1(X_get)
Y_get = normalizing_0_to_1(Y_get)
X_test_get = normalizing_0_to_1(X_test_get)
Y_test_get = normalizing_0_to_1(Y_test_get)


Input = Layer(785,100)
Output = Layer(100,10)
layers = [Input,Output]
train_steps = 50000
batch_size = 32

global_step = 0
#学习率指数下降
lr = 0.66
decay_step = 1e4
decay_rate = 0.9999

while(1):
    for i in range(train_steps):
        global_step = global_step + 1
        start = i * batch_size % len(X_get)
        end = min(start + batch_size, len(X_get))
        X = X_get[start: end, :]
        Y = Y_get[:,start: end]
        forward_propagation(layers)
        back_propagation(layers)
        # tester(layers,X_get,Y_get)
        if i % 1000 == 0:
            print("Error: {0}".format(np.mean(np.abs(Output.output - Y.T))),end="   ")
            tester(layers,X_get,Y_get)
        #if global_step % decay_step == 0 and global_step != 0:
            #lr = lr * decay_rate**(global_step / decay_step)




