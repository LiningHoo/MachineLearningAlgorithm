import numpy as np

class Layer():
    delta = None
    net = None
    W_C = None
    output = None
    x = None
    def __init__(self,inputs,ouputs):
        self.W = (np.random.random(size = (inputs,ouputs)) - 0.5) * 2

def activation_func(x):
    if (x >= 0).all():
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))

def dactivation_func(x):
    return x * (1 - x)


def forward_propagation(layers):
    length = len(layers)
    layers[0].x = X
    layers[0].net = np.dot(X,layers[0].W)
    layers[0].output = activation_func(layers[0].net)
    for index in range(1,length):
        layers[index].x = layers[index - 1].output
        layers[index].net = np.dot(layers[index].x,layers[index].W)
        layers[index].output = activation_func(layers[index].net)

def back_propagation(layers):
    length = len(layers)
    layers[length - 1].delta = (Y.T - layers[length - 1].output) * dactivation_func(layers[length - 1].output)
    layers[length - 1].W_C = lr * (layers[length - 1].x.T).dot(layers[length - 1].delta)
    layers[length - 1].W += layers[length - 1].W_C
    for index in range(length - 2,-1,-1):
        layers[index].delta = layers[index + 1].delta.dot(layers[index + 1].W.T) * dactivation_func(layers[index].output)
        layers[index].W_C = lr * (layers[index].x.T).dot(layers[index].delta)
        layers[index].W += layers[index].W_C

def save_Ws(Accuracy,layers):
    file = open(r"./Appropriate Ws.out","a+")
    file.writelines("Accuracy " + str(Accuracy) + "\n")
    for layer in layers:
        file.write("layer[" + str(layers.index(layer)) + "]'s Ws:\n")
        for line in layer.W:
            for w in line:
                file.write(str(w) + " ")    
        file.write("\n")
    file.close()

def tester(layers,X_test,Y_test):
    correct = 0
    length = len(layers)
    layers[0].x_test = X_test
    layers[0].net_test = np.dot(layers[0].x_test,layers[0].W)
    layers[0].output_test = activation_func(layers[0].net_test)
    for index in range(1,length):
        layers[index].x_test = layers[index - 1].output_test
        layers[index].net_test = np.dot(layers[index].x_test,layers[index].W)
        layers[index].output_test = activation_func(layers[index].net_test)
    Accuracy = correct / layers[length - 1].output_test.shape[0]
    for index in range(layers[length - 1].output_test.shape[0]):
        if np.mean(np.abs(layers[length - 1].output_test[index] - (Y_test.T)[index])) < 0.05:
            correct += 1
    if correct / layers[length - 1].output_test.shape[0] > Accuracy:
        save_Ws(correct / layers[length - 1].output_test.shape[0],layers)
    print("Accuracy: {0}".format(correct / layers[length - 1].output_test.shape[0]))
    return None

statistics = np.genfromtxt("data.csv",delimiter = ',')
X = statistics[:,:2]
X = np.append(np.ones(shape = [X.shape[0],1]),X,axis = 1)
Y = statistics[:,2,np.newaxis]
Y = Y.T

Input = Layer(3,4)
Output = Layer(4,1)
layers = [Input,Output]
lr = 0.11


for i in range(20000000):
    forward_propagation(layers)
    back_propagation(layers)
    if i % 1000 == 0:
        print("Error: {0}".format(np.mean(np.abs(Output.output - Y.T))))
        tester(layers,X,Y)
    if np.mean(np.abs(Output.output - Y.T)) < 0.005:
        break

