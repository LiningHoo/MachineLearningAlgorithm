import numpy as np
import random
import datetime
import struct 


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
        self.W = ((np.random.random(size = (inputs,ouputs)) - 0.5) * 2) * 0.01
    #def __init__(self, parameter_path):
        #self.W = np.load(parameter_path)

# 激活函数
def activation_func(x, mode):
    if(mode=="sigmoid"):
        if (x >= 0).all():
            return 1 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))
    if(mode=="relu"):
        #print(np.maximum(0.1*x, x))
        return np.maximum(0.1*x, x)


# 激活函数的导数
def dactivation_func(x, mode):
    if(mode=="sigmoid"):
        return x * (1 - x)
    if(mode=="relu"):
        result = (x < 0).astype(np.float32)
        result *= -0.9
        result = result + 1
        #print(result)
    return result



# 前向传播
def forward_propagation(layers):
    length = len(layers)
    layers[0].x = X
    layers[0].net = np.dot(X,layers[0].W)
    layers[0].output = activation_func(np.array(layers[0].net), mode="relu")
    for index in range(1,length):
        layers[index].x = layers[index - 1].output
        layers[index].net = np.dot(layers[index].x,layers[index].W)
        layers[index].output = activation_func(layers[index].net, mode="relu")
        

# 反向传播
def back_propagation(layers):
    length = len(layers)
    #layers[length - 1].delta = (Y.T - layers[length - 1].output) * dactivation_func(layers[length - 1].output, mode="sigmoid")
    layers[length - 1].delta = (Y.T - layers[length - 1].output) * dactivation_func(layers[length - 1].net, mode="relu")
    layers[length - 1].W_C = lr * (layers[length - 1].x.T).dot(layers[length - 1].delta)
    layers[length - 1].W += layers[length - 1].W_C
    for index in range(length - 2,-1,-1):
        #layers[index].delta = layers[index + 1].delta.dot(layers[index + 1].W.T) * dactivation_func(layers[index].output, mode="sigmoid")
        layers[index].delta = layers[index + 1].delta.dot(layers[index + 1].W.T) * dactivation_func(layers[index].net, mode="relu")
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
def normalizing_minus1_to_1(images):
    for index in range(images.shape[0]):
        images[index] = (images[index] - np.mean(images[index])) / (np.max(images[index]) - np.min(images[index]))
    return images

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
def tester(layers,X_test,Y_test):
    global Accuracy
    global global_step
    correct = 0
    length = len(layers)
    layers[0].x_test = X_test
    layers[0].net_test = np.dot(layers[0].x_test,layers[0].W)
    layers[0].output_test = activation_func(layers[0].net_test, mode = "relu")
    for index in range(1,length):
        layers[index].x_test = layers[index - 1].output_test
        layers[index].net_test = np.dot(layers[index].x_test,layers[index].W)
        layers[index].output_test = activation_func(layers[index].net_test, mode = "relu")
    output_len = layers[length - 1].output_test.shape[0]
    index = 0
    while(index < output_len):
        if np.argmax(layers[length - 1].output_test[index]) == np.argmax((Y_test.T)[index]):
            correct += 1
        index += 1
    if correct / layers[length - 1].output_test.shape[0] > Accuracy:
        # save_Ws(correct / layers[length - 1].output_test.shape[0],layers)
        np.save("w1.log",Input.W)
        np.save("w2.log",Output.W)
        #print("Error: {0}".format(np.mean(np.abs(Output.output - Y.T))),end="   ")
        #print("Accuracy: {0}%".format(round(((correct / layers[length - 1].output_test.shape[0]) * 100),2)),end="   ")
        #print("global_step = {0}".format(global_step))
        #Accuracy = correct / layers[length - 1].output_test.shape[0]
    print("Accuracy: {0}%".format(round(((correct / layers[length - 1].output_test.shape[0]) * 100),2)))
    return correct / layers[length - 1].output_test.shape[0]


train_images_idx3_ubyte_file = './MNIST_data/train-images.idx3-ubyte'
train_labels_idx1_ubyte_file = './MNIST_data/train-labels.idx1-ubyte'
test_images_idx3_ubyte_file = './MNIST_data/t10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = './MNIST_data/t10k-labels.idx1-ubyte'

def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>iiii' 
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  
    # print(offset)
    fmt_image = '>' + str(image_size) + 'B' 
    # print(fmt_image,offset,struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows * num_cols))
    for i in range(num_images):
        # if (i + 1) % 10000 == 0:
            # print('已解析 %d' % (i + 1) + '张')
            # print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    # print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        # if (i + 1) % 10000 == 0:
            # print ('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)

X_get = load_train_images()
Y_get = load_train_labels()
X_test_get = load_test_images()
Y_test_get = load_test_labels()
# mnist = fetch_mldata('MNIST Original',data_home = r'./datasets')
# X = mnist['data']
# Y = mnist['target']
#statistics = np.genfromtxt("train.csv",delimiter = ',')
#test_statistics = np.genfromtxt("test.csv",delimiter = ',')
#X_get = statistics[:,1:]
#Y_get = statistics[:,0]
#X_test_get = test_statistics[:,1:]
#Y_test_get = test_statistics[:,0]
#X_get = X_get[:,:]
#Y_gte = Y_get[:]
Y_get = new_Y(Y_get) # 转换为one hot编码
Y_test_get = new_Y(Y_test_get)
Y_get = Y_get.T
Y_test_get = Y_test_get.T
# 特征矩阵添加偏置值
X_get = np.append(np.ones(shape = [X_get.shape[0],1]),X_get,axis = 1)
X_test_get = np.append(np.ones(shape = [X_test_get.shape[0],1]),X_test_get,axis = 1)
# 归一化处理数据
X_get = normalizing_minus1_to_1(X_get)
Y_get = normalizing_minus1_to_1(Y_get)
X_test_get = normalizing_minus1_to_1(X_test_get)
Y_test_get = normalizing_minus1_to_1(Y_test_get)

Input = Layer(785, 500)
Output = Layer(500, 10)
#Input = Layer("w1.log.npy")
#Output = Layer("w2.log.npy")

layers = [Input, Output]
train_steps = 60000
batch_size = 64

global_step = 0.0
#学习率指数下降
lr = 1e-3
decay_step = X_get.shape[0] / batch_size
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
        if(global_step % 100 == 0):
            print("#", end="", flush = True)
        if global_step % 1000 == 0:
            print("\nlr = {} Error: {}".format(lr, np.mean(np.abs(Output.output - Y.T))),end="   ")
            #print("\ntrain: ", end="")
            #tester(layers,X_get[: 1000],Y_get[: 1000])
            #print("test: ", end="")
            print("\ntrain: ", end="")
            tester(layers,X_get[: 1000],Y_get[: 1000])
            print("test: ", end="")
            accuracy = tester(layers,X_test_get[: 1000],Y_test_get[: 1000])
        #if global_step % decay_step == 0 and global_step != 0:
            #lr = lr * decay_rate**(global_step / decay_step)




