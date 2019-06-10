import numpy as np
import matplotlib.pyplot as plt
import struct



# 读取数据

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
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        # if (i + 1) % 10000 == 0:
            # print('已解析 %d' % (i + 1) + '张')
            # print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
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

def normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img

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
    def __init__(self,parameter_path):
        self.W = np.load(parameter_path)

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
        result = x < 0
        result * -0.9
        result = result + 1
        #print(result)
    return result


# 前向传播
def forward_propagation(X, layers):
    length = len(layers)
    layers[0].x = X
    layers[0].net = np.dot(X,layers[0].W)
    layers[0].output = activation_func(np.array(layers[0].net), mode="relu")
    for index in range(1,length):
        layers[index].x = layers[index - 1].output
        layers[index].net = np.dot(layers[index].x,layers[index].W)
        layers[index].output = activation_func(layers[index].net, mode="relu")
    return layers[length - 1].output
        

# 反向传播
def back_propagation(Y, layers):
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


def predict(X):
    return np.argmax(forward_propagation(np.append(np.ones(shape=[1, 1]), X.reshape(1, 784), axis = 1), layers))

Input = Layer("w1.log.npy")
Output = Layer("w2.log.npy")
layers = [Input,Output]

# 载入数据集
train_images = load_train_images()
train_labels = load_train_labels()
test_images = load_test_images()
test_labels = load_test_labels()

# 归一化
train_images = normalize_image(train_images)
#train_labels = normalize_image(train_labels)
test_images = normalize_image(test_images)
#test_labels = normalize_image(test_labels)

correct = 0
#print(forward_propagation(np.append(np.ones(shape=[1, 1]), train_images[0].reshape(1, 784), axis = 1), layers))
for index in range(0, test_images.shape[0]):
    if(predict(test_images[index])==test_labels[index]):
        correct += 1
    #print("{} {}".format(predict(train_images[index]), train_labels[index]))
    plt.title("guess : {}".format(predict(test_images[index])))
    plt.gray()
    plt.imshow(test_images[index])
    plt.show()
print("accuracy: {} / {}".format(correct, test_images.shape[0]))



