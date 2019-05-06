import numpy as np
from scipy.signal import fftconvolve as conv2
import matplotlib.pyplot as plt
import struct


k1_seed = np.sqrt(6 / ((1 + 6) * 5**2))
k2_seed = np.sqrt(6 / ((6 + 12) * 5**2))
W_seed = np.sqrt(6 / (192 + 10))
b1 = np.array([np.random.uniform(0.0, 0.01) for i in range(6)])
b2 = np.array([np.random.uniform(0.0, 0.01) for i in range(12)])
b = np.array([np.random.uniform(0.0, 0.01) for i in range(10)]).reshape(-1, 1)

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

# 初始化参数

def k1_init():
    k1 = [None for i in range(6)]
    for p in range(6):
        k1[p] = np.random.uniform(-0.5, 0.5, size=[5, 5])
    return np.array(k1)

def k2_init():
    k2 = [[None for i in range(12)] for i in range(6)]
    for p in range(6):
        for q in range(12):
            k2[p][q] = np.random.uniform(-0.5, 0.5, size=[5, 5])
    return np.array(k2)



def W_init():
    return np.random.uniform(-0.5, 0.5, size=[10, 192])

def normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img

def one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[int(label[i])] = 1
    return lab

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def sigmoid_derivative(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return ds

def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis = 0, keepdims = True)
    s = x_exp / x_sum    
    return s

def cal_C1(I, k1):
    C1 = np.array([None for i in range(6)])
    for p in range(6):
        C1[p] = sigmoid(conv2(I, k1[p], mode='valid') + b1[p])
    return C1

def cal_S1(C1):
    S1 = [[[None for j in range(12)] for i in range(12)] for p in range(6)]
    for p in range(6):
        for i in range(12):
            for j in range(12):
                S1[p][i][j] = 0
                for v in range(2):
                    for u in range(2):
                        S1[p][i][j] += C1[p][2 * i - u, 2 * j - v] / 4
    return np.array(S1)

def cal_C2(S1, k2, b2):
    C2 = np.array([None for i in range(12)])
    for q in range(12):
       sum = [[0.0 for j in range(8)] for i in range(8)]
       for p in range(6):
           sum += conv2(S1[p], k2[p][q], mode='valid')
       C2[q] = sigmoid(sum + b2[q])
    return C2

def cal_S2(C2):
    S2 = [[[None for j in range(4)] for i in range(4)] for p in range(12)]
    for q in range(12):
        for i in range(4):
            for j in range(4):
                S2[q][i][j] = 0
                for v in range(2):
                    for u in range(2):
                        S2[q][i][j] += C2[q][2 * i - u, 2 * j - u] / 4
    return np.array(S2)

def ravel(x):
    return x.ravel(order=1).reshape(-1, 1)

def vectorizatiion(S2):
    vectors = [None for i in range(12)]
    for q in range(12):
        vectors[q] = ravel(S2[q])
    return np.array(vectors)

def concatenation(vectors):
    return vectors.reshape(-1, 1)

def convolve_and_pooling(I):
    C1 = cal_C1(I, k1)
    S1 = cal_S1(C1)
    C2 = cal_C2(S1, k2, b2)
    S2 = cal_S2(C2)
    vectors = vectorizatiion(S2)
    f = concatenation(vectors).reshape(192, 1)
    return C1, S1, C2, S2, f



def get_train_images_batches(train_images, batch_size):
    start = 0
    batches = []
    while(start <= train_images.shape[0] - batch_size):
        batch = [None for i in range(batch_size)]
        end = start + batch_size
        for curve in range(batch_size):
            C1, S1, C2, S2, f = convolve_and_pooling(train_images[start + curve])
            batch[curve] = f
        start = end 
        batches.append(np.array(batch).T)
    return np.array(batches)

def get_train_labels_batches(train_labels, batch_size):
    start = 0
    batches = []
    while(start <= train_labels.shape[0] - batch_size):
        batch = [None for i in range(batch_size)]
        end = start + batch_size
        for curve in range(batch_size):
            batch[curve] = train_labels[start + curve]
        start = end 
        batches.append(np.array(batch).T)
    return np.array(batches)

def get_y_pred(f):
    return sigmoid(np.dot(W, f) + b)

def get_delta_y_pred(y_pred, y):
    return (y_pred - y) * y_pred * (1 - y_pred)

def get_delta_W(delta_y_pred, f):
    return np.dot(delta_y_pred, f.T)

def get_delta_b(delta_y_pred):
    return delta_y_pred

def get_delta_f(W, delta_y_pred):
    return np.dot(W.T, delta_y_pred)

def square_transpose(Mat):
    length = Mat.shape[0]
    for i in range(length):
        for j in range(length):
            if i < j :
                temp = Mat[i][j]
                Mat[i][j] = Mat[j][i]
                Mat[j][i] = temp
    return Mat


def get_delta_S2(delta_f):
    delta_S2 = np.transpose(delta_f).reshape([12, 4, 4])
    for q in range(12):
       delta_S2[q] = delta_S2[q].T
    return delta_S2

def get_delta_C2(delta_S2):
    delta_C2 = np.array([[[None for j in range(8)] for i in range(8)] for q in range(12)])
    for q in range(12):
        for i in range(8):
            for j in range(8):
                delta_C2[q][i][j] = delta_S2[q][int(i / 2)][int(j / 2)] / 4
    return delta_C2

def get_delta_C2_sigma(delta_C2, C2):
    delta_C2_sigma = np.array([[[None for j in range(8)] for i in range(8)] for q in range(12)])
    for q in range(12):
        for i in range(8):
            for j in range(8):
                delta_C2_sigma[q][i][j] = delta_C2[q][i][j] * C2[q][i][j] * (1 - C2[q][i][j])
    return np.array(delta_C2_sigma)


def fz(a):
    return a[::-1]

def rotator(mat):
    return np.array(fz(list(map(fz, mat))))

def get_S1_rot(S1):
    S1_rot = np.array([[[None for j in range(12)] for i in range(12)] for p in range(6)])
    for p in range(6):
        S1_rot[p] = rotator(S1[p])
    return S1_rot

def get_delta_k2(S1_rot, delta_C2_sigma):
    delta_k2 = np.array([[[[None for j in range(5)] for i in range(5)] for q in range(12)] for p in range(6)])
    for p in range(6):
        for q in range(12):
            delta_k2[p][q] = conv2(S1_rot[p], delta_C2_sigma[q], mode='valid')
    return delta_k2

def get_delta_b2(delta_C2_sigma):
    delta_b2 = np.array([None for q in range(12)])
    for q in range(12):
        delta_b2[q] = np.sum(delta_C2_sigma[q])
    return delta_b2

def get_k2_rot(k2):
    k2_rot = np.array([[None for q in range(12)] for p in range(6)])
    for p in range(6):
        for q in range(12):
            k2_rot[p][q] = rotator(k2[p][q])
    return k2_rot

def get_delta_S1(delta_C2_sigma, k2_rot):
    delta_S1 = np.array([[[0.0 for j in range(12)] for i in range(12)] for p in range(6)])
    for p in range(6):
        for q in range(12):
            delta_S1[p] += conv2(delta_C2_sigma[q], k2_rot[p][q], mode='full') 
    return delta_S1

def get_delta_C1(delta_S1):
    delta_C1 = np.array([[[None for j in range(24)] for i in range(24)] for p in range(6)])
    for p in range(6):
        for i in range(24):
            for j in range(24):
                delta_C1[p][i][j] = delta_S1[p][int(i / 2)][int(j / 2)] / 4
    return delta_C1

def get_delta_C1_sigma(delta_C1, C1):
    delta_C1_sigma = np.array([[[None for j in range(24)] for i in range(24)] for p in range(6)])
    for p in range(6):
        delta_C1_sigma[p] = delta_C1[p] * C1[p] * (1 - C1[p])
    return delta_C1_sigma

def get_delta_k1(I, delta_C1_sigma):
    delta_k1 = np.array([[[None for j in range(5)] for i in range(5)] for p in range(6)])
    for p in range(6):
        delta_k1[p] = conv2(rotator(I), delta_C1_sigma[p], mode='valid')
    return delta_k1

def get_delta_b1(delta_C1_sigma):
    delta_b1 = np.array([None for p in range(6)])
    for p in range(6):
        delta_b1[p] = np.sum(delta_C1_sigma[p])
    return delta_b1

def error(y_pred, y):
    return np.mean(np.abs(y_pred - y)) 

def cal_accuracy(test_images, one_hot_test_labels, test_size):
    correct = 0
    for index in range(test_size):
        # index = np.random.randint(0, test_images.shape[0])
        I = test_images[index]
        C1, S1, C2, S2, f = convolve_and_pooling(I)
        y_pred = get_y_pred(f)
        if np.argmax(y_pred) == np.argmax(one_hot_test_labels[index]):
            correct += 1
    accuracy = correct / test_size
    return "{0} / {1}".format(correct, test_size)

train_images = load_train_images()
train_labels = load_train_labels()
test_images = load_test_images()
test_labels = load_test_labels()

# 生成独热码
one_hot_train_labels = one_hot_label(train_labels)
one_hot_test_labels = one_hot_label(test_labels)

# 归一化
train_images = normalize_image(train_images)
train_labels = normalize_image(train_labels)
test_images = normalize_image(test_images)
test_labels = normalize_image(test_labels)

k1 = k1_init()
k2 = k2_init()
W = W_init()


# train_images_batches =  get_train_images_batches(train_images, batch_size=100)
# train_labels_batchs = get_train_labels_batches(train_labels, batch_size=100)


train_steps = 500

global_step = 0
lr_c_base = 0.01
lr_d_base = 0.01
lr_c = lr_c_base
lr_d = lr_d_base
decay_rate_c = 0.999
decay_rate_d = 0.999
decay_step_c = 100
decay_step_d = 100

while(1):
    for index in range(len(train_images)):
        global_step += 1
        I = train_images[index]
        C1, S1, C2, S2, f = convolve_and_pooling(I)
        y_pred = get_y_pred(f)
        delta_y_pred = get_delta_y_pred(y_pred, train_labels[index])
        delta_W = get_delta_W(delta_y_pred, f)
        delta_b = get_delta_b(delta_y_pred)
        delta_f = get_delta_f(W, delta_y_pred)
        delta_S2 = get_delta_S2(delta_f)
        delta_C2 = get_delta_C2(delta_S2)
        delta_C2_sigma = get_delta_C2_sigma(delta_C2, C2)
        S1_rot = get_S1_rot(S1)
        delta_k2 = get_delta_k2(S1_rot, delta_C2_sigma)
        delta_b2 = get_delta_b2(delta_C2_sigma)
        k2_rot = get_k2_rot(k2)
        delta_S1 = get_delta_S1(delta_C2_sigma, k2_rot)
        delta_C1 = get_delta_C1(delta_S1)
        delta_C1_sigma = get_delta_C1_sigma(delta_C1, C1)
        delta_k1 = get_delta_k1(I, delta_C1_sigma)  
        delta_b1 = get_delta_b1(delta_C1_sigma)

        
        # 更新参数
        k1 = k1 - lr_c * delta_k1
    
        b1 = b1 - lr_c * delta_b1
        k2 = k2 - lr_c * delta_k2
        b2 = b2 - lr_c * delta_b2
        W = W - lr_d * delta_W
        b = b - lr_d * delta_b

        
        if global_step % 10 == 0:
            print(np.argmax(y_pred),np.argmax(one_hot_train_labels[index]))
            print("Error:{0}".format(error(y_pred, one_hot_train_labels[index])))
        if global_step % 100 == 0:
            Accuracy = cal_accuracy(test_images, one_hot_test_labels, test_size=100)
            print("After {0} steps, Accuracy:{1}\n".format(global_step, Accuracy))
            with open("log.txt", "a+") as f:
                 f.write(str(Accuracy) + "\n")
        # if global_step % decay_step_c == 0:
            # lr_c = lr_c_base * decay_rate_c**(global_step / decay_step_c)
        # if global_step % decay_step_d == 0:   
            # lr_d = lr_d_base * decay_rate_d**(global_step / decay_step_d)
         
        