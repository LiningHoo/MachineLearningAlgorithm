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

def tester(train_imgs, train_labels, test_imgs, test_labels):
    correct = 0.0
    for index in range(len(test_imgs)):
        print("正在测试第{0}张图片".format(index + 1))
        x = predict(test_imgs[index], train_images, train_labels, 5)
        #print("{0} {1}".format(x, int(round(255 * test_labels[index]))))
        if x == int(round(255 * test_labels[index])):
            correct += 1
    accuracy = correct / len(test_imgs) * 100
    return ('%.2f' % accuracy)

train_images = load_train_images()
train_labels = load_train_labels()
test_images = load_test_images()
test_labels = load_test_labels()

# 归一化
train_images = normalize_image(train_images)
train_labels = normalize_image(train_labels)
test_images = normalize_image(test_images)
test_labels = normalize_image(test_labels)

def predict(img, k_imgs, k_labels, k):
    errors = np.array([np.inf for i in range(k)])
    mins = np.array([None for i in range(k)])
    for index in range(len(k_imgs)):
        L = np.sum(np.abs(img - k_imgs[index]))
        for index_k in range(k):
            if L <= errors[index_k]:
                errors[index_k] = L
                mins[index_k] = int(round(k_labels[index] * 255))
                break
    mins = mins.astype(np.int64)
    return np.argmax(np.bincount(mins))

print("Accuracy:{0}%".format(tester(train_images, train_labels, test_images[:100], test_labels[:100])))

#for index in range(len(train_images)):
    #print("{0} {1}".format(predict(train_images[index], train_images, train_labels), int(round(255 * train_labels[index]))))
    