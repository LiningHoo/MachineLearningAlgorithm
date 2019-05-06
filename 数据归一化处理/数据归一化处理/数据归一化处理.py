import numpy as np
import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D

list = np.genfromtxt("data.csv",delimiter = ',')
x1 = list[:,0,np.newaxis]
x2 = list[:,1,np.newaxis]
y = list[:,2,np.newaxis]
print("原数据：\n")
print(list,"\n",end = '\n')

def normalizing_0_to_1(array):
    new_array = []
    max_num = max(array)
    min_num = min(array)
    try:
        for item in array:
            temp = (item - min_num) / (max_num - min_num)
            new_array.append(temp)
    except ZeroDivisionError:
        for item in array:
            temp = 0
            new_array.append(temp)
    return new_array

def normalizing_minus1_to_1(array):
    new_array = []
    max_num = max(array)
    min_num = min(array)
    try:
        for item in array:
            temp = ((item - min_num) / (max_num - min_num) - 0.5) * 2
            new_array.append(temp)
    except ZeroDivisionError:
        for item in array:
            temp = 0
            new_array.append(temp)
    return new_array

new_x1 = normalizing_0_to_1(x1)
new_x2 = normalizing_0_to_1(x2)
new_y = normalizing_0_to_1(y)
new_list = np.append(new_x1,new_x2,axis = 1)
new_list = np.append(new_list,new_y,axis = 1)
print("从零到一归一数据：\n")
print(new_list)

print("\n")

newest_x1 = np.array(normalizing_minus1_to_1(x1))
newest_x2 = np.array(normalizing_minus1_to_1(x2))
newest_y = np.array(normalizing_minus1_to_1(y))
newest_list = np.append(newest_x1,newest_x2,axis = 1)
newest_list = np.append(newest_list,newest_y,axis = 1)
print("从负一到一归一数据：\n")
print(newest_list)
