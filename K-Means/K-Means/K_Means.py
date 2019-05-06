import numpy as np
import matplotlib.pyplot as plt

def normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img

data = np.genfromtxt("kmeans.txt", delimiter=' ')
#data = np.genfromtxt("train.csv", delimiter=',')
#data = normalize_image(data[:, 1:])
plt.scatter(data[:, 0],data[:, 1])

k = 4

def barycenter_init(k, data):
    width = data.shape[1]
    barycenter = np.array([[None for j in range(data.shape[1])] for i in range(k)])
    for index in range(k):
        barycenter[index] = np.random.random([1, width])[0]
    return barycenter


def cal_SSE(barycenter, cluster):
    SSE = 0.0
    for i in range(len(cluster)):
        for j in range(len(cluster[i])):
            SSE += np.sum(np.square(barycenter[i] - cluster[i][j]))
    return SSE

def barycenter_update(barycenter, data, k):
    cluster = [[] for i in range(k)] 
    errors = np.array([None for i in range(k)])
    for data_index in range(len(data)):
        for k_index in range(k):
            errors[k_index] = np.sum(np.abs(data[data_index] - barycenter[k_index]))
        cluster[np.argmin(errors)].append(data[data_index])
    for k_index in range(k):
        for j in range(data.shape[1]):
            if(len([array[j] for array in cluster[k_index]])):
                barycenter[k_index][j] = np.mean([array[j] for array in cluster[k_index]])
    
    return barycenter, cluster


barycenter = barycenter_init(k, data)

global_step = 0
SSE = np.inf
while(1):
    global_step += 1
    barycenter, cluster = barycenter_update(barycenter, data, k)
    if SSE == cal_SSE(barycenter, cluster):
        print("SSE: %.3f" % (SSE))
        print("After {0} steps, find the barycenters.".format(global_step))
        break
    else:
        SSE = cal_SSE(barycenter, cluster)
    print("SSE: %.3f" % (SSE))

plt.scatter(barycenter[:,0],barycenter[:,1],marker = "+")
plt.show()
