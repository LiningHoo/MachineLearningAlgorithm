import numpy as np


def original_group(dna_length, n):
    group = np.array([[None for j in range(dna_length)] for i in range(n)])
    for index in range(n):
        group[index] = np.random.randint(0, 2, size=dna_length)
    return group

def multiply(father, n):
    son = np.array([[None for j in range(father.shape[0])] for i in range(n)])
    for index in range(n):
        son[index] = father
    # 基因突变
    for index in range(n):
        for epochs in range(1):
            point = np.random.randint(son[index].shape[0])
            son[index][point] = int(not son[index][point])
    return son

# 适应性函数
def adaptability_func(son, target):
    adaptability = np.array([None for i in range(son.shape[0])])
    for index in range(son.shape[0]):
        adaptability[index] = son.shape[1] - np.abs(np.sum(son[index]) - target)
    return adaptability

# 存活可能性
def compute_probability(adaptability):
    probability = np.array([None for i in range(adaptability.shape[0])])
    sum = np.sum(adaptability)
    for index in range(probability.shape[0]):
        probability[index] = adaptability[index] / sum
    return probability

# 物竞天择，适者生存
def select(son, probability):
    #turntable = np.array([0.0 for i in range(probability.shape[0] + 1)])
    #for index in range(1, turntable.shape[0], 1):
        #turntable[index] = turntable[index - 1] + probability[index - 1]
    #turntable = turntable * 100
    #rand_n = np.random.random() * 100
    #for index in range(1, turntable.shape[0]):
        #if rand_n < turntable[index]:
            #new_father = son[index - 1]
            #break
    new_father = son[np.argmax(probability)]
    return new_father

def evolution(father, n):
    son = multiply(father, n)
    adaptability = adaptability_func(son, target)
    probability = compute_probability(adaptability)
    father = select(son, probability)
    return father



dna_length = 1000
target = int(input("Please input the target integer number(from 0 to {0}):".format(dna_length)))
group = original_group(dna_length, 200)
adaptability = adaptability_func(group, target)
probability = compute_probability(adaptability)
father = select(group, probability)
global_generation = 0

while(1):
    global_generation += 1
    father = evolution(father, 200)
    if global_generation % 1 == 0:
         print("generation {0} : {1}".format(global_generation, np.sum(father)))