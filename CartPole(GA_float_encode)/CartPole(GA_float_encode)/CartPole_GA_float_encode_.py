import gym
import numpy as np
import copy
import time
import datetime

global env
env = gym.make('CartPole-v0')


def randf_between(a, b):
    delta = np.abs(a - b)
    return min(a, b) + np.random.ranf() * delta


def original_group(dna_length, n):
    group = np.array([[None for j in range(dna_length)] for i in range(n)])
    for index in range(n):
        group[index] = np.random.random(size=dna_length)
    return group

def multiply(father, mother, n):
    son = np.array([[None for j in range(father.shape[0])] for i in range(n)])
    # 克隆
    son[0] = father
    son[1] = mother
    # 交叉互换
    for index in range(2, n):
        if np.random.random() > 0.5:
            son[index] = father
        else:
            son[index] = mother
        point = np.random.randint(0, father.shape[0])
        son[index][point] = randf_between(father[point], mother[point])
    # 基因突变
    for index in range(n):
       if np.random.random() < 0.2:
           for gene_point in range(son[index].shape[0]):
               son[index][gene_point] += randf_between(-0.1, 0.1)
    return son

def adjust(indiviual, observation):
    return np.sum(indiviual * observation)


def evalue(individual):
    observation = env.reset()   # 初始化本场游戏的环境
    episode_reward = 0  # 初始化本场游戏的得分
    # 一场游戏分为一个个时间步
    while(1):
        #env.render()    # 更新并渲染游戏画面
        #time.sleep(0.02)
        observation, reward, done, info = env.step(int(adjust(individual, observation) >= 0))  # 获取本次行动的反馈结果
        episode_reward += reward
        if done:
            env.close()
            return episode_reward
    return episode_reward

# 适应性函数
def adaptability_func(son):
    adaptability = np.array([None for i in range(son.shape[0])])
    for index in range(son.shape[0]):
        adaptability[index] = evalue(son[index])
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
    #rand_n = np.random.random() * 100
    #for index in range(1, turntable.shape[0]):
        #if rand_n < turntable[index]:
            #new_mother = son[index - 1]
            #break
    new_father = son[np.argmax(probability)]
    new_mother = son[np.sum(np.argwhere(probability == np.sort(probability)[son.shape[0] - 2])[0])]
    return new_father, new_mother

def evolution(father, mother, n):
    son = multiply(father, mother, n)
    adaptability = adaptability_func(son)
    probability = compute_probability(adaptability)
    father, mother = select(son, probability)
    return father, mother

dna_length = 4
group = original_group(dna_length, 50)
adaptability = adaptability_func(group)
probability = compute_probability(adaptability)
father, mother = select(group, probability)
global_generation = 0


while(1):
    global_generation += 1
    father, mother = evolution(father, mother, 50)
    if global_generation % 1 == 0:
         value = evalue(father)
         print("generation {0} : {1}".format(global_generation, value))
         if value >= 200:
             observation = env.reset()
             while(1):
                 env.render()    # 更新并渲染游戏画面
                 time.sleep(0.01)
                 observation, reward, done, info = env.step(int(adjust(father, observation) > 0))  # 获取本次行动的反馈结果
                 if done:
                     break
             env.close()
             #break

