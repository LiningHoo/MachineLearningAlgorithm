import gym
import numpy as np
import copy
import time
import datetime

global env
env = gym.make('CartPole-v0')

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
        #if np.random.random() < 0.8:
            for epochs in range(1):
                point = np.random.randint(son[index].shape[0])
                son[index][point] = int(not son[index][point])
    return son


env.reset()

def evalue(father):
    env_copy = copy.deepcopy(env)
    #observation = env.reset()   # 初始化本场游戏的环境
    episode_reward = 0  # 初始化本场游戏的得分
    # 一场游戏分为一个个时间步
    for t in father:
        #env.render()    # 更新并渲染游戏画面
        # action = father[index] # 随机决定小车运动的方向
        observation, reward, done, info = env_copy.step(t)  # 获取本次行动的反馈结果
        episode_reward += reward
        if done:
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
    new_father = son[np.argmax(probability)]
    return new_father

def evolution(father, n):
    son = multiply(father, n)
    adaptability = adaptability_func(son)
    probability = compute_probability(adaptability)
    father = select(son, probability)
    return father


dna_length = 200
group = original_group(dna_length, 500)
adaptability = adaptability_func(group)
probability = compute_probability(adaptability)
father = select(group, probability)
global_generation = 0

while(1):
    global_generation += 1
    father = evolution(father, 500)
    if global_generation % 1 == 0:
         print("generation {0} : {1}".format(global_generation, evalue(father)))
         if evalue(father) == 200:
             env_copy = copy.deepcopy(env)
             for t in father:
                 env_copy.render()    # 更新并渲染游戏画面
                 time.sleep(0.1)
                 observation, reward, done, info = env_copy.step(t)  # 获取本次行动的反馈结果
                 if done:
                     break
             env_copy.close()
             break

env.close()


"""
max_number_of_steps = 200   # 每一场游戏的最高得分
#---------获胜的条件是最近100场平均得分高于195-------------
goal_average_steps = 195
num_consecutive_iterations = 100
#----------------------------------------------------------
num_episodes = 500 # 共进行5000场游戏
last_time_steps = np.zeros(num_consecutive_iterations)  # 只存储最近100场的得分（可以理解为是一个容量为100的栈）

# 重复进行一场场的游戏
for episode in range(num_episodes):
    observation = env.reset()   # 初始化本场游戏的环境
    episode_reward = 0  # 初始化本场游戏的得分
    # 一场游戏分为一个个时间步
    for t in range(max_number_of_steps):
        for index in range(father.shape[0]):
            env.render()    # 更新并渲染游戏画面
            action = father[index]   # 随机决定小车运动的方向
            observation, reward, done, info = env.step(action)  # 获取本次行动的反馈结果
            episode_reward += reward
            if done:
                print('%d Episode finished after %f time steps / mean %f' % (episode, t + 1, last_time_steps.mean()))
                last_time_steps = np.hstack((last_time_steps[1:], [episode_reward]))    # 更新最近100场游戏的得分stack
                break
    # 如果最近100场平均得分高于195
    if (last_time_steps.mean() >= goal_average_steps):
        print('Episode %d train agent successfuly!' % episode)
        break

print('Failed!')
"""
