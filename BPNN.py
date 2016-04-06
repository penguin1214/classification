# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import math
import random
import string

random.seed(0)
# 生成区间[a, b)内的随机数
def rand(a, b):
    return (b-a)*random.random() + a

# 生成大小 I*J 的矩阵，默认零矩阵 (当然，亦可用 NumPy 提速)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# 函数 sigmoid，这里采用 tanh，因为看起来要比标准的 1/(1+e^-x) 漂亮些
def sigmoid(x):
    return math.tanh(x)

# 函数 sigmoid 的派生函数, 为了得到输出 (即：y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    ''' 三层反向传播神经网络 '''
    def __init__(self, ni, nh, no):
        # 输入层、隐藏层、输出层的节点（数）
        self.ni = ni + 1 # 增加一个偏差节点
        self.nh = nh
        self.no = no

        # 激活神经网络的所有节点（向量）
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # 建立权重（矩阵）
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # 设为随机值
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # 最后建立动量因子（矩阵）
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('与输入层节点数不符！')

        # 激活输入层
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # 激活隐藏层
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # 激活输出层
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets, N, M):
        ''' 反向传播 '''
        # if len(targets) != self.no:
        #     raise ValueError('与输出层节点数不符！')

        # 计算输出层的误差
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # 计算隐藏层的误差
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # 更新输出层权重
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print(N*change, M*self.co[j][k])

        # 更新输入层权重
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # 计算误差
        error = 0.0
        # for k in range(len(targets)):
        #     error = error + 0.5*(targets[k]-self.ao[k])**2
        error += 0.5*(targets-self.ao[k])**2
        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('输入层权重:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('输出层权重:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=15, M=2):
        # N: 学习速率(learning rate)
        # M: 动量因子(momentum factor)
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0:4]
                targets = p[4]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            # if i % 100 == 0:
            print('误差 %-.5f' % error)


def demo():
    # 一个演示：教神经网络学习逻辑异或（XOR）------------可以换成你自己的数据试试
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    # 创建一个神经网络：输入层有两个节点、隐藏层有两个节点、输出层有一个节点
    n = NN(2, 2, 1)
    # 用一些模式训练它
    n.train(pat)
    # 测试训练的成果（不要吃惊哦）
    n.test(pat)
    # 看看训练好的权重（当然可以考虑把训练好的权重持久化）
    #n.weights()

import numpy as np
import pandas as pd

# features 0-3
# labels 4
def iris():
    # read dataset
    raw = pd.read_csv('iris.csv')
    raw_data = raw.values
    data = raw_data[0:,0:4]
    label = []
    label_int = 0
    for i in range(len(raw_data)):
        if raw_data[i][4] == 'Iris-setosa':
            label_int = 1
            label.append(label_int)
        elif raw_data[i][4] == 'Iris-versicolor':
            label_int = 2
            label.append(label_int)
        else:
            label_int = 3
            label.append(label_int)
    l = np.asarray(label).T
    print np.shape(l)
    print np.shape(data)
    training_set = np.c_[data, l]
    nn = NN(4,3,3)
    nn.train(training_set,iterations=1000)

if __name__ == '__main__':
    iris()

