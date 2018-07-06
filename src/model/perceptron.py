# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from src.util.activation_functions import Activation
from src.model.classifier import Classifier

# 记录样式的设置
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters 传入参数
    ----------
    train : list 训练集
    valid : list 验证集
    test : list 测试集
    learningRate : float 学习率
    epochs : positive int 迭代次数

    Attributes 类属性
    ----------
    learningRate : float 学习率
    epochs : int 迭代世代数
    trainingSet : list 训练集
    validationSet : list 验证集
    testSet : list 测试集
    weight : list 权重？在这只是一个列表？
    """
    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        # 初始化学习率
        self.learningRate = learningRate
        # 初始化迭代世代数
        self.epochs = epochs

        # 初始化训练集
        self.trainingSet = train
        # 初始化验证集
        self.validationSet = valid
        # 初始化测试集
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and 0.1
        # 利用随机函数 创建一个权重向量出来
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/10

        # 在权重向量中添加偏置值
        # add bias weights at the beginning with the same random initialize
        self.weight = np.insert(self.weight, 0, np.random.rand()/10)

    # 训练网络
    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.
        根据感知器学习算法来训练神经元
        Parameters
        ----------
        verbose : boolean，这是一个布尔值，如果为真的化就打印带有验证准确率的记录信息
            Print logging messages with validation accuracy if verbose is True.
        """

        # Try to use the abstract way of the framework
        # 使用框架的抽象方法
        from src.util.loss_functions import DifferentError
        # 损失等于DifferentError，实例化一个损失类的类对象
        loss = DifferentError()

        # 一个控制循环结束的标签
        learned = False
        # 迭代次数为0
        iteration = 0

        # 如果误差不是0，或者不满足要求的化，重复训练几次
        # Train for some epochs if the error is not 0
        while not learned:
            # 误差总和
            totalError = 0
            # 下面这个是这个语言的一种特殊形式，双重迭代，也就是两个一起，都在一轮循环中更新一次
            for input, label in zip(self.trainingSet.input,
                                    self.trainingSet.label):
                # todo：输出等于
                output = self.fire(input)
                # 如果输出不等于期望的标签的话：
                if output != label:
                    # 生成一个误差
                    error = loss.calculateError(label, output)
                    # 然后根据输入数据和所产生的误差来更新矩阵！
                    self.updateWeights(input, error)
                    # 更新总的误差值
                    totalError += error

            # 迭代记录器（数字）加一
            iteration += 1

            # 选择是否要打印出来迭代世代和误差率的信息出来
            if verbose:
                logging.info("Epoch: %i; Error: %i", iteration, -totalError)

            # 如果总的误差等于0 或者迭代次数大约训练预计的世代数率，改变循环标志，结束这个循环
            if totalError == 0 or iteration >= self.epochs:
                # stop criteria is reached
                learned = True

    # 分类/查询？
    def classify(self, testInstance):
        """Classify a single instance. 对单独的一个实例来进行分类。

        Parameters 传入参数
        ----------
        testInstance : list of floats 测试实例：一个浮点数的列表

        Returns 返回值的类型是一个布尔值
        为什么是数字7，和初始化注释里的这段文字有关：digit-7 recognizer
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        return self.fire(testInstance)

    # 评估
    def evaluate(self, test=None):
        """Evaluate a whole dataset. 评估一个完整的数据集合

        Parameters 参数
        ----------
        test : the data set to be classified 测试：用来分类的数据集合
        if no test data, the test set associated to the classifier will be used 如果没有测试数据，那么测试集合值得就是被用到的分类器

        Returns 返回值是一个列表
        -------
        List: 对于数据集的记录，分好了的类的决定
            List of classified decisions for the dataset's entries.
        """
        # 如果测试为空，有点不清楚这么写的意思
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test set.
        # 对于test集合里的每个参数，都运用classify函数，返回结果，这些结果作为元素，组成输出列表
        return list(map(self.classify, test))

    # 更新权重，在前面的train函数里面调用
    def updateWeights(self, input, error):
        # 权重 += 学习率 * 误差 * 输入
        self.weight += self.learningRate*error*input

    # 根据输入处理掉神经元的输出？
    # 根据点乘计算结果，那些大于0的，这个函数输出结果为1，小于的就为0
    # 有点类似于阈值的意思
    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        return Activation.sign(np.dot(np.array(input), self.weight))
        
        
