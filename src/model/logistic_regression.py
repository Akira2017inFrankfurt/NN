# -*- coding: utf-8 -*-

import sys
import logging
from src.report.evaluator import Evaluator
from src.data.mnist_seven import MNISTSeven

import numpy as np
from sklearn.metrics import accuracy_score

# from util.activation_functions import Activation
from src.model.classifier import Classifier
from src.model.logistic_layer import LogisticLayer

from src.util.loss_functions import *

# 对于记录形式的一些设置
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


# 继承分类器的类
class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm 基于逻辑回归算法

    Parameters 参数
    ----------
    train : list 训练列表
    valid : list 验证列表
    test : list 测试列表
    learningRate : float 学习率
    epochs : positive int 训练世代数

    Attributes 类属性
    ----------
    trainingSet : list 训练集，类型为列表
    validationSet : list 验证集， 列表
    testSet : list 测试集，列表
    learningRate : float 学习率，浮点数
    epochs : positive int 训练世代数，正整数
    performances: array of floats 表现：元素为浮点数的数组
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50, loss='bce'):

        # 学习率
        self.learningRate = learningRate
        # 世代数
        self.epochs = epochs

        # 初始化训练集
        self.trainingSet = train
        # 初始化验证集
        self.validationSet = valid
        # 初始化测试集
        self.testSet = test

        # 损失函数
        # 个人认为，直接指定就好，没必要写这么多
        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages 记录表现
        # e.g. plotting, reporting..
        # 记录每一轮表现的列表，打印输出也好，或者是作图也好，可能用得到
        self.performances = []

        # Use a logistic layer as one-neuron classification (output) layer
        # 使用一个逻辑斯蒂层类，创建一层神经网络
        # 实例化：为后面的多层的打下基础！
        self.layer = LogisticLayer(train.input.shape[1], 1,
                                   activation='sigmoid',
                                   isClassifierLayer=True)

        # add bias values ("1"s) at the beginning of all data sets
        # 还是加偏置项进来
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                           axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                             axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)

    def train(self, verbose=True):
        """Train the Logistic Regression.  训练逻辑斯蒂回归

        Parameters 参数的没有
        ----------
        verbose : boolean 打印训练过程中的信息
            Print logging messages with validation accuracy if verbose is True.
        """

        # 迭代世代数训练
        # Run the training "epochs" times, print out the logs
        for epoch in range(self.epochs):
            # 其实就是打印信息咯，你都默认写出来了，后面也没改。
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))
            # 调用下面的函数，先训练一次
            self._train_one_epoch()

            if verbose:
                # accuracy_score这个是从sklearn.metrics中导入的
                accuracy = accuracy_score(self.validationSet.label,
                                          self.evaluate(self.validationSet))
                # 为后面的用法记录下表现
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                # 将准确率的值，添加到表现的列表中
                self.performances.append(accuracy)
                # 打印：在验证集上的表现率
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")

    # 只训练一个世代，迭代次数为1
    #
    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """

        for img, label in zip(self.trainingSet.input,
                              self.trainingSet.label):
            # Use LogisticLayer to do the job
            # Feed it with inputs

            # Do a forward pass to calculate the output and the error
            self.layer.forward(img)

            # Compute the derivatives w.r.t to the error
            # Please note the treatment of nextDerivatives and nextWeights
            # in case of an output layer
            # 这个函数需要好好看
            self.layer.computeDerivative(self.loss.calculateDerivative(
                label, self.layer.outp), 1.0)

            # Update weights in the online learning fashion
            # 额 这个更新有点厉害了
            self.layer.updateWeights(self.learningRate)

    def classify(self, test_instance):
        """Classify a single instance.  区分一个实例

        Parameters
        ----------
        test_instance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """

        # Here you have to implement classification method given an instance
        # 这里你需要在给定一个实例的情况下执行分类方法，但是好像不用我们管。。。 每次这个分类函数都有点神奇
        outp = self.layer.forward(test_instance)
        return outp > 0.5

    def evaluate(self, test=None):
        """Evaluate a whole dataset. 没变

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                             axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)


data = MNISTSeven("/Users/Haruki/PycharmProjects/NNPraktikum//data/mnist_seven.csv", 3000, 1000, 1000, oneHot=True)
myLRClassifier = LogisticRegression(data.trainingSet,
                                    data.validationSet,
                                    data.testSet,
                                    learningRate=0.005,
                                    epochs=30)

evaluator = Evaluator()
myLRClassifier.train()
lrPred = myLRClassifier.evaluate()
evaluator.printAccuracy(data.testSet, lrPred)
