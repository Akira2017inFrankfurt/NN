# -*- coding: utf-8 -*-

"""
Activation functions which can be used within neurons.
"""
import numpy as np
from numpy import exp
from numpy import divide
from numpy import ones
from numpy import asarray
import math


class Activation:
    """
    Containing various activation functions and their derivatives
    """

    @staticmethod
    def sign(netOutput, threshold=0):
        return netOutput >= threshold

    @staticmethod
    def sigmoid(netOutput):
        # use e^x from numpy to avoid overflow
        return 1 / (1 + np.exp(-1.0 * netOutput))

    @staticmethod
    def sigmoidPrime(netOutput):
        # Here you have to code the derivative of sigmoid function
        # s*(1-s)
        return Activation.sigmoid(netOutput) * (1.0 - Activation.sigmoid(netOutput))

    @staticmethod
    def tanh(netOutput):
        # return 2*Activation.sigmoid(2*netOutput)-1
        ex = np.exp(1.0 * netOutput)
        exn = np.exp(-1.0 * netOutput)
        return divide(ex - exn, ex + exn)  # element-wise division

    @staticmethod
    def tanhPrime(netOutput):
        return (1 - Activation.tanh(netOutput) ** 2)

    @staticmethod
    def rectified(netOutput):
        return asarray([max(0.0, i) for i in netOutput])

    @staticmethod
    def rectifiedPrime(netOutput):
        # reluPrime=1 if netOutput > 0 otherwise 0
        # print(type(netOutput))
        return netOutput > 0

    @staticmethod
    def identity(netOutput):
        return netOutput

    @staticmethod
    def identityPrime(netOutput):
        # identityPrime = 1
        return ones(netOutput.size)

    @staticmethod
    def softmax(netOutput):
        # # 以数组中的每个值作为指数的自然指数的值
        # exps = np.exp(netOutput)
        # # 返回的是一个列表，是已经处理好的列表
        # return exps / np.sum(exps)
        # 上面的这种写法 容易因为指数过大出现nan的错误
        """Compute the softmax in a numerically stable way."""
        # 利用这个性质 softmax(x) = softmax(x+c)
        x = netOutput - np.max(netOutput)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    @staticmethod
    def softmaxPrime(netOutput):
        netOutput_list = netOutput.tolist()
        i = netOutput_list.index(np.max(netOutput))
        a_i = Activation.softmax(netOutput_list[i])
        for j in range(len(netOutput_list)):
            a_j = Activation.softmax(netOutput_list[j])
            if j == i:
                netOutput_list[j] = a_j*(1-a_j)
            else:
                netOutput_list[j] = -a_j*a_i
        return np.array(netOutput_listndmin=2).T

    @staticmethod
    def getActivation(str):
        """
        Returns the activation function corresponding to the given string
        """

        if str == 'sigmoid':
            return Activation.sigmoid
        elif str == 'softmax':
            return Activation.softmax
        elif str == 'tanh':
            return Activation.tanh
        elif str == 'relu':
            return Activation.rectified
        elif str == 'linear':
            return Activation.identity
        else:
            raise ValueError('Unknown activation function: ' + str)

    @staticmethod
    def getDerivative(str):
        """
        Returns the derivative function corresponding to a given string which
        specify the activation function
        """

        if str == 'sigmoid':
            return Activation.sigmoidPrime
        elif str == 'softmax':
            return Activation.softmaxPrime
        elif str == 'tanh':
            return Activation.tanhPrime
        elif str == 'relu':
            return Activation.rectifiedPrime
        elif str == 'linear':
            return Activation.identityPrime
        else:
            raise ValueError('Cannot get the derivative of'
                             ' the activation function: ' + str)
