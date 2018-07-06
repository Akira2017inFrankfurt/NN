import time

import numpy as np

from src.util.activation_functions import Activation


class LogisticLayer():
    """
    A layer of neural 神经网络的一层

    Parameters 参数
    ----------
    nIn: int: number of units from the previous layer (or input data) 前面一层神经元的数量/输入
    nOut: int: number of units of the current layer (or output) 本层神经元的数量/输出
    activation: string: activation function of every units in the layer 激活函数：string类型，每层用到的
    isClassifierLayer: bool:  to do classification or regression 是分类的层吗，是做分类还是回归？：布尔类型

    Attributes 类属性
    ----------
    nIn : positive int:
        number of units from the previous layer 前层点数
    nOut : positive int:
        number of units of the current layer 本层点数
    weights : ndarray 权重：数组
        weight matrix 权值矩阵
    activation : functional 激活函数
        activation function
    activationString : string 激活函数的名字
        the name of the activation function
    isClassifierLayer: bool 是分类还是回归
        to do classification or regression
    deltas : ndarray 偏导数，多维数组形式
        partial derivatives
    size : positive int 形状：本层神经网络单数数，和前面的nOut应该是一样的
        number of units in the current layer
    shape : tuple 形状：元组，是这一层的样子+上一层的形状，也是权重矩阵的形状，
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, nIn, nOut, weights=None,
                 activation='sigmoid', isClassifierLayer=False):

        # Get activation function from string
        # 从参数给的字符串，得到本层神经元要使用的激活函数
        # 首先是得到激活函数的名字
        self.activationString = activation
        # 然后根据名字，调用激活函数的查找函数返回该激活函数
        self.activation = Activation.getActivation(self.activationString)
        # 根据名字，返回该激活函数的导函数
        self.activationDerivative = Activation.getDerivative(self.activationString)

        # 上面一层的节点数量，也是本层的输入节点数
        self.nIn = nIn
        # 本层的神经元的数量，也是输出的节点数
        self.nOut = nOut

        # 创建一个形状为(nIn+1, 1)的多维数组，后面的1表示1列，用来存储输入的数据，方便后面的点乘计算
        self.inp = np.ndarray((nIn + 1, 1))
        # 设置这个数组的第一个值为1
        self.inp[0] = 1
        # 创建一个形状为(nOut, 1)的多维数组，用处暂时不清楚，但是后面的求导倒是用到来
        self.outp = np.ndarray((nOut, 1))
        # 创建一个形状为(nOut, 1)的多维数组，用来存放偏导值（应该是用来更新权重矩阵的？后面看吧）
        self.deltas = np.zeros((nOut, 1))

        # You can have better initialization here 其实可以改成高斯分布的哪种
        # 如果一开始我没有传入指定的权值
        # 现在觉得这个就是针对第一层的
        rns = np.random.RandomState(int(time.time()))  # 使用随机的方式创建一个
        if weights is None:
            # 输入层
            self.weights = rns.uniform(size=(nIn + 1, nOut)) - 0.5
        else:
            # assert (weights.shape == (nIn + 1, nOut))  # 断言，看看是不是形状符合，符合的话就直接利用传进来的权重对本层和上层之间的权重矩阵（向量更合适）进行初始化
            # 其他层
            self.weights = rns.uniform(size=(nIn, nOut)) - 0.5

        self.isClassifierLayer = isClassifierLayer  # 意义不大，表示是分类器

        # Some handy properties of the layers
        self.size = self.nOut  # 本层的神经元有多少个
        self.shape = self.weights.shape  # 本层和上层之间的权重矩阵的形状是怎样的

    # 前向算法
    def forward(self, inp):
        """
        Compute forward step over the input using its weights 通过权重和输入来一步一步向前推进神经网络的训练

        Parameters 参数
        ----------
        inp : ndarray 输入：类型是一个nIn + 1行 1列的数组，代表了本层获得的输入参数x
            a numpy array (nIn + 1,1) containing the input of the layer

        Change outp 改变输出
        -------
        outp: ndarray 输出也是一个多维数组，但是多维数组需要通过计算
            a numpy array (nOut,1) containing the output of the layer
        """

        # Here you have to implement the forward pass
        # 上一层的输入信号
        # inp的形状是(nIn + 1,1)
        self.inp = inp
        # weights的形状是(nIn + 1, nOut)，为计算我们需要处理
        # w的形状是(nOut, nIn+1)
        # 接着是让二者做点乘,后使用激活函数处理
        # (nOut, nIn+1) * (nIn + 1,1)
        outp = self.activation(np.dot(self.weights.T, self.inp))
        self.outp = outp

        return outp

    # 根据后面一层的导数和权值，来更新本层的偏导数数组
    def computeDerivative(self, next_derivatives, next_weights):
        """
        Compute the derivatives (backward) 计算导数 （后向）

        Parameters 参数
        ----------
        next_derivatives: ndarray 下一个导数，多维数组。这一层的下面一层的
            a numpy array containing the derivatives from next layer 是一个包含了来自下一层的偏导！对啊！
        next_weights : ndarray 下一个权重
            a numpy array containing the weights from next layer 包含了来自后面一层的权重

        Change deltas 要改变的东西是deltas 偏导数：类型是多维数组
        -------
        deltas: ndarray
            a numpy array containing the partial derivatives on this layer 包含了本层的偏导数
        """

        # Here the implementation of partial derivative calculation

        # In case of the output layer, next_weights is array of 1
        # and next_derivatives - the derivative of the error will be the errors
        # Please see the call of this method in LogisticRegression.
        # self.deltas = (self.outp *
        #              (1 - self.outp) *
        #               np.dot(next_derivatives, next_weights))

        # Or more general: output*(1-output) is the derivatives of sigmoid
        # (sigmoid_prime)
        # self.deltas = (Activation.sigmoid_prime(self.outp) *
        #                np.dot(next_derivatives, next_weights))

        # Or even more general: doesn't care which activation function is used
        # dado: derivative of activation function w.r.t the output
        # dado表示的是激活函数 权重？输出
        dado = self.activationDerivative(self.outp)
        # 更新本层的偏导函数值
        self.deltas = (dado * np.dot(next_derivatives, next_weights))

        # Or you can explicitly calculate the derivatives for two cases
        # Page 40 Back-propagation slides
        # if self.isClassifierLayer:
        #     self.deltas = (next_derivatives - self.outp) * self.outp * \
        #                   (1 - self.outp)
        # else:
        #     self.deltas = self.outp * (1 - self.outp) * \
        #                   np.dot(next_derivatives, next_weights)
        # Or you can have two computeDerivative methods, feel free to call
        # the other is computeOutputLayerDerivative or such.
        # 返回本层的偏导函数值
        return self.deltas

    # 更新权重，参数只有一个学习率，这是人家写好的这个层里面自带的权重更新函数
    def updateWeights(self, learningRate):
        """
        Update the weights of the layer
        """
        # 使用梯度下降原则来更新
        # weight updating as gradient descent principle
        # 对于本层
        for neuron in range(0, self.nOut):
            self.weights[:, neuron] -= (learningRate *
                                        self.deltas[neuron] *
                                        self.inp)

    # 其实这次的_fire函数就很明显：就是直接计算上一层输出和权重矩阵进行点乘之后的结果
    # 然后将输出结果使用激活函数进行处理，直接就得到了本层的输出
    def _fire(self, inp):
        return self.activation(np.dot(inp, self.weights))
