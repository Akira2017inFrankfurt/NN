import sys
import numpy as np
from src.data.mnist_seven import MNISTSeven
from src.util.loss_functions import *
from src.model.logistic_layer import LogisticLayer
from src.model.classifier import Classifier

from sklearn.metrics import accuracy_score
from src.util.activation_functions import Activation
from src.report.evaluator import Evaluator


class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='bce', learningRate=0.01, epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm
        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int
        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask
        self.outputActivation = outputActivation

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

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
            raise ValueError('There is no predefined loss function ' + 'named ' + str)

        self.performances = []
        self.layers = layers
        self.layers = []

        # Input layer
        inputActivation = "sigmoid"
        self.layers.append(LogisticLayer(train.input.shape[1], 128, None, inputActivation, False))
        # Output layer
        outputActivation = "sigmoid"
        self.layers.append(LogisticLayer(128, 10, 1, outputActivation, True))

        self.inputWeights = inputWeights

        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1, axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1, axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)

    def train(self, verbose=True):
        """Train the Logistic Regression.  训练逻辑斯蒂回归

        Parameters 没有传入的参数
        ----------
        verbose : boolean 打印训练过程中的信息
            Print logging messages with validation accuracy if verbose is True.
        """

        # Run the training "epochs" times, print out the logs
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}..".format(epoch + 1, self.epochs))
            self._train_one_epoch()

            if verbose:
                accuracy = accuracy_score(self.validationSet.label, self.evaluate(self.validationSet))
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%".format((accuracy) * 100))
                print("-----------------------------")

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """

        for img, label in zip(self.trainingSet.input,
                              self.trainingSet.label):

            # output of input layer shape is (128,)
            hidden_outputs = self.layers[0].forward(img)
            hidden_array = np.array(hidden_outputs, ndmin=2).T
            # output of layer_output shape is (10,)
            final_outputs = self.layers[1].forward(hidden_outputs)
            final_array = np.array(final_outputs, ndmin=2).T

            # get targets
            targets_list = np.zeros(10) + 0.01
            targets_list[label] = 0.99
            targets = np.array(targets_list, ndmin=2).T

            # get the error
            # output_errors = targets - final_outputs
            output_errors_list = []
            for index in range(10):
                output_errors_list.append(targets[index] - final_outputs[index])
            # shape（10，1）
            output_errors = np.array(output_errors_list, ndmin=2)
            # shape（128，1)
            hidden_errors = np.dot(self.layers[1].weights, output_errors)

            # t1（1, 10）
            temp_1 = (output_errors * final_array * (1 - final_array)).T
            # t0（1, 128）
            temp_0 = (hidden_errors * hidden_array * (1.0 - hidden_array)).T

            # a (1, 128).T = (128,1)
            a = np.array(np.transpose(hidden_outputs), ndmin=2).T
            # b (1, 785).T = (785,1)
            b = np.array(np.transpose(img), ndmin=2).T

            # update the weights vector
            # w1 = [128, 10] = (128,1) * (1,10)
            self.layers[1].weights += self.learningRate * np.dot(a, temp_1)
            # w2 = [785, 128] = (785,1) * (1,128)
            self.layers[0].weights += self.learningRate * np.dot(b, temp_0)

    def classify(self, test_instance):
        outp = list(self.layers[1].forward(self.layers[0].forward(test_instance)))
        max_value = max(outp)
        index_class = outp.index(max_value)
        return index_class != test_instance[1]
        pass

    def evaluate(self, test=None):
        if test is None:
            test = self.testSet.input
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0, axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)


data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000, oneHot=True)
mlp = MultilayerPerceptron(data.trainingSet, data.validationSet, data.testSet, learningRate=0.005, epochs=50)
mlp.train()
mlpPred = mlp.evaluate()
evaluator = Evaluator()
evaluator.printAccuracy(data.testSet, mlpPred)
print(mlp.evaluate())
