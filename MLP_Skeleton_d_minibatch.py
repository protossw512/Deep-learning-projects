"""
Xinyao Wang
"""


from __future__ import division
from __future__ import print_function

import sys

import cPickle
import numpy as np


# np.random.seed(321)

# This is a class for a LinearTransform layer which takes an input

class LinearTransform(object):

    def __init__(self, W):
        # DEFINE __init function
        self.W = W

    def forward(self, x):
        # DEFINE forward function
        self.linear_forward = np.dot(x, self.W)


# This is a class for a ReLU layer max(x,0)
class ReLU(object):

    def forward(self, x):
        # DEFINE forward function
        self.x = x
        vmax = np.vectorize(max)
        self.relu_forward = vmax(self.x, 0.0)
        self.relu_forward = np.c_[np.ones(x.shape[0]), self.relu_forward]


    def backward(self):
        def _grad_calculator(x):
            if x > 0:
                return 1
            elif x == 0:
                return 0.5
            else:
                return 0
        vgrad = np.vectorize(_grad_calculator)
        return vgrad(self.x)


# DEFINE backward function
# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def forward(self, x, y):
        # DEFINE forward function
        self.sigmoid_forward = 1 / (1 + np.exp(-x))
        self.cross_entropy = - (y * np.log(self.sigmoid_forward) + (1 - y) * np.log(1 - self.sigmoid_forward))


# This is a class for the Multilayer perceptron
class MLP(object):
    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network
        self.input_dims = input_dims
        self.hidden_unites = hidden_units
        self.W1 = np.random.randn(self.input_dims + 1, hidden_units) * 0.1
        self.W2 = np.random.randn(self.hidden_unites + 1, 1) * 0.1
        self.W1Dt0 = 0
        self.W2Dt0 = 0

    def train(
        self,
        x_batch,
        y_batch,
        learning_rate,
        momentum,
        l2_penalty,
    ):
        def _gradients():
            # Initialize
            delta_W2 = np.zeros(self.hidden_unites + 1) # Add 1 dimension for bias term
            delta_W1 = np.zeros((self.input_dims + 1, self.hidden_unites)) # Add 1 dimension for bias term
            W2minus = np.delete(self.W2, 0, 0)
            # For each sample, calculate gradient matrix of W1 and W2, and sum them together
            # We calculated gradient for each linear trasnformation layer and activation function for each node
            for i in range(len(y_batch)):
                dEdL2 = (S2.sigmoid_forward[i] - y_batch[i])
                dL2dW2 = S1.relu_forward[i]
                dEdW2 = dEdL2 * dL2dW2
                delta_W2 += dEdW2  # delta_W2 is the summation of gradient matrix for W2
                dL2dS1 = W2minus
                dS1dL1 = ReLU_grad[i].reshape(self.hidden_unites, 1)
                # delta W1 is the summation of gradient matrix for W1
                # Here we reused the result from dEdL2
                delta_W1 += np.dot((dEdL2 * dL2dS1 * dS1dL1).reshape(self.hidden_unites, 1), x_batch[i].reshape(1, self.input_dims + 1)).T
            return delta_W1, delta_W2

        # Go through forward function to get out put of this set of mini-batch samples
        L1 = LinearTransform(self.W1)
        L1.forward(x_batch)
        S1 = ReLU()
        S1.forward(L1.linear_forward)
        ReLU_grad = S1.backward()
        L2 = LinearTransform(self.W2)
        L2.forward(S1.relu_forward)
        S2 = SigmoidCrossEntropy()
        S2.forward(L2.linear_forward, y_batch)


        delta_W1, delta_W2 = _gradients()
        # Update W1 and W2 by applying momentum and l2 regularization
        W1Dt1 = momentum * self.W1Dt0 - learning_rate * (delta_W1 + l2_penalty * self.W1)
        self.W1 += W1Dt1
        self.W1Dt0 = W1Dt1

        W2Dt1 = momentum * self.W2Dt0 - learning_rate * (delta_W2.reshape(self.hidden_unites + 1,1) + l2_penalty * self.W2)
        self.W2 += W2Dt1
        self.W2Dt0 = W2Dt1




    # INSERT CODE for training the network

    def evaluate(self, x, y):

        L1 = LinearTransform(self.W1)
        L1.forward(x)
        S1 = ReLU()
        S1.forward(L1.linear_forward)
        L2 = LinearTransform(self.W2)
        L2.forward(S1.relu_forward)
        S2 = SigmoidCrossEntropy()
        S2.forward(L2.linear_forward, y)
        self.evaluate_loss = np.sum(S2.cross_entropy)


    def epoch_evaluate(self, train_x, train_y, test_x, test_y):
        def _binary(x):
            return 1 if x >= 0.5 else 0
        def _accuracy(x, y):
            return 1 if x == y else 0

        # Create functions that apply _binary() and _accuracy() function to numpy array
        vbinary = np.vectorize(_binary)
        vaccuracy = np.vectorize(_accuracy)

        # Calculate training loss and training accuracy
        L1 = LinearTransform(self.W1)
        L1.forward(train_x)
        S1 = ReLU()
        S1.forward(L1.linear_forward)
        L2 = LinearTransform(self.W2)
        L2.forward(S1.relu_forward)
        S2 = SigmoidCrossEntropy()
        S2.forward(L2.linear_forward, train_y)
        self.train_loss = np.sum(S2.cross_entropy)
        train_predict = vbinary(S2.sigmoid_forward)
        self.train_acc = sum(vaccuracy(train_predict, train_y)) / len(train_x)

        # Calculate testing loss and testing accuracy
        L1bar = LinearTransform(self.W1)
        L1bar.forward(test_x)
        S1bar = ReLU()
        S1bar.forward(L1bar.linear_forward)
        L2bar = LinearTransform(self.W2)
        L2bar.forward(S1bar.relu_forward)
        S2bar = SigmoidCrossEntropy()
        S2bar.forward(L2bar.linear_forward, test_y)
        self.test_loss = np.sum(S2bar.cross_entropy)
        test_predict = vbinary(S2bar.sigmoid_forward)
        self.test_acc = sum(vaccuracy(test_predict, test_y)) / len(test_x)


    # INSERT CODE for testing the network
# ADD other operations and data entries in MLP if needed

if __name__ == '__main__':

    data = cPickle.load(open('cifar_2class_py2.p', 'rb'))
    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']

    num_examples, input_dims = train_x.shape
    # INSERT YOUR CODE HERE
    num_testings, _ = test_x.shape
    # YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    train_x_mean = np.array(np.mean(train_x, axis = 0))
    train_x_std = np.array(np.std(train_x, axis = 0))
    norm_x_train = (train_x - train_x_mean) / train_x_std
    norm_x_test = (test_x - train_x_mean) / train_x_std

    train_x = np.c_[np.ones(num_examples), norm_x_train]
    test_x = np.c_[np.ones(num_testings), norm_x_test]

    num_epochs = 50
    num_batches = 200
    hidden_units = 40
    batch_size = 64
    learning_rate = 0.0001
    momentum = 0.8
    l2_penalty = 0.4
    mlp = MLP(input_dims, hidden_units)

    train_loss_output = []
    train_acc_output = []
    test_loss_output = []
    test_acc_output = []
    for epoch in xrange(num_epochs):
        # Combine data and label together, then random shuffle the order
        combine = np.concatenate((train_y, train_x), axis = 1)
        np.random.shuffle(combine)
        train_y, train_x = np.hsplit(combine, [1])
        loopindex = 0
        total_loss = 0

        for b in xrange(num_batches):
            # Devide shuffled data into small batches, each loop takes one batch
            if loopindex + batch_size >= num_examples - 1:
                loopindex = 0
            x_batch = train_x[loopindex : loopindex + batch_size]
            y_batch = train_y[loopindex : loopindex + batch_size]
            loopindex += batch_size
            batch_index = np.random.randint(0,
                                            high = num_examples,
                                            size = batch_size)

            mlp.train(x_batch,
                      y_batch,
                      learning_rate,
                      momentum,
                      l2_penalty
                     )

            mlp.evaluate(x_batch, y_batch) # I talked to Dr. Li he said here we only need to calculate loss for batch.
            total_loss += mlp.evaluate_loss

            # MAKE SURE TO UPDATE total_loss
            print(
               '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
                   epoch + 1,
                   b + 1,
                   (total_loss / b) / batch_size,
               ),
               end='',
           )
            sys.stdout.flush()
        # INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
        # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
        mlp.epoch_evaluate(train_x, train_y, test_x, test_y)
        train_loss = mlp.train_loss
        train_accuracy = mlp.train_acc[0]
        test_loss = mlp.test_loss
        test_accuracy = mlp.test_acc[0]
        print()
        print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
            train_loss / num_examples,
            100. * train_accuracy,
        ))
        print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
            test_loss / num_testings,
            100. * test_accuracy,
        ))
        train_loss_output.append(train_loss)
        train_acc_output.append(train_accuracy)
        test_loss_output.append(test_loss)
        test_acc_output.append(test_accuracy)
    # Write test accuracy to txt file so that I can use it to build figures
    report = open('test_acc.txt', 'w')
    for item in test_acc_output:
        report.write('%s \n' % item)
    report.close()
