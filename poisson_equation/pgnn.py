import tensorflow as tf

from pgnn_config import *


class PhysicsGuidedNeuralNetwork:
    def __init__(self):
        self.name = 'pgnn_'

        self.n_layers = N_LAYERS
        self.n_inputs = N_INPUTS
        self.n_outputs = N_OUTPUTS
        self.n_units = N_UNITS

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.act_fn = tf.nn.sigmoid
        self.weights = {}
        self.biases = {}

        self.build_model()

    def build_model(self):
        for i in range(0, self.n_layers + 1):
            self.weights[self.name + str(i)] = tf.get_variable(
                name=self.name + 'weight_' + str(i),
                shape=[self.n_inputs if i == 0 else self.n_units[i - 1],
                       self.n_outputs if i == self.n_layers else self.n_units[i]],
                initializer=self.weight_initializer,
                dtype=tf.float64)

            self.biases[self.name + str(i)] = tf.get_variable(
                name=self.name + 'bias_' + str(i),
                shape=[self.n_outputs if i == self.n_layers else self.n_units[i]],
                initializer=self.weight_initializer,
                dtype=tf.float64)

    def calculate(self, given_input):
        """
        Calculates nn outputs from scratch.
        :param given_input: input vector
        :return: output vector
        """
        for i in range(0, self.n_layers):
            layer = tf.add(tf.matmul(given_input if i == 0 else layer,
                                     self.weights[self.name + str(i)]), self.biases[self.name + str(i)])
            layer = self.act_fn(layer)

        return tf.matmul(layer, self.weights[self.name + str(self.n_layers)]) + self.biases[
            self.name + str(self.n_layers)]

    def gradient(self, x):
        """
        Returns first derivative
        :param x: input
        :return: gradient
        """
        return tf.gradients(self.calculate(x), x)[0]

    def squared_gradient(self, x):
        """
        Returns second derivative
        :param x: input
        :return: squared gradient
        """
        grad = self.gradient(x)
        grad_grad = []
        for i in range(self.n_inputs):
            grad_grad.append(
                tf.slice(tf.gradients(tf.slice(grad, [0, i], [tf.shape(x)[0], 1]), x)[0], [0, i], [tf.shape(x)[0], 1]))

        return grad_grad
