import numpy as np


class PoissonEquation:
    def __init__(self):
        self.range = [0.0, 1.0]

    @staticmethod
    def get_g_x(x):
        """
        Returns Dirichlet boundary conditions vector for Poisson problem.
        g(x) = sin(pi*x1) * cos(pi*x2)
        :param x: input vector [x1 x2]
        :return: g(x)
        """
        g = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            g[i] = 1.0 * np.sin(np.pi * x[i, 0]) * np.cos(np.pi * x[i, 1])

        return g

    @staticmethod
    def get_f_x(x):
        """
        Returns analytical results of given Poisson problem.
        f(x) = 2 * pi^2 * sin(pi*x1) * cos(pi*x2)
        :param x: input vector [x1 x2]
        :return: f(x)
        """
        f = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            f[i] = 2.0 * np.pi * np.pi * np.sin(np.pi * x[i, 0]) * np.cos(np.pi * x[i, 1])

        return f
