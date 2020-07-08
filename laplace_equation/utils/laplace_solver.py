import numpy as np


class LaplaceSolver:
    """
    Iterative solution of Laplace equation with finite difference method,
    for given squared matrix with Dirichlet's boundary conditions
    """

    def __init__(self, matrix_dim, input_matrix: np.array):
        self.iterations = 350
        self.matrix_dim = matrix_dim
        self.matrix = input_matrix

    def calculate(self):
        for iteration in range(self.iterations):
            for row in range(1, self.matrix_dim - 1):
                for col in range(1, self.matrix_dim - 1):
                    self.matrix[row][col] = 0.25 * (self.matrix[row - 1][col] + self.matrix[row + 1][col] +
                                                    self.matrix[row][col - 1] + self.matrix[row][col + 1])
