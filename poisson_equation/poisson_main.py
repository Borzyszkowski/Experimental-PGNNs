import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data_loader import DataLoader
from pgnn import PhysicsGuidedNeuralNetwork
from pgnn_config import *
from poisson_equation import PoissonEquation

tf.set_random_seed(SEED)


def save_results(ranges, dim, solution, nn_solution):
    """
    Save results as three plots: mesh of analytical solution, mesh of nn solution and differences between them.
    :param ranges: start and stop value for mesh
    :param dim: number of points in each dimension
    :param solution: function for analytical solution
    :param nn_solution: function for nn solution
    """
    # build data
    x = np.linspace(ranges[0], ranges[1], dim)
    y = np.linspace(ranges[0], ranges[1], dim)
    mesh = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    # resolve exact solution
    sol = solution(mesh)
    sol_vec = np.reshape(sol, (dim, dim))
    plt.imshow(np.rot90(sol_vec), cmap='hot', interpolation='nearest',
               extent=[0.0, 1.0, 0.0, 1.0], aspect='auto')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.colorbar()
    plt.title('Exact results')
    plt.savefig('analytical.png')
    plt.clf()

    # resolve estimated solution
    nn_sol = nn_solution(mesh.astype(np.float64)).eval()
    nn_sol_vec = np.reshape(nn_sol, (dim, dim))
    plt.imshow(np.rot90(nn_sol_vec), cmap='hot', interpolation='nearest',
               extent=[0.0, 1.0, 0.0, 1.0], aspect='auto')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.colorbar()
    plt.title('Estimated results')
    plt.savefig('nn_estimation.png')
    plt.clf()

    # resolve mse of above solutions
    err = get_mse(dim, sol, nn_sol)
    plt.imshow(np.rot90(err), cmap='hot', interpolation='nearest',
               extent=[0.0, 1.0, 0.0, 1.0], aspect='auto')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.colorbar()
    plt.title('MSE')
    plt.savefig('mse.png')
    plt.clf()


def get_mse(dim, sol, nn_sol):
    """
    Returns mse vector of given solutions (exact and estimated)
    :param dim: number of points for each dimension
    :param sol: vector of exact solution
    :param nn_sol: vector of estimated solution
    :return: err_vec: mse vector
    """
    err_vec = np.zeros(dim * dim)
    for i in range(dim * dim):
        err_vec[i] = np.sqrt((sol[i] - nn_sol[i]) ** 2)

    return np.reshape(err_vec, (dim, dim))


def main():
    # ----------------------- PREPARE DATA -----------------------
    # load data
    data_loader = DataLoader('datasets/' + str(BATCH_SIZE), BATCH_SIZE)
    data_loader.load_dataset()

    # build model
    pgnn = PhysicsGuidedNeuralNetwork()
    poisson_eq = PoissonEquation()

    # data -> inside data + boundary conditions
    var_in = tf.placeholder(tf.float64, [None, N_INPUTS])
    val_in = pgnn.calculate(var_in)
    sol_in = tf.placeholder(tf.float64, [None, 1])

    var_bc = tf.placeholder(tf.float64, [None, N_INPUTS])
    val_bc = pgnn.calculate(var_bc)
    sol_bc = tf.placeholder(tf.float64, [None, 1])
    # -------------------------------------------------------------

    # ----------------------- LOSS FUNCTION -----------------------
    """
        loss function -> loss for inside data + loss for boundary conditions
        
        loss for inside data -> based on Poisson equation:
            -grad^2 u(x1, x2) = f (x1, x2) -> grad^2 u(x1, x2) + f(x1, x2) = 0
        
        loss for boundary conditions -> based on Dirichlet boundary conditions:
            u(x1, x2) = g(x1, x2) -> u(x1, x2) - g(x1, x2) = 0
    """
    gradients = sum(pgnn.squared_gradient(var_in))
    loss_in = tf.square(gradients + sol_in)
    loss_bc = tf.square(val_bc - sol_bc)
    loss = tf.reduce_mean(loss_in + loss_bc)
    # -------------------------------------------------------------

    # ----------------------- MODEL TRAINING -----------------------
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init)

        # get train data
        in_x, in_y = data_loader.get_inside_samples(BATCH_SIZE)
        in_x = np.reshape(in_x, (BATCH_SIZE, 1))
        in_y = np.reshape(in_y, (BATCH_SIZE, 1))
        in_data = np.concatenate([in_x, in_y], axis=1)

        bc_x, bc_y = data_loader.get_boundary_samples(BATCH_SIZE)
        bc_x = np.reshape(bc_x, (BATCH_SIZE, 1))
        bc_y = np.reshape(bc_y, (BATCH_SIZE, 1))
        bc_data = np.concatenate([bc_x, bc_y], axis=1)

        # define f(x)
        f_x = poisson_eq.get_f_x(in_data)
        f_x = np.reshape(np.array(f_x), (BATCH_SIZE, 1))

        # define g(x)
        g_x = poisson_eq.get_g_x(bc_data)
        g_x = np.reshape(np.array(g_x), (BATCH_SIZE, 1))

        # train model
        train_scipy = tf.contrib.opt.ScipyOptimizerInterface(
            loss, method='BFGS', options={'gtol': 1e-14, 'disp': True, 'maxiter': MAX_ITER})

        train_scipy.minimize(
            sess, feed_dict={sol_in: f_x, sol_bc: g_x, var_in: in_data, var_bc: bc_data})

        # save model
        model_name = 'models/layers_' + str(N_LAYERS) + '_batch_' + str(BATCH_SIZE) + '_max_iter_' + str(MAX_ITER)
        saver.save(sess, model_name)

        # plot results
        save_results(poisson_eq.range, 101, poisson_eq.get_g_x, pgnn.calculate)
    # -------------------------------------------------------------


if __name__ == '__main__':
    main()
