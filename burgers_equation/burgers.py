import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from lib.pinn import PINN
from lib.optimizer import L_BFGS_B


import tensorflow as tf

def build_neural_network(num_inputs=2, layers=[32, 16, 32], activation='tanh', num_outputs=1):
    # input layer
    inputs = tf.keras.layers.Input(shape=(num_inputs,))
    # hidden layers
    x = inputs
    for layer in layers:
        x = tf.keras.layers.Dense(layer, activation=activation,
            kernel_initializer='he_normal')(x)
    # output layer
    outputs = tf.keras.layers.Dense(num_outputs,
        kernel_initializer='he_normal')(x)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
def generate_test_data_placeholders(num_test_samples):
    t_flat = np.linspace(0, 1, num_test_samples)
    x_flat = np.linspace(-1, 1, num_test_samples)
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    
    return t_flat, x_flat, t, x, tx

def display_results(t, x, u, x_flat):
    # plot u(t,x) distribution as a color-map
    data = scipy.io.loadmat('./data/burgers_shock.mat')
    real_data = np.real(data['usol']).T
    x_real = data['x'].flatten()[:,None]
    fig = plt.figure(figsize=(7,4))
    gs = GridSpec(2, 3)
    plt.subplot(gs[0, :])
    plt.pcolormesh(t, x, u, cmap='rainbow')
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(t,x)')
    cbar.mappable.set_clim(-1, 1)
    # plot u(t=const, x) cross-sections
    t_cross_sections = [0.25, 0.5, 0.75]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
        u = neural_network.predict(tx, batch_size=num_test_samples)
        plt.plot(x_flat, u, label = 'prediction')
        plt.plot(x_real, real_data[int(t_cs*100), :], label='real data')
        plt.title('t={}'.format(t_cs))
        plt.xlabel('x')
        plt.ylabel('u(t,x)')
        plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    # Generate training and test data
    num_train_samples = 10000
    num_test_samples = 1000
    # kinematic viscosity
    nu = 0.01 / np.pi
    
    # create training input based on init and boundary conditions
    tx_eqn = np.random.rand(num_train_samples, 2)          # t_eqn =  0 ~ +1
    tx_eqn[..., 1] = 2 * tx_eqn[..., 1] - 1                # x_eqn = -1 ~ +1
    tx_ini = 2 * np.random.rand(num_train_samples, 2) - 1  # x_ini = -1 ~ +1
    tx_ini[..., 0] = 0                                     # t_ini =  0
    tx_bnd = np.random.rand(num_train_samples, 2)          # t_bnd =  0 ~ +1
    tx_bnd[..., 1] = 2 * np.round(tx_bnd[..., 1]) - 1      # x_bnd = -1 or +1
    # create training output
    u_eqn = np.zeros((num_train_samples, 1))               # u_eqn = 0
    u_ini = np.sin(-np.pi * tx_ini[..., 1, np.newaxis])    # u_ini = -sin(pi*x_ini)
    u_bnd = np.zeros((num_train_samples, 1))               # u_bnd = 0

    ### PINN - model with physical loss ###
    neural_network = build_neural_network()
    neural_network.summary()
    # build a PINN model
    pinn = PINN(neural_network, nu).build()

    # train the model using L-BFGS-B algorithm
    x_train = [tx_eqn, tx_ini, tx_bnd]
    y_train = [u_eqn, u_ini,  u_bnd]
    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
    lbfgs.fit()

    # predict u(t,x) distribution
    t_flat, x_flat, t, x, tx = generate_test_data_placeholders(num_test_samples)
    u = neural_network.predict(tx, batch_size=num_test_samples)
    u = u.reshape(t.shape)
    display_results(t, x, u, x_flat)    
    
    
    ### Model without physical loss ###
    x_train = [tx_ini, tx_bnd]
    y_train = [u_ini,  u_bnd]
    neural_network = build_neural_network()
    # initial condition input: (t=0, x)
    tx_ini = tf.keras.layers.Input(shape=(2,))
    # boundary condition input: (t, x=-1) or (t, x=+1)
    tx_bnd = tf.keras.layers.Input(shape=(2,))
    # initial condition output
    u_ini = neural_network(tx_ini)
    # boundary condition output
    u_bnd = neural_network(tx_bnd)
    # build the PINN model for Burgers' equation
    model = tf.keras.models.Model(inputs=[tx_ini, tx_bnd], outputs=[u_ini, u_bnd])
    lbfgs = L_BFGS_B(model=model, x_train=x_train, y_train=y_train)
    lbfgs.fit()
    
    # predict u(t,x) distribution
    t_flat, x_flat, t, x, tx = generate_test_data_placeholders(num_test_samples)
    u = neural_network.predict(tx, batch_size=num_test_samples)
    u = u.reshape(t.shape)
    display_results(t, x, u, x_flat)   
    