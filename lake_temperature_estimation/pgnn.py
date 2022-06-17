import numpy as np
import scipy.io as spio
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, TerminateOnNaN
from keras.losses import mean_squared_error


def density(temp):
    """
    Calculating density of water due to given temperature (Eq. 3.11)

    :param temp: temperature prediction Y[d, t] at depth d and time t
    :return: corresponding density prediction
    """
    return 1000 * (1 - ((temp + 288.9414) * (temp - 3.9863) ** 2) / (508929.2 * (temp + 68.12963)))


def root_mean_squared_error(y_true, y_pred):
    """
    Calculating RMSE. Used as metrics at model compiling
    :param y_true: measured value
    :param y_pred: predicted value
    :return: RMSE
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def physical_inconsistency(density_diff, lam):
    """
    Return physics-based loss function (Eq. 3.14)

    :param density_diff: difference in density estimates on time-stamp t (Eq. 3.13)
    :param lam: hyper-parameter; relative impact of physics-based loss due to empirical loss
    :return: physics-based loss function
    """
    def loss(y_true, y_pred):
        return K.mean(K.relu(density_diff))
    return loss


def loss_function(density_diff, lam):
    """
    Return full loss function for PGNN. Combination of empirical error and physics inconsistency (Eq. 2.4)

    :param density_diff: difference in density estimates on time-stamp t (Eq. 3.13)
    :param lam: hyper-parameter; relative impact of physics-based loss due to empirical loss
    :return: loss function
    """
    def loss(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) + lam * K.mean(K.relu(density_diff))
    return loss


class PhysicsGuidedNN:
    def __init__(self, n_layers, n_nodes, n_epochs, batch_size, train_size, lam):
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.train_size = train_size
        self.lam = lam

        self.train_x, self.train_y, self.test_x, self.test_y = self.read_data()
        self.model = self.create_model()

    def read_data(self):
        """
        Read data and split into train and test set

        :return: train_x, train_y, test_x, test_y
        """
        data = spio.loadmat("data\\lake.mat", squeeze_me=True, variable_names=['Y', 'Xc_doy', 'Modeled_temp'])
        inputs = data['Xc_doy']
        outputs = data['Y']

        return inputs[:self.train_size, :], outputs[:self.train_size], inputs[self.train_size:, :], outputs[
                                                                                                    self.train_size:]

    def create_model(self):
        """
        Build model with n_layers and n_nodes

        :return: keras.models.Sequential()
        """
        model = Sequential()

        model.add(Dense(self.n_nodes, activation='relu', input_shape=(np.shape(self.train_x)[1],)))
        for _ in range(self.n_layers):
            model.add(Dense(self.n_nodes, activation='relu'))
        model.add(Dense(1, activation='linear'))

        return model

    @staticmethod
    def read_unlabeled_data():
        data = spio.loadmat("data\\lake_sampled.mat", squeeze_me=True, variable_names=['Xc_doy1', 'Xc_doy2'])
        return data['Xc_doy1'], data['Xc_doy2']

    def physics_regularization(self):
        """
        Reads unlabeled data for physics-based loss.

        :return: total loss function, physics-based loss function
        """
        unlabeled_x, unlabeled_x_1 = self.read_unlabeled_data()

        unlabeled_x = K.constant(unlabeled_x)   # input at depth i
        unlabeled_y = self.model(unlabeled_x)   # model output at depth i

        unlabeled_x_1 = K.constant(unlabeled_x_1)   # input at depth i + 1
        unlabeled_y_1 = self.model(unlabeled_x_1)   # model output at depth i + 1

        density_diff = density(unlabeled_y) - density(unlabeled_y_1)   # difference in density estimates (Eq. 3.13)
        lam = K.constant(self.lam)   # regularization hyper-parameter

        return loss_function(density_diff, lam), physical_inconsistency(density_diff, lam)

    def run(self):
        # get loss functions
        total_loss, physics_loss = self.physics_regularization()

        # compile model with total loss as loss function and physics-based loss and rmse as metrics
        self.model.compile(loss=total_loss,
                           optimizer='AdaDelta',
                           metrics=[physics_loss, root_mean_squared_error])

        # to avoid over-fitting
        early_stopping = EarlyStopping(monitor='val_loss_1',
                                       patience=500,
                                       verbose=1)

        # train and evaluate
        history = self.model.fit(self.train_x, self.train_y,
                                 batch_size=self.batch_size,
                                 epochs=self.n_epochs,
                                 verbose=1,
                                 validation_split=0.1,
                                 callbacks=[early_stopping, TerminateOnNaN()])

        score = self.model.evaluate(self.test_x, self.test_y, verbose=0)

        self.model.save("model.h5")
        spio.savemat("results.mat",
                     {'train_loss_1': history.history['loss_1'], 'val_loss_1': history.history['val_loss_1'],
                      'train_rmse': history.history['root_mean_squared_error'],
                      'val_rmse': history.history['val_root_mean_squared_error'], 'test_rmse': score[2]})

