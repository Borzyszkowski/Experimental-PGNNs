import numpy as np


class DataLoader:
    def __init__(self, file_name, total_samples):
        self.file_name = file_name
        self.total_samples = total_samples
        self.dataset = []

        self.last_in = 0
        self.last_bc = 0

    def load_dataset(self):
        self.dataset = np.genfromtxt(self.file_name, delimiter=',')

    def update_last(self, last, batch_size):
        last += batch_size
        if last == self.total_samples:
            return 0
        return last

    def get_inside_samples(self, batch_size):
        in_x = self.dataset[self.last_in:(self.last_in + batch_size), 0]
        in_y = self.dataset[self.last_in:(self.last_in + batch_size), 1]
        self.last_in = self.update_last(self.last_in, batch_size)

        return in_x, in_y

    def get_boundary_samples(self, batch_size):
        bc_x = self.dataset[self.last_bc:(self.last_bc + batch_size), 2]
        bc_y = self.dataset[self.last_bc:(self.last_bc + batch_size), 3]
        self.last_bc = self.update_last(self.last_bc, batch_size)

        return bc_x, bc_y
