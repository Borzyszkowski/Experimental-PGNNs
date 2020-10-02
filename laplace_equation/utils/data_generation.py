import random

import numpy as np
import pandas as pd

import config
import result_generation


def generate_input_data(min_block_size, nb_samples=1000):
    """
    Generates input data for Laplace problem
    :param min_block_size: minimal block size in 2D space to be represented as 
    'warm' or 'cold' field
    :param nb_samples: number of generated samples per block size, total number of samples
    will be (config.LAPLACE_FIELD_SIZE - min_block_size) * nb_samples
    :return data in pandas DataFrame format
    """
    data = []
    record_size = 4 * config.LAPLACE_FIELD_SIZE - 4
    for block_size in range(min_block_size, config.LAPLACE_FIELD_SIZE):
        for sample in range(nb_samples):
            nb_blocks = record_size // block_size
            nominal_temperature = random.randint(0, config.MAX_TEMPERATURE)
            boundary_conditions = []
            for i in range(nb_blocks):
                block = [random.randint(0, config.MAX_TEMPERATURE)] * block_size
                boundary_conditions += block
            remainder = record_size % block_size
            if remainder:
                block = [random.randint(0, config.MAX_TEMPERATURE)] * remainder
                boundary_conditions += block
            data += [[nominal_temperature, boundary_conditions]]
    df = pd.DataFrame(data, columns=["nominal_temperature", "boundary_conditions"])
    return df


def convert_data_to_2d_array(data):
    converted = []
    for i, record in enumerate(data):
        record_2d = np.full((config.LAPLACE_FIELD_SIZE, config.LAPLACE_FIELD_SIZE), record[0])
        edges = [record[1][i:i + len(record[1])//4] for i in range(0, len(record[1]), len(record[1])//4)]
        record_2d[0, :-1] = edges[0]
        record_2d[:-1, -1] = edges[1]
        record_2d[-1, 1:] = edges[2][::-1]
        record_2d[1:, 0] = edges[3][::-1]
        converted += [record_2d]
    return converted
        
        
if __name__ == "__main__":
    file_name = "./data/input_data.csv"

    # create random cases
    data = generate_input_data(min_block_size=config.LAPLACE_MIN_BLOCK_SIZE, nb_samples=1000)
    data.to_csv(file_name)

    # resolve Laplace equation for each case
    result_generation.parse_input_file(file_name)
