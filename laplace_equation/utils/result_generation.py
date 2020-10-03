import numpy as np
import pandas as pd

from laplace_equation.utils import config
from laplace_equation.utils.laplace_solver import LaplaceSolver


def build_input_matrix(nominal_temp, boundary_cond):
    """
    Build 2D tensor which represents given Laplace problem.
    :param nominal_temp: str -> temperature inside matrix
    :param boundary_cond: str -> Dirichlet boundary conditions
    :return: 2D tensor
    """
    # convert str to List[float]
    boundary_cond = boundary_cond.replace('[', '')
    boundary_cond = boundary_cond.replace(']', '')
    boundary_cond = list(float(x) for x in boundary_cond.split(","))

    # check correctness of boundary conditions
    if len(boundary_cond) != 4 * config.LAPLACE_FIELD_SIZE - 4:
        raise Exception("Invalid number of boundary conditions!")

    # build matrix
    matrix = float(nominal_temp) * np.ones([config.LAPLACE_FIELD_SIZE, config.LAPLACE_FIELD_SIZE])
    matrix[0, :] = boundary_cond[:config.LAPLACE_FIELD_SIZE]
    matrix[config.LAPLACE_FIELD_SIZE - 1, :] = boundary_cond[-config.LAPLACE_FIELD_SIZE]

    for i in range(1, config.LAPLACE_FIELD_SIZE - 1):
        index = config.LAPLACE_FIELD_SIZE + i
        matrix[i, 0] = boundary_cond[index]
        matrix[i, config.LAPLACE_FIELD_SIZE - 1] = boundary_cond[index + 1]

    return matrix


def parse_input_file(input_file_name):
    """
    Read all cases from input data, perform finite difference method for each case and save results.
    :param input_file_name: input file with columns -> nominal_temperature, boundary_conditions
    """
    data = []
    output_data = []
    # read all cases
    input_df = pd.read_csv(input_file_name)
    for index, row in input_df.iterrows():
        try:
            # build 2D tensor
            nominal_temperature = row["nominal_temperature"]
            boundary_conditions = row["boundary_conditions"]
            matrix = build_input_matrix(nominal_temperature, boundary_conditions)

            # resolve Laplace equation
            solver = LaplaceSolver(config.LAPLACE_FIELD_SIZE, matrix)
            solver.calculate()

            data += [[nominal_temperature, boundary_conditions, solver.matrix]]
            output_data.append(solver.matrix)
            if index % 100 == 0:
                print(f"{index}/{len(input_df)} done")
        except Exception as e:
            print("Exception!! -> " + str(e))
            return -1

    output_data = np.asarray(output_data)
    np.save("output_data.npy", output_data)
    # save results
    output_df = pd.DataFrame(data, columns=["nominal_temperature", "boundary_conditions", "result"])
    output_df.to_csv("result.csv")
