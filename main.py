"""
Contains code to read from file, call algorithm, and save to file.
"""

import os
import argparse
import numpy as np
from glob import glob
join = os.path.join


def read_solomon(filename, data_dir='./data/instances/'):
    """
    Read Solomon (1987) VRPTW instance file(s) with specified filename.
    See README.TXT for more details.
    Filename may contain wildcards in order to load multiple files.

    Parameters:
    - filename: str (e.g. "C101.txt", "C1*.txt", "*.txt")
    - data_dir: str
    Returns:
    - instances: dict (containing information for each VRPTW instance)
    """
    instances = {}

    file_list = glob(join(data_dir, filename))
    for file in file_list:
        with open(file, 'r') as f:
            try:
                flines = f.read().splitlines()

                inst = {}
                inst['name'] = flines[0]
                inst['num_vehicles'], inst['capacity'] = flines[4].split()
                inst['num_vehicles'] = int(inst['num_vehicles'])
                inst['capacity'] = int(inst['capacity'])

                inst['customer_ids'] = []
                inst['x'], inst['y'], inst['demand'] = [], [], []
                inst['ready_time'], inst['due_time'] = [], []
                inst['service_time'] = []

                for i in range(9, len(flines)):
                    customer_line = flines[i].split()
                    customer_line = [int(x) for x in customer_line]
                    # note: customer ID of 0 indicates the depot
                    inst['customer_ids'] += [customer_line[0]]
                    inst['x'] += [customer_line[1]]
                    inst['y'] += [customer_line[2]]
                    inst['demand'] += [customer_line[3]]
                    inst['ready_time'] += [customer_line[4]]
                    inst['due_time'] += [customer_line[5]]
                    inst['service_time'] += [customer_line[6]]

                instances[inst['name']] = inst
            except:
                print("WARNING - file", file, "not successfully read. Skipping...")

    return instances

    
def read_ichoua(data_dir='./data/instances/'):
    """
    Read Ichoua et al. (2003) matrix files containing time-dependent
        travel times. See README.TXT for more details.
    Parameters:
    - data_dir: str
    Returns:
    - speed_mat: dict (containing speed matrices)
    - cat_mat: np.ndarray (category matrix)
    """

    # Load speed matrices
    speed_mat = {}
    for i, fname in enumerate(['t_dep.dat', 't_dep_2.dat', 't_dep_3.dat']):
        with open(join(data_dir, fname), "r") as f:
            flines = f.read().splitlines()
            mat_lines = [flines[j].split() for j in range(3, 6)]
            mat = np.array(mat_lines, dtype=np.float32)
            speed_mat[i + 1] = mat
    
    # Load category matrix
    with open(join(data_dir, "catheg.dat"), "r") as f:
        flines = f.read().splitlines()
        mat_lines = [flines[j].split() for j in range(1, 151)]
        cat_mat = np.array(mat_lines, dtype=np.int32)

    return speed_mat, cat_mat


def read_balseiro(data_dir='./data/instances/'):
    """
    Read Balseiro et al. (2010) matrix file containing (updated)
        category matrix. See README.TXT for more details.
    Parameters:
    - data_dir: str
    Returns:
    - cat_mat_2: np.ndarray (category matrix)
    """
    with open(join(data_dir, "catheg_2.dat"), "r") as f:
        flines = f.read().splitlines()
        mat_lines = [flines[j].split('\t') for j in range(1, 151)]
        cat_mat_2 = np.array(mat_lines, dtype=np.int32)
    return cat_mat_2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default='*.txt')
    args = parser.parse_args()

    instances = read_solomon(args.filename)
    speed_mat, cat_mat = read_ichoua()
    cat_mat_2 = read_balseiro()

    # TODO: run algorithm