"""
Contains code to load, preprocess, and run algorithm on benchmark data.

The benchmarks are:
- Solomon (1987)
- Ichoua et al. (2003)
- Balseiro et al. (2010)

See README.TXT under data/ for more details.
"""

import os
import argparse
import numpy as np
from glob import glob
join = os.path.join

from tdvrptw_snrpga2.tdvrptw_instance import TDVRPTWInstance

CURR_PATH = os.path.dirname(__file__)
DATA_DIR = "../data/instances/"


def read_solomon(filename):
    """
    Read Solomon (1987) VRPTW instance file with specified filename.
    Initializes object variables with instance information.

    Parameters:
    - filename: str (e.g. "C101.txt")
    Returns:
    - inst: dict (containing instance information)
    """

    with open(filename, 'r') as f:
        try:
            flines = f.read().splitlines()

            inst = {}
            inst['name'] = flines[0]

            inst['num_vehicles'], inst['capacity'] = flines[4].split()
            inst['num_vehicles'] = int(inst['num_vehicles'])
            inst['capacity'] = float(inst['capacity'])

            inst['customer_ids'] = []
            inst['x'], inst['y'], inst['demand'] = [], [], []
            inst['ready_time'], inst['due_time'] = [], []
            inst['service_time'] = []

            for i in range(9, len(flines)):
                customer_line = flines[i].split()
                customer_line = [float(x) for x in customer_line]
                # note: customer ID of '0' indicates the depot
                inst['customer_ids'] += [str(int(customer_line[0]))]
                inst['x'] += [customer_line[1]]
                inst['y'] += [customer_line[2]]
                inst['demand'] += [customer_line[3]]
                inst['ready_time'] += [customer_line[4]]
                inst['due_time'] += [customer_line[5]]
                inst['service_time'] += [customer_line[6]]

            inst['demand'] = np.array(inst['demand'])
            inst['ready_time'] = np.array(inst['ready_time'])
            inst['due_time'] = np.array(inst['due_time'])
            inst['service_time'] = np.array(inst['service_time'])
        
            return inst
        except:
            print("WARNING - file", filename, "not successfully read. Skipping...")
            return None


def read_ichoua():
    """
    Read Ichoua et al. (2003) matrix files containing time-dependent
    travel times.
    """

    # Load speed matrices
    speed_mats = []
    for fname in ['t_dep.dat', 't_dep_2.dat', 't_dep_3.dat']:
        with open(join(CURR_PATH, DATA_DIR, fname), "r") as f:
            flines = f.read().splitlines()
            mat_lines = [flines[j].split() for j in range(3, 6)]
            mat = np.array(mat_lines, dtype=np.float32)
            speed_mats += [mat]
    speed_mat = np.array(speed_mats)

    # Load category matrix
    with open(join(CURR_PATH, DATA_DIR, "catheg.dat"), "r") as f:
        flines = f.read().splitlines()
        mat_lines = [flines[j].split() for j in range(1, 151)]
        cat_mat = np.array(mat_lines, dtype=np.int32)

    return speed_mat, cat_mat


def read_balseiro():
    """
    Read Balseiro et al. (2010) matrix file containing (updated)
    category matrix.
    """
    with open(join(CURR_PATH, DATA_DIR, "catheg_2.dat"), "r") as f:
        flines = f.read().splitlines()
        mat_lines = [flines[j].split('\t') for j in range(1, 151)]
        cat_mat_2 = np.array(mat_lines, dtype=np.int32)
    return cat_mat_2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default='*.txt')
    # TODO: implement command-line support for setting hyperparameters
    # TODO: implement command-line support for loading pre-existing matrices
    # TODO: implement command-line support for specifying scenario
    args = parser.parse_args()

    speed_mat, cat_mat = read_ichoua()
    cat_mat_2 = read_balseiro()

    # Initialize and run all instances that match filename.
    fpaths = glob(join(CURR_PATH, DATA_DIR, args.filename))
    for fpath in fpaths:
        inst_s = read_solomon(fpath)
        if inst_s is None:
            continue

        inst = TDVRPTWInstance(fpath.split("/")[-1])
        inst.set_solomon(inst_s)
        # TODO: add support for enabling speed and category matrices
        #inst.set_ichoua(speed_mat, cat_mat)
        #inst.set_balseiro(cat_mat_2)

        sol_instr = inst.run()

        # TODO: do something with sol_instr