"""
Contains code to read from file, call algorithm, and save to file.
"""

import os
import time
import argparse
import numpy as np
from glob import glob
join = os.path.join

from tdvrptw_snrpga2.src.compute_inputs import *
from tdvrptw_snrpga2.src.genetic_op import eval_fitness
from tdvrptw_snrpga2.src.genetic_alg import snrpga2

CURR_PATH = os.path.dirname(__file__)


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

    file_list = glob(join(CURR_PATH, data_dir, filename))
    for file in file_list:
        with open(file, 'r') as f:
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
                    # note: customer ID of 0 indicates the depot
                    inst['customer_ids'] += [int(customer_line[0])]
                    inst['x'] += [customer_line[1]]
                    inst['y'] += [customer_line[2]]
                    inst['demand'] += [customer_line[3]]
                    inst['ready_time'] += [customer_line[4]]
                    inst['due_time'] += [customer_line[5]]
                    inst['service_time'] += [customer_line[6]]

                inst['demand'] = np.array(inst['demand'])
                inst['ready_time'] = np.array(inst['ready_time'])
                inst['due_time'] = np.array(inst['due_time'])

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
    - speed_mats: np.ndarray (concatenation of speed matrices)
    - cat_mat: np.ndarray (category matrix)
    """

    # Load speed matrices
    speed_mats = []
    for fname in ['t_dep.dat', 't_dep_2.dat', 't_dep_3.dat']:
        with open(join(CURR_PATH, data_dir, fname), "r") as f:
            flines = f.read().splitlines()
            mat_lines = [flines[j].split() for j in range(3, 6)]
            mat = np.array(mat_lines, dtype=np.float32)
            speed_mats += [mat]
    speed_mats = np.array(speed_mats)
    
    # Load category matrix
    with open(join(CURR_PATH, data_dir, "catheg.dat"), "r") as f:
        flines = f.read().splitlines()
        mat_lines = [flines[j].split() for j in range(1, 151)]
        cat_mat = np.array(mat_lines, dtype=np.int32)

    return speed_mats, cat_mat


def read_balseiro(data_dir='./data/instances/'):
    """
    Read Balseiro et al. (2010) matrix file containing (updated)
        category matrix. See README.TXT for more details.
    Parameters:
    - data_dir: str
    Returns:
    - cat_mat_2: np.ndarray (category matrix)
    """
    with open(join(CURR_PATH, data_dir, "catheg_2.dat"), "r") as f:
        flines = f.read().splitlines()
        mat_lines = [flines[j].split('\t') for j in range(1, 151)]
        cat_mat_2 = np.array(mat_lines, dtype=np.int32)
    return cat_mat_2


def reformat(sol_res):
    """
    Reformat algorithm output into sequential, more human-readable format.
    Parameters:
    - sol_res: tuple (output of genetic_alg.snrpga2())
    Returns:
    - sol_instr: list of list of dict (sequential instructions for each truck)
    """

    sol_instr = []
    for route_idx in range(len(sol_res[0])):
        truck_instr = []

        # To each delivery location
        for loc_idx in range(len(sol_res[0][route_idx])):
            loc_dict = {}
            loc_dict['loc_idx'] = sol_res[0][route_idx][loc_idx]
            loc_dict['deliv_amount'] = sol_res[1][route_idx][loc_idx]
            loc_dict['arrival_t'] = sol_res[2][route_idx][loc_idx]
            loc_dict['depart_t'] = sol_res[3][route_idx][loc_idx]

            truck_instr += [loc_dict]

        # To depot
        loc_dict = {}
        loc_dict['loc_idx'] = 0
        loc_dict['deliv_amount'] = 'N/A'
        loc_dict['arrival_t'] = sol_res[4][route_idx]
        loc_dict['depart_t'] = 'N/A'

        truck_instr += [loc_dict]
        sol_instr += [truck_instr]

    return sol_instr


def run(inst, window_size=None, speed_m=None, cat_m=None, scen=None,
        obj_func='dt'):
    """
    Run data preprocessing and algorithm.

    Parameters:
    - inst: dict  (containing instance information)
    - window_size: int (amount of unit time for each window)
    - speed_m (optional): np.ndarray
    - cat_m (optional): np.ndarray
    - scen (optional): int  (scenario number)
    - obj_func: str  (specifying objective function)
        -v: num vehicles. d: distance. t: time. dt: distance and time.

    Returns:
    - TODO
    """

    assert obj_func in ['v', 't', 'd', 'dt']

    if not window_size:
        window_size = compute_window_size(inst['ready_time'][0],
                                          inst['due_time'][0],
                                          len(inst['ready_time']))

    t2i = compute_t2i(inst['ready_time'][0], inst['due_time'][0], window_size)
    i2t = compute_i2t(t2i)

    D_m = compute_dist_matrix(inst['x'], inst['y'])
    T_r = compute_raw_time_matrix(D_m, scen, cat_m, speed_m, i2t)
    T_m = compute_time_matrix(T_r, inst['ready_time'], i2t)

    # Call algorithm
    begin_time = time.time()
    res, score = snrpga2(D_m, T_m, inst['service_time'], inst['demand'],
                         inst['due_time'], t2i, window_size,
                         inst['ready_time'][0], inst['due_time'][0],
                         inst['capacity'], mng=1000, init_size=100,
                         obj_func=obj_func, init='random_sample', w_t=0.1)
    end_time = time.time()

    d = eval_fitness('d', routes=res[0], dist_matrix=D_m,
                        depot_arrivals=res[4], g_start=inst['ready_time'][0])
    t = eval_fitness('t', routes=res[0], dist_matrix=D_m,
                        depot_arrivals=res[4], g_start=inst['ready_time'][0])

    sol_instr = reformat(res)

    print("--- RESULTS FOR {} ---".format(inst['name']))
    print("Score: {}".format(score))
    print("Distance to travel: {}".format(d))
    print("Total travel time: {}".format(t))
    print("Runtime: {} seconds".format(end_time - begin_time))
    print("Number of trucks used: {}/{}".format(len(sol_instr),
            inst['num_vehicles']))
    print(res[0])
    print()
    

def main(filename='*.txt'):
    instances = read_solomon(filename)
    speed_mats, cat_mat = read_ichoua()
    cat_mat_2 = read_balseiro()

    print("Number of instances: {}".format(len(instances)))
    for k in instances.keys():
        run(instances[k], obj_func='dt')
    # TODO: for time-dependent, run one scenario or all scenarios?


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default='*.txt')
    args = parser.parse_args()

    main(args.filename)