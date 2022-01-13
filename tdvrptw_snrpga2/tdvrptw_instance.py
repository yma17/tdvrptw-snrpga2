"""
Contains class definition of a TDVRPTW instance.
"""

import os
import time
import numpy as np
join = os.path.join

from .compute_inputs import *
from .genetic_op import eval_fitness
from .genetic_alg import snrpga2


def reformat(sol_res):
    """
    Reformat algorithm output into sequential, more human-readable format.
    Parameters:
    - sol_res: tuple (output of tdvrptw_snrpga2.genetic_alg.snrpga2())
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


class TDVRPTWInstance(object):
    """Instance of a TDVRPTW problem."""

    def __init__(self, inst_name='Instance 1'):
        """Initialize algorithm inputs and hyperparameters."""

        #
        # Hyperparameters
        #
        self.params = {}

        # Tournament selection probability
        self.params["ts_prob"] = 0.9
        # Crossover probability
        self.params["x_prob"] = 0.7
        # Mutation probability
        self.params["m_prob"] = 0.3
        # Weight of time in objective function
        self.params["w_t"] = 1.0
        # Maximum number of generations (e.g. number of alg iterations)
        self.params["mng"] = 1000
        # Number of individuals in initial population
        self.params["pop_size"] = 100
        # Method of initialization of first generation
        self.params["init_method"] = 'random_sample'
        #self.params["init_method"] = 'random_seq' 
        # Objective function
        self.params["obj_func"] = 'dt'  # weighted sum of distance and time
        #self.params["obj_func"] = 'd'  # distance only
        #self.params["obj_func"] = 't'  # time only
        #self.params["obj_func"] = 'v'  # number of vehicles
        # Granularity of keys of time to index cache
        self.params["cache_gran"] = 1.0

        #
        # Algorithm Inputs
        #

        self.name = inst_name  # instance name

        self.num_vehicles = 0  # (maximum) number of vehicles (optional)
        self.C = None  # maximum truck capacity

        self.customer_ids = []  # list of customer id names
        self.x, self.y = [], []  # coordinates for each location
        self.demand = []  # demand for each location
        self.ready_time = []  # start time for each location's time window
        self.due_time = []  # end time for each location's time window
        self.service_time = []  # necessary service time for each location

        self.D_m = None  # time-dependent (3D) distance matrix
        self.T_m = None  # time-dependent (3D) time matrix
        self.t2i = None  # map time to axis 0 index of self.D_m and self.T_m
        self.i2t = None  # list of matrix time values

        # Speed and category matrices (optional)
        # Ichoua et al. 2003, Balseiro et al. 2010 only
        self.speed_mat = None
        self.cat_mat = None


    def get_param(self, param_name):
        """Getter function for a hyperparameter."""
        assert param_name in self.params.keys()
        return self.params[param_name]


    def get_params(self):
        """Getter function for all hyperparameters."""
        return self.params


    def set_solomon(self, inst_s):
        """Set parameters from Solomon (1987) instance."""

        self.name = inst_s["name"]
        self.num_vehicles = inst_s["num_vehicles"]
        self.C = inst_s["capacity"]
        self.customer_ids = inst_s["customer_ids"]
        self.x, self.y = inst_s["x"], inst_s["y"]
        self.demand = inst_s["demand"]
        self.ready_time = inst_s["ready_time"]
        self.due_time = inst_s["due_time"]
        self.service_time = inst_s["service_time"]


    def set_ichoua(self, speed_mat, cat_mat):
        """Set matrices from Ichoua et al. (2003) data."""
        self.speed_mat = speed_mat
        self.cat_mat = cat_mat


    def set_balseiro(self, cat_mat):
        """Set matrix from Balseiro et al. (2010)."""
        self.cat_mat = cat_mat

    
    def set_param(self, param_name, value):
        """Setter function for a hyperparameter."""

        assert param_name in self.params.keys()
        if param_name == "obj_func":
            assert value in ['v', 't', 'd', 'dt']
        elif param_name == "init_method":
            assert value in ['random_sample', 'random_seq']
        
        self.params[param_name] = value
    

    def set_matrices(self, D_m, T_m, time_list):
        """
        Set time-dependent distance and time matrices,
        along with mapping from time value to axis 0 index.
        Parameters:
        D_m, T_m: np.ndarray with 3 dimensions
        time_list: list of time values corresponding to each index
                    of axis 0 in D_m and T_m
        """
        assert D_m.ndim == 3 and T_m.ndim == 3
        assert D_m.shape == T_m.shape
        assert D_m.shape[0] == len(time_list)
        assert D_m.shape[0] == len(self.x)

        self.D_m = D_m
        self.T_m = T_m
        self.i2t = time_list
        self.t2i = {self.i2t[i]:i for i in range(len(self.i2t))}
    

    def set_location_info(self,
                          x,
                          y,
                          customer_ids,
                          demand,
                          ready_time,
                          due_time,
                          service_time):
        """
        Set coordinate information, time window, service time, demand,
        and ids for locations.
        """
        assert len(x) == len(y)
        assert len(y) == len(customer_ids)
        assert len(customer_ids) == len(demand)
        assert len(demand) == len(ready_time)
        assert len(ready_time) == len(due_time)
        assert len(due_time) == len(service_time)

        self.x, self.y = x, y
        self.customer_ids = customer_ids
        self.demand = np.array(demand)
        self.ready_time = np.array(ready_time)
        self.due_time = np.array(due_time)
        self.service_time = np.array(service_time)


    def run(self, scen=None):
        """
        Run data preprocessing and algorithm.
        Parameter: scen: int (scenario number; required if speed matr used)
        Returns: sol_instr (output of reformat())
        """

        # Compute matrices, time indices if not explicitly specified
        if self.t2i is None:
            w_s = compute_window_size(self.ready_time[0],
                                      self.due_time[0],
                                      len(self.ready_time))
            self.t2i = compute_t2i(self.ready_time[0], self.due_time[0], w_s)
            D_r = compute_dist_matrix(self.x, self.y)
            self.D_m = np.repeat(D_r[np.newaxis, :, :], len(self.t2i), axis=0)
            self.i2t = compute_i2t(self.t2i)
            T_r = compute_raw_time_matrix(D_r,
                                          scen,
                                          self.cat_mat,
                                          self.speed_mat,
                                          self.i2t)
            self.T_m = compute_time_matrix(T_r, self.ready_time, self.i2t)

        # Call algorithm
        begin_time = time.time()
        res, score = snrpga2(self.D_m, self.T_m, self.service_time,
                                self.demand, self.due_time, self.ready_time[0],
                                self.t2i, self.i2t, self.C, self.params)
        end_time = time.time()

        d, t = [eval_fitness(f,
                             g_start=self.ready_time[0],
                             routes=res[0],
                             departure_times=res[3],
                             dist_matrix=self.D_m,
                             times=self.i2t,
                             t2i=self.t2i,
                             cache={},
                             granularity=self.params["cache_gran"],
                             depot_arrivals=res[4],
                             w_t=self.params["w_t"])
                             for f in ('d', 't')]

        sol_instr = reformat(res)

        print("--- RESULTS FOR {} ---".format(self.name))
        print("Score: {}".format(score))
        print("Distance to travel: {}".format(d))
        print("Total travel time: {}".format(t))
        print("Runtime: {} seconds".format(end_time - begin_time))
        print("Number of trucks used: {}/{}".format(len(sol_instr),
                self.num_vehicles))
        print(res[0])
        print()

        return sol_instr
