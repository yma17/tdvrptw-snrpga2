# tdvrptw-snrpga2
Python implementation of the genetic algorithm SNRPGA2 for the Time-Dependent Vehicle Routing Problem with Time Windows (TDVRPTW).

## Installation

In any case, the functionality of the algorithm can be accessed by installing the Python package `tdvrptw-snrpga2` ([PyPi link](https://pypi.org/project/tdvrptw-snrpga2/)). To install, simply run `pip install tdvrptw-snrpga2`.

## Usage (for your data)

### 1. Creating an instance

A TDVRPTW "instance" can be created as follows. This instance will store all algorithm inputs and hyperparameters, and is used to call the algorithm itself.

```
import tdvrptw_snrpga2 as ts
name = "GIVE_YOUR_INSTANCE_A_NAME"
inst = ts.TDVRPTWInstance(name)
```

### 2. Setting location info

The information necessary for each location are: the 2D coordinates (x/y, long/lat, etc.), the id (unique identifier string), demand, ready time (beginning of time window), due time (closing of time window), and service time (amount of unit time it takes to deliver after arriving). They can be initialized using the `set_location_info()` function as follows:

```
# These should all be Python lists of the same length
# The first index of each list MUST represent the depot
# The subsequent indices represent the customers

x = [0, 1, 0]
y = [0, 0, 1]
customer_ids = ["Depot", "Customer A", "Customer B"]
demand = [0, 10.3, 8.8]
ready_time = [0, 0, 5]
due_time = [10, 5, 10]
service_time = [0, 1, 2]

inst.set_location_info(x, y, customer_ids, demand, ready_time, due_time, service_time)
```

### 3. Setting distance matrix, time matrix, and time list

Both the (time-dependent) travel distance and travel time matrices are 3-dimensional numpy arrays. Axis 0 represents the (discretized) departure time, axis 1 represents source locations, and axis 2 represents destination locations.

Axis 0 can be understood to represent "snapshots" of distance and travel time values within the depot hours - a predetermined, discretized representation of a continuous value. The algorithm will round all input and computed times to the **closest time value** in this list of predetermined travel times.

Continuing the example in the previous section, let's set our list of predetermined times to be the following (note that they span the depot and closing time):

```
time_list = [0, 2, 4, 6, 8, 10]
```

Given that this time list is of length 6, and that there are 3 locations in our example, the distance (`D_m`) and travel time (`T_m`) matrices that we then pass in must both be of shape (6, 3, 3). Interpreting the values in these matrices:
* `D_m[1, 0, 1]` = the distance it takes to travel from location "Depot" (`customer_ids[0]`) to location "Customer A" (`customer_ids[1]`) if the vehicle departs at time 2, 3, 1.5, 2.5, or any number closest to `time_list[1]`.
* `T_m[4, 2, 0]` = the time it takes to travel from location "Customer B" (`customer_ids[2]`) to location "Depot" (`customer_ids[0]`) if the vehicle departs at time 8, 9, 7.5, 8.5, or any number closest to `time_list[4]`.

Note that the values in the time_list do not have to be equally spaced, but should nevertheless be in ascending order.

We then use the `set_matrices()` function to initialize the matrices and time list as follows:

```
inst.set_matrices(D_m, T_m, time_list)
```

### 4. Setting vehicle capacity (required)

```
inst.C = DESIRED_CAPACITY_VALUE
```

### 4.5. Setting the maximum number of vehicles (optional)

```
inst.num_vehicles = DESIRED_NV_VALUE
```

(This has no effect on the algorithm and only exists for result output)

### 5. Setting algorithm hyperparameters

The algorithm utilizes several hyperparameters to dictate its runtime steps. In order to modify a hyperparameter value from the default value, you can use:

```
inst.set_param(param_name, new_value)
```

A full list of hyperparameters, including name, explanation, default value, and acceptable range of values, is provided in `docs/hyperparameters.md`.

Additionally, current values of hyperparameters can be retrieved through calling `inst.get_params()` or `inst.get_param(param_name)`.

### 6. Running the algorithm

Once all previous required steps have been run, the following line of code will run the algorithm:

```
sol_instr = inst.run()
```

Upon completion, result statistics will be printed to console, and a data structure containing the results in the format of instructions for each vehicle (`sol_instr`) will be returned. See next section for the type/format of this data structure

### 7. Interpreting the results

`sol_instr` is a Python list, consisting of `T` sub-lists, where `T` is the number of vehicles used in the result, and each sub-list containing instructions on how to follow the algorithm's routing solution for a single vehicle. Specifically, each of these sub-lists consist of `U(T) + 2` dictionary structures that contain demand, arrival, departure, and other information, where `U(T)` is the number of customers that truck `T` delivers to.

Here is an outline of the structure, as well as datatypes of variables within it:

```
[
    [
        {
            "total_customers": int,
            "total_deliv_amount": float
        },  # Overall truck statistics

        {
            "loc_id": str,
            "deliv_amount": float,
            "arrival_t": float,
            "depart_t": float
        },  # Instructions for delivery to first customer

        {...}, ..., {...},  # Instructions for delivery to each subsequent customer
        
        {
            "loc_id": str,
            "arrival_t": float
        }  # Instructions for travel back to depot
    ],  # Data for first vehicle used

    [
        ...  # Data for each subsequent vehicle used
    ],

    ...
]
```

## Usage (for benchmark data)

This repository contains data for various benchmarks of the TDVRPTW problem, namely Solomon (1987), Ichoua et al. (2003), and Balseiro et al. (2010). By running a Python script with command line arguments, specified instance(s) may be run.

Under the `test` directory, run the command `python run_benchmarks.py -f <wildcard>` to run the algorithm all Solomon files/instances that obey `<wildcard>` (e.g. `C101.txt`, `C*.txt`, etc). The results will be printed to console.

Using default hyperparameters, expected runtime should be around 90-120 seconds with the Solomon instances, which have 100 customers each.

## Citations

### Implementation

This implementation of this repo is according to:

Nanda Kumar, Suresh & Panneerselvam, Ramasamy. (2017). Development of an Efficient Genetic Algorithm for the Time Dependent Vehicle Routing Problem with Time Windows. American Journal of Operations Research. 07. 1-25. 10.4236/ajor.2017.71001.

Note that the authors of this repo are not associated with the authors of the paper; the source code was implemented from scratch upon a read of the paper.

### Data

The data included in this repo for testing and evaluation is from:

Balseiro, S. & Loiseau, Irene & Ramonet, Juan. (2011). An Ant Colony algorithm hybridized with insertion heuristics for the Time Dependent Vehicle Routing Problem with Time Windows. Computers & OR. 38. 954-966. 10.1016/j.cor.2010.10.011. 

It is available for download at http://www.columbia.edu/~srb2155/papers/TDVRPTW-instances.zip
