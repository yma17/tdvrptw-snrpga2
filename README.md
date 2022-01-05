# tdvrptw-snrpga2
Python implementation of the genetic algorithm SNRPGA2 for the Time-Dependent Vehicle Routing Problem with Time Windows (TDVRPTW).

## Usage (package - recommended)

The contents of this repository can be accessed using the Python package `tdvrptw-snrpga2` ([PyPi link](https://pypi.org/project/tdvrptw-snrpga2/)). Run `pip install tdvrptw-snrpga2` in order to use it in your code.

This following code example demonstrates how to use the package within your Python program:

```
import tdvrptw_snrpga2 as ts

ts.main()  # run all test cases
ts.main('C*.txt')  # run all cases that fit this wildcard
ts.main('C101.txt')  # run one specific test case
```

## Usage (source)

If you wish, you may also run the code from source.

Run the command `python main.py` under the directory `tdvrptw_snrpga2` to run the algorithm for all Solomon files under `tdvrptw_snrpga2/data/instances/`, or `python main.py -f <wildcard>` to run all Solomon files that obey `<wildcard>` (e.g. `C101.txt`, `C*.txt`, etc).

`pip install -r requirements.txt` will install all libraries necessary to run the code, specifically the versions that were used to run and test it. It is recommended to install them and run the code from a virtual environment.

## Citations

### Implementation

This implementation of this repo is according to:

Nanda Kumar, Suresh & Panneerselvam, Ramasamy. (2017). Development of an Efficient Genetic Algorithm for the Time Dependent Vehicle Routing Problem with Time Windows. American Journal of Operations Research. 07. 1-25. 10.4236/ajor.2017.71001.

Note that the authors of this repo are not associated with the authors of the paper; the source code was implemented from scratch upon a read of the paper.

### Data

The data included in this repo for testing and evaluation is from:

Balseiro, S. & Loiseau, Irene & Ramonet, Juan. (2011). An Ant Colony algorithm hybridized with insertion heuristics for the Time Dependent Vehicle Routing Problem with Time Windows. Computers & OR. 38. 954-966. 10.1016/j.cor.2010.10.011. 

It is available for download at http://www.columbia.edu/~srb2155/papers/TDVRPTW-instances.zip
