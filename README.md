# tdvrptw-snrpga2
Python implementation of the genetic algorithm SNRPGA2 for the Time-Dependent Vehicle Routing Problem with Time Windows (TDVRPTW).

## Usage

Run the command `python main.py` under the root directory to run the algorithm for all Solomon files under `data/instances/`, or `python main.py -f <wildcard>` to run all Solomon files that obey `<wildcard>` (e.g. `C101.txt`, `C*.txt`, etc).

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
