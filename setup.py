from setuptools import setup, find_packages

VERSION = '0.1.0' 
DESCRIPTION = 'SNRPGA2 algorithm to solve TDVRPTW'
LONG_DESCRIPTION = 'Implementation of the genetic algorithm SNRPGA2 for ' + \
     'the Time-Dependent Vehicle Routing Problem with Time Windows ' + \
     '(TDVRPTW), as proposed by: \n\nNanda Kumar, Suresh & Panneerselvam, ' + \
     'Ramasamy. (2017). Development of an Efficient Genetic Algorithm for' + \
     'the Time Dependent Vehicle Routing Problem with Time Windows. ' + \
     'American Journal of Operations Research. 07. 1-25. 10.4236/ajor.2017.71001.'

setup(
    name="tdvrptw_snrpga2", 
    version=VERSION,
    author="Yingchen Ma",
    author_email="ericma@comcast.net",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_package_data=True,
    install_requires=['numpy', 'scipy', 'tqdm'],
    keywords=['python', 'tdvrptw', 'vrp', 'tdvrp', 'vrptw'],
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)