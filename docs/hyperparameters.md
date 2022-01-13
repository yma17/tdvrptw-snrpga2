# Hyperparameters

"obj_func"
* Objective function for algorithm to minimize
* Data type: `str
* Acceptable values:
  * `"v"` (number of vehicles used)
  * `"d"` (total distance travelled)
  * `"t"` (total time taken)
  * `"dt"` (weighted sum of distance and time - see `"w_t"` below)
* Default value: `"dt"`

"w_t"
* Relative weight of time in objective function with respect to distance
* Needed if `obj_func == dt`
* Data type: `float`
* Acceptable values: `[0, +inf)`
* Default value: `1.0`

"mng"
* Maximum number of generations, aka number of iterations in algorithm
* Data type: `int`
* Acceptable values: `(0, +inf)`
* Default value: `1000`

"pop_size"
* Size of genetic population
* Data type: `int`
* Acceptable values: `(0, +inf)`
* Default value: `100`

"init_method"
* Method to generate initial population
* Data type: `str`
* Acceptable values:
  * `"random_sample"`: sample from locations without replacement
  * `"random_seq"`: sample a location index, then set to ordered list of locations that begins at that index and wraps around to beginning of location list and back to that index
* Default value: `"random_sample"`

"ts_prob"
* Tournament selection probability in genetic algorithm
* Data type: `float`
* Acceptable values: `[0, 1]`
* Default value: `0.9`

"x_prob"
* Crossover probability in genetic algorithm
* Data type: `float`
* Acceptable values: `[0, 1]`
* Default value: `0.7`

"m_prob"
* Mutation probability in genetic algorithm
* Data type: `float`
* Acceptable values: `[0, 1]`
* Default value: `0.3`

"cache_gran"
* Numerical granularity of keys in cache used in `utils.round_nearest()`
* Data type: `float`
* Default value: `1.0`