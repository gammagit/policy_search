This repository explores various alogirthms searching for good (even optimal) decision boundaries for an integrate-to-threshold model. It creates an animation of an observer moving through the _reward landscape_ as it makes decisions and experiences reward rates at various positions in this landscape.


### Running the code
To run the code simply use the notebook `test_dynamics.ipynb` for a set of default parameters. Change the parameters of the observer in `observer.py` (`get_gen_params()`) and `generator.py` (`get_params()`)


### Description of files

- `test_dynamics.ipynb`: Use this notebook to create the animations. Most of the code here deals with plotting functions. The crucial function call is `obs.update_bound()`, which is called on every frame and updates the policy based on current policy and RR, etc.

- `observer.py`: Contains code for the "observer" who sets a policy, experiences a set of random walks at this policy and then jumps to a different policy to search search for better reward rate. Crucial function here is `get_gen_params()`, which contains various parameters for each search algorithm.

- `generator.py`: Contains code for the process (external to the observer) that generates the random walks. The function `gen_params()` contains various parameter setting (drift, inter-trial interval, etc) for these random walks.


### Adding more search algorithms
- The function `update_bound()` in `observer.py` calls the search algorithm to iteratively search for decision policies. You can use the functions `gradient_descent()` or `greedy_search()` as templates, if you would like to experiment with a search algorithm.


### Requirements
See requirements.txt
I ran the code using Python 3.8