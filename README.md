# COM-516 Mini Project

## Installation

Install dependencies listed in `requirements.txt`.

```python
pip install -r requirements.txt
```

## Running

Simply run the solver with:

```python
python runner.py
```

You can use predefined datasets `g1` and `g2` and even enter customize number of cities **n** and **seed** (for reproducibility) as follows:

```python
python runner.py dataset=g2 dataset.n=10 dataset.seed=5
```

To run the solver over various lambdas:

```python
python runner.py --multirun lmbd=0,0.25,0.5,0.75,1.0
```

### MCMC Options

Default solver is naive solver. Use MCMC solver as follows:

```python
python runner.py solver=mcmc
```

MCMC solver provides the following options with given default values:

```yaml
beta: 1 # initial beta value
step: 2000 # number of steps
start: empty # 'empty set' or 'binomial'
seed: null # random seed for reproducibility
scheduler: # beta scheduling
  checkpoints: [500, 1000, 1500] # steps to change beta values
  betas: [5, 25, 125] # beta values used after checkpoints
use_best: true # if True, uses the best value over all steps instead of final state
num_trials: 10 # if > 1, runs the chain with different seeds and reports best results found
visualize: False # visualizes the run of chain (only the last trial)
```

Here is a more complicated example running MCMC solver over 5 instances of G2 dataset with various lambda values:

```python
python runner.py --multirun lmbd=0.6,0.8,1.0,1.2,1.4 dataset=g2 dataset.seed=0,1,2,3,4 solver=mcmc solver.num_trials=5 solver.use_best=true solver.seed=0
```

## Logging

Results our automatically stored under the `outputs` folder. Depending on the task, `hydra` will store the results under either `outputs/run` or `outputs/multirun` folders. Each run is encapsuled with another file named after current date and time (e.g. `2020-12-14_15-53-02`) which contains result logs and experiment parameters.

Learn more about using `hydra` from [here](https://hydra.cc/).

## Contributing

1. Create a new `Solver` class under `src/solvers.py` by implementing `_solve` function.
2. Define a new hydra solver configuration under `conf/solver` similar to `conf/solver/naive.yaml`
3. Run hydra with the new solver `solver=newsolver`