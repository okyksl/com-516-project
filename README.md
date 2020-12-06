# COM-516 Mini Project

## Installation

Install dependencies listed in `requirements.txt`.

```python
pip install -r requirements.txt
```

## Run

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

## Contributing

1. Create a new `Solver` class under `src/solvers.py` by implementing `_solve` function.
2. Define a new hydra solver configuration under `conf/solver` similar to `conf/solver/naive.yaml`
3. Run hydra with the new solver `solver=mcmc`