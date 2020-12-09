import itertools
import numpy as np

from abc import ABC, abstractmethod
from typing import Callable, Tuple, List, Optional, Any
from omegaconf import DictConfig


from src.dataset import Dataset
from src.geometry import find_max_dist
from src.mcmc import MCMCPowerOptimizer
from src.utils import rand, rand_bool, rand_bool_n

class Solver(ABC):
    dataset: Dataset = None
    lmbd: float = None

    def solve(self, dataset: Dataset, lmbd: float) -> Tuple[list, float]:
        """Saves dataset and lambda, uses internal solver to report results"""
        # save global params
        self.dataset = dataset
        self.lmbd = lmbd
        
        # solve the problem with a specific solver
        return self._solve()

    @abstractmethod
    def _solve(self) -> Tuple[list, float]:
        """Internal solver function, to be implemented by subclasses"""
        pass

    def inside(self, center: np.ndarray, radius: float) -> list:
        """Calculates the points inside the proposed circle"""
        S = []
        for i in range(self.dataset.n):
            if Solver.dist(center, self.dataset.coords[i]) <= radius:
                S.append(i)
        return S
    
    def objective(self, S: list, radius: float) -> float:
        """Calculates objective given the set S and cost radius"""
        return np.sum(self.dataset.vals[S]) - self.lmbd * self.dataset.n * np.pi * (radius ** 2)

    @staticmethod
    def dist(pt1: np.ndarray, pt2: np.ndarray) -> np.float:
        return np.linalg.norm(pt1 - pt2, ord=2)

class NaiveSolver(Solver):
    """Implements naive O(n^3) brute-force solution"""

    S: list = None
    best: float = None

    def _consider(self, center: np.ndarray, radius: float) -> None:
        """Updates best solution according to the proposed circle"""
        S = self.inside(center, radius)
        obj = self.objective(S, radius)
        if obj > self.best:
            self.best = obj
            self.S = S

    def _solve(self) -> Tuple[list, float]:
        # container vars for best found solution
        self.S = []
        self.best = 0

        # list composed of three (0,n-1) ranges
        ranges = [range(self.dataset.n) for i in range(3)]

        # single point solutions
        for (i) in itertools.product(*ranges[:1]):
            self._consider(self.dataset.coords[i], 0)
        
        # two points solutions
        for (i,j) in itertools.product(*ranges[:2]):
            if i < j:
                center = (self.dataset.coords[i] + self.dataset.coords[j]) / 2
                radius = Solver.dist(self.dataset.coords[i], center)
                self._consider(center, radius)

        # return the best results
        return self.S, self.best

class MCMCSolver(Solver):
    """Implements naive MCMC solution with a base chain on power set of cities"""

    def __init__(self, beta: float, step: int, start: str, scheduler: Optional[Any]) -> None:
        self.beta = beta
        self.step = step
        self.start = start

        if scheduler is not None:
            self.beta_n = len(scheduler.checkpoints)
            self.checkpoints = scheduler.checkpoints
            self.betas = scheduler.betas
            self.betas.insert(0, self.beta)

            def scheduler(t: int) -> float:
                i = 0
                while i < self.beta_n and t > self.checkpoints[i]:
                    i += 1
                return self.betas[i]

            self.scheduler = scheduler

    def _solve(self) -> Tuple[list, float]:
        def f(i: List[bool]) -> float:
            points = self.dataset.coords[i, :]
            radius = find_max_dist(points) / 2
            return -self.objective(i, radius)

        if self.start == 'empty':
            state = [False for i in range(self.dataset.n)]
        elif self.start == 'binom':
            state = rand_bool_n(self.dataset.n)
        else:
            raise ValueError

        chain = MCMCPowerOptimizer(self.dataset.n, f, beta=self.beta, cache=True)
        res = chain.simulate(state, self.step, scheduler=self.scheduler)

        objectives = -np.array(chain.objectives)
        states = [ np.sum(np.array(trajectory)) for trajectory in chain.trajectory ]

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=2, ncols=1)
        axs[0].plot(objectives)
        axs[0].set_ylabel(f'$f(S)$')
        axs[1].plot(states)
        axs[1].set_ylabel(f'$|S|$')
        plt.show()

        return np.arange(self.dataset.n)[res], -f(res)
