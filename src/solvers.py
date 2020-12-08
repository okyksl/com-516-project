import itertools
import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple

from src.dataset import Dataset

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
