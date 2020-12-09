from typing import TypeVar, Generic, Callable, List, Optional
from abc import ABC, abstractmethod

import numpy as np

from src.utils import rand

T = TypeVar('T')

class MCMC(Generic[T]):
    """Generic MCMC for """

    trajectory = []

    @abstractmethod
    def base(self, i: T, j: T) -> float:
        """Returns the base chain transition probability :math:`p(X_1 = j | X_0 = i)`"""
        pass

    def neighbour(self, i: T, j: T) -> bool:
        """Returns whether there is a path from i to j :math:`p(X_1 = j | X_0 = i) > 0`"""
        return self.base(i, j) > 0

    @abstractmethod
    def propose(self, i: T) -> T:
        """Returns a neighbour state based on base transition probabilities :math:`j ~ p(X_1 | X_0=i)`"""
        pass

    @abstractmethod
    def stationary(self, i: T, j: T) -> float:
        """Returns the stationary dist. ratios of states :math:`\frac{\pi_{j}}{\pi_{i}}`"""
        pass

    def acceptance(self, i: T, j: T) -> float:
        """Returns the acceptance probability of the transition :math:`i \rightarrow j`"""
        return np.minimum(1., self.stationary(i, j) * self.base(j, i) / self.base(i, j))

    def step(self, i: T, t: int) -> T:
        """Takes a step from the current state"""
        j = self.propose(i)
        if rand() <= self.acceptance(i, j):
            return j
        return i

    def simulate(self, i: T, n: int) -> T:
        """Simulate the MCMC from the given start state"""
        self.record(i, 0)
        for t in range(1, n+1):
            i = self.step(i, t)
            self.record(i, t)
        return i

    def record(self, i: T, t: int) -> None:
        """Called to record useful information for the current step"""
        self.trajectory.append(i)

class MCMCUniform(MCMC[T]):
    """MCMC with the uniform base chain"""
    
    @abstractmethod
    def cardinality(self, i: T) -> int:
        """Returns the number of the neighbours a state have"""
        pass

    @abstractmethod
    def neighbour(self, i: T, j: T) -> bool:
        """Returns whether there is a path from i to j :math:`p(X_1 = j | X_0 = i) > 0`"""
        pass

    def base(self, i: T, j: T) -> float:
        if self.neighbour(i, j):
            return 1. / self.cardinality(i)
        else:
            return 0.
    
class MCMCPower(MCMCUniform[List[bool]]):
    """MCMC on the power set of a finite set with the uniform base chain"""

    def __init__(self, n: int) -> None:
        self.n = n

    def cardinality(self, i: List[bool]) -> float:
        return self.n
    
    def neighbour(self, i: List[bool], j: List[bool]) -> bool:
        cnt = 0
        for k in range(self.n):
            if i[k] != j[k]:
                cnt += 1
            if cnt > 1:
                return False
        return True
    
    def propose(self, i: List[bool]) -> List[bool]:
        j = list(i)
        k = np.random.randint(low=0, high=self.n)
        j[k] = not i[k]
        return j
    
    def acceptance(self, i: List[bool], j: List[bool], check=False) -> float:
        """Returns the acceptance probability of the transition :math:`i \rightarrow j`"""
        if check and not self.neighbour(i, j):
            return 0.
        return np.minimum(1., self.stationary(i, j))

class MCMCOptimizer(MCMC[T]):
    """MCMC to optimize a given generic function"""
    objectives = []

    def __init__(self, f: Callable, beta: float = 10, cache: bool = True) -> None:
        self.f = f
        self.beta = beta

        if cache:
            self._cur = None
            self._cur_val = None
            self._prev = None
            self._prev_val = None

            def _f(i: T) -> float:
                if i == self._cur:
                    return self._cur_val
                elif i == self._prev:
                    return self._prev_val
                else:
                    self._prev = self._cur
                    self._prev_val = self._cur_val
                    self._cur = i
                    self._cur_val = f(i)
                return self._cur_val
            self.f = _f
    
    def stationary(self, i: T, j: T) -> float:
        exponent = self.beta * (self.f(i) - self.f(j))
        exponent = np.clip(exponent, -500, +500)
        return np.exp(exponent)

    def simulate(self, i: T, n: int, scheduler: Optional[Callable] = None) -> T:
        self.scheduler = scheduler
        return super().simulate(i, n)

    def record(self, i: T, t: int) -> None:
        """Records useful information for the current step"""
        super().record(i, t)
        self.objectives.append(self.f(i))

        if self.scheduler is not None:
            self.beta = self.scheduler(t)

class MCMCPowerOptimizer(MCMCOptimizer[List[bool]], MCMCPower):
    """MCMC optimization on a uniform power set"""

    def __init__(self, n: int, f: Callable, beta: float, cache: bool = True) -> None:
        MCMCPower.__init__(self, n)
        MCMCOptimizer.__init__(self, f, beta, cache)
