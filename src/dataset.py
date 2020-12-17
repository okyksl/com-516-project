import csv
import numpy as np
from typing import Any, Callable, Optional, Type, TypeVar

T = TypeVar('T', bound='Dataset')

class Dataset:
    """A generic dataset handling common dataset operations"""

    n: int = None
    vals: np.ndarray = None # Shape: (n)
    coords: np.ndarray = None # Shape: (n,2)
    dist: np.ndarray = None # Shape: (n,n)

    def __init__(
        self,
        n: int,
        vals: np.ndarray,
        coords: np.ndarray,
        precalc_dist: bool = True) -> None:
        self.n = n
        self.vals = vals
        self.coords = coords

        # precalculate euclidean distances between points
        if precalc_dist:
            coords_row = np.repeat(self.coords[:, np.newaxis, ...], self.n, axis=1)
            coords_col = np.repeat(self.coords[np.newaxis, ...], self.n, axis=0)
            coords_diff = (coords_row - coords_col)
            self.dist = np.linalg.norm(coords_diff, ord=2, axis=-1)

    def dist(i, j) -> np.float:
        """Calculates distance between two cities"""
        if self.dist:
            return self.dist[i,j]
        else:
            return np.linalg.norm(self.coords[i] - self.coords[j], ord=2)

    @classmethod
    def load(cls: Type[T], root: str, **kwargs: Any) -> T:
        """Loads a dataset from the disk"""
        vals = np.load(root + '/vals.npy')
        coords = np.load(root + '/coords.npy')
        n = coords.shape[0]
        return cls(n, vals, coords, **kwargs)

    @classmethod
    def save(cls: type, root: str) -> None:
        """Saves a dataset to disk"""
        np.save(root + '/vals.npy', dataset.vals)
        np.save(root + '/coords.npy', dataset.coords)

class GeneratorDataset(Dataset):
    """Dataset with explicit generators"""

    def __init__(
        self,
        n: int,
        seed: int,
        val_gen: Callable,
        coord_gen: Callable,
        **kwargs: Any) -> None:

        # set seed for reproducibility
        np.random.seed(seed)

        # initialize dataset
        super().__init__(
            n,
            val_gen(n),
            coord_gen(n),
            **kwargs
        )

class G1Dataset(GeneratorDataset):
    """Implements given G1 generative model"""
    def __init__(
        self,
        n: int = 100,
        seed: int = 1,
        **kwargs: Any) -> None:

        val_gen = lambda n: np.random.uniform(low=0, high=1, size=n) 
        coord_gen = lambda n: np.random.uniform(low=0, high=1, size=(n,2))

        super().__init__(
            n,
            seed,
            val_gen,
            coord_gen,
            **kwargs
        )

class G2Dataset(GeneratorDataset):
    """Implements given G2 generative model"""
    def __init__(
        self,
        n: int = 100,
        seed: int = 1,
        **kwargs: Any) -> None:

        val_gen = lambda n: np.exp(np.random.normal(loc=-0.85, scale=1.3, size=n))
        coord_gen = lambda n: np.random.uniform(low=0, high=1, size=(n,2))

        super().__init__(
            n,
            seed,
            val_gen,
            coord_gen,
            **kwargs
        )

class CSVDataset(Dataset):
    def __init__(
        self,
        path: str,
    ) -> None:
        with open(path, 'r') as file:
            reader = csv.reader(file)
            rows = []
            for r in reader:
                rows.append(r)
        rows = rows[1:]
            
        self.ids = np.array([ r[0] for r in rows ])
        super().__init__(
            n=len(rows),
            vals=np.array([ float(r[1]) for r in rows ]),
            coords=np.array([ [float(r[2]), float(r[3])] for r in rows ])
        )

    def output_csv(
        self,
        path: str,
        cities: np.ndarray
    ) -> None:
        out = [['id', 'include']]
        for i in range(len(self.ids)):
            out.append([ self.ids[i], (1 if i in cities else 0) ])

        with open(path, 'w+') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows(out)
