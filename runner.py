import os
import hydra
import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from src.utils import display_config
from src.dataset import Dataset
from src.solvers import Solver

@hydra.main(config_path='conf', config_name='solve')
def run(cfg: DictConfig):
    display_config(cfg)

    # instantiate from config file
    dataset: Dataset = instantiate(cfg.dataset)
    solver: Solver = instantiate(cfg.solver)
    lmbd: float = cfg.lmbd

    logger = logging.getLogger()
    logger.info('Begin solving')
    S, obj = solver.solve(dataset, lmbd)
    logger.info('End solving')
    logger.info('=' * 40 + '\n')

    logger.info(f'Objective: {obj}')
    logger.info(f'Set S: {S}')

if __name__ == "__main__":
    run()