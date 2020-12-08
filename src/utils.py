import logging
from typing import List

import numpy as np
from omegaconf import DictConfig, OmegaConf

def rand() -> float:
    return np.random.rand()

def rand_bool() -> bool:
    if rand() < 0.5:
        return True
    return False

def rand_bool_n(n: int) -> List[bool]:
    return [ rand_bool() for i in range(n) ]

def display_config(cfg: DictConfig) -> None:
    """Displays the configuration"""
    logger = logging.getLogger()
    logger.info("Configuration:\n")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 40 + "\n")
