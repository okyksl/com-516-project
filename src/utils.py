import logging
from typing import List

import numpy as np
from omegaconf import DictConfig, OmegaConf

def rand() -> float:
    """Generates a random float in [0,1)"""
    return np.random.rand()

def rand_bool() -> bool:
    """Generates a random bool value"""
    if rand() < 0.5:
        return True
    return False

def rand_bool_n(n: int) -> List[bool]:
    """Generates a random bool list"""
    return [ rand_bool() for i in range(n) ]

def rand_int(n) -> int:
    """Generates a random integer in [0,n)"""
    return np.random.randint(n)

def display_config(cfg: DictConfig) -> None:
    """Displays the configuration"""
    logger = logging.getLogger()
    logger.info("Configuration:\n")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 40 + "\n")
