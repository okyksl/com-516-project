import logging
from omegaconf import DictConfig, OmegaConf

def display_config(cfg: DictConfig) -> None:
    """Displays the configuration"""
    logger = logging.getLogger()
    logger.info("Configuration:\n")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 40 + "\n")
