from __future__ import annotations

from omegaconf import DictConfig, ListConfig, OmegaConf
from .data_module import OICDataModule


def get_datamodule(config: DictConfig | ListConfig) -> OICDataModule:
    config_data = OmegaConf.to_container(config.data)
    datamodule = OICDataModule(**config_data)
    
    return datamodule

