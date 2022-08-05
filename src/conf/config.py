from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from src.conf.ml.default import (
    MlConfig, 
    TrainConfig, 
    ImageDatasetConfig,
    PlotConfig,
    ModelConfig
)

@dataclass
class Config:
    ml: MlConfig = MlConfig()
    train: TrainConfig = TrainConfig()
    imdata: ImageDatasetConfig = ImageDatasetConfig()
    plot: PlotConfig = PlotConfig()
    model: ModelConfig = ModelConfig()
    
cs = ConfigStore.instance()
cs.store(name="config", node=Config)

@hydra.main(version_base=None, config_name="config")
def config_print(cfg: Config) -> None:
    cfg = OmegaConf.to_yaml(cfg)
    print(cfg)
    

if __name__ == "__main__":
    # python src/conf/config.py ml.batch_size=24
    # ml.batch_sizeが24に変更される
    config_print()
    