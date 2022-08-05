from pathlib import Path
import neptune
import hydra
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint

from src.util.logger import get_logger
logger = get_logger()

from src.conf.config import Config
from src.dataset.loader import DataModule
from src.model.patchcore import PatchCoreModelModule

SEED = 3407
seed_everything(SEED, workers=True)

class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @rank_zero_only
    def _del_model(self, *_):
        pass
    
    def _save_model(self, *_):
        pass

@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):
    print(cfg)
    cfg.ml.batch_size = 1
    try:
        device = "gpu" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    
    dataset = DataModule(cfg)
    model = PatchCoreModelModule(cfg)
    
    trainer = Trainer(
        max_epochs=cfg.train.epochs,
        gpus=1,
        logger=False,
        callbacks=[ModelCheckpoint(f"./data/{cfg.plot.save_dir}")]
    )
    
    trainer.fit(
        model,
        dataset
    )
    trainer.test(model, dataloaders=dataset)
    

if __name__ == "__main__":
    main()