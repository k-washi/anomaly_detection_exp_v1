from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from src.util.logger import get_logger
logger = get_logger()

from src.conf.config import Config


def data_loader_setup(
    cfg: Config
):
    logger.info(f"Phase:{cfg.ml.phase}, name: {cfg.ml.dataset}")
    if cfg.ml.dataset == "mvtecad":
        from src.dataset.mvtecad import (
            MVTecAdDataset,
            get_img_transform,
            get_gt_transform
        )
        imt = get_img_transform(cfg)
        gtt = get_gt_transform(cfg)
        dataset = MVTecAdDataset(cfg, imt, gtt)
    else:
        raise NotImplementedError(f"{cfg.ml.dataset} データセットについて実装していません。")
    if cfg.ml.phase == "train":
        return DataLoader(
            dataset,
            batch_size=cfg.ml.batch_size,
            shuffle=True,
            num_workers=cfg.ml.num_workers,
            pin_memory=cfg.ml.pin_memory,
            drop_last=True
        )
    else:
        return DataLoader(
            dataset,
            batch_size=cfg.ml.batch_size,
            shuffle=False,
            num_workers=cfg.ml.num_workers,
            pin_memory=cfg.ml.pin_memory,
            drop_last=False
        )


class DataModule(LightningDataModule):
    def __init__(self, cfg:Config) -> None:
        super().__init__()
        self.cfg = cfg
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.cfg.ml.phase = "train"
        if stage == "test":
            self.cfg.ml.phase = "test"
            
    def train_dataloader(self):
        self.cfg.ml.phase = "train"
        return data_loader_setup(self.cfg)
    
    def test_dataloader(self):
        self.cfg.ml.phase = "test"
        return data_loader_setup(self.cfg)