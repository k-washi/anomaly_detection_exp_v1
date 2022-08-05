import torch

from src.conf.config import Config

def model_setup(cfg:Config):
    if cfg.model.use_torchhub:
        return torch.hub.load(
            cfg.model.torch_hub_version,
            cfg.model.torch_hub_model,
            pretrained=cfg.model.pretrained
        )
    else:
        raise NotImplementedError("モデルの設定に関して、実装されていません。")
    