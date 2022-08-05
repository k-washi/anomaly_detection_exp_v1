from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainConfig:
    seed:int =  3407
    learning_rate: float = 0.001
    optimizer: str = 'AdamW'

@dataclass
class MlConfig:
    phase: str = "train" # test
    dataset: str = "mvtecad"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class ModelConfig:
    torch_hub_version: str = 'pytorch/vision:v0.9.0'
    torch_hub_model: str = 'wide_resnet50_2'
    pretrained: bool = True

@dataclass
class ImageDatasetConfig:
    img_dir: str = "./data/mvtec_ad/bottle"
    img_mean: List[float] = field(default_factory=lambda:[0.485, 0.456, 0.406])
    img_std: List[float] = field(default_factory=lambda:[0.229, 0.224, 0.225])
    inv_img_mean: List[float] = field(default_factory=lambda:[-0.485/0.229, -0.456/0.224, -0.406/0.255])
    inv_img_std: List[float] = field(default_factory=lambda: [1/0.229, 1/0.224, 1/0.255])
    load_size: int = 256
    input_size: int = 254

class PlotConfig:
    save_dir: str = "./data/mvtecad_result"