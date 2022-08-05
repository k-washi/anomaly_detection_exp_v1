from torch.utils.data import Dataset
from pathlib import Path
from typing import Any, Optional, List
import torch
from torchvision import transforms
from PIL import Image

from src.util.logger import get_logger
logger = get_logger()

from src.dataset.util.io import load_img_paths_from_dir
from src.conf.config import Config

MVTEC_GOOD_DIR_NAME = "good"
MVTEC_IMG_EXT_TYPE = "png"

def get_img_transform(cfg: Config) -> transforms.Compose:
    return transforms.Compose([
                transforms.Resize((cfg.imdata.load_size, cfg.imdata.load_size), Image.ANTIALIAS),
                transforms.ToTensor(),
                transforms.CenterCrop(cfg.imdata.input_size),
                transforms.Normalize(mean=cfg.imdata.img_mean, std=cfg.imdata.img_std)
            ])

def get_gt_transform(cfg: Config) -> transforms.Compose:
    return transforms.Compose([
                transforms.Resize((cfg.imdata.load_size, cfg.imdata.load_size)),
                transforms.ToTensor(),
                transforms.CenterCrop(cfg.imdata.input_size)
            ])

class MVTecAdDataset(Dataset):
    def __init__(
        self, 
        cfg: Config, 
        transform: Optional[transforms.Compose], 
        gt_transform: Optional[transforms.Compose]
    ) -> None:
        super().__init__()
        root = Path(cfg.imdata.img_dir).absolute()
        logger.info(f"DATA ROOT DIR: {root}")
        assert root.exists(), f"{root}は存在しません。"
        if cfg.ml.phase == 'train':
            self.img_path =  root / "train"
        else:
            # test
            self.img_path = root / "test"
            self.gt_path = root / "ground_truth"
        
        self.cfg = cfg
        self.transform = transform
        self.gt_transform = gt_transform
        
        self.img_paths, self.gt_img_paths, self.labels, self.img_types = self.load_dataset()
    
    def load_dataset(self):
        img_paths: List[str] = [] # 入力画像のパス
        gt_img_paths: List[str] = [] #異常位置のマスクのパス
        labels: List[int] = [] # 0:正常, 1:異常
        defect_types_list: List[str] = [] # 正常:goot, 異常:異常に関するコメント
        
        defect_types = self.img_path.iterdir()
        
        for defect_type in defect_types:
            if not defect_type.is_dir():
                continue
            # 入力画像パスのロード
            img_path = self.img_path / defect_type
            logger.debug(f"Load dir: {img_path}")
            _img_paths = load_img_paths_from_dir(img_path, ext=MVTEC_IMG_EXT_TYPE)
            _img_paths.sort()
            img_paths.extend([str(fp) for fp in _img_paths])
            
            if str(defect_type.name) == MVTEC_GOOD_DIR_NAME:
                # 訓練データ(正常データ)は、goodフォルダに入っている
                gt_img_paths.extend([""]*len(_img_paths))
                label_id = 0
            else:
                # 異常データに関するデータまとめ
                _gt_paths = load_img_paths_from_dir(self.gt_path / defect_type, ext=MVTEC_IMG_EXT_TYPE)
                _gt_paths.sort()
                gt_img_paths.extend([str(fp) for fp in _gt_paths])
                label_id = 1
            
            
            labels.extend([label_id]*len(_img_paths))
            defect_types_list.extend([str(defect_type)]*len(_img_paths))
        assert len(img_paths) > 0, f"{self.img_path}ないの画像数は0です。"
        assert len(img_paths) == len(gt_img_paths) == len(labels) == len(defect_types_list), f"データ情報の読み込み数が一致していません。"
        
        return img_paths, gt_img_paths, labels, defect_types_list
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> Any:
        img_path, gt_path, label, img_type = self.img_paths[index], self.gt_img_paths[index], self.labels[index], self.img_types[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None: img=self.transform(img)
        if len(gt_path) == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-1]])
        else:
            gt = Image.open(gt_path).convert('L')
            gt = self.gt_transform(gt)
        
        assert img.size()[1:] == gt.size()[1:], f"Image.size:{img.size()} != gt.size: {gt.size()}"
        return img, gt, label, img_type


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    cfg = Config()
    cfg.ml.phase = "test"
    cfg.ml.batch_size = 2
    imt = get_img_transform(cfg)
    gtt = get_gt_transform(cfg)
    
    dataset = MVTecAdDataset(cfg, imt, gtt)
    img_loader = DataLoader(dataset, batch_size=cfg.ml.batch_size, shuffle=False, drop_last=False)
    for batch in img_loader:
        batch_img, batch_gt, batch_label, betch_img_type = batch
        print(batch_img.shape, batch_gt.shape)
    
    np_img = batch_img.cpu().numpy()[0]
    np_gt = batch_gt.cpu().numpy()[0]
    
    