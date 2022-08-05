from ctypes import Union
from typing import Union, List
from pathlib import Path

def load_img_paths_from_dir(data_dir: Union[str, Path], ext="png") -> List[Path]:
    _data_dir = Path(data_dir)
    assert _data_dir.exists(), f"{data_dir}は存在しません。"
    
    return list(_data_dir.glob(f"*.{ext}"))
    