import torch
from typing import List
from torchvision import transforms
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path

def inv_transform(x:torch.Tensor, inv_mean:List[float], inv_std:List[float]) -> torch.Tensor:
    """正規化した画像テンソルをもとに戻す

    Args:
        x (torch.Tensor): _description_
        inv_mean (List[float]): [-0.485/0.229, -0.456/0.224, -0.406/0.255]
        inv_std (List[float]): [1/0.229, 1/0.224, 1/0.255]

    Returns:
        torch.Tensor: _description_
    """
    return transforms.Normalize(
        mean=inv_mean, std=inv_std
    )(x)


def cvt2heatmap(gray:np.ndarray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap:np.ndarray, image:np.ndarray):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image:np.ndarray):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    


def cal_confusion_matrix(y_true, y_pred_no_thresh, thresh, img_path_list):
    pred_thresh = []
    false_n = []
    false_p = []
    for i in range(len(y_pred_no_thresh)):
        if y_pred_no_thresh[i] > thresh:
            pred_thresh.append(1)
            if y_true[i] == 0:
                false_p.append(img_path_list[i])
        else:
            pred_thresh.append(0)
            if y_true[i] == 1:
                false_n.append(img_path_list[i])

    cm = confusion_matrix(y_true, pred_thresh)
    print(cm)
    print('false positive')
    print(false_p)
    print('false negative')
    print(false_n)

def save_anomaly_map(save_dir, anomaly_map, input_img, gt_img, file_name, x_type):
    if anomaly_map.shape != input_img.shape:
        anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
    anomaly_map_norm = min_max_norm(anomaly_map)
    anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)

    # anomaly map on image
    heatmap = cvt2heatmap(anomaly_map_norm*255)
    hm_on_img = heatmap_on_image(heatmap, input_img)

    # save images
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    print(str(save_dir / f'{x_type}_{file_name}.jpg'))
    cv2.imwrite(str(save_dir / f'{x_type}_{file_name}.jpg'), input_img)
    cv2.imwrite(str(save_dir / f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
    cv2.imwrite(str(save_dir / f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
    cv2.imwrite(str(save_dir / f'{x_type}_{file_name}_gt.jpg'), gt_img)

