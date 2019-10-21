"""Util functions"""
import random

import torch
import numpy as np

def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

CLASS_LABELS = {
    'VOC': {
        'all': set(range(1, 21)),
        0: set(range(1, 21)) - set(range(1, 6)),
        1: set(range(1, 21)) - set(range(6, 11)),
        2: set(range(1, 21)) - set(range(11, 16)),
        3: set(range(1, 21)) - set(range(16, 21)),
    },
    'COCO': {
        'all': set(range(1, 81)),
        0: set(range(1, 81)) - set(range(1, 21)),
        1: set(range(1, 81)) - set(range(21, 41)),
        2: set(range(1, 81)) - set(range(41, 61)),
        3: set(range(1, 81)) - set(range(61, 81)),
    }
}

def get_bbox(fg_mask, inst_mask):
    """
    Get the ground truth bounding boxes
    """

    fg_bbox = torch.zeros_like(fg_mask, device=fg_mask.device)
    bg_bbox = torch.ones_like(fg_mask, device=fg_mask.device)

    inst_mask[fg_mask == 0] = 0
    area = torch.bincount(inst_mask.view(-1))
    cls_id = area[1:].argmax() + 1
    cls_ids = np.unique(inst_mask)[1:]

    mask_idx = np.where(inst_mask[0] == cls_id)
    y_min = mask_idx[0].min()
    y_max = mask_idx[0].max()
    x_min = mask_idx[1].min()
    x_max = mask_idx[1].max()
    fg_bbox[0, y_min:y_max+1, x_min:x_max+1] = 1

    for i in cls_ids:
        mask_idx = np.where(inst_mask[0] == i)
        y_min = max(mask_idx[0].min(), 0)
        y_max = min(mask_idx[0].max(), fg_mask.shape[1] - 1)
        x_min = max(mask_idx[1].min(), 0)
        x_max = min(mask_idx[1].max(), fg_mask.shape[2] - 1)
        bg_bbox[0, y_min:y_max+1, x_min:x_max+1] = 0
    return fg_bbox, bg_bbox
