from pysot.core.config import cfg
from pysot.datasets.dataset import TrkDataset
from torch.utils.data import DataLoader
import numpy as np
import cv2

train_dataset = TrkDataset()
train_loader = DataLoader(train_dataset,
                              # batch_size=cfg.TRAIN.BATCH_SIZE,
                              batch_size=2,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True)
for idx, data in enumerate(train_loader):
    print(data['bbox'].shape)
    # bbox = data['bbox'].unsqueeze(-2)
    bbox = data['bbox']
    print(bbox.shape)
    gt_bboxes = bbox.split(1, dim=0)
    print(len(gt_bboxes))
    print(gt_bboxes[0].shape)
    print(gt_bboxes[0])
    break
