from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.dataset import TrkDataset
from torch.utils.data import DataLoader

import numpy as np
import cv2



cfg.merge_from_file('/home/shanyu/git/pysot/experiments/siamgarpn/config.yaml')

# create model
model = ModelBuilder().cuda()

train_dataset = TrkDataset()

train_loader = DataLoader(train_dataset,
                              # batch_size=cfg.TRAIN.BATCH_SIZE,
                              batch_size=2,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True)

for idx, data in enumerate(train_loader):
    output= model(data)
    print(output)
    break
