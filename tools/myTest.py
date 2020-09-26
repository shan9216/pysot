from mmdet.models import FPN
from mmdet.models.backbones.resnet import ResNet
from mmdet.models.anchor_heads.ga_rpn_head import GARPNHead
import torch
from pysot.models.head import GARPN
from pysot.models.backbone.resnet_atrous import resnet50
import numpy as np



bbox = torch.ones((32,4))
print(bbox)
bbox = bbox.unsqueeze(-2)
print(bbox.shape)

bboxes = bbox.split(1, dim=0)
print(len(bboxes))
print(bboxes[0])