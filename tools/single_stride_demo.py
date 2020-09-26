import torch
from pysot.models.backbone.resnet_atrous import resnet50
from pysot.models.neck import AdjustAllLayer
from pysot.models.head import GARPN
import numpy as np
from mmcv import Config



img_search = torch.ones((1, 3 , 255, 255)).cuda()
img_template = torch.ones((1, 3 , 127, 127)).cuda()
backbone = resnet50(used_layers=[3]).cuda()

adjust = AdjustAllLayer(in_channels=[1024], out_channels=[256]).cuda()
cfg = Config.fromfile('/home/shanyu/git/pysot/experiments/siamgarpn/garpn.py')

z_f = backbone(img_template)
z_f = adjust(z_f)
print('after backbone z_f, len:', len(z_f))
for i in range(len(z_f)):
    print('after backbone z_f, shape:', z_f[i].shape)


x_f = backbone(img_search)
x_f = adjust(x_f)
print('after backbone x_f, len:', len(z_f))
for i in range(len(x_f)):
    print('after backbone x_f, shape:', x_f[i].shape)

garpn = GARPN(in_channels=256,
        feat_channels=256,
        octave_base_scale=8,
        scales_per_octave=3,
        octave_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8],
        anchor_base_sizes=None,
        anchoring_means=[.0, .0, .0, .0],
        anchoring_stds=[0.07, 0.07, 0.14, 0.14],
        target_means=(.0, .0, .0, .0),
        target_stds=[0.07, 0.07, 0.11, 0.11],
        loc_filter_thr=0.01,
        loss_loc=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)).cuda()
cls_score, bbox_pred, shape_pred, loc_pred = garpn(z_f, x_f)

print('cls_score, len:', len(cls_score))
for i in range(len(cls_score)):
    print('cls_score, shape:', cls_score[i].shape)

print('bbox_pred, len:', len(bbox_pred))
for i in range(len(bbox_pred)):
    print('bbox_pred, shape:', bbox_pred[i].shape)

print('shape_pred, len:', len(shape_pred))
for i in range(len(shape_pred)):
    print('shape_pred, shape:', shape_pred[i].shape)

print('loc_pred, len:', len(loc_pred))
for i in range(len(loc_pred)):
    print('loc_pred, shape:', loc_pred[i].shape)

bboxes = [[10, 10, 20, 20]]
bboxes_list = []
bboxes_list.append(torch.from_numpy(np.array(bboxes, dtype=np.float32)).cuda())
losses = garpn.loss(cls_score,bbox_pred,shape_pred,loc_pred,bboxes_list,[{'pad_shape':[255, 255],"img_shape":[255, 255]}], cfg.train_cfg.rpn)
print(losses)