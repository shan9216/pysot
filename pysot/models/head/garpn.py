from mmdet.models import GARPNHead

import torch
import torch.nn as nn
import torch.nn.functional as F


class GARPN(GARPNHead):
    def __init__(self, out_channels=256, **kwargs):
        super(GARPN, self).__init__( **kwargs)
        # self.cls = DepthwiseXCorr(in_channels, out_channels, self.num_anchors * self.cls_out_channels)
        self.cls = DepthwiseXCorr(kwargs['in_channels'], out_channels, self.num_anchors * self.cls_out_channels)
        self.loc = DepthwiseXCorr(kwargs['in_channels'], out_channels, self.num_anchors * 4)

    def _init_layers(self):
        super(GARPN, self)._init_layers()

    def init_weights(self):
        super(GARPN, self).init_weights()

    def forward_single(self, z_f, x_f):
        x_f = self.rpn_conv(x_f)
        x_f = F.relu(x_f, inplace=True)

        loc_pred = self.conv_loc(x_f)
        shape_pred = self.conv_shape(x_f)
        x_f = self.feature_adaption(x_f, shape_pred)

        z_f = self.rpn_conv(z_f)
        z_f = F.relu(z_f, inplace=True)

        cls_score = self.cls(z_f, x_f)
        bbox_pred = self.loc(z_f, x_f)

        return cls_score, bbox_pred, shape_pred, loc_pred


    def forward(self, z_f, x_f):
        cls_scores=[]
        bbox_preds=[]
        shape_preds=[]
        loc_preds=[]
        # for z, x in zip(z_f, x_f):
        #     cls_score, bbox_pred, shape_pred, loc_pred = self.forward_single(z, x)
        #     cls_scores.append(cls_score)
        #     bbox_preds.append(bbox_pred)
        #     shape_preds.append(shape_pred)
        #     loc_preds.append(loc_pred)
        cls_score, bbox_pred, shape_pred, loc_pred = self.forward_single(z_f, x_f)
        if isinstance(cls_score, list):
            return cls_score, bbox_pred, shape_pred, loc_pred
        cls_scores.append(cls_score)
        bbox_preds.append(bbox_pred)
        shape_preds.append(shape_pred)
        loc_preds.append(loc_pred)
        return cls_scores, bbox_preds, shape_preds, loc_preds

    def loss(self,
             cls_scores,
             bbox_preds,
             shape_preds,
             loc_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        return super(GARPN, self).loss(
            cls_scores,
            bbox_preds,
            shape_preds,
            loc_preds,
            gt_bboxes,
            img_metas,
            cfg,
            gt_bboxes_ignore=None)


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out

class MultiGARPN(nn.Module):
    def __init__(self, in_channels, weighted=False, **kwargs):
        super(MultiGARPN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('garpn'+str(i+2),
                    GARPN(in_channels[i], in_channels[i], **kwargs))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.bbox_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        cls = []
        bbox =[]
        shape = []
        loc = []
        # cls_score, bbox_pred, shape_pred, loc_pred
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            rpn = getattr(self, 'garpn'+str(idx))
            c, b, s, l = rpn(z_f, x_f)
            cls.append(c)
            loc.append(l)
            bbox.append(b)
            shape.append(s)

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            bbox_weight = F.softmax(self.bbox_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(bbox, bbox_weight), avg(shape), avg(loc)
        else:
            return avg(cls), avg(bbox), avg(shape), avg(loc)


    def loss(self,
             cls_scores,
             bbox_preds,
             shape_preds,
             loc_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        rpn = getattr(self, 'garpn2')
        return rpn.loss(cls_scores,
             bbox_preds,
             shape_preds,
             loc_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore)


def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel, padding=2)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out