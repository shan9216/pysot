import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

# class demo_module(nn.Module):
#
#     def __init__(self):
#         super(demo_module, self).__init__()
#         # self.pad = nn.ConstantPad2d(0)
#         self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3,padding=1)
#
#     def forward(self, data):
#         return self.conv(data)
#
#
# m = demo_module()
# d = torch.ones([1,3,5,5])
# print(m(d))

# data = torch.from_numpy(np.array([[[[1.,2.,3.],[4.,5., 6.],[7., 8., 9.]],[[1.,2.,3.],[4.,5., 6.],[7., 8., 9.]]]]))
data = torch.from_numpy(np.array([[[[1.,2.,3.],[4.,5., 6.],[7., 8., 9.]]]]))
score = data.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
score1 = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
print(data)
print(score)
print(score1)