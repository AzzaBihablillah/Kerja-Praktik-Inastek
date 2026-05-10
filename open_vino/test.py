import torch
import numpy as np
from utils.detector_utils import non_max_suppression

# x = torch.rand(4)
y = torch.rand(2,2)
print(y)
c = y[:,1:2] * 2
print(c)
# print(torch.cat((x,y), 0))

# print(x)
# print(x.view(-1)[[True, True, False, False]])
# print(x[[True, True], [True, True]])
# print(x.reshape(-1))
# print(x[..., 1])
# print(4 < y < 6)
# print(x[..., 1:2])

x = torch.rand(1,10,10)
print(x)

x = non_max_suppression(x, conf_thres=1, iou_thres=0.5)
print(x)
print(x[0].shape)


