import torch
import numpy as np
import mechanistic_model as model

model_yf = model.yf
cloud_num = 10
x_dim = 2
edge_iter = 100
device_num = 3
f_x = torch.rand(
    (
        cloud_num + device_num * edge_iter,
        x_dim,
    ),
    requires_grad=False,
)
t = torch.tensor(
    np.zeros((cloud_num + device_num * edge_iter), dtype=int),
    requires_grad=False,
)
f_y_1 = model_yf(f_x, t, edge_iter, 0, 0)
# 产生和数据对应的事件t
for i in range(edge_iter):
    begin = cloud_num + i * device_num
    end = cloud_num + (i + 1) * device_num
    t[begin:end] = i
f_y = model_yf(f_x, t, edge_iter, 0, 0)
t.shape
f_x.shape
f_y.shape
