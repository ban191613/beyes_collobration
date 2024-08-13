import torch
import math
import numpy as np


class data_set:
    """产生数据集，管理数据集"""

    def __init__(
        self,
        model_yf,
        value_dim: int,
        cloud_num: int,
        device_num: int,
        edge_iter: int,
    ):
        """

        Args:
            edge_num (int): 边侧每次更新需要的数据量
            model_yf (): 边缘侧模拟数据产生数据的函数
            cloud_num (int): 云侧每次更新需要的数据量
            device_num (int): 边缘侧每次设备数据数量，就是主动学习选取池子的数据数量
            value_dim (int): 参数维度
            cloud_iter:int,:云端迭代次数
            edge_iter (int, optional): 边缘迭代次数
            active_learning_nums(int): 主动学习选取后的数据数量
        """
        super(data_set, self).__init__()
        self.dtype = torch.float64
        self.cloud_num = cloud_num

        self.device_num = device_num
        self.edge_iter = edge_iter
        self.x_dim = value_dim
        self.model_yf = model_yf

        self.edge_i = 0

        self.f_x = None  # 产生的训练数据
        self.f_y = None

        self.pre_f_x = None  # 产生用于预测的数据
        self.pre_f_y = None

    def generate_data(self):
        """产生仿真的所有数据"""
        self.f_x = torch.rand(
            (
                self.cloud_num + self.device_num * self.edge_iter,
                self.x_dim,
            ),
            dtype=self.dtype,
            requires_grad=False,
        )
        t = np.zeros((self.cloud_num + self.device_num * self.edge_iter, 1), dtype=int)
        t = torch.tensor(t, requires_grad=False)
        for i in range(self.edge_iter):
            t[self.cloud_num + i * self.edge_iter : (i + 1) * self.edge_iter, :] = i
        self.f_y = self.model_yf(self.f_x, t, self.edge_iter)
        # 产生用于预测的数据
        self.pre_f_x = torch.rand(
            (self.edge_iter, self.x_dim),
            dtype=self.dtype,
            requires_grad=False,
        )
        self.pre_f_y = self.model_yf(self.pre_f_x, t, self.edge_iter)

    def cloud_get(self):
        """每次云侧迭代的数据
        Returns:
            torch: 云缘侧第i次训练数据
        """
        begin = self.cloud_num + (self.edge_i) * self.device_num - self.cloud_num
        end = self.cloud_num + (self.edge_i) * self.device_num
        return (
            self.f_x[begin:end, :],
            self.f_y[begin:end, :],
        )

    def edge_get(self, i: int):
        """每次边缘侧迭代的数据

        Args:
            i (int): 第i次边缘侧迭代

        Returns:
            torch: 边缘侧第i次训练数据
        """
        self.edge_i = i

        begin = self.cloud_num + i * self.device_num
        end = self.cloud_num + (i + 1) * self.device_num
        return (
            self.f_x[begin:end, :],
            self.f_y[begin:end, :],
        )

    def edge_pre(self, i):
        return self.pre_f_x[i, :].reshape(1, -1), self.pre_f_y[i, :].reshape(1, -1)

    def save_generate_data(self, file_name):
        data = {
            "f_x": self.f_x.numpy(),
            "f_y": self.f_y.numpy(),
        }
        np.savez(file_name, **data)
        print(f"数据已成功保存到 {file_name}")

    def load_generate_data(self, file_name):
        loaded_data = np.load(file_name)
        self.f_x = torch.tensor(
            loaded_data["f_x"],
            dtype=self.dtype,
            requires_grad=False,
        )
        self.f_y = torch.tensor(
            loaded_data["f_y"],
            dtype=self.dtype,
            requires_grad=False,
        )
        # print(self.edge_f_x.shape, self.edge_f_y.shape)
        # print(self.pre_f_x, self.pre_f_y)
        print(f"数据已成功加载到 dataset")

    # def random_get(self, f_x, f_y, h_x, h_y, active_learning_nums):

    #     random_indices = torch.randperm(len(h_x))[:active_learning_nums]
    #     edge_f_x = h_x[random_indices,]
    #     edge_f_y = h_y[random_indices,]
    #     f_x = torch.cat((edge_f_x, f_x))
    #     f_y = torch.cat((edge_f_y, f_y))
    #     return (f_x, f_y)

    # def edge_history_get(self, i: int):
    #     begin = (
    #         self.cloud_num
    #         + self.edge_iter * self.device_num
    #         + self.edge_i * self.history_num
    #     )
    #     end = (
    #         self.cloud_num
    #         + self.edge_iter * self.device_num
    #         + (self.edge_i + 1) * self.history_num
    #     )
    #     return (
    #         self.f_x[begin:end, :],
    #         self.f_y[begin:end, :],
    #     )

    # def is_history(self):
    #     if self.cloud_num + self.edge_i * self.device_num > self.history_num:
    #         return True
    #     return False

    # def edge_pre(self, i):
    #     return self.pre_f_x[i, :].reshape(1, -1), self.pre_f_y[i, :].reshape(1, -1)


# def yf(x, mean=0, std=0.1):
#     """物理模型

#     Args:
#         x (torch.tensor): 变量
#         u (torch.tensor): 参数

#     Returns:
#         torch.tensor:因变量
#     """
#     y = (
#         0.6
#         + 1.54 * torch.sin(x[:, 0] * math.pi)
#         + 3 * 0.703 * (x[:, 1] ** 3)
#         + std * torch.randn(x.shape[0])
#         + mean
#     )
#     return y.reshape(-1, 1)


# cloud_num = 100  # 云端每次矫正使用真实数据集大小
# history_num = 200  # 边端历史数据集大小
# dim_x = 2  # 参数维度
# edge_iter = 100  # 边端矫正次数
# edge_num = 15  # 边缘端每次矫正使用真实数据集大小
# active_learning_nums = 15  # 主动学习采样数据量
# cloud_m_num = 100  # 云端采样数据量
# cloud_iter = 2  # 云端矫正次数
# edge_iter = 100  # 边端矫正次数
# dataSet = data_set(
#     edge_num=edge_num,
#     edge_yf=yf,
#     cloud_num=cloud_num,
#     history_num=history_num,
#     dim=dim_x,
#     edge_iter=edge_iter,
#     active_learning_nums=active_learning_nums,
# )
# dataSet.generate_data()
# dataSet.save_generate_data("generate_data.npz")
# dataSet.load_generate_data("generate_data.npz")
# class time_data_set:
#     """产生数据集，管理数据集"""

#     def __init__(
#         self,
#         edge_num: int,
#         edge_yf,
#         cloud_num: int,
#         history_num: int,
#         dim: int,
#         edge_iter: int = 100,
#         active_learning_nums=20,
#     ):
#         """

#         Args:
#             edge_num (int): 边侧每次更新需要的数据量
#             edge_yf (_type_): 边缘侧模拟数据产生数据的函数
#             cloud_num (int): 云侧每次更新需要的数据量
#             history_num (int): 边侧历史数据每次更新需要的数据量
#             dim (int): 数据维度
#             edge_iter (int, optional): 边缘侧默认迭代100次. Defaults to 100.
#         """
#         super(time_data_set, self).__init__()
#         self.dtype = torch.float64
#         self.cloud_num = cloud_num

#         self.edge_num = edge_num
#         self.edge_iter = edge_iter
#         self.x_dim = dim

#         self.edge_f_x = torch.rand(
#             (cloud_num + edge_num * edge_iter, dim),
#             dtype=self.dtype,
#             requires_grad=False,
#         )
#         self.edge_f_y = edge_yf(self.edge_f_x[0:cloud_num,], 0)
#         for i in range(edge_iter):
#             begin = cloud_num + i * edge_num
#             end = cloud_num + (i + 1) * edge_num
#             self.edge_f_y = torch.cat(
#                 (
#                     self.edge_f_y,
#                     edge_yf(
#                         self.edge_f_x[begin:end,],
#                         i,
#                     ),
#                 ),
#             )

#         self.pre_f_x = torch.rand(
#             (edge_iter, dim),
#             dtype=self.dtype,
#             requires_grad=False,
#         )
#         for i in range(edge_iter):
#             if i == 0:
#                 self.pre_f_y = edge_yf(self.pre_f_x[i,].unsqueeze(0), i)
#             else:
#                 self.pre_f_y = torch.cat(
#                     (self.pre_f_y, edge_yf(self.pre_f_x[i,].unsqueeze(0), i))
#                 )

#         self.edge_i = 0

#         self.history_num = history_num
#         self.active_learning_nums = active_learning_nums
#         # self.no_al_f_x = torch.rand(
#         #     (edge_iter * active_learning_nums, dim),
#         #     dtype=self.dtype,
#         #     requires_grad=False,
#         # )
#         # self.no_al_f_y = edge_yf(self.no_al_f_x)

#     def edge_input(self):
#         k = torch.range(self.edge_iter * self.x_dim)
#         for _ in range(self.x_dim):
#             y = torch.sin(2 * math.pi * k / 15) + torch.sin(2 * math.pi * k / 25)
#         return y.reshape(-1, 2)

#     def edge_get(self, i: int):
#         self.edge_i = i

#         begin = self.cloud_num + i * self.edge_num
#         end = self.cloud_num + (i + 1) * self.edge_num
#         return (
#             self.edge_f_x[begin:end, :],
#             self.edge_f_y[begin:end, :],
#         )

#     def no_al(self, i):
#         begin = self.cloud_num + i * self.edge_num - self.active_learning_nums
#         end = self.cloud_num + (i + 1) * self.edge_num
#         return (
#             self.edge_f_x[begin:end, :],
#             self.edge_f_y[begin:end, :],
#         )

#     def edge_history_get(self, i: int):
#         begin = self.cloud_num + i * self.edge_num - self.history_num
#         end = self.cloud_num + i * self.edge_num
#         return (
#             self.edge_f_x[begin:end, :],
#             self.edge_f_y[begin:end, :],
#         )

#     def is_history(self):
#         if self.edge_i * self.edge_num > self.history_num:
#             return True
#         return False

#     def cloud_get(self):

#         begin = self.cloud_num + (self.edge_i) * self.edge_num - self.cloud_num
#         end = self.cloud_num + (self.edge_i) * self.edge_num
#         return (
#             self.edge_f_x[begin:end, :],
#             self.edge_f_y[begin:end, :],
#         )

#     def edge_pre(self, i):
#         return self.pre_f_x[i, :].reshape(1, -1), self.pre_f_y[i, :].reshape(1, -1)
