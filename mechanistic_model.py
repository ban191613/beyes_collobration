import torch
import math


def ym(x, u):
    """机理模型

    Args:
        x (torch.tensor): 变量
        u (torch.tensor): 参数

    Returns:
        torch.tensor: 因变量
    """
    y = (
        0.3
        + 1.5 * torch.sin(x[:, 0] * u[:, 0] * 3 * math.pi)
        + 3 * u[:, 1] * (x[:, 1] ** 3)
    )
    return y.reshape(-1, 1)


def yf(x, t, n, mean=0, std=0.1):
    """物理模型  插入一个阶跃函数

    Args:
        x (torch.tensor): 变量
        t (torch.tensor): 时间
        n (int): 时间阈值
        mean (float, optional): 均值. 默认为 0.
        std (float, optional): 标准差. 默认为 0.1.
    Returns:
        torch.tensor:因变量
    """
    # t=n/2 (0.1 if t >= n / 2 else 0)
    step_function = torch.where(t >= n / 2, torch.tensor(0.1), torch.tensor(0.0))

    y = (
        0.6
        + 1.54 * torch.sin(x[:, 0] * (1 - 0.3 * step_function) * math.pi)
        + 3 * (0.703 + step_function) * x[:, 1] ** 3
        + std * torch.randn(x.shape[0])
        + mean
    )
    return y.reshape(-1, 1)
