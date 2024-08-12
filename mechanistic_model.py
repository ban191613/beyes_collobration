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


def yf(x, mean=0, std=0.1):
    """物理模型

    Args:
        x (torch.tensor): 变量
        u (torch.tensor): 参数

    Returns:
        torch.tensor:因变量
    """
    y = (
        0.6
        + 1.54 * torch.sin(x[:, 0] * math.pi)
        + 3 * 0.703 * (x[:, 1] ** 3)
        + std * torch.randn(x.shape[0])
        + mean
    )
    return y.reshape(-1, 1)
