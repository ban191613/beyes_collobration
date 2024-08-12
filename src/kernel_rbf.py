import torch


def kernel_rbf(X1, X2, log_length_scale, log_scale):

    N = X1.shape[0]
    X1 = X1 / torch.exp(log_length_scale)
    X2 = X2 / torch.exp(log_length_scale)
    X1_norm2 = torch.sum(X1 * X1, dim=1).view(-1, 1)
    X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)
    K = (
        -2.0 * X1 @ X2.t()
        + X1_norm2.expand(X1.size(0), X2.size(0))
        + X2_norm2.t().expand(X1.size(0), X2.size(0))
    )  # this is the effective Euclidean distance matrix between X1 and X2.
    return torch.exp(log_scale) * torch.exp(-K)
