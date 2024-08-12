import torch
import math
import numpy as np
from tqdm import tqdm


from kernel_rbf import kernel_rbf


class gaussian_process(torch.nn.Module):
    def __init__(self, x_dim):
        super(gaussian_process, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        self.JITTER = 1e-6
        self.x_dim = x_dim
        self.train_x = None
        self.train_y = None

        # 如长度尺度（length scale）和信号方差（signal variance）。
        # 如果长度尺度设置得过大，会导致所有输入点之间的协方差接近，协方差矩阵接近对角占优或出现大量相同的值.
        # 降低其条件数，可能使矩阵接近奇异。
        # 反之，如果长度尺度过小，可能导致协方差矩阵过于稀疏，同样影响其可逆性。
        self.log_length_scale = torch.nn.Parameter(
            torch.tensor(
                np.zeros(1),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=True,
        )
        self.log_scale = torch.nn.Parameter(
            torch.tensor(np.zeros(1), dtype=self.dtype, device=self.device),
            requires_grad=True,
        )
        self.log_beta = torch.nn.Parameter(
            torch.tensor(np.ones(1) * -4, dtype=self.dtype, device=self.device),
            requires_grad=True,
        )
        # a large noise by default. Smaller value makes larger noise variance.

    def reset_hyperparameter(self):
        self.log_length_scale = torch.nn.Parameter(
            torch.tensor(
                np.zeros(self.x_dim),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=True,
        )
        self.log_scale = torch.nn.Parameter(
            torch.tensor(np.zeros(1), dtype=self.dtype, device=self.device),
            requires_grad=True,
        )
        self.log_beta = torch.nn.Parameter(
            torch.tensor(np.ones(1) * -4, dtype=self.dtype, device=self.device),
            requires_grad=True,
        )

    def kernel(self, x1, x2):
        return kernel_rbf(x1, x2, self.log_length_scale, self.log_scale)

    def kernel_noise(self, x1, x2):
        return kernel_rbf(
            x1, x2, self.log_length_scale, self.log_scale
        ) + self.log_beta.exp().pow(-1) * torch.eye(x1.size(0))

    def negative_log_likelihood(self):
        y_num = self.train_y.size(0)
        Sigma = (
            kernel_rbf(
                self.train_x,
                self.train_x,
                self.log_length_scale,
                self.log_scale,
            )
            + self.log_beta.exp().pow(-1) * torch.eye(self.train_x.size(0))
            + self.JITTER * torch.eye(self.train_x.size(0))
        )
        # u,info = torch.linalg.cholesky(k_x,upper=True)
        # u_inv = torch.inverse(u)
        # kx_inv = u_inv @ u_inv.t()
        # log_det_k_x = 2 * torch.sum(torch.log(torch.diag(u)))
        # print(Sigma)
        L = torch.linalg.cholesky(Sigma)  # upper=False(defate)
        # option 1 (use this if torch supports)
        gamma = torch.linalg.solve_triangular(L, self.train_y, upper=False)
        nll = (
            0.5 * (gamma**2).sum()
            + L.diag().log().sum()
            + 0.5 * y_num * torch.log(2 * torch.tensor(math.pi))
        )
        return nll

    def train_lbfgs(
        self,
        x,
        y,
        iter=100,
        lr=0.001,
    ):
        self.train_x = x
        self.train_y = y
        optimizer = torch.optim.LBFGS(
            [self.log_length_scale, self.log_scale, self.log_beta],
            lr=lr,
            max_iter=20,
            history_size=7,
            line_search_fn="strong_wolfe",
        )
        # optimizer.zero_grad()
        for _ in range(iter):

            def closure():
                optimizer.zero_grad()
                loss = self.negative_log_likelihood()
                if loss.requires_grad:
                    loss.backward()
                return loss

            optimizer.step(closure)

    def train_adam(self, x, y, iter=100, lr=1):
        self.train_x = x
        self.train_y = y
        optimizer = torch.optim.Adam(
            [self.log_beta, self.log_length_scale, self.log_scale],
            lr=lr,
            # weight_decay=1e-4
        )
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        optimizer.zero_grad()

        # for _ in bar:
        for _ in range(iter):
            optimizer.zero_grad()
            # self.update()
            loss = self.negative_log_likelihood()
            loss.backward()
            optimizer.step()
            # scheduler.step()
        print(
            "nnl:{:.9f}".format(loss.item()),
            "likelihood:{:.9f}\n".format((-loss).exp().item()),
            f"log_length_scale:{[i.item() for i in self.log_length_scale]}\n",
            f"log_scale:{self.log_scale.item()}\n",
            f"log_beta:{self.log_beta.item()}",
        )

    def forward(self, x_predict):
        with torch.no_grad():
            n_predict = x_predict.size(0)
            Sigma_x_train = (
                kernel_rbf(
                    self.train_x, self.train_x, self.log_length_scale, self.log_scale
                )
                + self.log_beta.exp().pow(-1) * torch.eye(self.train_x.size(0))
                + self.JITTER * torch.eye(self.train_x.size(0))
            )
            #  add JITTER here to avoid singularity
            K_x_predict = kernel_rbf(
                self.train_x, x_predict, self.log_length_scale, self.log_scale
            )
            L = torch.linalg.cholesky(Sigma_x_train)

            # option 1
            mean = K_x_predict.t() @ torch.cholesky_solve(self.train_y, L)
            # torch.linalg.cholesky()
            # option 2
            # mean = kx @ torch.L.t().inverse() @ L.inverse() @ Y

            # LinvKx = L.inverse() @ kx.t()  #  the inverse for L should be cheap. check this.
            # torch.cholesky_solve(kx.t(), L)
            LinvKx = torch.linalg.solve_triangular(L, K_x_predict, upper=False)
            # option 1, standard way

            # K_xx_predict = kernel_rbf(
            #     x_predict,
            #     x_predict,
            #     self.log_length_scale,
            #     self.log_scale,
            #     self.log_beta,
            #     True,
            # ) + self.JITTER * torch.eye(x_predict.size(0))
            # var_diag = (
            #     K_xx_predict
            #     - LinvKx.t() @ LinvKx
            #     # + self.log_beta.exp().pow(-1) * torch.eye(n_predict)
            # )
            # option 2, a faster way
            var_diag = (
                self.log_scale.exp().expand(n_predict, 1)
                - (LinvKx**2).sum(dim=0).view(-1, 1)
                + self.log_beta.exp().pow(-1)
            )
        return mean, var_diag

    def max_entropy_predicate(self, x, x_predict, noise=True):
        with torch.no_grad():
            n_predict = x_predict.size(0)
            Sigma_x_train = (
                kernel_rbf(
                    x,
                    x,
                    self.log_length_scale,
                    self.log_scale,
                )
                + self.log_beta.exp().pow(-1) * torch.eye(x.size(0))
                + self.JITTER * torch.eye(x.size(0))
            )
            # # add JITTER here to avoid singularity
            K_x_predict = kernel_rbf(
                x,
                x_predict,
                self.log_length_scale,
                self.log_scale,
            )
            L = torch.linalg.cholesky(Sigma_x_train)

            # torch.linalg.cholesky()
            # option 2
            # mean = kx @ torch.L.t().inverse() @ L.inverse() @ Y

            # LinvKx = L.inverse() @ kx.t()  #  the inverse for L should be cheap. check this.
            # torch.cholesky_solve(kx.t(), L)
            LinvKx = torch.linalg.solve_triangular(L, K_x_predict, upper=False)
            # option 1, standard way

            # K_xx_predict = kernel_rbf(
            #     x_predict,
            #     x_predict,
            #     self.log_length_scale,
            #     self.log_scale,
            #     self.log_beta,
            #     flag_noise=noise,
            # ) + self.JITTER * torch.eye(x_predict.size(0))
            # var_diag = (
            #     K_xx_predict
            #     - LinvKx.t() @ LinvKx
            #     # + self.log_beta.exp().pow(-1) * torch.eye(n_predict)
            # )
            # option 2, a faster way
            var_diag = (
                self.log_scale.exp().expand(n_predict, 1)
                - (LinvKx**2).sum(dim=0).view(-1, 1)
                + self.log_beta.exp().pow(-1)
            )
        return var_diag

    def set_hyperparameter(self, log_length_scale, log_scale, log_beta):
        self.log_length_scale.data = torch.tensor(
            log_length_scale,  # 使用你想要的新长度尺度初始值数组
            dtype=self.dtype,
            device=self.device,
        )
        self.log_scale.data = torch.tensor(
            log_scale,  # 使用你想要的新长度尺度初始值数组
            dtype=self.dtype,
            device=self.device,
        )
        self.log_beta.data = torch.tensor(
            log_beta,  # 使用你想要的新长度尺度初始值数组
            dtype=self.dtype,
            device=self.device,
        )

    def get_hyperparameter(self):
        return (
            [i.item() for i in self.log_length_scale],
            self.log_scale.item(),
            self.log_beta.item(),
        )


# # torch.manual_seed(789)
# from matplotlib import pyplot as plt

# # train_set
# tr_n = 20
# xtr = torch.rand(tr_n, 1)
# ytr = ((6 * xtr - 2) ** 2) * torch.sin(12 * xtr - 4) + torch.randn(
#     tr_n, 1, dtype=torch.float64
# ) * 1

# # test_set
# xte = torch.linspace(0, 1, 100, dtype=torch.float64).view(-1, 1)
# yte = ((6 * xte - 2) ** 2) * torch.sin(12 * xte - 4)
# plt.plot(xtr.numpy(), ytr.numpy(), "b+")
# plt.plot(xte.numpy(), yte.numpy(), "r-", alpha=0.5)
# plt.show()
# # torch.manual_seed(789)
# gp = gaussian_process(1)
# # gp.train_adam(xtr, ytr,iter=1000, lr=0.01)
# gp.train_lbfgs(xtr, ytr, iter=1000, lr=0.001)
# print(gp.log_length_scale, gp.log_scale, gp.log_beta)
# # # gp fit
# # gp.log_length_scale= torch.nn.Parameter(torch.tensor([-1.1535],dtype=torch.float64,requires_grad=True))
# # gp.log_scale= torch.nn.Parameter(torch.tensor([3.3099],dtype=torch.float64, requires_grad=True))
# # gp.log_beta= torch.nn.Parameter(torch.tensor([0.0586], dtype=torch.float64,requires_grad=True))

# ypred, yvar = gp(xte)


# print(yvar.sqrt().squeeze().detach().numpy())
# # plot the data


# plt.errorbar(
#     xte.numpy().reshape(100),
#     ypred.detach().numpy().reshape(100),
#     yerr=yvar.sqrt().squeeze().detach().numpy(),
#     fmt="r-.",
#     alpha=0.2,
# )
# plt.plot(xtr.numpy(), ytr.numpy(), "b+")
# plt.plot(xte.numpy(), yte.numpy(), "r-", alpha=0.3)
# plt.show()
# print("parameter：")
# print(gp.log_length_scale, gp.log_scale, gp.log_beta.exp().pow(-1))
