import torch
import math
import numpy as np
from scipy.stats import multivariate_normal

from gaussian_process import gaussian_process
from MetropolisHastings import MetropolisHastings


class edge_koh_b:
    def __init__(
        self,
        dim_x,
        u,
        random_walk: list,
        mcmc_sample_nums: int,
        prior_mean,
        prior_cov,
        prior_lower,
        prior_upper,
        burn_ratio,
    ):
        super(edge_koh_b).__init__()
        self.JITTER = 1e-6
        self.dtype = torch.float64

        self.dim_x = dim_x
        self.dim_u = len(u)

        self.b_gp = gaussian_process(dim_x)
        # self.m_gp = None
        self.random_walk = random_walk
        self.mcmc = MetropolisHastings(
            sample_nums=mcmc_sample_nums, random_walk=random_walk, burn_ratio=burn_ratio
        )

        self.u = torch.tensor(u, dtype=self.dtype).view(-1)

        self.cov_m = None
        self.cov_b = None
        self.sample = None
        self.m_xu = None
        self.sigma_m = None
        self.m_xu = None

        self.mean = prior_mean
        self.cov = prior_cov
        self.lwr = prior_lower
        self.upr = prior_upper

    def yb_train(self, f_x, f_y, m_gp, iter, lr):
        self.f_x = f_x

        f_xu = torch.cat(
            (
                f_x,
                torch.ones((f_x.shape[0], len(self.u))) * self.u,
            ),
            1,
        )
        self.f_y = f_y
        self.m_gp = m_gp
        m_ypred, m_yvar = m_gp(f_xu)
        cloud_b_y = self.f_y - m_ypred
        self.b_gp.train_lbfgs(f_x, cloud_b_y, iter=iter, lr=lr)

    def sigma_b(
        self,
        x,
        y,
    ):
        return self.b_gp.kernel_noise(x, y)

    def log_parameter_prior(self, u) -> float:

        for i in range(self.dim_u):
            if u[i] < self.lwr[i]:
                return -math.inf
            if u[i] > self.upr[i]:
                return -math.inf
        dist = multivariate_normal(mean=self.mean, cov=self.cov)
        return dist.logpdf(u)

    def negative_log_likelihood(self, u) -> float:
        u = torch.tensor(u)
        xfu = torch.hstack((self.f_x, torch.ones((self.f_x.shape[0], len(u))) * u))
        cov_mf = self.sigma_m(self.m_xu, xfu, noise=False)
        cov_f = self.sigma_m(xfu, xfu, noise=True) + self.cov_b

        cov = torch.cat(
            (
                torch.cat((self.cov_m, cov_mf), dim=1),
                torch.cat((cov_mf.transpose(0, 1), cov_f), dim=1),
            ),
            dim=0,
        )
        cov = cov + self.JITTER * torch.eye(cov.shape[0])
        L = torch.linalg.cholesky(cov)  # upper=False(defate)
        # option 1 (use this if torch supports)
        gamma = torch.linalg.solve_triangular(L, self.y, upper=False)
        nll = -0.5 * (gamma**2).sum() - L.diag().log().sum()
        return nll

    def parameter_mcmc(self, cov_m, m_xu, m_y, sigma_m):
        u = self.u.numpy()
        self.m_xu = m_xu
        self.m_y = m_y
        self.cov_m = cov_m
        self.cov_b = self.sigma_b(self.f_x, self.f_x)
        self.sigma_m = sigma_m
        self.y = torch.vstack((self.m_y, self.f_y))
        self.sample = self.mcmc.sampling(
            u, self.log_parameter_prior, self.negative_log_likelihood
        )
        self.set_u(np.mean(self.sample, axis=0))
        self.set_random_walk(np.cov(self.sample, rowvar=False))

    def predict(self, x):

        xu = torch.cat(
            (
                x,
                torch.ones((x.shape[0], len(self.u))) * self.u,
            ),
            1,
        )
        (m_mean, m_var) = self.m_gp(xu)
        (b_mean, b_var) = self.b_gp(x)

        return m_mean, b_mean, m_var, b_var

    def predict_y_condition_x_u(self, x, u, m_gp):

        xu = torch.cat(
            (
                x,
                torch.ones((x.shape[0], len(u))) * u,
            ),
            1,
        )
        (m_mean, m_var) = m_gp(xu)
        (b_mean, b_var) = self.b_gp(x)

        return m_var + b_var

    def set_random_walk(self, random_walk):
        self.random_walk = random_walk

    def set_b_hyperparameter(self, log_length_scale, log_scale, log_beta):
        self.b_gp.set_hyperparameter(log_length_scale, log_scale, log_beta)

    def get_m_hyperparameter(self):
        return self.b_gp.get_hyperparameter()

    def set_u(self, u):
        self.u = torch.tensor(u, dtype=self.dtype).view(-1)

    def set_prior(self, mean, cov):
        self.mean = mean
        self.cov = cov
        # self.lwr = [-0.5, -0.5]
        # self.upr = [1.5, 1.5]

    def plot_sample(self):
        # for i in range(self.dim_u):
        #     elements = self.sample[:, i]
        #     plt.scatter(range(0, self.mcmc.iter), elements)
        #     plt.show()
        first_elements = self.sample[:, 0]
        second_elements = self.sample[:, 1]
        plt.scatter(first_elements, second_elements)
        plt.show()
