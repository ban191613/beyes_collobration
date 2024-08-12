import torch
from gaussian_process import gaussian_process
import numpy as np

# from kernel import kernel_rbf


class Disagreement:
    def __init__(self, ym: gaussian_process, b: gaussian_process, koh_b):
        super(Disagreement, self).__init__()
        self.ym = ym
        self.b = b
        self.koh_b = koh_b

    def compute_sigma(self, basic_data_x, choose_data_x, u):
        """预测分布的最大熵

        Args:
            x_r (_type_): basic_data
            x (_type_): choose_data

        Returns:
            _type_: _description_
        """
        u = torch.tensor(u)
        basic_data_xu = torch.hstack(
            (basic_data_x, torch.ones((basic_data_x.shape[0], len(u))) * u)
        )
        choose_data_xu = torch.hstack(
            (choose_data_x, torch.ones((choose_data_x.shape[0], len(u))) * u)
        )
        sigma = self.ym.max_entropy_predicate(
            basic_data_xu, choose_data_xu
        ) + self.b.max_entropy_predicate(basic_data_x, choose_data_x)

        return sigma

    def calculate_multivariate_gaussian_entropy(self, covariance_matrix, dimension=1):
        """计算多维度高斯分布的熵。

        Args:
            dimension (float): 随机变量的维度数 d
            covariance_matrix (float): 协方差矩阵 Sigma。

        Returns:
            float: 多维度高斯分布的熵。
        """ """
        """
        # 熵的计算公式: H(X) = (d/2) * log(2*pi*e) + (1/2) * log(|Sigma|)
        # 注意：log在这里使用自然对数ln
        # entropy = 0.5 * dimension * np.log(2 * np.pi * np.e) + 0.5 * np.log(
        #     np.linalg.det(covariance_matrix)
        # )
        # 这里y只有一维情况
        entropy = 0.5 * np.log(2 * np.pi * np.e * covariance_matrix)  # 一次算n个
        return entropy

    def max_entropy(self, basic_data_x, choose_data_x, u):
        cov = self.compute_sigma(basic_data_x, choose_data_x, u)
        entropy = self.calculate_multivariate_gaussian_entropy(cov)
        return entropy

    def conditional_entropy(self, x, u):
        # p(y|x,u)
        cov = self.koh_b.predict_y_condition_x_u(x, u, self.ym)
        entropy = self.calculate_multivariate_gaussian_entropy(cov)
        return entropy

    def expectation_conditional_entropy(self, choose_data_x, n=10):
        # 3  取n个u 请平均
        arr_u = np.random.multivariate_normal(self.koh_b.mean, self.koh_b.cov, size=n)
        _sum_entropy = torch.zeros((choose_data_x.shape[0], 1))
        for u in arr_u:
            _sum_entropy += self.conditional_entropy(choose_data_x, u)
        return _sum_entropy / arr_u.shape[0]

    def compute_criteria(self, basic_data_x, choose_data_x, u):
        return self.max_entropy(
            basic_data_x, choose_data_x, u
        ) - self.expectation_conditional_entropy(choose_data_x)

    def acquisition_function(
        self,
        basic_data_x,
        basic_data_y,
        choose_data_x,
        choose_data_y,
        u,
        choose_num: int,
    ):
        train_data_x = basic_data_x
        train_data_y = basic_data_y
        for _ in range(choose_num):
            m_criteria = self.compute_criteria(train_data_x, choose_data_x, u)
            m_index = torch.argmax(m_criteria)
            train_data_x = torch.vstack((train_data_x, choose_data_x[m_index, :]))
            train_data_y = torch.vstack((train_data_y, choose_data_y[m_index, :]))
            choose_data_x = torch.cat(
                (choose_data_x[:m_index, :], choose_data_x[m_index + 1 :, :])
            )
            choose_data_y = torch.cat(
                (choose_data_y[:m_index, :], choose_data_y[m_index + 1 :, :])
            )
        return train_data_x, train_data_y
