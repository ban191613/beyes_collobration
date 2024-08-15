import os
import numpy as np
from cloud_koh import cloud_koh

from edge_koh_b import edge_koh_b
from torch.utils.tensorboard import SummaryWriter

# from matplotlib import pyplot as plt
from Disagreement import Disagreement
from Disagreement_st import Disagreement_st


class cloud_edge:
    def __init__(
        self,
        data_set,
        dim_x,
        dim_u,
        cloud_ym,
        cloud_m_num,
        cloud_u,
        cloud_iter,
        edge_u,
        edge_iter,
        prior_mean,
        prior_cov,
        prior_lower,
        prior_upper,
        mcmc_sample_num,
        random_walk,
        mcmc_br,
        gp_iter,
        gp_lr,
        active_learning,
        active_learning_num,
        filename,
    ):
        """边云协同整体算法

        Args:
            data_set (_type_): 数据集
            dim_x (_type_): 自变量维度
            dim_u (_type_): 参数维度
            cloud_ym (_type_): 机理模型
            cloud_m_num (_type_): 机理模型采样数量
            cloud_u (_type_): 云端初始值
            cloud_iter (_type_): 云端校正次数
            cloud_mcmc_sample_num (_type_): 云端MCMC次数
            edge_u (_type_): 边段初始值
            edge_iter (_type_): 边段迭代次数
            edge_mcmc_sample_num (_type_): 边段mcmc次数
            active_learning (_type_): 主动学习是否开启
            active_learning_num (_type_): 主动学习数据数量
        """
        super(cloud_edge, self).__init__()

        self.dim_x = dim_x
        self.dim_u = dim_u

        self.cloud_ym = cloud_ym
        self.cloud_u = cloud_u
        self.cloud_koh = cloud_koh(  # 云端模型初始化
            dim_x,
            cloud_m_num,
            cloud_u,
            random_walk=random_walk,
            mcmc_sample_nums=mcmc_sample_num,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            prior_lower=prior_lower,
            prior_upper=prior_upper,
            burn_ratio=mcmc_br,
        )

        self.edge_u = edge_u
        self.edge_koh = edge_koh_b(  # 边段残差模型初始化
            dim_x,
            edge_u,
            random_walk=random_walk,
            mcmc_sample_nums=mcmc_sample_num,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            prior_lower=prior_lower,
            prior_upper=prior_upper,
            burn_ratio=mcmc_br,
        )

        self.gp_iter = gp_iter
        self.gp_lr = gp_lr

        self.edge_iter = edge_iter

        self.data_set = data_set

        self.active_learning_num = active_learning_num

        self.JITTER = 1e-4

        # tensor board

        index = 0
        while os.path.exists(filename + str(index) if index > 0 else filename):
            index = index + 1
        self.writer = SummaryWriter(
            log_dir=(filename + str(index) if index > 0 else filename)
        )
        self.active_learning = None
        # 主动学习
        if active_learning == "Disagreement":
            self.active_learning = Disagreement(
                self.cloud_koh.m_gp, self.edge_koh.b_gp, self.edge_koh
            )
        elif active_learning == "Disagreement+":
            self.active_learning = Disagreement_st(
                self.cloud_koh.m_gp, self.edge_koh.b_gp, self.edge_koh
            )

    def cloud(self):
        self.cloud_koh.data_m(self.cloud_ym)  # 对机理模型进行抽样
        self.cloud_koh.ym_train(iter=self.gp_iter, lr=self.gp_lr)
        (f_x, f_y) = self.data_set.cloud_get()  # 获取数据
        self.cloud_koh.yb_train(f_x, f_y, iter=self.gp_iter, lr=self.gp_lr)
        self.cloud_koh.parameter_mcmc()
        # 求后验的参数均值和方差
        mean = np.mean(self.cloud_koh.sample, axis=0)
        cov = np.cov(self.cloud_koh.sample, rowvar=False) + self.JITTER * np.eye(
            self.dim_x
        )
        self.cloud_koh.set_u(mean)
        self.cloud_koh.set_prior(mean, cov)

        # self.cloud_koh.set_random_walk(cov)

        # self.cloud_koh.plot_sample()
        # 设置边缘侧的参数
        log_length_scale, log_scale, log_beta = self.cloud_koh.get_b_hyperparameter()
        self.edge_koh.set_b_hyperparameter(log_length_scale, log_scale, log_beta)
        return mean, cov

    def edge(self, i: int):
        (f_x, f_y) = self.data_set.edge_get(i)  # 获取数据
        if self.active_learning and i:
            f_x, f_y = self.active_learning.acquisition_function(
                None, None, f_x, f_y, self.edge_koh.mean, self.active_learning_num
            )
        self.edge_koh.yb_train(
            f_x, f_y, self.cloud_koh.m_gp, iter=self.gp_iter, lr=self.gp_lr
        )
        self.edge_koh.parameter_mcmc(
            self.cloud_koh.cov_m,
            self.cloud_koh.m_xu,
            self.cloud_koh.m_y,
            self.cloud_koh.sigma_m,
        )
        # 求后验的参数均值和方差
        mean = np.mean(self.edge_koh.sample, axis=0)
        cov = np.cov(self.edge_koh.sample, rowvar=False) + self.JITTER * np.eye(
            self.dim_x
        )
        self.edge_koh.set_u(mean)
        self.edge_koh.set_prior(mean, cov)

        # self.edge_koh.set_random_walk(cov)

        # self.edge_koh.plot_sample()

        return mean, cov

    def forward(self, cloud_iter: int):
        for i in range(self.edge_iter):
            if i % cloud_iter == 0:
                # 云端训练
                cloud_mean, cloud_cov = self.cloud()  # 云侧执行一次
            # 边端训练
            edge_mean, edge_cov = self.edge(i)

            # 预测
            self.pre(i)
            # 记录数据 并且画图
            self.writer.add_scalars(
                "parameter0",
                {"Edge": edge_mean[0], "Cloud": cloud_mean[0]},
                i,
            )
            self.writer.add_scalars(
                "parameter1",
                {"Edge": edge_mean[1], "Cloud": cloud_mean[1]},
                i,
            )
        self.writer.close()

    def pre(self, i):
        (edge_tre_x, edge_tre_y) = self.data_set.edge_pre(i)
        (edge_m_mean, edge_b_mean, edge_m_var, edge_b_var) = self.edge_koh.predict(
            edge_tre_x
        )
        edge_pre_mean = edge_m_mean + edge_b_mean
        edge_pre_var = edge_m_var + edge_b_var
        # 记录数据
        self.writer.add_scalars(
            "prediction",
            {
                "true": edge_tre_y,
                "predict": edge_pre_mean,
                "error": edge_tre_y - edge_pre_mean,
                "m_mean": edge_m_mean,
                "b_mean": edge_b_mean,
            },
            i,
        )
        self.writer.add_scalars(
            "prediction_var",
            {"var": edge_pre_var, "m_var": edge_m_var, "b_var": edge_b_var},
            i,
        )

    # def plot_cloud_edge(self):
    #     x = range(self.edge_iter)
    #     fig, axs = plt.subplots(6, 1, figsize=(20, 10))
    #     # 参数一
    #     cloud = [row[0] for row in self.record_cloud_mean]
    #     edge = [row[0] for row in self.record_edge_mean]
    #     axs[0].plot(x, cloud)
    #     axs[0].plot(x, edge)
    #     # 参数二
    #     cloud = [row[1] for row in self.record_cloud_mean]
    #     edge = [row[1] for row in self.record_edge_mean]
    #     axs[1].plot(x, cloud)
    #     axs[1].plot(x, edge)

    #     # 参数一方差
    #     cloud = [matrix[0] for matrix in self.record_cloud_cov]
    #     edge = [matrix[0] for matrix in self.record_edge_cov]
    #     axs[2].plot(x, cloud)
    #     axs[2].plot(x, edge)

    #     # 参数二方差

    #     cloud = [matrix[1, 1] for matrix in self.record_cloud_cov]
    #     edge = [matrix[1, 1] for matrix in self.record_edge_cov]
    #     axs[3].plot(x, cloud)
    #     axs[3].plot(x, edge)

    #     # m，b，预测
    #     edge_pre_m_mean = [row.item() for row in self.record_edge_pre_m_mean]
    #     edge_pre_m_var = [row.item() for row in self.record_edge_pre_m_var]
    #     edge_pre_b_mean = [row.item() for row in self.record_edge_pre_b_mean]
    #     edge_pre_b_var = [row.item() for row in self.record_edge_pre_b_var]
    #     edge_pre_mean = [row.item() for row in self.record_edge_pre_mean]
    #     edge_pre_var = [row.item() for row in self.record_edge_pre_var]

    #     edge_tre_y = [row.item() for row in self.record_edge_tre_y]
    #     edge_error = [t - p for t, p in zip(edge_tre_y, edge_pre_mean)]

    #     axs[4].errorbar(
    #         x,
    #         edge_pre_m_mean,
    #         edge_pre_m_var,
    #         label="Computational model value",
    #         fmt="g-",
    #         alpha=0.2,
    #     )
    #     axs[4].errorbar(
    #         x,
    #         edge_pre_b_mean,
    #         yerr=edge_pre_b_var,
    #         label="Computational model",
    #         fmt="b-",
    #         alpha=0.2,
    #     )

    #     axs[4].errorbar(
    #         x,
    #         edge_pre_mean,
    #         yerr=edge_pre_var,
    #         label="Predictive model",
    #         fmt="o-",
    #         alpha=0.2,
    #     )

    #     # 预测，真实，误差值
    #     axs[5].errorbar(
    #         x,
    #         edge_pre_mean,
    #         yerr=edge_pre_var,
    #         label="Predictive model",
    #         fmt="g-",
    #         alpha=0.2,
    #     )

    #     axs[5].plot(x, edge_tre_y, label="True value", color="orange")
    #     axs[5].plot(
    #         x,
    #         edge_error,
    #         label="Error value",
    #         color="red",
    #     )

    #     plt.show()
