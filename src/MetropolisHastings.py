import random
from tqdm import tqdm
import numpy as np
import math


class MetropolisHastings:
    def __init__(
        self,
        random_walk,
        sample_nums,
        burn_ratio,
    ):
        self.iter = sample_nums
        self.burn_ratio = burn_ratio
        self.sample = None
        self.random_walk = random_walk

    def sampling(self, u, log_prior, log_likelihood) -> list:
        self.sample = [[] for _ in range(self.iter)]
        self.sample[0] = np.array(u)
        last_log_poster = log_prior(u) + log_likelihood(u)
        # bar = tqdm(range(1, self.iter), desc="MetropolisHastings")
        for i in range(1, self.iter):
            self.sample[i] = np.random.multivariate_normal(
                self.sample[i - 1], self.random_walk
            )

            prior = log_prior(self.sample[i])
            if math.isinf(prior):
                self.sample[i] = self.sample[i - 1]
                continue
            log_poster = prior + log_likelihood(self.sample[i])
            if random.random() > math.exp(log_poster - last_log_poster):
                self.sample[i] = self.sample[i - 1]
            else:
                last_log_poster = log_poster

        return np.array(self.sample[(int)(self.burn_ratio * self.iter) :])

    def set_random_walk(self, random_walk):
        self.random_walk = random_walk


# def test():
#     # from matplotlib import pyplot as plt
#     from scipy.stats import multivariate_normal

#     def log_prior(x):
#         return 0

#     def log_likelihood(x):
#         return multivariate_normal(mean=[1, 2], cov=[[2.0, 0.3], [0.3, 0.5]]).logpdf(x)

#     MH = MetropolisHastings(random_walk=[[0.1, 0], [0, 0.1]])
#     MH.sampling([0.1, 0.1], log_prior, log_likelihood)
#     first_elements = [row[0] for row in MH.sample]
#     second_elements = [row[1] for row in MH.sample]

#     plt.hist(first_elements, bins=60, density=True, alpha=0.6, color="g")
#     plt.show()
#     plt.hist(second_elements, bins=60, density=True, alpha=0.6, color="g")
#     plt.show()


# test()
