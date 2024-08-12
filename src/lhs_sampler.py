from scipy.stats.qmc import LatinHypercube


class lhs_sampler:
    def __init__(self, n_samples: int, dimensions: int):
        super(lhs_sampler, self).__init__()
        self.sampler = LatinHypercube(dimensions)
        self.n_samples = n_samples

    def sampling(self) -> list:
        return self.sampler.random(self.n_samples)


# def test():
#     lhs = lhs_sampler(2)
#     print(type(lhs.sampling(100)))


# test()
