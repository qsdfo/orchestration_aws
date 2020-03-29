import numpy as np
import torch


class ARDataset:
    def __init__(self, phis, length, c=0):
        # 0.9
        # 0.3
        # 0.3 0.3
        # 0.9 0.8
        self.phis = phis
        self.c = c
        self.p = len(phis)
        self.length = length

    def generator(self, batch_size, d_model=512):
        while True:
            X = [np.random.randn(batch_size, 1, 1)]
            for j in range(1, self.length):
                Y = 0
                for i in range(max(j - self.p, 0), j):
                    Y += self.phis[j - i - 1] * X[i]
                Y += self.c
                Y += np.random.randn(batch_size, 1, 1)
                X.append(Y)
            X = np.concatenate(X, axis=1)
            X = torch.from_numpy(X).float()

            # X = X.repeat(1, 1, d_model)
            # X += torch.randn_like(X) * 0.01
            yield X

    def data_loaders(self, batch_size):
        return (self.generator(batch_size),
                self.generator(batch_size),
                self.generator(batch_size))

