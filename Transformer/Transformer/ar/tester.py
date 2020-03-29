import torch
from statsmodels.tsa.arima_model import ARMA
import seaborn as sns

from Transformer.ar.ar_dataset import ARDataset
import matplotlib.pyplot as plt
import numpy as np

from Transformer.helpers import cuda_variable


class Tester:
    def __init__(self, flow):
        self.flow = flow
        self.flow.load()
        self.flow.to('cuda')

    def test_ar(self):
        # batch_size, length, d_model = 16, 32, 512
        batch_size = 1
        length = 128
        d_model = 8
        noise = cuda_variable(torch.randn(batch_size, length, d_model))
        # todo to remove
        # noise *= 0.5
        y = self.flow.inverse(y=noise)

        # x = x.permute(0, 2, 1).contiguous().view(batch_size * d_model, length)
        # x = x.permute(0, 2, 1).contiguous().view(batch_size * d_model * length)
        # y = y[:, :, 0].view(-1)
        y = y[0, :, 0].view(-1)

        y = y.detach().cpu().numpy()
        print(y.shape)

        self.plot_sequence(y=y)

        ar = ARMA(y, order=(1, 0))
        model_fit = ar.fit(trend='nc')
        print(model_fit.params)

    def plot_sequence(self, y):
        sns.set()
        x = np.linspace(0, seq_len, y.shape[0])
        plt.plot(x, y)
        plt.show()

        g, _, _ = ARDataset(phis=[0.9],
                            length=seq_len,
                            c=c, noise_dim=d_model).data_loaders(16)

        # y = next(g)[:, :, 0].view(-1)
        y = next(g)[0, :, 0].view(-1)
        y = y.detach().cpu().numpy()
        plt.plot(x, y)
        plt.show()

        ar = ARMA(y, order=(1, 0))
        model_fit = ar.fit(trend='nc')
print(model_fit.params)