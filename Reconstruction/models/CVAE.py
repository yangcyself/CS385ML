import torch
from torch.autograd import Variable
import torch.nn as nn

from itertools import chain
import numpy as np
import time

class Encoder(nn.Module):
    """
    Class Encoder

    """
    def __init__(self, z_dim, hidden_size, num_channels, num_labels):
        """
        Initialization
        x -> Linear -> ReLU x 2 -> (mu, sigma) -> z

        """
        super(Encoder, self).__init__()

        self.main = nn.Sequential(
                    nn.Linear(num_channels + num_labels, hidden_size),
                    nn.ReLU(True)
                    )
        self.mu_ = nn.Linear(hidden_size, z_dim)
        self.sigma_ = nn.Linear(hidden_size, z_dim)

    def idx2onehot(self, idx, n):
        """
        Convert a label to a one-hot vector

        """
        if idx.dim() == 1:
            idx = idx.unsqueeze(1)

        onehot = torch.zeros(idx.size(0), n)
        onehot.scatter_(1, idx, 1)

        return onehot

    def forward(self, x, c):
        """
        Definition of forward process

        """
        c = self.idx2onehot(c, n=10)
        x = torch.cat((x, c), dim=-1)

        h = self.main(x)
        mu = self.mu_(h)
        sigma = self.sigma_(h)

        return mu, sigma

class Decoder(nn.Module):
    """
    Class Decoder

    """
    def __init__(self, z_dim, hidden_size, num_channels, num_labels):
        """
        Initialization
        z -> Linear -> ReLU -> Linear -> Sigmoid -> x_hat

        """
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
                    nn.Linear(z_dim + num_labels, hidden_size),
                    nn.ReLU(True),
                    nn.Linear(hidden_size, num_channels),
                    nn.Sigmoid()
                    )

    def idx2onehot(self, idx, n):
        """
        Convert a label to a one-hot vector

        """
        if idx.dim() == 1:
            idx = idx.unsqueeze(1)

        onehot = torch.zeros(idx.size(0), n)
        onehot.scatter_(1, idx, 1)

        return onehot

    def forward(self, x, c):
        """
        Definition of forward process

        """
        c = self.idx2onehot(c, n=10)
        x = torch.cat((x, c), dim=-1)
        x = self.main(x)
        return x

class CVAE(nn.Module):
    """
    Class Conditional-VAE containing Encoder & Decoder using Linear layers

    """
    def __init__(self, z_dim, hidden, num_channels, num_labels):
        """
        Initialization of CVAE

        """
        super(CVAE, self).__init__()

        self.Encoder = Encoder(z_dim, hidden, num_channels, num_labels)
        self.Decoder = Decoder(z_dim, hidden, num_channels, num_labels)
        self.z_dim = z_dim
        self.model_name = "CVAE"
        self.mu = None
        self.log_sigma = None

    def sample_from_q(self, mu, log_sigma):
        """
        VAE sample from Normal(mu, sigma)

        """
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False).type(torch.FloatTensor)
        sigma = torch.exp(log_sigma / 2)
        return mu + sigma * epsilon

    def forward(self, x, c):
        """
        Definition of forward process

        """
        self.mu, self.log_sigma = self.Encoder(x, c)
        z = self.sample_from_q(self.mu, self.log_sigma)
        return self.Decoder(z, c)

    def load(self, path):
        """
        load model from given path

        """
        self.load_state_dict(torch.load(path))

    def save(self, dataset, name=None):
        """
        save model to given path with time as name

        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_' + str(self.z_dim) + '_' + dataset + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name
