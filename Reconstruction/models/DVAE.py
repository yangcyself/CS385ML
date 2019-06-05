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
    def __init__(self, z_dim, hidden_size, num_channels):
        """
        Initialization
        x -> Linear -> ReLU x 2 -> (mu, sigma) -> z

        """
        super(Encoder, self).__init__()

        self.main = nn.Sequential(
                    nn.Linear(num_channels, hidden_size),
                    nn.ReLU(True)
                    )
        self.mu_ = nn.Linear(hidden_size, z_dim)
        self.sigma_ = nn.Linear(hidden_size, z_dim)

    def forward(self, x):
        """
        Definition of forward process

        """
        h = self.main(x)
        mu = self.mu_(h)
        sigma = self.sigma_(h)

        return mu, sigma

class Decoder(nn.Module):
    """
    Class Decoder

    """
    def __init__(self, z_dim, hidden_size, num_channels):
        """
        Initialization
        z -> Linear -> ReLU -> Linear -> Sigmoid -> x_hat

        """
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
                    nn.Linear(z_dim, hidden_size),
                    nn.ReLU(True),
                    nn.Linear(hidden_size, num_channels),
                    nn.Sigmoid()
                    )
    def forward(self, x):
        """
        Definition of forward process

        """
        x = self.main(x)
        return x

class DVAE(nn.Module):
    """
    Class VAE containing Encoder & Decoder using Linear layers

    """
    def __init__(self, z_dim, hidden, num_channels, std=0.01):
        """
        Initialization of DVAE

        """
        super(DVAE, self).__init__()

        self.Encoder = Encoder(z_dim, hidden, num_channels)
        self.Decoder = Decoder(z_dim, hidden, num_channels)
        self.z_dim = z_dim
        self.model_name = "DVAE"
        self.mu = None
        self.log_sigma = None
        self.std = std

    def sample_from_q(self, mu, log_sigma):
        """
        VAE sample from Normal(mu, sigma)

        """
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False).type(torch.FloatTensor)
        sigma = torch.exp(log_sigma / 2)
        return mu + sigma * epsilon

    def forward(self, x):
        """
        Definition of forward process

        """
        x += Variable(torch.randn(x.size()) * self.std, requires_grad=False).type(torch.FloatTensor)
        self.mu, self.log_sigma = self.Encoder(x)
        z = self.sample_from_q(self.mu, self.log_sigma)
        return self.Decoder(z)

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
