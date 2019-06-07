import torch
import torch.nn as nn
import time
import numpy as np

from itertools import chain
from torch.autograd import Variable

class littleConv(torch.nn.Module):
    """
    Class encoder

    """
    def __init__(self, c_dim, z_dim=200, num_channels=1, std=0.02):
        """
        Initialization
        conv -> ReLU x 4 -> (mu, sigma)

        """
        super(littleConv, self).__init__()
        self.main = nn.Sequential(
			nn.Conv2d(num_channels, 64, 4, 2, 1, bias=False), # b x 64 x 14 x 14
			nn.ReLU(True),
			nn.BatchNorm2d(64),
			nn.Conv2d(64, 128, 4, 2, 1, bias=False), # b x 128 x 7 x 7
			nn.ReLU(True),
			nn.BatchNorm2d(128),
			nn.Conv2d(128, 256, 4, 2, 1, bias=False), # b x 256 x 3 x 3
			nn.ReLU(True),
			nn.BatchNorm2d(256),
			nn.Conv2d(256, 128, 4, 2, 1, bias=False), # b x 128 x 1 x 1
			nn.ReLU(True),
			nn.BatchNorm2d(128)
			)

        self.c_dim = c_dim
        self.z_dim = z_dim
        self.num_channels = num_channels
        self.std = std
        self.linear= nn.Linear(int(np.prod(c_dim)), z_dim)
        self.init_weights()

    def init_weights(self):
        """
        Weight Initialization

        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, self.std)
                m.bias.data.normal_(0.0, self.std)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, self.std)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, self.std)
                m.bias.data.zero_()

    def forward(self, input):
        """
        Definition of forward process

        """
        h = self.main(input)
        h = h.resize(h.size(0), h.size(1) * h.size(2) * h.size(3))
        return self.linear(h)


class littlePoolingConv(torch.nn.Module):
    """
    Class encoder

    """
    def __init__(self, c_dim, z_dim=200, num_channels=1, std=0.02):
        """
        Initialization
        conv -> ReLU x 4 -> (mu, sigma)

        """
        super(littlePoolingConv, self).__init__()
        self.main = nn.Sequential(
			nn.Conv2d(num_channels, 64, 3, 1, 1, bias=True), # b x 64 x 14 x 14
			nn.ReLU(True),
            nn.MaxPool2d(2),
			nn.BatchNorm2d(64),
			nn.Conv2d(64, 128, 3, 1, 1, bias=False), # b x 128 x 7 x 7
			nn.ReLU(True),
            nn.MaxPool2d(2),
			nn.BatchNorm2d(128),
			nn.Conv2d(128, 256, 3, 1, 1, bias=False), # b x 256 x 3 x 3
			nn.ReLU(True),
            nn.MaxPool2d(2),
			nn.BatchNorm2d(256),
			nn.Conv2d(256, 128, 3, 1, 1, bias=False), # b x 128 x 1 x 1
			nn.ReLU(True),
            nn.MaxPool2d(2),
			nn.BatchNorm2d(128)
			)

        self.c_dim = c_dim
        self.z_dim = z_dim
        self.num_channels = num_channels
        self.std = std
        self.linear= nn.Linear(int(np.prod(c_dim)), z_dim)
        self.init_weights()

    def init_weights(self):
        """
        Weight Initialization

        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, self.std)
                m.bias.data.normal_(0.0, self.std)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, self.std)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, self.std)
                m.bias.data.zero_()

    def forward(self, input):
        """
        Definition of forward process

        """
        h = self.main(input)
        h = h.resize(h.size(0), h.size(1) * h.size(2) * h.size(3))
        return self.linear(h)

