import torch
from torch.autograd import Variable
import torch.nn as nn

from itertools import chain
import numpy as np

class Encoder(torch.nn.Module):
    """
    Class encoder

    """
    def __init__(self, c_dim, z_dim=256, num_channels, std=0.02):
        """
        Initialization
        conv -> ReLU x 4 -> (mu, sigma)

        """
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
			nn.Conv2d(num_channels, 64, 4, 2, 1, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(64),
			nn.Conv2d(64, 128, 4, 2, 1, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(128),
			nn.Conv2d(128, 256, 4, 2, 1, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(256),
			nn.Conv2d(256, 128, 4, 2, 1, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(128)
			)

        self.c_dim = c_dim
        self.z_dim = z_dim
        self.num_channels = num_channels
        self.std = std
        self.linear_mu = nn.Linear(int(np.prod(c_dim)), z_dim)
		self.sigma = nn.Linear(int(np.prod(c_dim)), z_dim)
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
		return self.linear_mu(h), self.sigma(h)

class Decoder(nn.Module):
	"""
	Class decoder

	"""
	def __init__(self, c_dim, z_dim, num_channels, std):
        """
        Initialization
        (mu, sigma) -> Linear -> ReLU x 2 -> conv -> ReLU x 4 -> sigmoid

        """
		super(Decoder, self).__init__()
		self.main_1 = nn.Sequential(
			nn.Linear(z_dim, int(prod(c_dim))),
			nn.ReLU(),
			nn.Linear(int(prod(c_dim)), int(prod(c_dim))),
			nn.ReLU()
			)
		self.main_2 = nn.Sequential(
			nn.ConvTranspose2d(128, 256, 4, 2, 1, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(256),
			nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(128),
			nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(64),
			nn.ConvTranspose2d(64, num_channels, 4, 2, 1, bias=False),
			nn.Sigmoid()
			)

        self.c_dim = c_dim
        self.z_dim = z_dim
        self.num_channels = num_channels
        self.std = std
		self.init_weights()

	def init_weights(self):
        """
        Weight Initialization

        """
		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0.0, self.std)
				m.bias.data.normal_(0.0, self.std)
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				m.weight.data.normal_(0.0, self.std)

	def forward(self, input):
        """
        Definition of forward process

        """
		h = self.main_1(input)
		h = h.resize(input.size(0), *self.c_dim)
		x = self.main_2(h)
		return x

class ConvVAE(nn.Module):
	"""
	Class VAE containing encoder & decoder using Conv layers

	"""
	def __init__(self, c_dim, z_dim, num_channels, std):
        """
        Initialization of VAE
        x -> encoder -> (mu, sigma) -> decoder -> x_hat

        """
		super(ConvVAE, self).__init__()
		self.c_dim = c_dim
        self.z_dim = z_dim
        self.num_channels = num_channels
        self.std = std

        self.model_name = "ConvVAE"
		self.Encoder = Encoder(c_dim, z_dim, num_channels, std)
		self.Decoder = Decoder(c_dim, z_dim, num_channels, std)

	def parameters(self):
        """
        Getting parameters

        """
		return chain(self.Encoder.parameters(), self.Decoder.parameters())

	def sample_from_q(self, mu, log_sigma_sq):
        """
        VAE sample from Normal(mu, sigma)

        """
		epsilon = Variable(torch.randn(mu.size()), requires_grad=False).type(torch.FloatTensor)
		sigma = torch.exp(log_sigma_sq / 2)
		return mu + sigma * epsilon

	def forward(self, input):
        """
        Definition of forward process

        """
		self.mu, self.log_sigma_sq = self.Encoder(input)
		z = self.sample_from_q(self.mu, self.log_sigma_sq)
		return self.Decoder(z)

    def load(self, path):
        """
        load model from given path

        """
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """
        save model to given path with time as name

        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name
