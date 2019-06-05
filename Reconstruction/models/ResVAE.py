import torch
import torch.nn as nn
import numpy as np
import cv2, pickle, os, time
import time
import numpy as np

from itertools import chain
from torch.autograd import Variable
from torch.nn import functional as F

class ResizeConv2d(nn.Module):
    """
    Upsampling the network instead of using ConvTranspose2d
    because of the checkerboard effect

    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        # print(x.shape)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        return x

class ResidualBlockDec(nn.Module):
    """
    Residual Block for Decoder

    """
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlockDec, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            ResizeConv2d(inchannel, outchannel, 3, stride),
            nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        # print(self.left)
        # print(self.right)
        # print(out.shape, residual.shape)
        out += residual
        return F.relu(out)

class ResidualBlockEnc(nn.Module):
    """
    Residual Block for Encoder

    """
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlockEnc, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=False))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class Encoder(nn.Module):
    """
    Class Encoder with ResNet architecture

    """
    def __init__(self, c_dim=(512, 2, 2), z_dim=256, num_channels=3, std=0.02):
        """
        Initialization

        """
        super(Encoder, self).__init__()
        # b x 3 x 64 x 64
        self.pre = nn.Sequential(
            nn.Conv2d(num_channels, 64, 7, 2, 3, bias=False), # b x 64 x 32 x 32
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)) # b x 64 x 16 x 16

        self.layer1 = self._make_layer(64, 128, 2) # b x 128 x 16 x 16
        self.layer2 = self._make_layer(128, 256, 2, stride=2) # b x 256 x 8 x 8
        self.layer3 = self._make_layer(256, 512, 2, stride=2) # b x 512 x 4 x 4
        self.layer4 = self._make_layer(512, 512, 2, stride=2) # b x 512 x 2 x 2

        self.c_dim = c_dim
        self.z_dim = z_dim
        self.std = std
        self.num_channels = num_channels
        self.linear_mu = nn.Linear(int(np.prod(c_dim)), z_dim)
        self.sigma = nn.Linear(int(np.prod(c_dim)), z_dim)
        self.init_weights()

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        """
        Constructing Layers

        """
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel))

        layers = []
        layers.append(ResidualBlockEnc(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlockEnc(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        h = self.layer4(x)
        # print(h.shape)
        h = h.resize(h.size(0), h.size(1) * h.size(2) * h.size(3))

        return self.linear_mu(h), self.sigma(h)

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


class Decoder(nn.Module):
    """
    Class Decoder with ResNet architecture

    """
    def __init__(self, c_dim=(512, 2, 2), z_dim=256, num_channels=3, std=0.02):
        """
        Initialization

        """
        super(Decoder, self).__init__()
        # b x 1 x 256(z_dim)
        self.pre = nn.Sequential(
            nn.Linear(z_dim, int(np.prod(c_dim))),
            nn.ReLU(),
            nn.Linear(int(np.prod(c_dim)), int(np.prod(c_dim))),
            nn.ReLU()) # b x 512 x 2 x 2

        self.layer1 = self._make_layer(512, 512, 2, stride=2) # b x 512 x 4 x 4
        self.layer2 = self._make_layer(512, 256, 2, stride=2) # b x 256 x 8 x 8
        self.layer3 = self._make_layer(256, 128, 2, stride=2) # b x 128 x 16 x 16
        self.layer4 = self._make_layer(128, 64, 2) # b x 64 x 16 x 16

        self.main = ResizeConv2d(64, num_channels, kernel_size=3, scale_factor=2) # b x 3 x 32 x 32

        self.c_dim = c_dim
        self.z_dim = z_dim
        self.std = std
        self.num_channels = num_channels
        self.linear_mu = nn.Linear(int(np.prod(c_dim)), z_dim)
        self.sigma = nn.Linear(int(np.prod(c_dim)), z_dim)
        self.init_weights()

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        """
        Constructing Layers

        """
        shortcut = nn.Sequential(
            ResizeConv2d(inchannel, outchannel, 3, stride),
            nn.BatchNorm2d(outchannel))

        layers = []
        layers.append(ResidualBlockDec(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlockDec(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = x.resize(x.size(0), *(self.c_dim))
        x = F.interpolate(x, scale_factor=2)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        h = self.layer4(x)
        # print(h.shape)
        h = self.main(h)
        # print(h.shape)
        return h

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

class ResVAE(nn.Module):
    """
    Class VAE with ResNet architecture

    """
    def __init__(self, c_dim=(512, 2, 2), z_dim = 256, num_channels = 3, std=0.02):
        """
        Initialization

        """
        super(ResVAE, self).__init__()
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.num_channels = num_channels
        self.std = std
        self.gpu_avaliable = torch.cuda.is_available()

        self.model_name = "ResVAE"
        self.Encoder = Encoder(c_dim, z_dim, num_channels, std)
        self.Decoder = Decoder(c_dim, z_dim, num_channels, std)

    def parameters(self):
        """
        Overwrite parameters

        """
        return chain(self.Encoder.parameters(), self.Decoder.parameters())

    def sample_from_q(self, mu, log_sigma):
        """
        Simulation of sample process

        """
        if self.gpu_avaliable and isinstance(mu, torch.cuda.FloatTensor):
            epsilon = Variable(torch.randn(mu.size()), requires_grad=False).type(torch.FloatTensor).cuda()
        else:
            epsilon = Variable(torch.randn(mu.size()), requires_grad=False).type(torch.FloatTensor)
        sigma = torch.exp(log_sigma / 2)
        return mu + sigma * epsilon

    def forward(self, x):
        """
        Definition of forward process

        """
        self.mu, self.log_sigma = self.Encoder(x)
        z = self.sample_from_q(self.mu, self.log_sigma)
        #print(z)
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

if __name__ == '__main__':
    model = Decoder()
    a = np.zeros((1, 256))
    a = torch.from_numpy(a).float()
    print(model(a).shape)
