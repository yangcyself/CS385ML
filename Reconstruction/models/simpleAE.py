import torch
import time
import torch.nn.functional as F
from itertools import chain

class Encoder(torch.nn.Module):
    """
    Encoder class

    """
    def __init__(self, n_feature, n_hidden):
        super(Encoder, self).__init__()
        self.main = torch.nn.Sequential(
                    torch.nn.Linear(n_feature, n_hidden),
                    torch.nn.ReLU()
                    )

    def forward(self, x):
        """
        Definition of forward process

        """
        x = self.main(x)
        return x

class Decoder(torch.nn.Module):
    """
    Decoder class

    """
    def __init__(self, n_feature, n_output):
        super(Decoder, self).__init__()
        self.main = torch.nn.Sequential(
                    torch.nn.Linear(n_feature, n_output),
                    torch.nn.ReLU()
                    )

    def forward(self, x):
        """
        Definition of forward process

        """
        x = self.main(x)
        return x

class simpleAE(torch.nn.Module):
    """
    A simple Auto Encoder model with very easy structure as follow
    x -> dense -> z -> dense -> x_hat

    """
    def __init__(self, n_feature, n_hidden, n_output):
        """
        Initialization with encoder & decoder
        n_feature : input feature dimension
        n_hidden : hidden size of the network which indicates the dimension of z
        n_output : output feature dimension, here is equal to n_feature for Auto-encoder

        """
        super(simpleAE, self).__init__()
        self.Encoder = Encoder(n_feature, n_hidden)
        self.Decoder = Decoder(n_hidden, n_output)
        self.model_name = 'simpleAE'

    def parameters(self):
        """
        Getting parameters

        """
        return chain(self.Encoder.parameters(), self.Decoder.parameters())

    def forward(self, x):
        """
        Definition of forward process

        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
