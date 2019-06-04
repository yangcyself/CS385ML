import torch
import torchvision.transforms as transforms
import argparse, os, pickle, glob, cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.loss_function import select_loss_function
from utils.get_cdim import update_code_dim

from data.dataset import StanfordDog
from Solver import LinearSolver, ConvSolver

from models.simpleAE import simpleAE
from models.LinearVAE import LinearVAE
from models.ConvVAE import ConvVAE

from torchvision.datasets import MNIST
from scipy.stats import norm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='simpleAE', help='Choosing model')
parser.add_argument('--dataset', type=str, default='mnist', help='Choosing dataset')
parser.add_argument('--datasize', type=int, default=32, help='Choosing data size')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='n-history')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--max_epoch', type=int, default=100, help='max number of epochs to train for')
parser.add_argument('--optim', default='adam', choices=['adadelta', 'sgd', 'adam', 'rmsprop'],
                    help='choose an optimizer')

opt = parser.parse_args()
model_name = opt.model
datasize = opt.datasize
learning_rate = opt.learning_rate
max_epoch = opt.max_epoch
optimizer = opt.optim
hidden = opt.hidden_size
batch_size = opt.batch_size
dataset = opt.dataset

if dataset == 'mnist':
    channel = 1
    train_loader = torch.utils.data.DataLoader(dataset=MNIST('./data/{0}'.format(dataset), train=True, transform=transforms.Compose([transforms.Resize(datasize), transforms.ToTensor()])), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=MNIST('./data/{0}'.format(dataset), train=False, transform=transforms.Compose([transforms.Resize(datasize), transforms.ToTensor()])), batch_size=batch_size, shuffle=True)
elif dataset == 'dog':
    channel = 3
    train_loader = torch.utils.data.DataLoader(dataset=StanfordDog(root='./data', train=True, size=datasize), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=StanfordDog(root='./data', train=False, size=datasize), batch_size=batch_size, shuffle=True)

if model_name.lower() == 'simpleae':
    model = simpleAE(n_feature=32*32*channel, n_hidden=hidden, n_output=32*32*channel)
elif model_name.lower() == 'linearvae':
    model = LinearVAE(z_dim=hidden, hidden=512, num_channels=32*32*channel)
elif model_name.lower() == 'convvae':
    model = ConvVAE(c_dim=update_code_dim(128, 32, 4), z_dim=128, num_channels=channel)

params = model.parameters()
if optimizer.lower() == 'adam':
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8) # (beta1, beta2)
elif optimizer.lower() == 'sgd':
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.5)
elif optimizer.lower() == 'rmsprop':
    optimizer = torch.optim.RMSprop(params, lr=learning_rate)

print("---------Training Phase----------")
vae = model.cuda().train()
bce_loss = torch.nn.BCELoss(False)

for epoch in range(max_epoch):
    if epoch >= 1:
        print("\n[%2.2f]" % (time.time() - t0), end="\n")
    t0 = time.time()

    for step, (images, _) in enumerate(data_loader):
        if step >= 1:
            print("[%i] [%i] [%2.2f] [%2.2f]" % (epoch, step, time.time() - t1, (loss).data.cpu().numpy()), end="\r")
        t1 = time.time()

        batch_size = images.size(0)

        x = Variable(images.type(torch.cuda.FloatTensor))
        x_r = vae(x) # reconstruction

        loss_r = bce_loss(x_r, x) / batch_size
        loss_kl = torch.mean(.5 * torch.sum((vae.mu**2) + torch.exp(vae.log_sigma) - 1 - vae.log_sigma, 1))
        loss = loss_r + loss_kl
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    model.save()
