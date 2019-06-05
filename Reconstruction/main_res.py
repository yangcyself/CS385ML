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
from models.ResVAE import ResVAE

from torchvision.datasets import MNIST
from scipy.stats import norm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist', help='Choosing dataset')
parser.add_argument('--datasize', type=int, default=64, help='Choosing data size')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='n-history')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--max_epoch', type=int, default=100, help='max number of epochs to train for')
parser.add_argument('--loss_function', type=str, default='MSELoss', help='loss function')
parser.add_argument('--optim', default='adam', choices=['adadelta', 'sgd', 'adam', 'rmsprop'],
                    help='choose an optimizer')

opt = parser.parse_args()
datasize = opt.datasize
learning_rate = opt.learning_rate
max_epoch = opt.max_epoch
loss_func = opt.loss_function
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

model = ResVAE(z_dim=hidden, num_channels=channel)

loss_function = select_loss_function(loss_func)

params = model.parameters()
if optimizer.lower() == 'adam':
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8) # (beta1, beta2)
elif optimizer.lower() == 'sgd':
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.5)
elif optimizer.lower() == 'rmsprop':
    optimizer = torch.optim.RMSprop(params, lr=learning_rate)

solver = ConvSolver(model, loss_function, optimizer)

print(model)
solver.train_and_decode(train_loader, test_loader, test_loader, max_epoch=max_epoch)
print("training phase finished")
model.save(dataset=dataset)
