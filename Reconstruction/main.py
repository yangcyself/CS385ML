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
parser.add_argument('--learning_rate', type=float, default=1e-3, help='n-history')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--max_epoch', type=int, default=100, help='max number of epochs to train for')
parser.add_argument('--loss_function', type=str, default='MSELoss', help='loss function')
parser.add_argument('--optim', default='adam', choices=['adadelta', 'sgd', 'adam', 'rmsprop'],
                    help='choose an optimizer')

opt = parser.parse_args()
model_name = opt.model
learning_rate = opt.learning_rate
max_epoch = opt.max_epoch
loss_func = opt.loss_function
optimizer = opt.optim
hidden = opt.hidden_size
batch_size = opt.batch_size
dataset = opt.dataset

if dataset == 'mnist':
    channel = 1
    train_loader = torch.utils.data.DataLoader(dataset=MNIST('./data/{0}'.format(dataset), train=True, transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()])), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=MNIST('./data/{0}'.format(dataset), train=False, transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()])), batch_size=batch_size, shuffle=True)
elif dataset == 'dog':
    channel = 3
    train_loader = torch.utils.data.DataLoader(dataset=StanfordDog(root='./data', train=True), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=StanfordDog(root='./data', train=False), batch_size=batch_size, shuffle=True)

if model_name.lower() == 'simpleae':
    model = simpleAE(n_feature=32*32*channel, n_hidden=hidden, n_output=32*32*channel)
elif model_name.lower() == 'linearvae':
    model = LinearVAE(z_dim=hidden, hidden=512, num_channels=32*32*channel)
elif model_name.lower() == 'convvae':
    model = ConvVAE(c_dim=update_code_dim(128, 32, 4), z_dim=128, num_channels=channel)

loss_function = select_loss_function(loss_func)

params = model.parameters()
if optimizer.lower() == 'adam':
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8) # (beta1, beta2)
elif optimizer.lower() == 'sgd':
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.5)
elif optimizer.lower() == 'rmsprop':
    optimizer = torch.optim.RMSprop(params, lr=learning_rate)

checkpoints = glob.glob(pathname='checkpoints/{0}_{1}_{2}*'.format(model_name, hidden, dataset))
if len(checkpoints) != 0:
    model.load(path=checkpoints[0])

    n = 15  # figure with 15x15 digits
    digit_size = 32
    figure = np.zeros((digit_size * n, digit_size * n, channel))
    figure1 = np.zeros((digit_size * n, digit_size * n))
    grid_x = np.linspace(-5, 5, n)
    grid_y = np.linspace(-5, 5, n)
    print(grid_x)

    eg, _ = next(iter(train_loader))
    print(eg.shape)
    if model.model_name == 'LinearVAE':
        mu, sigma = model.encoder(eg.view(eg.shape[0], 1, -1))
        epsilon = torch.autograd.Variable(torch.randn(mu.size()), requires_grad=False).type(torch.FloatTensor)
        sigma = torch.exp(sigma / 2)
        print(sigma.shape)
        z = (mu + sigma * epsilon).detach().numpy()[0, 0, :]
        print(z)
    elif model.model_name == 'ConvVAE':
        mu, sigma = model.Encoder(eg)
        epsilon = torch.autograd.Variable(torch.randn(mu.size()), requires_grad=False).type(torch.FloatTensor)
        sigma = torch.exp(sigma / 2)
        print(sigma.shape)
        z = (mu + sigma * epsilon).detach().numpy()[0, :].reshape(1, -1)

    output = model.decoder(torch.from_numpy(z).float()).reshape([channel, 32, 32])
    print(output.shape)
    plt.subplot(121)
    plt.imshow(transforms.ToPILImage()(output[:, :, :]).convert('RGB'))
    plt.subplot(122)
    plt.imshow(transforms.ToPILImage()(eg[0, :, :, :]).convert('RGB'))
    plt.show()

    if model.model_name == 'LinearVAE':
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                print(xi, yi)
                z_sample = np.array([xi, yi] + list(z[2:]))
                print(z_sample)
                x_decoded = model.decoder(torch.from_numpy(z_sample).float()).reshape([channel, 32, 32])
                digit = x_decoded.detach().numpy().reshape(32, 32)
                # plt.imshow(digit)
                # plt.show()
                figure1[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit
    elif model.model_name == 'ConvVAE':
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([xi, yi] + list(z[0, 2:])).reshape(1, -1)
                x_decoded = model.Decoder(torch.from_numpy(z_sample).float()).reshape([channel, 32, 32])
                digit = transforms.ToPILImage()(x_decoded[:, :, :]).convert('RGB')
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size, :] = digit

    if dataset == 'mnist':
        plt.figure(figsize=(10, 10))
        plt.imshow(figure1, cmap='gray_r')
        plt.show()
    elif dataset == 'dog':
        plt.figure(figsize=(10, 10))
        plt.imshow(figure)
        plt.show()
else:
    if model_name.lower() == 'simpleae' or model_name.lower() == 'linearvae':
        solver = LinearSolver(model, loss_function, optimizer, channel)
    else:
        solver = ConvSolver(model, loss_function, optimizer)

    print(model)
    solver.train_and_decode(train_loader, test_loader, test_loader, max_epoch=max_epoch)
    print("training phase finished")
    model.save(dataset=dataset)
