import torch
import torchvision.transforms as transforms
import argparse, os, pickle, glob, cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.loss_function import select_loss_function
from Solver import LinearSolver
from models.simpleAE import simpleAE
from models.LinearVAE import LinearVAE
from torchvision.datasets import MNIST
from scipy.stats import norm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='simpleAE', help='Choosing dataset')
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
dataset = 'mnist'

train_loader = torch.utils.data.DataLoader(dataset=MNIST('./data/{0}'.format(dataset), train=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=MNIST('./data/{0}'.format(dataset), train=False, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)

if model_name == 'simpleAE':
    model = simpleAE(n_feature=28*28, n_hidden=hidden, n_output=28*28)
elif model_name == 'LinearVAE':
    model = LinearVAE(z_dim=256, hidden=512, num_channels=28*28)

loss_function = select_loss_function(loss_func)

params = model.parameters()
if optimizer.lower() == 'adam':
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8) # (beta1, beta2)
elif optimizer.lower() == 'sgd':
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.5)
elif optimizer.lower() == 'rmsprop':
    optimizer = torch.optim.RMSprop(params, lr=learning_rate)

checkpoints = glob.glob(pathname='checkpoints/{0}*'.format(model_name))
if len(checkpoints) != 0:
    model.load(path=checkpoints[0])

    noise = np.random.normal(size=(1, hidden))
    output = model.decoder(torch.from_numpy(noise).float()).detach().numpy().reshape(28, 28)
    # cv2.imshow("output", output)
    # cv2.waitKey(0)

    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    eg, _ = next(iter(train_loader))
    mu, sigma = model.encoder(eg.view(eg.shape[0], 1, -1))
    epsilon = torch.autograd.Variable(torch.randn(mu.size()), requires_grad=False).type(torch.FloatTensor)
    sigma = torch.exp(sigma / 2)
    z = (mu + sigma * epsilon).detach().numpy()[0, 0, :]

    output = model.decoder(torch.from_numpy(z).float()).detach().numpy().reshape(28, 28)
    plt.subplot(121)
    plt.imshow(output)
    plt.subplot(122)
    plt.imshow(eg[0, 0, :, :].numpy())
    plt.show()

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([xi, yi] + list(z[2:]))
            x_decoded = model.decoder(torch.from_numpy(z_sample).float()).detach().numpy()
            digit = x_decoded.reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
else:
    if model_name == 'simpleAE':
        solver = LinearSolver(model, loss_function, optimizer)
    elif model_name == 'LinearVAE':
        solver = LinearSolver(model, loss_function, optimizer)
    print(model)
    solver.train_and_decode(train_loader, test_loader, test_loader, max_epoch=max_epoch)
    print("training phase finished")
    model.save()
