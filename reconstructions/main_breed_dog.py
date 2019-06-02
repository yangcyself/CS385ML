import torch, time
import torchvision.transforms as transforms
import argparse, os, pickle, glob, cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.loss_function import select_loss_function
from utils.get_cdim import update_code_dim
from PIL import Image

from data.dataset import StanfordDog
from Solver import LinearSolver, ConvSolver

from models.simpleAE import simpleAE
from models.LinearVAE import LinearVAE
from models.ConvVAE import ConvVAE

from torchvision.datasets import MNIST
from scipy.stats import norm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='ConvVAE', help='Choosing model')
parser.add_argument('--datasize', type=int, default=32, help='Choosing data size')
parser.add_argument('--breed', type=str, default='Chihuahua', help='Choosing dog breed')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='n-history')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--max_epoch', type=int, default=100, help='max number of epochs to train for')
parser.add_argument('--loss_function', type=str, default='MSELoss', help='loss function')
parser.add_argument('--optim', default='adam', choices=['adadelta', 'sgd', 'adam', 'rmsprop'],
                    help='choose an optimizer')

opt = parser.parse_args()
model_name = opt.model
datasize = opt.datasize
learning_rate = opt.learning_rate
max_epoch = opt.max_epoch
loss_func = opt.loss_function
optimizer = opt.optim
breed = opt.breed
hidden = opt.hidden_size
batch_size = opt.batch_size

channel = 3
dataset = StanfordDog(root='./data', train=True)
imgs = np.array(dataset.getBreed('Chihuahua'))
print(imgs.shape)
data = []
for img in imgs:
    # print(type(img))
    tmp = cv2.resize(np.array(img), (32, 32))
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(tmp)
    image = transforms.ToTensor()(image)
    data.append(image)
# print(imgs[0])

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

# vae = model
# vae.load(path='./checkpoints/ConvVAE_128_dog_0602_21:03:54.pth')
# test = data[0].reshape(1, 3, 32, 32)
# mu, sigma = vae.Encoder(test)
# epsilon = torch.autograd.Variable(torch.randn(mu.size()), requires_grad=False).type(torch.FloatTensor)
# sigma = torch.exp(sigma / 2)
# z = (mu + sigma * epsilon).detach().numpy()[0, :].reshape(1, -1)
#
# output = model.Decoder(torch.from_numpy(z).float()).reshape([channel, 32, 32])
# print(output.shape, type(output))
# plt.subplot(121)
# plt.imshow(transforms.ToPILImage()(output.squeeze()).convert('RGB'))
# plt.subplot(122)
# plt.imshow(transforms.ToPILImage()(test.squeeze()).convert('RGB'))
# plt.show()

print("---------Training Phase----------")
vae = model.train()
print(vae)
bce_loss = torch.nn.BCELoss(size_average=False)

for epoch in range(max_epoch):
    if epoch >= 1:
        print("\n[%2.2f]" % (time.time() - t0), end="\n")
    t0 = time.time()

    for step, (images) in enumerate(data):
        if step >= 1:
            print("[%i] [%i] [%2.2f] [%2.2f]" % (epoch, step, time.time() - t1, (loss).data.detach().numpy()), end="\r")
        t1 = time.time()

        images = torch.autograd.Variable(images.reshape(1, 3, 32, 32))
        batch_size = images.size(0)

        x = torch.autograd.Variable(images.type(torch.FloatTensor))
        x_r = vae(x) # reconstruction

        loss_r = bce_loss(x_r, x)
        loss_kl = torch.mean(.5 * torch.sum((vae.mu**2) + torch.exp(vae.log_sigma) - 1 - vae.log_sigma, 1))
        # print(loss_r, loss_kl)
        loss = loss_r + loss_kl
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

model.save("dog")
