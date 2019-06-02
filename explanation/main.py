from VAE_grad_cam import *
import sys
sys.path.append("../Reconstruction/")
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import shutil
import time

import argparse, os, pickle, glob, cv2
import matplotlib.pyplot as plt
import numpy as np


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
# import torch.utils.data
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

from utils.loss_function import select_loss_function
from utils.get_cdim import update_code_dim

from data.dataset import StanfordDog
from Solver import LinearSolver, ConvSolver

from models.simpleAE import simpleAE
from models.LinearVAE import LinearVAE
from models.ConvVAE import ConvVAE

from torchvision.datasets import MNIST
from scipy.stats import norm


channel = 3
model = ConvVAE(c_dim=update_code_dim(128, 32, 4), z_dim=128, num_channels=channel)
model = model.cuda()

checkpoints = glob.glob(pathname='checkpoints/{0}_{1}*'.format("ConvVAE", "dog"))
if len(checkpoints) != 0:
    model.load(path=checkpoints[0])

def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image

images = []
raw_images = []
print("Images:")
image_paths = ["../Reconstruction/data/Images/n02090622-borzoi/n02090622_3615.jpg"]#,"cat_dog.png"]
for i, image_path in enumerate(image_paths):
    print("\t#{}: {}".format(i, image_path))
    image, raw_image = preprocess(image_path)
    images.append(image)
    raw_images.append(raw_image)

images = torch.stack(images).cuda()

bp = BackPropagation(model=model)
out = bp.forward(images)
bp.backward(pos = np.array([[20,20]]), size = (10,10))
gradients = bp.generate()
gradient = gradients[0]
gradient = gradient.cpu().numpy().transpose(1, 2, 0)
gradient -= gradient.min()
gradient /= gradient.max()
gradient *= 255.0
plt.imshow(gradient)
plt.show()

