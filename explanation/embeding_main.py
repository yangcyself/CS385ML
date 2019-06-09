import argparse
import os
import shutil
import time

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

import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np


from copy import deepcopy
from gradCam.VAE_grad_cam import *
from gradCam.vis_util import *

sys.path.append('../classification')
from models.littleConv import littleConv

sys.path.append('../Reconstruction')
from  utils.get_cdim import update_code_dim
from data.dataset import StanfordDog


from embedding.embedor import convEmbeder


channel_num = 3
batchsize = 32
traindataset = StanfordDog(root='../Reconstruction/data/', train=True, already = False,size = 32)
train_loader = torch.utils.data.DataLoader(dataset= traindataset,batch_size=batchsize, shuffle=True)
# val_loader = torch.utils.data.DataLoader(dataset=StanfordDog(root='../Reconstruction/data/', train=True,  already = False), batch_size=batchsize, shuffle=True)
model = littleConv(c_dim=update_code_dim(128, 32, 4), z_dim=200 ,num_channels = channel_num)
model = model.cuda()
picReslu = 32
resume = "../classification/little_.checkpoint.pth.tar"
print("=> loading checkpoint '{}'".format(resume))
checkpoint = torch.load(resume)
start_epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {})"
  .format(resume, checkpoint['epoch']))

embeder = convEmbeder(model,["main.11"])


labels = []
embeds = []
for inpt, target in val_loader:
    embeder.refresh()
    target = target.cuda(non_blocking=True)
    inpt = inpt.cuda()

    _ = embeder.forward(inpt)
    ebds = embeder.generate("main.11").cpu().numpy()
    print(ebds.shape) # 32 * 128 * 1 * 1

    embeds.append(ebds.reshape(batch_size, 128))
    labels += target

LOG_DIR = 'logs'
metadata = os.path.join(LOG_DIR, 'metadata.tsv')
embeds = np.concatenate(embeds,axis = 0)



import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector



with open(metadata, 'w') as metadata_file: # record the metadata
    for row in labels:
        metadata_file.write('%d\n' % row)


# embeds = embeds
with tf.Session() as sess:
    saver = tf.train.Saver([embeds])

    # sess.run(images.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embeds"
    # Link this tensor to its metadata file (e.g. labels).
    # embedding.metadata_path = metadata
    embedding.metadata_path = 'metadata.tsv'
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

        
    








