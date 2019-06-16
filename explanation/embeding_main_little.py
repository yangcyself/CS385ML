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
from torchvision import models
from torch.utils.data.sampler import SubsetRandomSampler

import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np


from copy import deepcopy
from gradCam.VAE_grad_cam import *
from gradCam.vis_util import *

sys.path.append('../classification')
from models.littleConv import littleConv,littlePoolingConv

sys.path.append('../Reconstruction')
from  utils.get_cdim import update_code_dim
from data.dataset import StanfordDog


from embedding.embedor import convEmbeder
from gradCam.vis_util import cvt255




def create_sprite_image(images,size = None):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    if(size is None):
        img_h = images.shape[1]
        img_w = images.shape[2]
    else:
        img_h,img_w = size[0], size[1]

    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    
    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots,3))
    
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                if(size is not None):
                    this_img = cv2.resize(this_img,size)
                this_img = cvt255(this_img)
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w,:] = this_img
    
    return cvt255(spriteimage)


channel_num = 3
batchsize = 32
picReslu = 96
# traindataset = StanfordDog(root='../Reconstruction/data/', train=True, already = False,size = picReslu)
# train_loader = torch.utils.data.DataLoader(dataset= traindataset,batch_size=batchsize, shuffle=True)

model = littlePoolingConv(c_dim=update_code_dim(128, picReslu, 4), z_dim=200 ,num_channels = channel_num)
############## CUDA ###########
# model = model.cuda()
resume = "../classification/littlePoolingConvckpt_96/littlePoolingConv_epoch1_ckpt.pth.tar_1559877797"
# resume = "littleDropoutConv_epoch19_ckpt.pth.tar_1559983996"
# resume = "littleDropoutConv_epoch19_ckpt.pth.tar_1559983129"
print("=> loading checkpoint '{}'".format(resume))
checkpoint = torch.load(resume,  map_location='cpu')
start_epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {})"
  .format(resume, checkpoint['epoch']))


val_loader = torch.utils.data.DataLoader(dataset=StanfordDog(root='../Reconstruction/data/', train=True,  already = True), batch_size=batchsize, shuffle=True)

# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

# data_dir = "../classification/data/"
# dset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])

         
# val_loader = torch.utils.data.DataLoader(dset, batch_size=32,
#                                                shuffle=True, num_workers=4)
batch_size = 32
# model = models.resnet18(pretrained=True).cuda()

Name = None
for name, module in model.named_modules():
    print (name)
    # if(model is)
    Name = name
# Name = "layer4.1.conv2"
Name = "main.11"
embeder = convEmbeder(model,[Name])


labels = []
embeds = []
images = []
for inpt, target in val_loader:
    embeder.refresh()
    ##################### CUDA ##############
    # target = target.cuda(non_blocking=True)
    # target = target.cuda(non_blocking=True)
    # inpt = inpt.cuda()
    # inpt = inpt

    _ = embeder.forward(inpt)
    ebds = embeder.generate(Name).cpu().numpy()
    # print(ebds.shape) # 32 * 128 * 1 * 1

    embeds.append(ebds.reshape(*(ebds.shape[:2])))
    # print(target)
    # print(target.shape)
    labels.append (target.cpu().numpy())
    tmp = inpt.cpu().numpy().transpose(0,2,3,1)
    # print(tmp.dtype,tmp.min(),tmp.max())
    images.append( tmp)

LOG_DIR = 'logs'
metadata = os.path.join(LOG_DIR, 'metadata.tsv')
embeds = np.concatenate(embeds,axis = 0)
labels = np.concatenate(labels,axis = 0)
images = np.concatenate(images,axis = 0)


import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

path_for_sprites = "sprites.png"

# sprite = images_to_sprite(images).astype(np.uint8)
# plt.imsave(path_for_sprites,sprite)
sprite_image = create_sprite_image(images,size = (32,32))

plt.imsave(os.path.join(LOG_DIR, path_for_sprites),sprite_image)
# sprite.save('./oss_data/' + sprite_name)



with open(metadata, 'w') as metadata_file: # record the metadata
    for row in labels:
        metadata_file.write('%d\n' % row)


# embeds = embeds
with tf.Session() as sess:
    embeds = tf.Variable(embeds, name = "embeds") 
    saver = tf.train.Saver([embeds ])

    sess.run(embeds.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embeds"
    # Link this tensor to its metadata file (e.g. labels).
    # embedding.metadata_path = metadata
    embedding.metadata_path = 'metadata.tsv'
    embedding.sprite.image_path = path_for_sprites
    embedding.sprite.single_image_dim.extend([32,32])
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

        
    








