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
from torchvision.datasets import MNIST


from models.littleConv import littleConv,littlePoolingConv
import sys
sys.path.append('../Reconstruction')
from  utils.get_cdim import update_code_dim
from data.dataset import StanfordDog



QUICK = False

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training - Stochastic Downsampling')

parser.add_argument("--train_path", type=str, default="~/dataset/vgg100")
parser.add_argument("--test_path", type=str, default="~/dataset/vgg100")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=115, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-vb', '--val-batch-size', default=1024, type=int,
                    metavar='N', help='validation mini-batch size (default: 1024)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-r', '--resolution', default=96, type=int,
                    metavar='R', help='resolution of picture')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=400, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-o', '--nodownsample', dest='nodownsample', action='store_true',
                    help='turn off the down sampling function of the model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('-m','--message', default='', type=str,
                    help='the message used for naming the outputfile')
parser.add_argument('--val-results-path', default='val_results.txt', type=str,
                    help='filename of the file for writing validation results')
parser.add_argument("--torch_version", dest="torch_version", action="store", type=float, default=0.4)


args = parser.parse_args()

DEBUG = True

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after the 40th, 75th, and 105th epochs"""
    lr = args.lr
    for e in [40,75,105]:
        if epoch >= e:
            lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    # dataset=DataSet(torch_v=args.torch_version)
    ############### USE MINST AS Dataset #######################
    # channel_num = 1
    # train_loader = torch.utils.data.DataLoader(dataset=MNIST('~/dataset/Mnist', train=True, transform=transforms.Compose([transforms.Resize(64),transforms.ToTensor()]) ,download=True), batch_size=args.batch_size, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(dataset=MNIST('~/dataset/Mnist', train=False, transform=transforms.Compose([transforms.Resize(64),transforms.ToTensor()]),download=True), batch_size=args.batch_size, shuffle=True)
    # train_loader = dataset.loader(args.train_path,batch_size = args.batch_size)
    # val_loader = dataset.test_loader(args.test_path,batch_size = args.batch_size)
    ################# USE STANFORD DOGS ########################
    # channel_num = 3
    # resolution = 96
    # train_loader = torch.utils.data.DataLoader(dataset=StanfordDog(root='../Reconstruction/data/', size = resolution , train=True, already = True),batch_size=args.batch_size, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(dataset=StanfordDog(root='../Reconstruction/data/', size = resolution , train=False,  already = True),batch_size=args.batch_size, shuffle=True)
    ################# RANDOM DATA GEN for DOGS ########################
    traindir = 'data/train'
    valdir = 'data/validation'
    resolution = 96
    channel_num = 3
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # transforms.RandomAffine(1, translate=(0.1,0.1), scale=(0.9,1.1), shear=0.2, resample=False),
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(True),
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    ################# Tiny Image Net ########################
    # channel_num = 3
    # train_loader = torch.utils.data.DataLoader(dataset=TinyImageNet('/home/ycy/dataset/tiny-imagenet-200/', train=True), batch_size=args.batch_size, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(dataset=TinyImageNet('/home/ycy/dataset/tiny-imagenet-200/', train=False), batch_size=args.batch_size, shuffle=True)


    #model = pipeNet(100).cuda() # The Second argument of pipenet Changes Channel Wise DS rate
    # model = cdsresnext50(inputChannels = channel_num, dsProbability = 0.75).cuda()
    # model = resnext50(inputChannels = 1).cuda()
    # model = littleConv(c_dim=update_code_dim(128, resolution, 4), z_dim=200 ,num_channels = channel_num)
    model = littlePoolingConv(c_dim=update_code_dim(128, resolution, 4), z_dim=200 ,num_channels = channel_num)
    print("model: littlePoolingConv")
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            

 

    if(args.evaluate):
        validate(train_loader,val_loader,model,criterion,None,None)
    else:
        main_train(args,optimizer,train_loader,val_loader,model,criterion)
def main_train(args,optimizer,train_loader,val_loader,model,criterion):
    special_epochs = [1,3,5,7,9,11,13,15,17,19]
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        validate(train_loader,val_loader,model,criterion,None,None)
        if(epoch in special_epochs):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            },"littleDropoutConvckpt_%d/littleDropoutConv_epoch%d_ckpt.pth.tar_%d"%(args.resolution, epoch,int(time.time())))

def save_checkpoint(state, filename='{}_{}.checkpoint.pth.tar'.format("little",args.message)):
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input = input.cuda()
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    if(DEBUG):
        return

def validate(train_loader, val_loader, model, criterion, blockID, ratio,):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input )
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
            if(QUICK):
                break

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg

if __name__ == '__main__':
    main()
    
