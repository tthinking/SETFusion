import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
import glob
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from Networks.Mymodules_out4_basicBloV5 import MODEL as net
# from CEL import CEL
from torch.utils.tensorboard import SummaryWriter
from losses import VIFF_Public,L1_loss


device = torch.device('cuda:0')
use_gpu = torch.cuda.is_available()

if use_gpu:
    print('GPU Mode Acitavted')
else:
    print('CPU Mode Acitavted')


def parse_args():
    parser = argparse.ArgumentParser()
    # 增加属性
    parser.add_argument('--name', default='', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)

    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--alpha', default=300, type=int,
                        help='number of new channel increases per depth (default: 300)')
    parser.add_argument('-w', '--wavename', default='haar', type=str,
                        help='wavename: haar, dbx, biorx.y, et al')
    args = parser.parse_args()
    return args



class AverageMeter(object):
    """Computes and stores the average and current value 计算并存储平均值和当前值"""
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


def train(args, train_loader_ir,train_loader_vi, model, criterion_viff,criterion_L1_loss,optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    losses_ir = AverageMeter()
    losses_vi = AverageMeter()
    losses_viff = AverageMeter()
    weight = args.weight

    writer = SummaryWriter()
    model.train()

    for i, (input, ir, vi) in tqdm(enumerate(train_loader_ir), total=len(train_loader_ir)):

        if use_gpu:
            input = input.cuda()

            ir = ir.cuda()
            vi = vi.cuda()

        else:
            input = input
            ir = ir
            vi = vi


        out = model(input)

        loss_ir = criterion_L1_loss(ir, out)
        loss_vi =  criterion_L1_loss(vi, out)
        loss_viff= criterion_viff(ir,vi,out)
        loss = loss_ir + loss_vi+loss_viff
        losses_ir.update(loss_ir.item(), input.size(0))
        losses_vi.update(loss_vi.item(), input.size(0))
        losses_viff.update(loss_viff.item(), input.size(0))
        losses.update(loss.item(), input.size(0))


        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader_ir) + i)

        if (i + 1) % 50 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}, Loss_ir: {:.4f}, Loss_vi: {:.4f}, Loss_viff: {:.4f}'
                  .format(epoch + 1, args.epochs, loss.item(), loss_ir.item(), loss_vi.item(), loss_viff.item()))

    writer.close()
    log = OrderedDict([
        ('loss', losses.avg),
        ('loss_ir', losses_ir.avg),
        ('loss_vi', losses_vi.avg),
        ('loss_viff', losses_viff.avg),
    ])

    return log



def main():
    args = parse_args()

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    cudnn.benchmark = True

    train_loader_ir = "..."

    train_loader_vi = ".../"
    model = net(in_channel=2)
    if use_gpu:
        model = model.cuda()
        model.cuda()

    else:
        model = model

    criterion_viff = VIFF_Public
    criterion_L1_loss = L1_loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps,weight_decay=args.weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    log = pd.DataFrame(index=[],
                       columns=['epoch',
                                'lr',
                                'loss',
                                'loss_ir',
                                'loss_vi',
                                'loss_viff',
                                ])

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch+1, args.epochs))

        train_log = train(args, train_loader_ir,train_loader_vi, model, criterion_viff,  criterion_L1_loss, optimizer, epoch)
        print('loss: %.4f - loss_ir: %.4f - loss_vi: %.4f  - loss_viff: %.4f'
              % (train_log['loss'],
                 train_log['loss_ir'],
                 train_log['loss_vi'],
                 train_log['loss_viff'],
                 ))

        tmp = pd.Series([
            epoch + 1,
            scheduler.get_lr()[0],
            train_log['loss'],
            train_log['loss_ir'],
            train_log['loss_vi'],
            train_log['loss_viff'],

        ], index=['epoch','lr', 'loss', 'loss_ir', 'loss_vi', 'loss_viff'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)

        scheduler.step()
        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), 'models/%s/model_{}.pth'.format(epoch+1) %args.name)


if __name__ == '__main__':
    main()


